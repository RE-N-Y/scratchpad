import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Integer, Array

import optax
import numpy as onp
import equinox
from equinox import nn, static_field as buffer, filter_jit as jit, Module
from einops import rearrange, reduce, repeat, pack
from compression import VQGAN
from layers import Embedding, SinusodialEmbedding, Projection, Selformer, SelfAttention, CrossAttention, Layernorm, MLP, Crossformer
from toolkit import *
from dataloader import *


class Timeformer(Module):
    attention:SelfAttention
    time:Projection
    prenorm:Layernorm
    postnorm:Layernorm
    mlp:MLP

    def __init__(self, features:int, time:int, heads:int=12, dropout=0., bias=True, key=None):
        key = RNG(key)
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.time = Projection(time, features, bias=bias, key=next(key))
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], t:Float[Array, "t"], key=None):
        akey, okey = jr.split(key)
        x = self.attention(self.prenorm(x) + self.time(t), key=akey) + x
        x = self.mlp(self.postnorm(x) + self.time(t), key=okey) + x

        return x

class Transformer(Module):
    embedding:Embedding
    wpe:jnp.ndarray
    tpe:SinusodialEmbedding
    encoder:list
    fusion:Crossformer
    decoder:list
    layernorm:Layernorm
    cls:Projection
    features:int
    time:int

    def __init__(self, vocab:int=8192, length:int=1024, features:int=768, time:int=512, heads:int=12, depth:int=12, dropout=0., bias=True, key=None):
        key = RNG(key)
        self.features, self.time = features, time
        self.embedding = Embedding(vocab, features, key=next(key))
        self.wpe = jnp.zeros((length, features))
        self.tpe = SinusodialEmbedding(time, base=16, key=next(key))

        self.encoder = [Timeformer(features, time, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.fusion = Crossformer(features, features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.decoder = [Timeformer(features, time, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.layernorm = Layernorm([features])
        self.cls = Projection(features, vocab, bias=bias, key=next(key))

    @forward
    def __call__(self, x, t, masks=None, key=None):
        key = RNG(key)

        time = self.tpe(t)
        hiddens = self.embedding(x) * rearrange(masks, 'n -> n 1')
        hiddens = hiddens + self.wpe
        
        for layer in self.encoder : hiddens = layer(hiddens, time, key=next(key))
        hiddens = self.fusion(hiddens, hiddens, key=next(key))
        for layer in self.decoder : hiddens = layer(hiddens, time, key=next(key))
        logits = self.cls(self.layernorm(hiddens))

        return logits

class GPT(Module):
    embedding:Embedding
    sos:jnp.ndarray
    wpe:jnp.ndarray
    encoder:list
    layernorm:Layernorm
    cls:Projection
    length:int

    def __init__(self, vocab:int=8192, length:int=1024, features:int=768, heads:int=12, depth:int=12, dropout=0., bias=True, key=None):
        key = RNG(key)
        self.length = length
        self.embedding = Embedding(vocab, features, key=next(key))
        self.sos = jr.normal(next(key), (features,))
        self.wpe = jnp.zeros((length, features))
        self.encoder = [Selformer(features, heads=heads, dropout=dropout, causal=True, bias=bias, key=next(key)) for _ in range(depth)]
        self.layernorm = Layernorm([features])
        self.cls = Projection(features, vocab, bias=bias, key=next(key))

    @forward
    def __call__(self, tokens, key=None):
        key = RNG(key)
        hiddens = self.embedding(tokens)
        hiddens, _ = pack([self.sos, hiddens[:-1]], '* d')
        hiddens = hiddens + self.wpe[:len(hiddens)]
        for layer in self.encoder : hiddens = layer(hiddens, key=next(key))
        logits = self.cls(self.layernorm(hiddens))

        return logits

def adjust(model, depth=12, scale=.02, key=None):
    key = RNG(key)
    target = lambda c,m : isinstance(m,c)

    surgery = lambda module : [scale * jr.normal(next(key), module.weight.shape)]
    model = initialise(model, partial(target, Projection), ["weight"], surgery)

    surgery = lambda module : [module.output.weight / math.sqrt(2)]
    model = initialise(model, partial(target, MLP), ["output.weight"], surgery)

    surgery = lambda module : [module.out.weight / math.sqrt(2)]
    model = initialise(model, partial(target, (SelfAttention, CrossAttention)), ["out.weight"], surgery)
    
    return model

import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image

def schedule(ratio): return jnp.cos(.5 * jnp.pi * ratio)

@batch
def sampling(logits, tau=1, k=128, key=None):
    _, idxes = jax.lax.top_k(-logits, k=len(logits)-k)
    logits = logits.at[idxes].set(-jnp.inf)
    return jr.categorical(key, logits / tau)
    
def generate(G, batch=4, tau=1, key=None):
    key = RNG(key)
    tokens = jnp.zeros((batch, G.length), dtype=int)
    for idx in range(G.length):
        logits = jit(G)(tokens, key=next(key))
        samples = sampling(logits[:,idx], tau=tau, key=next(key))
        tokens = tokens.at[:,idx].set(samples)

    return tokens

@click.command()
@click.option("--compressor", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=10000)
@click.option("--lr", type=float, default=6e-6)
@click.option("--batch", default=4, type=int)
@click.option("--size", default=256, type=int)
@click.option("--patch", default=8, type=int)
@click.option("--length", type=int, default=1024)
@click.option("--features", type=int, default=768)
@click.option("--vocab", type=int, default=8192)
@click.option("--heads", type=int, default=12)
@click.option("--depth", type=int, default=12)
@click.option("--bias", type=bool, default=False)
@click.option("--dropout", type=float, default=0)
@click.option("--label-smoothing", type=float, default=0.1)
@click.option("--precision", type=str, default="single")
def train(**cfg):
    wandb.init(project = "autogressive", config = cfg)
    folder = Path(f"autoregressive/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(42)
    key = RNG(key)

    dataset = Path("dataset/megabooru")
    augmentations = [
        T.Resize((cfg["size"], cfg["size"])),
        T.RandomApply([T.RandomResizedCrop((cfg["size"], cfg["size"]), scale=(0.7,1.0))]), 
        T.RandomHorizontalFlip(0.2), 
        T.RandomAdjustSharpness(2,0.3), 
        T.RandomAutocontrast(0.3)
    ]

    dataset = ImageDataset(dataset, augmentations=augmentations)
    loader = dataloader(dataset, cfg["batch"])
    
    ntoken = cfg["size"]//cfg["patch"]
    C = VQGAN.load(cfg["compressor"], size=cfg["size"], patch=cfg["patch"], pages=cfg["vocab"], depth=4, bias=False, key=next(key))
    G = GPT(cfg["vocab"], ntoken ** 2, features=cfg["features"], heads=cfg["heads"], depth=cfg["depth"], dropout=cfg["dropout"], bias=cfg["bias"], key=next(key))
    G = adjust(G, depth=cfg["depth"], key=next(key))

    grads = partial(gradients, precision=cfg["precision"])
    optimisers = optax.adamw(cfg["lr"])
    states = optimisers.init(G)

    G, states = replicate(G), replicate(states)

    @ddp
    def Gstep(G, images, states, key=None):
        key = RNG(key)
        @grads
        def loss(G, C, images):
            _, _, tokens = C.encode(images, key=next(key))

            # labels
            labels = jax.nn.one_hot(tokens, cfg["vocab"])
            labels = optax.smooth_labels(labels, cfg["label_smoothing"])

            # cross entropy
            logits = G(tokens, key=next(key))
            ce = optax.softmax_cross_entropy(logits, labels)

            # accuracy
            predictions = jr.categorical(next(key), logits)
            accuracy = predictions == tokens

            return ce.mean(), { "ce":ce.mean(), 'accuracy':accuracy.mean() }

        (l, metrics), gradients = loss(G, C, images)
        updates, states = optimisers.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    for idx in tqdm(range(cfg["steps"])):
        batch = next(loader)
        every = lambda n : idx % n == 0

        G, states, Gloss, metrics = Gstep(G, batch, states, key=next(key))
        if every(256): wandb.log({ "loss":onp.mean(Gloss), "accuracy":onp.mean(metrics["accuracy"]) })
        if every(256):
            tokens = generate(unreplicate(G), tau=1, key=next(key))
            tensors = C.decode(C.lookup(tokens), key=next(key))
            tensors = rearrange(tensors, 'b h w c -> h (b w) c')
            dataset.tensor_to_image(tensors).save("sample.png")
    
    wandb.finish()

if __name__ == "__main__":
    train()
