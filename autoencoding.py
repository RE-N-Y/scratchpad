import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Integer, Array

import optax
import numpy as onp
import equinox
from equinox import nn, static_field, Module
from einops import rearrange, reduce, repeat, pack
from layers import (
    Convolution,
    Layernorm,
    Projection,
    MLP,
    GLU,
    SelfAttention,
    Sequential,
    SinusodialEmbedding,
    Embedding,
)
from toolkit import *
from dataloader import *

load = equinox.tree_deserialise_leaves
save = equinox.tree_serialise_leaves

class Transformer(Module):
    attention: SelfAttention
    mlp: GLU
    prenorm: Layernorm
    postnorm: Layernorm

    def __init__(
        self,
        features: int,
        heads: int = 12,
        bias = False,
        dropout: float = 0,
        key=None,
    ):
        key = RNG(key)
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(
            features, heads=heads, dropout=dropout, bias=bias, key=next(key)
        )
        self.mlp = GLU(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x: Float[Array, "n d"], key=None):
        key = RNG(key)
        x = self.attention(self.prenorm(x), key=next(key)) + x
        x = self.mlp(self.postnorm(x), key=next(key)) + x

        return x


class Decoder(Module):
    layers: Sequential
    embedding:Embedding
    wpe:jnp.ndarray
    layernorm:Layernorm
    head:Projection


    def __init__(
        self,
        features: int,
        vocab: int = 8192,
        layers: int = 24,
        dropout: float = 0,
        bias=False,
        key=None,
    ):
        key = RNG(key)
        self.embedding = Embedding(vocab, features, key=next(key))
        self.wpe = jr.normal(next(key), (256, features))
        self.layers = Sequential(
            [
                Transformer(features, dropout=dropout, bias=bias, key=next(key))
                for _ in range(layers)
            ]
        )
        self.layernorm = Layernorm([features])
        self.head = Projection(features, vocab, bias=bias, key=next(key))

    def masking(self, confidences, masklen, tau=1, key=None):
        confidences = jnp.log(confidences) + tau * jr.gumbel(key, confidences.shape)
        sorted = jnp.sort(confidences, axis=-1)
        cutoff = sorted[...,masklen]
        return confidences < repeat(cutoff, 'b -> b 256')


    def generate(self, n:int, T:int=16, tau=0.7, key=None):
        key = RNG(key)
        
        offset = 1
        x = jr.randint(next(key), (n, 256), minval=0, maxval=8192)
        masks = jnp.zeros((n, 256), dtype=int)
        schedule = lambda r : math.cos(0.5 * math.pi * r)
        masklens = [ [t/T, math.ceil(schedule(t/T) * 256)] for t in range(offset, T + offset)]
       
        for ratio, masklen in tqdm(masklens):
            logits = self(x, masks, jr.split(next(key),n))
            tokens = jr.categorical(next(key), logits)

            p = jax.nn.softmax(logits, axis=-1)
            p, tokens = rearrange(p, 'b n c -> (b n) c'), rearrange(tokens, 'b n -> (b n)')
            confidences = batch(lambda x,idx:x[idx])(p, tokens)
            tokens, confidences = rearrange(tokens, '(b n) -> b n', b=n), rearrange(confidences, '(b n) -> b n', b=n)
            
            # confidences = jnp.where(masks == 0, confidences, jnp.inf)
            x = jnp.where(masks == 0, tokens, x)
            masks = jnp.where(self.masking(confidences, masklen, tau=(1-ratio)*tau, key=next(key)), 0, 1)
            
        return x

    @forward
    def __call__(self, x: Integer[Array, "n"], masks: Integer[Array, "n"], key=None):
        x = self.embedding(x)
        x = jnp.where(rearrange(masks == 1, 'n -> n 1'), x, jnp.zeros(x.shape, x.dtype))
        x = x + self.wpe
        x = self.layers(x, key=key)
        x = self.layernorm(x)
        x = self.head(x)

        return x

import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image


def schedule(ratio):
    return jnp.cos(0.5 * jnp.pi * ratio)


@batch
def masks(idxes, key=None):
    key = RNG(key)
    ratio = schedule(jr.uniform(next(key)))
    masks = jr.uniform(next(key), idxes.shape)
    masks = jnp.where(masks > ratio, 1, 0)  # 1 = NO MASK, 0 = MASK
    masks = masks.at[0].set(0)
    masks = jr.permutation(next(key), masks)

    return masks

@batch
def accuracy(logits, labels, masks):
    imasks = 1 - masks
    logits = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(jnp.where(logits == labels, 1, 0) * imasks)
    total = jnp.sum(imasks)
    return correct / total


@click.command()
@click.option("--dataset", type=Path)
@click.option("--compressor", type=Path)
@click.option("--steps", default=256, type=int)
@click.option("--warmup", default=32, type=int)
@click.option("--cooldown", type=float, default=224)
@click.option("--lr", type=float, default=1e-4)
@click.option("--batch", default=256, type=int)
@click.option("--features", type=int, default=768)
@click.option("--vocab", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=24)
@click.option("--dropout", type=float, default=0)
@click.option("--seed", type=int, default=42)
@click.option("--workers", default=32)
@click.option("--precision", default="half")
def train(**cfg):
    wandb.init(project="MASKGIT", config=cfg)
    folder = Path(f"checkpoints/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(cfg["seed"])
    key = RNG(key)
    dsplit = lambda key: jr.split(key, jax.device_count())

    def tform(sample):
        return onp.array(sample)

    loader, length = dataloader(
        "hub://reny/animefaces", 
        tensors=["16x16"], 
        batch_size=cfg["batch"],
        transform={"16x16":tform},
        num_workers=cfg["workers"],
        buffer_size=8192,
        shuffle=True
    )

    G = Decoder(cfg["features"], cfg["vocab"], cfg["depth"], cfg["dropout"], key=next(key))
    
    cfg["warmup"], cfg["steps"], cfg["cooldown"] = cfg["warmup"] * length, cfg["steps"] * length, cfg["cooldown"] * length
    grads = partial(gradients, precision=cfg["precision"])
    lr = optax.warmup_cosine_decay_schedule(0, cfg["lr"], cfg["warmup"], cfg["steps"], cfg["cooldown"])
    optimisers = optax.lion(lr, b1=0.95, b2=0.98, weight_decay=0.1)
    states = optimisers.init(parameters(G))

    G, states = replicate(G), replicate(states)

    @ddp
    def Gstep(G, batch, states, key=None):
        key = RNG(key)

        @grads
        def cross_entropy_loss(G, batch):
            m = masks(batch, jr.split(next(key), len(batch)))
            logits = G(batch, m, jr.split(next(key), len(batch)))
            labels = jax.nn.one_hot(batch, cfg["vocab"])
            labels = optax.smooth_labels(labels, 0.1)
            loss = optax.softmax_cross_entropy(logits, labels) * (1 - m)
            loss = jnp.sum(loss) / jnp.sum(1 - m)

            return loss, { "accuracy" : accuracy(logits, batch, m), "masking" : 1 - m }

        (loss, metrics), gradients = cross_entropy_loss(G, batch)
        updates, states = optimisers.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, loss, metrics

    for idx in tqdm(range(cfg["steps"])):
        batch = next(loader)
        G, states, Gloss, metrics = Gstep(G, batch["16x16"], states, dsplit(next(key)))
        wandb.log({
            "loss":onp.mean(Gloss),
            "accuracy":onp.mean(metrics["accuracy"]),
            "masking":onp.mean(metrics["masking"])
        })

        if idx % 4096 == 0:
            save(folder / "G.weight", unreplicate(G))
            save(folder / "states.ckpt", unreplicate(states))

    wandb.finish()


import compression as Q

@click.command()
@click.option("--denoiser", type=Path)
@click.option("--compressor", type=Path)
@click.option("--seed", type=int, default=42)
def generate(denoiser:Path, compressor:Path, seed:int=42):
    seed = jr.PRNGKey(seed)
    key = RNG(seed)
    wandb.init(project="MASKGIT")

    VQ = Q.VQVAE(features=768, pages=8192, depth=12, dropout=0., bias=False, size=256, key=next(key))
    VQ = load(compressor / "G.weight", VQ)
    G = Decoder(features=768, vocab=8192, layers=24, dropout=0., key=next(key))
    G = load(denoiser / "G.weight", G)

    tokens = G.generate(n=64, T=12, tau=4, key=next(key))
    images = VQ.decode(tokens, jr.split(next(key), 64))
    wandb.log({ "samples": [wandb.Image(i) for i in Q.t2i(images)] }) 
    wandb.finish()

if __name__ == "__main__":
    generate()
