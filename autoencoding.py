import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Integer, Bool, Array

import optax
import numpy as onp
import equinox
from equinox import nn, static_field as buffer, Module
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


class Encoder(Module):
    layers: Sequential
    embedding:Embedding
    wpe:jnp.ndarray
    layernorm:Layernorm
    length:int = buffer()
    drops:int = buffer()
    cls:int = buffer()
    mask:int = buffer()
    vocab:int = buffer()

    def __init__(
        self,
        length:int=256,
        droprate:int=0.5,
        features:int=1024,
        heads:int=16,
        vocab:int=8192,
        layers:int=24,
        dropout:float=0,
        bias=False,
        key=None,
    ):
        CLS = MASK = 1
        key = RNG(key)
    
        self.drops = int(length * droprate)
        self.length = length
        self.vocab = vocab
        self.cls, self.mask = vocab, vocab + 1
        self.embedding = Embedding(CLS + MASK + vocab, features, key=next(key))
        self.wpe = jr.normal(next(key), (CLS + self.length - self.drops, features))
        self.layers = Sequential([Transformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(layers)])
        self.layernorm = Layernorm([features])

    def sampler(self, x:Integer[Array,"n"], key=None):
        key = RNG(key)
        CLS, MASK = jnp.array([self.cls]), jnp.array([self.mask])
        dtype = self.wpe.dtype

        mr = 0.25 * jr.truncated_normal(next(key), -0.2, 1.8, dtype=dtype) + 0.55
        rolls = jr.uniform(next(key), [self.drops], minval=0.5, maxval=1, dtype=dtype)
        idxes = jr.permutation(next(key), jnp.arange(self.length))
        drops, keeps = idxes[:self.drops], idxes[self.drops:]
        masks = rolls < mr
        
        x = x[keeps]
        x = jnp.where(masks, MASK, x) 
        x = jnp.concatenate([CLS, x]) 

        return x, masks, keeps, drops         


    def __call__(self, x: Integer[Array, "n"], key=None):
        key = RNG(key)

        x, masks, keeps, drops = self.sampler(x, next(key))
        x = self.embedding(x)
        x = x + self.wpe
        
        x = self.layers(x, key=next(key))
        x = self.layernorm(x)

        return x, masks, keeps, drops

class Decoder(Module):
    layers:Sequential
    projection:Projection
    wpe:jnp.ndarray
    layernorm:Layernorm
    head:Projection
    length:int=buffer()
    features:int=buffer()


    def __init__(
        self,
        length:int=256,
        channels:int=1024,
        features:int=512,
        heads:int=8,
        vocab:int=8192,
        layers:int=8,
        dropout:float=0,
        bias=False,
        key=None,
    ):
        key = RNG(key)
        self.features = features
        self.length = length
        self.projection = Projection(channels, features, bias=bias, key=next(key))
        self.wpe = jr.normal(next(key), (256, features))
        self.layers = Sequential([Transformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(layers)])
        self.layernorm = Layernorm([features])
        self.head = Projection(features, vocab, bias=bias, key=next(key))

    def __call__(self, x:Integer[Array,"n"], masks:Bool[Array,"n"], keeps:Integer[Array,"n"], drops:Integer[Array,"n"], key=None):
        key = RNG(key)
        x = self.projection(x)
        CLS, x = x[0], x[1:]

        slate = jnp.zeros((self.length, self.features), dtype=x.dtype)
        masks = repeat(masks, 'n -> n d', d=self.features)
        kept = jnp.where(masks, CLS, x)

        slate = slate.at[keeps].set(kept)
        slate = slate.at[drops].set(CLS)
        hits = jnp.all(slate == CLS, axis=-1)

        x = slate + self.wpe
        x = self.layers(x, key=next(key))
        x = self.layernorm(x)
        x = self.head(x)

        return x, hits

class MAGE(Module):
    encoder:Encoder
    decoder:Decoder

    def __init__(self, features:int=1024, heads:int=16, vocab:int=8192, layers:int=24, dropout:float=0, bias=False, key=None):
        key = RNG(key)
        self.encoder = Encoder(key=next(key))
        self.decoder = Decoder(key=next(key))

    @forward
    def __call__(self, x:Integer[Array,"n"], key=None):
        key = RNG(key)
        x, masks, keeps, drops = self.encoder(x, next(key))
        x = self.decoder(x, masks, keeps, drops, next(key))
        return x


import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image


def sample(ratio):
    return jnp.cos(0.5 * jnp.pi * ratio)


@batch
def masks(idxes, key=None):
    key = RNG(key)
    ratio = sample(jr.uniform(next(key)))
    masks = jr.uniform(next(key), idxes.shape)
    masks = jnp.where(masks > ratio, 1, 0)  # 1 = NO MASK, 0 = MASK

    return masks

@batch
def accuracy(logits, labels, masks):
    logits = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(jnp.where(logits == labels, 1, 0) * masks)
    total = jnp.sum(masks)
    return correct / total


@click.command()
@click.option("--dataset", type=Path)
@click.option("--compressor", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=4096, type=int)
@click.option("--cooldown", type=float, default=1e-6)
@click.option("--lr", type=float, default=3e-4)
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

    loader = dataloader(
        "hub://reny/animefaces", 
        tensors=["16x16"], 
        batch_size=cfg["batch"],
        transform={"16x16":tform},
        num_workers=cfg["workers"], 
        shuffle=True
    )

    G = MAGE(key=next(key))

    grads = partial(gradients, precision=cfg["precision"])
    optimisers = optax.adabelief(cfg["lr"])
    states = optimisers.init(G)

    G, states = replicate(G), replicate(states)

    @ddp
    def Gstep(G, batch, states, key=None):
        key = RNG(key)
        
        @grads
        def cross_entropy_loss(G, batch):
            logits, hits = G(batch, jr.split(next(key), len(batch)))
            labels = jax.nn.one_hot(batch, cfg["vocab"])
            labels = optax.smooth_labels(labels, 0.1)
            loss = optax.softmax_cross_entropy(logits, labels) * hits
            loss = loss.sum() / hits.sum()
            return loss, {}

        (loss, metrics), gradients = cross_entropy_loss(G, batch)
        updates, states = optimisers.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, loss, metrics

    for idx in tqdm(range(cfg["steps"])):
        batch = next(loader)
        G, states, Gloss, metrics = Gstep(G, batch["16x16"], states, dsplit(next(key)))
        wandb.log({"loss": onp.mean(Gloss)})

        if idx % 4096 == 0:
            save(folder / "G.weight", unreplicate(G))
            save(folder / "states.ckpt", unreplicate(states))

    wandb.finish()


if __name__ == "__main__":
    train()
