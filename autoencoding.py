import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Integer, Array

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
    head:Projection
    dr:int = buffer()
    cls:int = buffer()
    mask:int = buffer()
    vocab:int = buffer()

    def __init__(
        self,
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
    
        self.dr = 128
        self.vocab = vocab
        self.cls, self.mask = vocab, vocab + 1
        self.embedding = Embedding(CLS + MASK + vocab, features, key=next(key))
        self.wpe = jr.normal(next(key), (CLS + 256, features))
        self.layers = Sequential([Transformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(layers)])
        self.layernorm = Layernorm([features])

    def sampler(self, x:Integer[Array,"n"], key=None):
        key = RNG(key)
        mr = 0.25 * jr.truncated_normal(next(key), -0.2, 1.8, dtype=x.dtype) + 0.55
        idxes = jr.permutation(next(key), jnp.arange(len(x)))
        
        drops = idxes[:self.dr]
        masks = idxes[self.dr:]
        
        rolls = jnp.linspace(0.5, 1, len(masks), dtype=x.dtype)
        keeps = masks[rolls > mr]
        masks = masks[rolls < mr]
        masks = jnp.concatenate([drops, masks])

        x = x.at[masks].set(self.mask)
        x = x[keeps]

        CLS = jnp.array([self.cls])
        x = jnp.concatenate([CLS, x])

        return x, masks, keeps, drops         


    @forward
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


    def __init__(
        self,
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
        self.projection = Projection(channels, features, bias=bias, key=next(key))
        self.wpe = jr.normal(next(key), (256, features))
        self.layers = Sequential([Transformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(layers)])
        self.layernorm = Layernorm([features])
        self.head = Projection(features, vocab, bias=bias, key=next(key))

    @forward
    def __call__(self, x:Integer[Array,"n"], masks:Integer[Array,"n"], keeps:Integer[Array,"n"], key=None):
        key = RNG(key)
        x = self.projection(x)
        CLS, x = x[:,:1], x[:,1:]

        slate = jnp.zeros((256,512), dtype=x.dtype)
        slate = slate.at[keeps].set(x)
        slate = slate.at[masks].set(CLS)

        x = x + self.wpe
        x = self.layers(x, key=next(key))
        x = self.layernorm(x)
        x = self.head(x)

        return x

class MAGE(Module):
    encoder:Encoder
    decoder:Decoder

    def __init__(self, features:int=1024, heads:int=16, vocab:int=8192, layers:int=24, dropout:float=0, bias=False, key=None):
        key = RNG(key)
        self.encoder = Encoder(next(key))
        self.decoder = Decoder(next(key))

    @forward
    def __call__(self, x:Integer[Array,"n"], key=None):
        key = RNG(key)
        x, masks, keeps, drops = self.encoder(x, key=next(key))
        x = self.decoder(x, masks, keeps, key=next(key))
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

    G = Decoder(cfg["features"], cfg["vocab"], cfg["depth"], cfg["dropout"], key=next(key))

    grads = partial(gradients, precision=cfg["precision"])
    optimisers = optax.adamw(cfg["lr"], b1=0.9, b2=0.96)
    states = optimisers.init(G)

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
            return loss.mean(), {}

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
