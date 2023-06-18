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
        self.wpe = jnp.zeros((256, features))
        self.layers = Sequential(
            [
                Transformer(features, dropout=dropout, bias=bias, key=next(key))
                for _ in range(layers)
            ]
        )
        self.layernorm = Layernorm([features])
        self.head = Projection(features, vocab, bias=bias, key=next(key))

    @forward
    def __call__(self, x: Integer[Array, "n"], masks: Integer[Array, "n"], key=None):
        key = RNG(key)
        x = self.embedding(x)
        x = jnp.where(rearrange(masks == 0, 'n -> n 1'), x, jnp.zeros(x.shape, x.dtype))
        x = x + self.wpe
        x = self.layers(x, key=next(key))
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


def sample(ratio):
    return jnp.cos(0.5 * jnp.pi * ratio)


@batch
def masks(idxes, key=None):
    key = RNG(key)
    ratio = sample(jr.uniform(next(key)))
    masks = jr.uniform(next(key), idxes.shape)
    masks = jnp.where(masks > ratio, 1, 0)  # 1 = NO MASK, 0 = MASK

    return masks


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

    wandb.finish()


if __name__ == "__main__":
    train()
