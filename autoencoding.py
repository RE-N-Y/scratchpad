import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Integer, Array

import optax
import numpy as onp
import equinox
from equinox import nn, static_field, Module
from equinox.serialisation import tree_serialise_leaves as save
from equinox.serialisation import tree_deserialise_leaves as load
from einops import rearrange, reduce, repeat, pack
from .layers import (
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
from .toolkit import *
from .dataloader import *


class Transformer(Module):
    attention: SelfAttention
    mlp: GLU
    prenorm: Layernorm
    width: int
    height: int

    def __init__(
        self,
        features: int,
        heads: int,
        width: int,
        height: int,
        dropout: float = 0,
        key=None,
    ):
        key = RNG(key)
        self.width, self.height = width, height
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(
            features, heads=heads, dropout=dropout, bias=bias, key=next(key)
        )
        self.mlp = GLU(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x: Float[Array, "n d"], key=None):
        key = RNG(key)
        x = self.prenorm(x)
        x = x + self.attention(x, key=next(key)) + self.mlp(x, key=next(key))

        return x


class Decoder(Module):
    layers: Sequential
    wpe: SinusodialEmbedding

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
        self.embedding = Embedding(vocab, features, dropout=dropout, key=next(key))
        self.wpe = SinusodialEmbedding(features, dropout=dropout, key=next(key))
        self.layers = Sequential(
            [
                Transformer(features, dropout=dropout, bias=bias, key=next(key))
                for _ in layers
            ]
        )
        self.layernorm = Layernorm([features])
        self.head = Projection(features, vocab, bias=bias, key=next(key))

    @forward
    def __call__(self, x: Integer[Array, "n"], masks: Integer[Array, "n"], key=None):
        key = RNG(key)
        x = self.embedding(x)
        x = jnp.where(masks == 0, x, jnp.zeros(x.shape, x.dtype))
        x = x + self.wpe(x)
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
@click.option("--batch", default=4200, type=int)
@click.option("--minibatch", default=6, type=int)
@click.option("--features", type=int, default=768)
@click.option("--vocab", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=24)
@click.option("--dropout", type=float, default=0)
@click.option("--length", type=int, default=1024)
def train(**cfg):
    wandb.init(project="MASKGIT", config=cfg)
    folder = Path(f"VQGAN/autoencoding/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(cfg["seed"])
    key = RNG(key)
    dsplit = lambda key: jr.split(key, jax.device_count())

    loader = dataloader("hub://reny/animefaces", tensors=["16x16"], batch_size=cfg["batch"], num_workers=cfg["workers"], shuffle=False)

    G = Decoder(cfg["features"], cfg["vocab"], cfg["depth"], cfg["dropout"], key=next(key))

    grads = partial(gradients, precision=cfg["precision"])
    G, states = replicate(G), replicate(states)

    optimisers = optax.adamw(cfg["lr"])
    states = optimisers.init(G)

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

    for idx in tqdm(range(cfg["total_steps"])):
        batch = next(loader)
        G, states, Gloss, metrics = Gstep(G, batch, states, dsplit(next(key)))

        wandb.log({"loss": onp.mean(Gloss)})

    wandb.finish()


if __name__ == "__main__":
    train()
