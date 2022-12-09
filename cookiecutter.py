import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

import numpy as onp
import optax
import equinox
from equinox import nn, static_field, Module
from equinox.serialisation import tree_serialise_leaves as save
from equinox.serialisation import tree_deserialise_leaves as load
from einops import rearrange, reduce, repeat
from .layers import convolve, normalise, Convolution, Projection, Embedding, Selformer
from .toolkit import *
from .dataloader import *
from .lpips import LPIPS

import wandb
import click
import math
from pathlib import Path
from tqdm import tqdm
from functools import partial

@click.command()
@click.option("--dataset", type=Path)
@click.option("--steps", default=100000, type=int)
@click.option("--lr", type=float, default=3e-4)
@click.option("--batch", default=16, type=int)
@click.option("--features", type=int, default=768)
@click.option("--pages", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=12)
@click.option("--dropout", type=float, default=0)
@click.option("--use-bias", type=bool, default=False)
@click.option("--ftype", type=str, default="single")
def train(**config):
    wandb.init(project = "MASKGIT", config = config)
    folder = Path(f"VQGAN/autoencoding/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(42)
    key = RNG(key)

    ftype = config.get("ftype")
    dataset = Path("dataset/megabooru")
    dataset = ImageDataset(dataset, size=256)
    loader = dataloader(dataset, config["batch"], num_workers=4)
    
    M = GPT42(...)
    optimisers = optax.adamw(config["lr"], 0.9, 0.95)
    states = optimisers.init(parameterise(M))
    M, states = replicate(M), replicate(states)

    @ddp
    def Gstep(G, D, batch, states, key=None):
        key = RNG(key)
        @gradients
        def loss(G):
            # ... compute loss
            return loss, {}

        (l, metrics), gradients = loss(G)
        updates, states = optimisers.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    for idx in tqdm(range(config["steps"])):
        batch = cast(ftype)(next(loader))
        M, states, loss, metrics = Gstep(M, batch, states, key=next(key))

        if idx % 4200 == 0:
            ckpt = folder / idx
            ckpt.mkdir()

            save(ckpt / "model.weight", unreplicate(M))
            save(ckpt / "optimisers.ckpt", optimisers)
            save(ckpt / "states.ckpt", states)

        wandb.log({ "loss":loss.mean() })

    wandb.finish()

if __name__ == "__main__":
    train()