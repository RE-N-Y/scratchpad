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
from .layers import Convolution, Layernorm, MLP, CrossAttention
from .toolkit import *
from .dataloader import *

class Resnet(Module):
    depthwise:Convolution
    layernorm:Layernorm
    mlp:MLP
    
    def __init__(self, features, bias=True, key=None):
        key = RNG(key)
        self.depthwise = Convolution(features, features, kernel=7, padding=3, groups=features, bias=bias, key=next(key))
        self.layernorm = Layernorm((features,))
        self.mlp = MLP(features, bias=bias, key=next(key))

    def __call__(self, x, key=None):
        x = self.mlp(self.layernorm(self.depthwise(x))) + x
        return x

class Upsample(Module):
    layer:Convolution
    scale:float = static_field()

    def __init__(self, nin, non, scale=2, bias=True, key=None):
        self.scale = scale
        self.layer = Convolution(nin, non, kernel=3, padding=1, bias=bias, key=key)
    
    def __call__(self, x, key=None):
        h, w, c = x.shape
        x = jax.image.resize(x, (self.scale * h, self.scale * w, c), mode="nearest")
        x = self.layer(x)
        return x

class Downsample(Module):
    layer:Convolution
    scale:float = static_field()

    def __init__(self, nin, non, scale=2, bias=True, dtype=float, key=None):
        self.scale = scale
        self.layer = Convolution(nin, non, kernel=3, stride=scale, padding=1, bias=bias, dtype=dtype, key=key)
    
    def __call__(self, x, key=None):
        x = self.layer(x)
        return x

class UNet(Module):
    input:Convolution
    encoder:list
    bridge:list
    decoder:list
    output:Convolution

    def __init__(self, vocab:int, features:list, blocks:int=3, heads=8, bias=True, key=None):
        key = RNG(key)
        [first, *_, last] = features
        ninnon = list(zip(features[:-1], features[1:]))
        
        down = lambda nin, non : nn.Sequential(
            [Resnet(nin, bias=bias, key=next(key)) for _ in range(blocks)] + \
            [Downsample(nin, non, bias=bias, key=next(key))]
        )

        up = lambda nin, non : nn.Sequential(
            [Resnet(non+non, bias=bias, key=next(key)) for _ in range(blocks)] + \
            [Upsample(non+non, nin, bias=bias, key=next(key))]
        )

        self.input = Convolution(vocab, first, kernel=3, padding=1, bias=bias, key=next(key))
        self.encoder = [down(nin,non) for nin,non in ninnon]
        self.bridge = CrossAttention(features=last, context=last, heads=heads, bias=bias, key=next(key))
        self.decoder = [up(nin,non) for nin,non in ninnon] # reverse
        self.output = Convolution(first, vocab, kernel=3, padding=1, bias=bias, key=next(key))

    @forward
    def __call__(self, x, masks, key=None):
        x, hiddens = self.input(x), []

        for idx, layer in enumerate(self.encoder): 
            hiddens.append(x := layer(x))

        x = self.attention(x,x)
        
        for idx, layer in enumerate(self.decoder):
            x, _ = pack([layer(x),hiddens[idx]], 'h w *')

        x = self.output(x)

        return x


import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image

def sample(ratio): return jnp.cos(.5 * jnp.pi * ratio)

@batch
def create_tensor_masks(idxes, key=None):
    key = RNG(key)
    ratio = sample(jr.uniform(next(key)))
    masks = jr.uniform(next(key), idxes.shape)
    masks = jnp.where(masks > ratio, 1, 0) # 1 = NO MASK, 0 = MASK

    return masks

@click.command()
@click.option("--dataset", type=Path)
@click.option("--compressor", type=Path)
@click.option("--total-steps", default=3042, type=int)
@click.option("--warmup-steps", default=300, type=int)
@click.option("--decay-steps", default=2742, type=int)
@click.option("--lr", type=float, default=3e-4)
@click.option("--batch", default=4200, type=int)
@click.option("--minibatch", default=6, type=int)
@click.option("--features", type=int, default=768)
@click.option("--vocab", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=24)
@click.option("--dropout", type=float, default=0)
@click.option("--length", type=int, default=1024)
@click.option("--label-smoothing", type=float, default=0.1)
def train(**cfg):
    wandb.init(project = "MASKGIT", config = cfg)
    folder = Path(f"VQGAN/autoencoding/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(42)
    key = RNG(key)

    dataset = Path("dataset/megabooru")
    dataset = ImageDataset(dataset, size=256)
    loader = dataloader(dataset, cfg["batch"], num_workers=4)
    loader = cycle(loader)

    grads = partial(gradients, precision=cfg["precision"])
    G, states = replicate(G), replicate(states)

    optimisers = optax.adam(cfg["lr"], 0.9, 0.96)
    states = optimisers.init(G)

    @ddp
    def Gstep(G, batch, masks, states, key=None):
        key = RNG(key)

        @grads
        def cross_entropy_loss(G, batch, masks):
            logits = G(batch, key=next(key))
            labels = jax.nn.one_hot(batch, cfg["vocab"])
            labels = optax.smooth_labels(labels, cfg["label_smoothing"])
            loss = optax.softmax_cross_entropy(logits, labels)
            return loss.mean(), { }

        (loss, metrics), gradients = cross_entropy_loss(G, batch)
        updates, states = optimisers.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, loss, metrics


    for idx in tqdm(range(cfg["total_steps"])):
        batch = next(loader)
        G, states, Gloss, metrics = Gstep(G, batch, masks, states, key=next(key))

        if idx % 16 == 0:
            checkpoint = { "T":unreplicate(G), "states":unreplicate(states), "optimisers":optimisers }
            save(folder / f"{idx}.nox", checkpoint)

        wandb.log({ "loss":onp.mean(Gloss) })

    wandb.finish()

if __name__ == "__main__":
    train()
