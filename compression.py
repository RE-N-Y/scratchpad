import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

import equinox
from equinox import nn, static_field, Module
from equinox.serialisation import tree_serialise_leaves as save
from equinox.serialisation import tree_deserialise_leaves as load

import optax
import numpy as onp
from einops import rearrange, reduce, repeat
from layers import convolve, normalise, Activation, Convolution, Projection, Layernorm, Embedding, MLP, SelfAttention
from toolkit import *
from augmentations import *
from models import LPIPS

import torch
import torchvision.transforms as T
from dataloader import ImageDataset, dataloader


import wandb
import click
import math
from pathlib import Path
from tqdm import tqdm
from functools import partial

sg = jax.lax.stop_gradient
lrelu = partial(jax.nn.leaky_relu, negative_slope=.2)

class Downsample(Module):
    filter:jnp.ndarray = static_field()
    stride:int = static_field()

    def __init__(self, filter=[1.,2.,1.], stride:int=2, key=None):
        filter = jnp.array(filter)
        filter = filter[:,None] * filter[None,:]
        self.filter = filter / filter.sum()
        self.stride = stride

    def __call__(self, x:Float[Array, "h w c"], key=None):
        h, w, c = x.shape
        depthwise = repeat(self.filter.astype(x.dtype), 'h w -> h w i o', i=1, o=c)
        x = rearrange(x, 'h w c -> 1 h w c')
        x = convolve(x, depthwise, stride=self.stride, padding=1, groups=c)
        x = rearrange(x, '1 h w c -> h w c')
        return x


class Resnet(Module):
    input:Convolution
    downsample:Downsample
    residual:Convolution
    output:Convolution

    def __init__(self, nin:int, non:int, bias=True, key=None):
        key = RNG(key)
        self.input = Convolution(nin, nin, kernel=3, padding=1, bias=bias, key=next(key))
        self.downsample = Downsample()
        self.residual = Convolution(nin, non, kernel=1, padding=0, bias=False, key=next(key))
        self.output = Convolution(nin, non, kernel=3, padding=1, bias=bias, key=next(key))


    def __call__(self, x:Float[Array, "h w c"], key=None):
        r = x
        x = lrelu(self.input(x))
        x = self.downsample(x)
        x = lrelu(self.output(x))
        out = x + self.residual(self.downsample(r)) / math.sqrt(2)

        return out

class Discriminator(Module):
    stem:Convolution
    layers:list
    out:Convolution
    middle:Projection
    cls:Projection

    def __init__(self, features, bias=True, key=None):
        key = RNG(key)
        [first, *_, last] = features
        ninnon = list(zip(features[:-1], features[1:]))

        self.stem = Convolution(3, first, kernel=3, padding=1, key=next(key))
        self.layers = [Resnet(nin, non, bias=bias, key=next(key)) for nin, non in ninnon]
        self.out = Convolution(last, last, kernel=3, padding=1, key=next(key))
        self.middle = Projection(4 * 4 * last, last, bias=bias, key=next(key))
        self.cls = Projection(last, 1, bias=bias, key=next(key))

    @forward
    def __call__(self, x:Float[Array, "h w c"], key=None) -> Float[Array, ""]:
        x = self.stem(x)
        for layer in self.layers: x = layer(x)
        x = rearrange(lrelu(self.out(x)), '... -> (...)')
        x = self.cls(lrelu(self.middle(x)))

        return x.squeeze()


def compute_euclidean_distance(samples:Float[Array, "n d"], codes:Float[Array, "c d"]) -> Float[Array, "n c"]:
    distance = rearrange(samples, 'n d -> n () d') - rearrange(codes, 'c d -> () c d')
    distance = reduce(distance ** 2, 'n c d -> n c', 'sum')
    return distance

class VectorQuantiser(Module):
    input:Module
    output:Module
    codebook:Module
    pages:int = static_field()
    beta:float = static_field()

    def __init__(self, features:int, codes:int, pages:int, beta:float=0.25, bias=True, key=None):
        key = RNG(key)

        self.codebook = Embedding(pages, codes, key=next(key))
        self.input = Projection(features, codes, bias=bias, key=next(key))
        self.output = Projection(codes, features, bias=bias, key=next(key))
        self.pages = pages
        self.beta = beta

    def __call__(self, z:Float[Array, "n d"], key=None):
        z, codes = normalise(self.input(z)), normalise(self.codebook.weight)

        distance = compute_euclidean_distance(z, codes)
        idxes = distance.argmin(axis=-1)
        codes = normalise(self.codebook(idxes))

        loss = self.beta * jnp.mean((sg(z) - codes) ** 2) + \
                           jnp.mean((z - sg(codes)) ** 2)
        
        codes = z + sg(codes - z)
        codes = self.output(codes)

        return codes, loss, idxes

class Imageformer(Module):
    attention:SelfAttention
    mixer:Convolution
    mlp:MLP
    prenorm:Layernorm
    postnorm:Layernorm
    mixnorm:Layernorm
    width:int
    height:int

    def __init__(self, features:int, heads:int, width:int, height:int, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.width, self.height = width, height
        self.prenorm, self.postnorm, self.mixnorm = Layernorm([features]), Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.mixer = Convolution(features, features, 3, padding=1, groups=features, bias=bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], key=None):
        key = RNG(key)
        x = self.attention(self.prenorm(x), key=next(key)) + x
        x = self.mlp(self.postnorm(x), key=next(key)) + x
        x = rearrange(x, "(h w) d -> h w d", h=self.height, w=self.width)
        x = self.mixer(self.mixnorm(x)) + x
        x = rearrange(x, "h w d -> (h w) d")

        return x

class ViTQuantiser(Module):
    epe:jnp.ndarray
    dpe:jnp.ndarray
    input:Convolution
    encoder:nn.Sequential
    quantiser:VectorQuantiser
    decoder:nn.Sequential
    output:Convolution
    size:int
    patch:int


    def __init__(self, features:int=768, codes:int=16, pages:int=8192, heads:int=12, depth:int=12, patch:int=8, size:int=256, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.size = size
        self.patch = patch

        ntoken = size//patch
        self.epe = jnp.zeros((ntoken ** 2, features))
        self.dpe = jnp.zeros((ntoken ** 2, features))

        # patchify
        self.input = Convolution(3, features, patch, stride=patch, bias=bias, key=next(key))
        # encoder
        transformers = [Imageformer(features, width=ntoken, height=ntoken, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.encoder = nn.Sequential([*transformers, Layernorm([features]), MLP(features, activation="tanh", dropout=dropout, bias=bias, key=next(key))])
        # quantiser
        self.quantiser = VectorQuantiser(features, codes, pages, bias=bias, key=next(key))
        # decoder
        transformers = [Imageformer(features, width=ntoken, height=ntoken, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.decoder = nn.Sequential([*transformers, Layernorm([features]), MLP(features, activation="tanh", dropout=dropout, bias=bias, key=next(key))])
        # pixelshuffle
        self.output = nn.Sequential([Layernorm([features]), Activation("tanh"), Convolution(features, 3 * patch * patch, 1, bias=bias, key=next(key))])

    @forward
    def __call__(self, x:Float[Array, "h w c"], key=None):
        key = RNG(key)
        compression = self.size // self.patch
        x = rearrange(self.input(x), 'h w c -> (h w) c')

        x = self.encoder(x + self.epe, key=next(key))
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(codes + self.dpe, key=next(key))

        x = rearrange(x, '(h w) c -> h w c', h=compression, w=compression)
        x = rearrange(self.output(x), 'h w (hr wr c) -> (h hr) (w wr) c', hr=self.patch, wr=self.patch) 

        return x, codes, loss, idxes



@click.command()
@click.option("--dataset", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=10000)
@click.option("--lr", type=float, default=6e-7)
@click.option("--batch", default=4, type=int)
@click.option("--features", type=int, default=768)
@click.option("--pages", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=12)
@click.option("--dropout", type=float, default=0)
@click.option("--bias", type=bool, default=True)
@click.option("--workers", type=int, default=4)
@click.option("--seed", type=int, default=42)
@click.option("--precision", type=str, default="single")
def train(**config):
    wandb.init(project = "MASKGIT", config = config)
    folder = Path(f"VQGAN/autoencoding/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(config["seed"])
    key = RNG(key)

    precisions = { "single":jnp.float32, "half":jnp.bfloat16 }
    ftype = precisions[config["precision"]]

    dataset = Path("dataset/megabooru")
    augmentations = [
        T.RandomApply([T.RandomResizedCrop((256,256), scale=(0.7,1.0))]), 
        T.RandomHorizontalFlip(0.2), 
        T.RandomAdjustSharpness(2,0.3), 
        T.RandomAutocontrast(0.3)
    ]

    dataset = ImageDataset(dataset, augmentations=augmentations)
    loader = dataloader(dataset, config["batch"], num_workers=config["workers"])
    
    lpips = LPIPS.load()
    augmentation = nn.Sequential([RandomBrightness(), RandomSaturation(), RandomContrast(), RandomAffine(), RandomCutout()])
    augmentation = CoinFlip(augmentation)

    D = Discriminator(features=[128,256,512,512,512,512,512], bias=config["bias"], key=next(key))
    G = ViTQuantiser(config["features"], heads=config["heads"], depth=config["depth"], dropout=config["dropout"], bias=config["bias"], key=next(key))
    D, G, lpips = cast(ftype)(D), cast(ftype)(G), cast(ftype)(lpips)

    schedule = optax.warmup_cosine_decay_schedule(0, config["lr"], config["warmup"], config["steps"] - config["warmup"])
    Doptim, Goptim = optax.adamw(schedule, 0.9, 0.99), optax.adamw(schedule, 0.9, 0.99)
    Doptim, Goptim = optax.MultiSteps(Doptim, 1), optax.MultiSteps(Goptim, 1)
    Dstates, Gstates = Doptim.init(parameters(D)), Goptim.init(parameters(G))
    D, G, Dstates, Gstates = replicate(D), replicate(G), replicate(Dstates), replicate(Gstates)

    @ddp
    def Gstep(model, x, states, key=None):
        key = RNG(key)
        G,D = model

        @gradients
        def loss(G):
            r, codes, loss, idxes = G(x, key=next(key))
            l2, l1 = jnp.square(x - r), jnp.abs(x - r)
            adversarial = jax.nn.softplus(-D(r))
            perceptual = lpips(x,r)

            loss = loss.mean() + l2.mean() + .25 * (perceptual ** 2).mean() + .25 * adversarial.mean()
            return loss, { "l2":l2.mean(), "perceptual":perceptual.mean(), "adversarial":adversarial.mean() }

        (l, metrics), g = loss(G)
        updates, states = Goptim.update(g, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    @ddp
    def Dstep(model, x, states, augmentation=identity, key=None):
        key = RNG(key)
        G,D = model

        @gradients
        def loss(D):
            fakes, codes, loss, idxes = G(x, key=next(key))
            fscores, rscores = D(augmentation(fakes, key=next(key))), D(augmentation(x, key=next(key)))
            loss = jax.nn.softplus(fscores) + jax.nn.softplus(-rscores)
            return loss.mean(), {}

        (l, metrics), g = loss(D)
        updates, states = Doptim.update(g, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics


    @ddp
    def DRstep(model, x, states, interval:int=32, key=None):
        G,D = model

        @gradients
        def loss(D):
            y, pullback = jax.vjp(D,x)
            (g,) = pullback(jnp.ones(y.shape, dtype=ftype))
            loss = 1000000 * interval * reduce(g ** 2, 'b h w c -> b', "sum")
            return loss.mean(), {}

        (l, metrics), g = loss(D)
        updates, states = Doptim.update(g, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics

    for idx in tqdm(range(config["steps"])):
        batch = cast(ftype)(next(loader))
        every = lambda n : idx % n == 0

        G, Gstates, Gloss, metrics = Gstep((G,D), batch, Gstates, key=next(key))
        D, Dstates, Dloss, _ = Dstep((G,D), batch, Dstates, augmentation=augmentation, key=next(key))
        if every(4): D, Dstates, DRloss, _ = DRstep((G,D), batch, Dstates, interval=4, key=next(key))

        if every(10000):
            ckpt = folder / str(idx)
            ckpt.mkdir()

            save(ckpt / "G.weight", unreplicate(G))
            save(ckpt / "D.weight", unreplicate(D))
            save(ckpt / "optimisers.ckpt", {"G":Goptim, "D":Doptim})
            save(ckpt / "states.ckpt", {"G":unreplicate(Gstates), "D":unreplicate(Dstates)})
            
        if every(2500):
            N = 16
            fakes, codes, loss, idxes = unreplicate(G)(batch[:N], key=next(key))
            comparison = jnp.concatenate((batch[:N], fakes))
            comparison = rearrange(comparison, '(b d) h w c -> (b h) (d w) c', d = 2)
            dataset.tensor_to_image(comparison).save(folder / f"{idx}.png")

            print("reals", batch[:N].mean(), batch[:N].std())
            print("fakes", fakes.mean(), fakes.std())

        wandb.log({ 
            "G":Gloss.mean().item(), 
            "D":Dloss.mean().item(), 
            "DR":DRloss.mean().item(),
            "l2":metrics["l2"].mean().item(), 
            "perceptual":metrics["perceptual"].mean().item(),
            "adversarial":metrics["adversarial"].mean().item(),
        })

    wandb.finish()

if __name__ == "__main__":
    train()