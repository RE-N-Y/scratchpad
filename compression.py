import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer

import equinox
from equinox import static_field as buffer, Module

import optax
import numpy as onp
from einops import rearrange, reduce, repeat, einsum, pack
from layers import convolve, normalise, Sequential, Convolution, Projection, Layernorm, Embedding, MLP, GLU, SelfAttention
from toolkit import *
from augmentations import *
from models import LPIPS

import torch
import torchvision.transforms as T
from dataloader import dataloader

import deeplake
import wandb
import click
import math
from pathlib import Path
from tqdm import tqdm
from functools import partial

load = equinox.tree_deserialise_leaves
save = equinox.tree_serialise_leaves

sg = jax.lax.stop_gradient
lrelu = partial(jax.nn.leaky_relu, negative_slope=.2)

class Downsample(Module):
    filter:jnp.ndarray = buffer()
    stride:int = buffer()

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


class VectorQuantiser(Module):
    input:Module
    output:Module
    codebook:Module
    pages:int = buffer()
    beta:float = buffer()

    def __init__(self, features:int, codes:int, pages:int, beta:float=0.25, bias=True, key=None):
        key = RNG(key)

        self.codebook = Embedding(pages, codes, key=next(key))
        self.input = Projection(features, codes, key=next(key))
        self.output = Projection(codes, features, key=next(key))
        self.pages = pages
        self.beta = beta

    def __call__(self, z:Float[Array, "n d"], key=None):
        z, codes = normalise(self.input(z)), normalise(self.codebook.weight)

        distance = reduce(z ** 2, 'N d -> N 1', 'sum') - 2 * einsum(z, codes, 'N d, C d -> N C') + reduce(codes ** 2, 'C d -> 1 C', 'sum')
        idxes = distance.argmin(axis=-1)
        codes = normalise(self.codebook(idxes))

        loss = self.beta * (z - sg(codes)) ** 2 + (sg(z) - codes) ** 2

        codes = z + sg(codes - z)
        codes = self.output(codes)

        return codes, loss, idxes

class Imageformer(Module):
    attention:SelfAttention
    mlp:GLU
    prenorm:Layernorm
    postnorm:Layernorm
    width:int
    height:int

    def __init__(self, features:int, heads:int, width:int, height:int, bias=False, dropout:float=0, key=None):
        key = RNG(key)
        self.width, self.height = width, height
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.mlp = GLU(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], key=None):
        key = RNG(key)
        x = self.attention(self.prenorm(x), key=next(key)) + x
        x = self.mlp(self.postnorm(x), key=next(key)) + x
        
        return x

class VQVAE(Module):
    epe:jnp.ndarray
    dpe:jnp.ndarray
    input:Convolution
    encoder:nn.Sequential
    quantiser:VectorQuantiser
    decoder:nn.Sequential
    output:Convolution
    size:int
    patch:int


    def __init__(self, features:int=768, codes:int=32, pages:int=8192, heads:int=12, depth:int=12, patch:int=16, size:int=256, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.size = size
        self.patch = patch

        ntoken = size//patch
        self.epe = jnp.zeros((ntoken ** 2, features))
        self.dpe = jnp.zeros((ntoken ** 2, features))

        # patchify
        self.input = Convolution(3, features, patch, stride=patch, bias=bias, key=next(key))
        # encoder
        transformers = [Imageformer(features, width=ntoken, height=ntoken, heads=heads, dropout=dropout, key=next(key)) for _ in range(depth)]
        self.encoder = Sequential([*transformers, Layernorm([features]), MLP(features, activation="tanh", dropout=dropout, bias=bias, key=next(key))])
        # quantiser
        self.quantiser = VectorQuantiser(features, codes, pages, bias=bias, key=next(key))
        # decoder
        transformers = [Imageformer(features, width=ntoken, height=ntoken, heads=heads, dropout=dropout, key=next(key)) for _ in range(depth)]
        self.decoder = Sequential([*transformers, Layernorm([features]), MLP(features, activation="tanh", dropout=dropout, bias=bias, key=next(key))])
        # pixelshuffle
        self.output = Convolution(features, 3 * patch * patch, 1, bias=bias, key=next(key))
    
    @forward
    def decode(self, idxes:Integer[Array, "n"], key=None):
        key = RNG(key)
        ratio = self.size // self.patch
        codes = self.quantiser.codebook(idxes)
        codes = self.quantiser.output(codes)

        x = self.decoder(codes + self.dpe, key=next(key))
        x = rearrange(x, '(h w) c -> h w c', h=ratio, w=ratio)
        x = rearrange(self.output(x), 'h w (hr wr c) -> (h hr) (w wr) c', hr=self.patch, wr=self.patch)

        return x

    @forward
    def __call__(self, x:Float[Array, "h w c"], key=None):
        key = RNG(key)
        ratio = self.size // self.patch
        x = rearrange(self.input(x), 'h w c -> (h w) c')

        x = self.encoder(x + self.epe, key=next(key))
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(codes + self.dpe, key=next(key))

        x = rearrange(x, '(h w) c -> h w c', h=ratio, w=ratio)
        x = rearrange(self.output(x), 'h w (hr wr c) -> (h hr) (w wr) c', hr=self.patch, wr=self.patch)

        return x, codes, loss, idxes

def t2i(x:Float[Array, "b h w c"]) -> onp.ndarray:
    x = x.clip(-1, 1)
    x = x * 0.5 + 0.5
    x = onp.asarray(x * 255, dtype=onp.uint8)
    return x



@click.command()
@click.option("--batch", default=64, type=int)
@click.option("--size", default=256, type=int)
@click.option("--features", type=int, default=768)
@click.option("--pages", type=int, default=8192)
@click.option("--heads", type=int, default=12)
@click.option("--depth", type=int, default=12)
@click.option("--dropout", type=float, default=0)
@click.option("--bias", type=bool, default=False)
@click.option("--seed", type=int, default=42)
@click.option("--workers", type=int, default=16)
@click.option("--checkpoint", type=Path)
def encode(**cfg):
    key = jr.PRNGKey(cfg["seed"])
    key = RNG(key)
    dsplit = lambda key : jr.split(key, jax.device_count())
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(cfg["size"], antialias=True),
        T.ConvertImageDtype(torch.float), 
        T.Normalize(0.5, 0.5)
    ])

    model = VQVAE(cfg["features"], pages=cfg["pages"], dropout=cfg["dropout"], bias=cfg["bias"], size=cfg["size"], key=next(key))
    model = load(cfg["checkpoint"] / "G.weight", model)
    model = replicate(model)

    ds = deeplake.load("hub://reny/animefaces")
    ds.create_tensor("16x16")
    loader = ds.pytorch(tensors=["images"], transform={"images":transform}, batch_size=cfg["batch"], num_workers=cfg["workers"], shuffle=False)
    
    @ddp
    def quantise(G, images, key=None):
        fakes, codes, loss, idxes = G(images, jr.split(key, len(images)))
        return idxes

    with ds:
        for batch in tqdm(loader):
            batch = rearrange(batch["images"], "... c h w -> ... h w c")
            batch = jnp.asarray(batch)
            idxes = quantise(model, batch, dsplit(next(key)))
            idxes = rearrange(idxes, "n mb ... -> (n mb) ...")
            ds["16x16"].extend(onp.asarray(idxes))

@click.command()
@click.option("--dataset", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=4096, type=int)
@click.option("--lr", type=float, default=1e-5)
@click.option("--cooldown", type=float, default=0)
@click.option("--batch", default=64, type=int)
@click.option("--size", default=256, type=int)
@click.option("--patch", type=int, default=8)
@click.option("--features", type=int, default=768)
@click.option("--pages", type=int, default=8192)
@click.option("--heads", type=int, default=12)
@click.option("--depth", type=int, default=12)
@click.option("--dropout", type=float, default=0)
@click.option("--bias", type=bool, default=False)
@click.option("--workers", type=int, default=16)
@click.option("--seed", type=int, default=42)
@click.option("--checkpoint", type=Path, default=None)
@click.option("--precision", type=str, default="half")
def train(**cfg):
    wandb.init(project = "VQGAN", config = cfg)
    folder = Path(f"checkpoints/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(cfg["seed"])
    key = RNG(key)
    dsplit = lambda key : jr.split(key, jax.device_count())
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(cfg["size"], antialias=True),
        T.RandomResizedCrop(cfg["size"], scale=(0.8, 1.0), antialias=True),
        T.RandomHorizontalFlip(0.3), T.RandomAdjustSharpness(2,0.3), T.RandomAutocontrast(0.3),
        T.ConvertImageDtype(torch.float), T.Normalize(0.5, 0.5)
    ])

    loader = dataloader(
        "hub://reny/animefaces",
        tensors=["images"],
        batch_size=cfg["batch"],
        transform={"images":transform},
        decode_method={"images":"numpy"},
        num_workers=cfg["workers"],
        buffer_size=8192,
        shuffle=True
    )

    lpips = LPIPS.load()
    augmentation = Sequential([RandomBrightness(), RandomSaturation(), RandomContrast(), RandomAffine(), RandomCutout()])
    augmentation = CoinFlip(augmentation)

    grads = partial(gradients, precision=cfg["precision"])
    G = VQVAE(cfg["features"], pages=cfg["pages"], patch=cfg["patch"], heads=cfg["heads"], dropout=cfg["dropout"], bias=cfg["bias"], size=cfg["size"], key=next(key))

    Goptim = optax.warmup_cosine_decay_schedule(0, cfg["lr"], cfg["warmup"], cfg["steps"], cfg["cooldown"])
    Goptim = optax.lion(Goptim, b1=0.9, b2=0.99, weight_decay=1.)
    Gstates = Goptim.init(parameters(G))

    if cfg["checkpoint"] is not None:
        G = load(cfg["checkpoint"] / "G.weight", G)
        Gstates = load(cfg["checkpoint"] / "states.ckpt", Gstates)


    G, Gstates = replicate(G), replicate(Gstates)

    @ddp
    def Gstep(G, reals, states, key=None):
        key = RNG(key)

        @grads
        def loss(G, lpips, reals):
            fakes, codes, loss, idxes = G(reals, jr.split(next(key), len(reals)))
            l2, l1 = jnp.square(reals - fakes), jnp.abs(reals - fakes)
            perceptual = lpips(reals, fakes)

            l = loss.mean() + l2.mean() + .1 * perceptual.mean()

            return l, { "loss":loss.mean(), "l2":l2.mean(), "perceptual":perceptual.mean() }

        (l, metrics), g = loss(G, lpips, reals)
        updates, states = Goptim.update(g, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    for idx in tqdm(range(cfg["steps"])):
        batch = next(loader)
        batch = rearrange(batch["images"], "... c h w -> ... h w c")

        G, Gstates, Gloss, metrics = Gstep(G, batch, Gstates, dsplit(next(key)))

        if idx % 8192 == 0:
            save(folder / "G.weight", unreplicate(G))
            save(folder / "states.ckpt", unreplicate(Gstates))

        if idx % 2048 == 0:
            reals = batch[:32]
            fakes, loss, distance, idxes = unreplicate(G)(reals, jr.split(next(key), len(reals)))
            wandb.log({ "fakes":[wandb.Image(i) for i in t2i(fakes)], "reals":[wandb.Image(i) for i in t2i(reals)] }, commit=False)

        wandb.log({
            "G":Gloss.mean().item(),
            "l2":metrics["l2"].mean().item(),
            "loss":metrics["loss"].mean().item(),
            "perceptual":metrics["perceptual"].mean().item(),
        })

    wandb.finish()

if __name__ == "__main__":
    train()
