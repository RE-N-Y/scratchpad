import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

import equinox
from equinox import nn, static_field as buffer, Module
from equinox.serialisation import tree_serialise_leaves as save
from equinox.serialisation import tree_deserialise_leaves as load

import optax
import numpy as onp
from einops import rearrange, reduce, repeat
from layers import *
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

class Imageformer(Module):
    attention:SelfAttention
    mixer:Convolution
    mlp:MLP
    prenorm:Layernorm
    postnorm:Layernorm
    mixnorm:Layernorm
    ctxnorm:Layernorm
    width:int
    height:int

    def __init__(self, features:int, heads:int, width:int, height:int, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.width, self.height = width, height
        self.prenorm, self.postnorm, self.mixnorm = Layernorm([features]), Layernorm([features]), Layernorm([features])
        self.srcnorm, self.ctxnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.mixer = Convolution(features, features, 3, padding=1, groups=features, bias=bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n c"], ctx:Float[Array, "m d"], key=None):
        key = RNG(key)
        x = self.attention(self.prenorm(x), key=next(key)) + x
        x = self.fusion(self.srcnorm(x), self.ctxnorm(ctx), key=next(key)) + x
        x = self.mlp(self.postnorm(x), key=next(key)) + x

        x = rearrange(x, "(h w) d -> h w d", h=self.height, w=self.width)
        x = self.mixer(self.mixnorm(x)) + x
        x = rearrange(x, "h w d -> (h w) d")

        return x

class Mapping(Module):
    features:int
    layers:list

    def __init__(self, features:int, depth:int=12, bias=False, key=None):
        self.layers = [ Projection(features, features, bias=bias, key=next(key)) for _ in range(depth) ]

    def __call__(self, z, key=None):
        w = normalise(z)
        for layer in self.layers : w = layer(w)
        return w


class Synthesis(Module):
    latent:jnp.ndarray
    wpe:jnp.ndarray
    fpe:jnp.ndarray = buffer()
    mlp:MLP
    layers:list
    output:Convolution
    size:int
    patch:int

    def __init__(self, features:int=768, context:int=512, heads:int=12, depth:int=12, patch:int=8, size:int=256, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.size = size
        self.patch = patch
        ntoken = size//patch

        self.latent = jr.normal(next(key), (ntoken ** 2, features))
        self.wpe = jnp.zeros((ntoken ** 2, features))
        self.fpe = fpe(resolutions=(size,size))

        _, d = self.wpe.shape
        self.mlp = MLP(features=d, dropout=dropout, bias=False, key=next(key))
        self.layers = [Imageformer(features, heads=heads, width=ntoken, height=ntoken, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.decoder = Crossformer(d * patch * patch, context, heads=1, dropout=dropout, bias=bias, key=next(key))
        self.output = Convolution(features, 3, 1, bias=bias, key=next(key))

    def __call__(self, styles:Float[Array, "l m d"], key=None):
        key = RNG(key)
        ntoken = self.size // self.patch
        hiddens = self.latent + self.wpe

        for idx, layer in enumerate(self.layers):
            hiddens = layer(hiddens, styles[idx], key=next(key))

        fpe = self.mlp(self.fpe, key=next(key))
        hiddens = self.decoder(fpe, hiddens, key=next(key))
        hiddens = rearrange(hiddens, '(h w) (hr wr c) -> (h hr) (w wr) c', h=ntoken, w=ntoken, hr=self.patch, wr=self.patch) 
        out = self.output(hiddens)

        return out

class Generator(Module):
    mapping:Mapping
    synthesis:Synthesis
    nlayers:int

    def __init__(self, mapping:Mapping, synthesis:Synthesis, key=None):
        self.mapping = mapping
        self.synthesis = synthesis
        self.nlayers = len(synthesis.layers)


    @forward
    def __call__(self, key=None):
        key = RNG(key)
        
        z = jr.normal(next(key), ())
        styles = self.mapping(z, next(key))
        styles = repeat(styles, '... d -> l ... d', l=self.nlayers)
        fakes = self.synthesis(styles, key=next(key))

        return fakes

@click.command()
def distil(**cfg):
    pass

@click.command()
@click.option("--dataset", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=10000)
@click.option("--lr", type=float, default=6e-7)
@click.option("--batch", default=4, type=int)
@click.option("--L", type=int, default=768)
@click.option("--W", type=int, default=512)
@click.option("--pages", type=int, default=8192)
@click.option("--heads", type=int, default=8)
@click.option("--depth", type=int, default=12)
@click.option("--dropout", type=float, default=0)
@click.option("--bias", type=bool, default=True)
@click.option("--workers", type=int, default=4)
@click.option("--seed", type=int, default=42)
@click.option("--precision", type=str, default="single")
def train(**cfg):
    wandb.init(project = "MASKGIT", config = cfg)
    folder = Path(f"VQGAN/autoencoding/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(cfg["seed"])
    key = RNG(key)

    precisions = { "single":jnp.float32, "half":jnp.bfloat16 }
    ftype = precisions[cfg["precision"]]

    dataset = Path("dataset/megabooru")
    augmentations = [
        T.RandomApply([T.RandomResizedCrop((256,256), scale=(0.7,1.0))]), 
        T.RandomHorizontalFlip(0.2), 
        T.RandomAdjustSharpness(2,0.3), 
        T.RandomAutocontrast(0.3)
    ]

    dataset = ImageDataset(dataset, augmentations=augmentations)
    loader = dataloader(dataset, cfg["batch"], num_workers=cfg["workers"])
    
    augmentation = nn.Sequential([RandomBrightness(), RandomSaturation(), RandomContrast(), RandomAffine(), RandomCutout()])
    augmentation = CoinFlip(augmentation)

    D = Discriminator(features=[128,256,512,512,512,512,512], bias=cfg["bias"], key=next(key))
    S = Synthesis(cfg["L"], heads=cfg["heads"], depth=cfg["depth"], dropout=cfg["dropout"], bias=cfg["bias"], key=next(key))
    M = Mapping(cfg[""])
    G = Generator(M, S, key=next(key))
    D, G = cast(ftype)(D), cast(ftype)(G)

    Doptim, Goptim = optax.adabelief(cfg["lr"], 0.9, 0.99), optax.adabelief(cfg["lr"], 0.9, 0.99)
    Doptim, Goptim = optax.MultiSteps(Doptim, 1), optax.MultiSteps(Goptim, 1)
    Dstates, Gstates = Doptim.init(parameters(D)), Goptim.init(parameters(G))
    D, G, Dstates, Gstates = replicate(D), replicate(G), replicate(Dstates), replicate(Gstates)
    ma = replicate(jnp.nan)

    @ddp
    def Gstep(model, reals, states, key=None):
        (G,D), key = model
        key = RNG(key)

        @gradients
        def loss(G):
            fakes = G(x, key=next(key))
            adversarial = jax.nn.softplus(-D(fakes))

            return adversarial.mean(), {}

        (l, metrics), g = loss(G)
        updates, states = Goptim.update(g, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    @ddp
    def Dstep(model, reals, states, augmentation=identity, key=None):
        (G,D), key = model, RNG(key)

        @gradients
        def loss(D):
            fakes = G(x, key=next(key))
            fscores, rscores = D(augmentation(fakes, key=next(key))), D(augmentation(reals, key=next(key)))
            loss = jax.nn.softplus(fscores) + jax.nn.softplus(-rscores)
            return loss.mean(), {}

        (l, metrics), g = loss(D)
        updates, states = Doptim.update(g, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics


    @ddp
    def DRstep(model, reals, states, interval:int=32, key=None):
        G,D = model

        @gradients
        def loss(D):
            y, pullback = jax.vjp(D, reals)
            (g,) = pullback(jnp.ones(y.shape, dtype=ftype))
            loss = .1 * interval * reduce(g ** 2, 'b h w c -> b', "sum")
            return loss.mean(), {}

        (l, metrics), g = loss(D)
        updates, states = Doptim.update(g, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics

    @ddp
    def GRstep(model, reals, states, ma, interval=32, key=None):
        (G,D), key = model, RNG(key)
        b,h,w,c = reals.shape

        @gradients
        def ppl(G):
            styles = G.mapping(jr.normal(next(key), (), dtpye=ftype))
            tangent = jr.normal(next(key), styles.shape, dtype=ftype)
            generate = partial(G.synthesis, key=next(key))

            _, g = jax.jvp(generate, styles, normalise(tangent))
            lengths = reduce(g ** 2, 'b h w c -> b', "sum") / math.sqrt(h*w)

            ma = ema(ma, lengths.mean())
            loss = .1 * interval * (lengths - ma) ** 2

            return loss.mean(), ma

        (l, ma), g = ppl(G)
        updates, states = Goptim.update(g, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, ma

    for idx in tqdm(range(cfg["steps"])):
        batch = cast(ftype)(next(loader))
        every = lambda n : idx % n == 0

        G, Gstates, Gloss, _ = Gstep((G,D), batch, Gstates, augmentation=augmentation, key=next(key))
        D, Dstates, Dloss, _ = Dstep((G,D), batch, Dstates, augmentation=augmentation, key=next(key))
        if every(32): G, Gstates, GRloss, _ = GRstep((G,D), batch, Gstates, interval=32, key=next(key))
        if every(4): D, Dstates, DRloss, ma = DRstep((G,D), batch, Dstates, interval=4, key=next(key))

        if every(10000):
            ckpt = folder / str(idx)
            ckpt.mkdir()

            save(ckpt / "G.weight", unreplicate(G))
            save(ckpt / "D.weight", unreplicate(D))
            save(ckpt / "optimisers.ckpt", {"G":Goptim, "D":Doptim})
            save(ckpt / "states.ckpt", {"G":unreplicate(Gstates), "D":unreplicate(Dstates), "ma":unreplicate(ma)})
            
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
        })

    wandb.finish()

if __name__ == "__main__":
    train()