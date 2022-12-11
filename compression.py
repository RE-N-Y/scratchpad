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
from .models.lpips import LPIPS

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

    def __init__(self, filter=[1.,3.,3.,1.], stride:int=2, key=None):
        filter = jnp.array(filter)
        filter = filter[:,None] * filter[None,:]
        self.filter = filter / filter.sum()
        self.stride = stride

    def __call__(self, x:Float[Array, "h w c"], key=None):
        h, w, c, dtype = x.shape, x.dtype
        depthwise = repeat(self.filter.astype(dtype), 'h w -> h w i o', i=1, o=c)
        x = convolve(x, depthwise, stride=self.stride, padding="same", groups=c)

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

        weight = jr.normal(next(key), (pages, codes))
        self.codebook = Embedding(pages, codes, weight=weight, key=next(key))

        self.input = Projection(features, codes, bias=bias, key=next(key))
        self.output = Projection(codes, features, bias=bias, key=next(key))
        self.pages = pages
        self.beta = beta

    def __call__(self, z:Float[Array, "n d"], key=None):
        z, codes = normalise(self.input(z)), normalise(self.codebook.weight)

        distance = compute_euclidean_distance(z, codes)
        idxes = jnp.argmin(distance)
        codes = normalise(self.codebook(idxes))

        loss = self.beta * jnp.mean((sg(z) - codes) ** 2) + \
                           jnp.mean((z - sg(codes)) ** 2)
        
        codes = z + sg(codes - z)
        codes = self.output(codes)

        return codes, loss, idxes

class ViTQuantiser(Module):
    input:Convolution
    encoder:nn.Sequential
    quantiser:VectorQuantiser
    decoder:nn.Sequential
    output:Convolution
    size:int = static_field()
    patch:int = static_field()


    def __init__(self, features:int=768, codes:int=16, pages:int=8192, heads:int=12, depth:int=12, patch:int=8, size:int=256, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.size = size
        self.patch = patch
        self.input = Convolution(3, features, patch, stride=patch, bias=bias, key=next(key))
        self.encoder = nn.Sequential([Selformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)])
        self.quantiser = VectorQuantiser(features, codes, pages, bias=bias, key=next(key))
        self.decoder = nn.Sequential([Selformer(features, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)])
        self.output = Convolution(features, 3 * patch * patch, patch, stride=patch, bias=bias, key=next(key))

    @forward
    def __call__(self, x:Float[Array, "h w c"], key=None):
        compression = self.size // self.patch
        x = rearrange(self.input(x), 'h w c -> (h w) c')
        
        x = self.encoder(x)
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(codes)

        x = rearrange(x, '(h w) c -> h w c', h=compression, w=compression)
        x = rearrange(self.output(x), 'h w (hr wr c) -> (h hr) (w wr) c', hr=self.patch, wr=self.patch) 

        return x, codes, loss, idxes



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
    
    
    lpips = LPIPS.load("weights/lpips.weights")
    D = Discriminator(features=[128,256,512,512,512,512,512], bias=config["bias"], key=next(key))
    G = ViTQuantiser(config["features"], heads=config["heads"], depth=config["depths"], dropout=config["dropout"], bias=config["bias"], key=next(key))
    D, G, lpips = cast(ftype)(D), cast(ftype)(G), cast(ftype)(lpips)

    Doptim, Goptim = optax.adamw(config["lr"], 0.9, 0.95), optax.adamw(config["lr"], 0.9, 0.95)
    Dstates, Gstates = Doptim.init(parameterise(D)), Goptim.init(parameterise(G))
    D, G, Dstates, Gstates = replicate(D), replicate(G), replicate(Dstates), replicate(Gstates)

    @ddp
    def Gstep(G, D, x, states, key=None):
        key = RNG(key)
        @gradients
        def loss(G):
            r, codes, loss, idxes = G(x, key=next(key))
            l2, l1 = jnp.square(x - r), jnp.abs(x - r)
            adversarial = jax.nn.softplus(-D(r))
            perceptual = lpips(x,r)

            loss = loss.mean() + l2.mean() + .3 * perceptual.mean() + .3 * adversarial.mean() + l1.mean()
            return loss, {}

        (l, metrics), gradients = loss(G)
        updates, states = Goptim.update(gradients, states, G)
        G = equinox.apply_updates(G, updates)

        return G, states, l, metrics

    @ddp
    def Dstep(G, D, x, states, augmentation, key=None):
        key = RNG(key)
        @gradients
        def loss(D):
            fakes, codes, loss, idxes = G(x, key=next(key))
            fscores, rscores = D(augmentation(fakes, key=next(key))), D(augmentation(x, key=next(key)))
            loss = jax.nn.softplus(fscores) + jax.nn.softplus(-rscores)
            return loss.mean(), {}

        (l, metrics), gradients = loss(D)
        updates, states = Doptim.update(gradients, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics


    @ddp
    def DRstep(G, D, x, states, interval:int=32, key=None):
        @gradients
        def loss(D):
            y, pullback = jax.vjp(D, x)
            (gradients,) = pullback(jnp.ones(y.shape, dtype=ftype))
            gradients = reduce(gradients ** 2, 'b h w c -> b', 'sum')
            loss = .5 * .1 * interval * (gradients.sqrt() - 1) ** 2
            return loss.mean(), {}

        (l, metrics), gradients = loss(D)
        updates, states = Doptim.update(gradients, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics 


    for idx in tqdm(range(config["steps"])):
        batch = cast(ftype)(next(loader))
        G, Gstates, Gloss, metrics = Gstep(G, D, batch, Gstates, key=next(key))
        D, Dstates, Dloss, metrics = Dstep(G, D, batch, Dstates, key=next(key))
        
        if idx % 32 == 0:
            D, Dstates, DRloss, metrics = DRstep(G, D, batch, Dstates, interval=32, key=next(key))

        if idx % 4200 == 0:
            ckpt = folder / idx
            ckpt.mkdir()

            save(ckpt / "G.weight", unreplicate(G))
            save(ckpt / "D.weight", unreplicate(D))
            save(ckpt / "optimisers.ckpt", {"G":Goptim, "D":Doptim})
            save(ckpt / "states.ckpt", {"G":unreplicate(Gstates), "D":unreplicate(Dstates)})

        wandb.log({ "G":Gloss.mean(), "D":Dloss.mean(), "DR":DRloss.mean() })

    wandb.finish()

if __name__ == "__main__":
    train()