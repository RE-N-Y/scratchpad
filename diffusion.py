import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Integer, Array

import optax
import numpy as onp
import equinox
from equinox import nn, static_field as buffer, Module
from einops import rearrange, reduce, repeat, pack
from compression import ViTQuantiser
from layers import Convolution, Projection, SelfAttention, CrossAttention, Layernorm, Groupnorm, MLP, Crossformer, SinusodialEmbedding
from toolkit import *
from dataloader import *

class Diffusion(Module):
    betas:jnp.ndarray
    alphas:jnp.ndarray
    palphas:jnp.ndarray
    steps:int

    # 0.0001 #  0.02
    def __init__(self, start=0.00085, end=0.0120, steps:int=1024, key=None):
        self.betas = jnp.linspace(start, end, steps)
        self.alphas = 1 - self.betas
        self.palphas = jnp.cumproduct(self.alphas)
        self.steps = steps
        
    def q(self, data, t):
        mean = jnp.sqrt(self.palphas[t]) * data
        std = jnp.sqrt(1 - self.palphas[t])
        return mean, std

    @forward
    def qsample(self, data, t, eps=None, key=None):
        eps = default(eps, jr.normal(key, data.shape, dtype=data.dtype))
        mean, std = self.q(data, t=t)
        return mean + std * eps
    
    @forward
    def psample(self, xt, t, prediction, key=None):
        alpha, palpha = self.alphas[t], self.palphas[t]
        std = jnp.sqrt(self.betas[t])

        deps = (1 - alpha) / jnp.sqrt(1 - palpha)
        mean = 1 / jnp.sqrt(alpha) * (xt - deps * prediction)
        eps = jr.normal(key, xt.shape, dtype=xt.dtype)
        
        return mean + std * eps


class Resnet(Module):
    depthwise:Convolution
    time:Projection
    prenorm:Groupnorm
    postnorm:Groupnorm
    projection:Convolution
    mlp:MLP
    
    def __init__(self, nin:int, non:int, time:int=256, groups=8, bias=True, key=None):
        key = RNG(key)
        self.depthwise = Convolution(nin, nin, kernel=7, padding=3, groups=nin, bias=bias, key=next(key))
        self.time = Projection(time, nin, bias=bias, key=next(key))
        self.prenorm, self.postnorm = Groupnorm(groups, nin), Groupnorm(groups, nin)
        self.mlp = MLP(nin, activation="swish", bias=bias, key=next(key))
        self.projection = Convolution(nin, non, kernel=7, padding=3, groups=nin, bias=bias, key=next(key))

    def __call__(self, x, t, key=None):
        time = self.time(t)
        x = self.depthwise(self.prenorm(x + time)) + x
        x = self.mlp(self.postnorm(x + time), key=key) + x
        x = self.projection(x)

        return x

class Upsample(Module):
    layer:Convolution
    scale:float = buffer()

    def __init__(self, nin, non, scale=2, bias=True, key=None):
        self.scale = scale
        self.layer = Convolution(nin, non, kernel=3, padding=1, bias=bias, key=key)
    
    def __call__(self, x, key=None):
        h, w, c = x.shape
        x = jax.image.resize(x, (self.scale * h, self.scale * w, c), method="bilinear")
        x = self.layer(x)
        return x

class Downsample(Module):
    layer:Convolution
    scale:float = buffer()

    def __init__(self, nin, non, scale=2, bias=True, key=None):
        self.scale = scale
        self.layer = Convolution(nin, non, kernel=3, stride=scale, padding=1, bias=bias, key=key)
    
    def __call__(self, x, key=None):
        x = self.layer(x)
        return x

class Upstage(Module):
    upsample:Upsample
    resnets:list

    def __init__(self, nin:int, non:int, time:int, blocks:int=3, groups:int=8, up=True, bias=True, key=None):
        key = RNG(key)
        self.resnets = [Resnet(nin+nin, nin+nin, time, groups=groups, bias=bias, key=next(key)) for _ in range(blocks)]
        self.upsample = Upsample(nin+nin, non, scale=(2 if up else 1), bias=bias, key=next(key))

    def __call__(self, x, r, t, key=None):
        key = RNG(key)
        x = rearrange([x,r], 'n h w c -> h w (n c)')
        for layer in self.resnets : x = layer(x, t, key=next(key))
        x = self.upsample(x, key=next(key))
        return x

class Downstage(Module):
    downsample:Downsample
    resnets:list

    def __init__(self, nin:int, non:int, time:int, blocks:int=3, groups:int=8, down=True, bias=True, key=None):
        key = RNG(key)
        self.downsample = Downsample(nin, non, scale=(2 if down else 1), bias=bias, key=next(key))
        self.resnets = [Resnet(non, non, time, groups=groups, bias=bias, key=next(key)) for _ in range(blocks)]

    def __call__(self, x, t, key=None):
        key = RNG(key)
        x = self.downsample(x, key=next(key))
        for layer in self.resnets : x = layer(x, t, key=next(key))
        return x

class Fusion(Module):
    down:Downsample
    middle:Crossformer
    up:Upsample

    def __init__(self, nin:int, non:int, context=None, heads:int=8, bias=True, key=None):
        key = RNG(key)
        self.down = Downsample(nin, non, bias=bias, key=next(key))
        self.middle = Crossformer(non, default(context, non), heads=heads, bias=bias, key=next(key))
        self.up = Upsample(non, nin, bias=bias, key=next(key))

    def __call__(self, x, ctx=None, key=None):
        h,w,c = x.shape

        x = self.down(x)
        x = rearrange(x, 'h w c -> (h w) c')
        x = self.middle(x, default(ctx,x), key=key)
        x = rearrange(x, '(h w) c -> h w c', h=h//2, w=w//2)
        x = self.up(x)

        return x

class UNet(Module):
    tpe:SinusodialEmbedding
    encoder:list
    fusion:Fusion
    decoder:list

    def __init__(self, features:list, channels:int, time:int=256, context=None, blocks:int=3, heads=8, bias=True, key=None):
        key = RNG(key)
        
        self.tpe = SinusodialEmbedding(time, key=next(key))

        [first, *rest, last, fusion] = features
        updowns = [first, *rest, last]
        ninnon = list(zip(updowns[:-1], updowns[1:]))
        
        input = Downstage(channels, first, time=time, blocks=blocks, down=False, bias=bias, key=next(key))
        encoder = [Downstage(nin, non, time=time, blocks=blocks, bias=bias, key=next(key)) for nin,non in ninnon]
        self.encoder = [input] + encoder

        self.fusion = Fusion(last, fusion, context=context, heads=heads, bias=bias, key=next(key))

        output = Upstage(first, channels, time=time, blocks=blocks, up=False, bias=bias, key=next(key))
        decoder = [Upstage(non, nin, time=time, blocks=blocks, bias=bias, key=next(key)) for nin,non in ninnon]
        self.decoder = [output] + decoder
        

    @forward
    def __call__(self, x, t, ctx=None, key=None):
        key = RNG(key)

        hiddens = []
        time = t.astype(x.dtype)
        time = self.tpe(time)
        
        for idx, stage in enumerate(self.encoder): 
            hiddens.append(x := stage(x, time, key=next(key)))

        x = self.fusion(x, ctx=ctx, key=next(key))

        for idx, stage in reversed(list(enumerate(self.decoder))):
            x = stage(x, hiddens[idx], time, key=next(key))

        return x

class Timeformer(Module):
    attention:SelfAttention
    time:Projection
    prenorm:Layernorm
    postnorm:Layernorm
    mlp:MLP

    def __init__(self, features:int, time:int, heads:int=12, dropout=0., bias=True, key=None):
        key = RNG(key)
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.time = Projection(time, features, bias=bias, key=next(key))
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], t:Float[Array, "t"], key=None):
        akey, okey = jr.split(key)
        time = self.time(t)
        x = self.attention(self.prenorm(x + time), key=akey) + x
        x = self.mlp(self.postnorm(x + time), key=okey) + x

        return x

class Transformer(Module):
    wpe:jnp.ndarray
    tpe:SinusodialEmbedding
    input:Projection
    stem:Crossformer
    encoder:list
    fusion:Crossformer
    decoder:list
    output:Projection

    def __init__(self, length:int=1024, channels:int=16, features:int=768, time:int=512, heads:int=12, depth:int=12, dropout=0., bias=True, key=None):
        key = RNG(key)
        
        self.wpe = jnp.zeros((length, features))
        self.tpe = SinusodialEmbedding(time)
        self.input = Projection(channels, features, bias=bias, key=next(key))
        self.stem = Crossformer(features, features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.encoder = [Timeformer(features, time, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.fusion = Crossformer(features, features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.decoder = [Timeformer(features, time, heads=heads, dropout=dropout, bias=bias, key=next(key)) for _ in range(depth)]
        self.output = Projection(features, channels, bias=bias, key=next(key))

    @forward
    def __call__(self, data, t, context=None, previous=None, key=None):
        key = RNG(key)
        
        time = t.astype(data.dtype)
        time = self.tpe(time)
        hiddens = self.input(data) + self.wpe

        hiddens = self.stem(hiddens, default(previous, hiddens), key=next(key))
        for layer in self.encoder : hiddens = layer(hiddens, time, key=next(key))
        hiddens = self.fusion(hiddens, default(context, hiddens), key=next(key))
        for layer in self.decoder : hiddens = layer(hiddens, time, key=next(key))

        data = self.output(hiddens)

        return data

def adjust(model, depth=12, scale=.02, key=None):
    key = RNG(key)
    input = fusion = 1
    depth = input + depth + fusion + depth
    target = lambda c,m : isinstance(m,c)

    surgery = lambda module : [scale * jr.normal(next(key), module.weight.shape)]
    model = initialise(model, partial(target, Projection), ["weight"], surgery)

    surgery = lambda module : [module.output.weight / math.sqrt(2)]
    model = initialise(model, partial(target, MLP), ["output.weight"], surgery)

    surgery = lambda module : [module.out.weight / math.sqrt(2)]
    model = initialise(model, partial(target, (SelfAttention, CrossAttention)), ["out.weight"], surgery)
    
    return model

import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image

@click.command()
@click.option("--compressor", type=Path)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=10000)
@click.option("--lr", type=float, default=6e-6)
@click.option("--batch", default=4, type=int)
@click.option("--length", type=int, default=1024)
@click.option("--features", type=int, default=768)
@click.option("--channels", type=int, default=16)
@click.option("--heads", type=int, default=12)
@click.option("--depth", type=int, default=12)
@click.option("--bias", type=bool, default=True)
@click.option("--dropout", type=float, default=0)
@click.option("--precision", type=str, default="single")
def train(**cfg):
    wandb.init(project = "diffusion", config = cfg)
    folder = Path(f"diffusion/{wandb.run.id}")
    folder.mkdir()

    key = jr.PRNGKey(42)
    key = RNG(key)

    dataset = Path("dataset/mnist")
    augmentations = [
        # T.RandomApply([T.RandomResizedCrop((256,256), scale=(0.7,1.0))]), 
        T.RandomHorizontalFlip(0.2), 
        T.RandomAdjustSharpness(2,0.3), 
        T.RandomAutocontrast(0.3),
        T.Resize((32,32))
    ]

    dataset = ImageDataset(dataset, augmentations=augmentations)
    loader = dataloader(dataset, cfg["batch"])

    grads = partial(gradients, precision=cfg["precision"])
    sampler = Diffusion(steps=1024)
    
    D = UNet(features=[16,32,64,128], channels=3, heads=cfg["heads"], bias=cfg["bias"], key=next(key))
    optimisers = optax.adamw(cfg["lr"])
    states = optimisers.init(D)

    D, states = replicate(D), replicate(states)

    def sample(denoise, steps:int=1024, shape=(4,32,32,3), key=None):
        b,h,w,d = shape
        key = RNG(key)

        xt = jr.normal(next(key), shape)

        for t in tqdm(reversed(range(steps)), total=steps):
            t = jnp.full((b,), t, dtype=int)
            noise = denoise(xt, t, key=next(key))
            xt = sampler.psample(xt, t, noise, key=next(key))

        xt = rearrange(xt, '(n m) h w c -> (n h) (m w) c', n=2, m=2)
        image = dataset.tensor_to_image(xt)
        image.save(folder / "samples.png")

    @ddp
    def step(D, images, states, key=None):
        key = RNG(key)

        @grads
        def loss(D, sampler, images):
            z = images
            b, *_ = images.shape
            noise = jr.normal(next(key), z.shape, dtype=z.dtype)

            t = jr.randint(next(key), (b,), 0, 1024)
            xt = sampler.qsample(z, t, eps=noise, key=next(key))

            y = D(xt, t, key=next(key))
            loss = optax.huber_loss(y, noise)
            
            return loss.mean(), {}

        (l, metrics), gradients = loss(D, sampler, images)
        updates, states = optimisers.update(gradients, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics

    batch = next(loader)
    for idx in tqdm(range(cfg["steps"])):
        every = lambda n : idx % n == 0

        D, states, loss, metrics = step(D, batch, states, key=next(key))
        if every(1042): sample(unreplicate(D), key=next(key))
        wandb.log({ "loss":onp.mean(loss) })

    wandb.finish()

if __name__ == "__main__":
    train()