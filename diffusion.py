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
from layers import Convolution, Projection, SelfAttention, CrossAttention, Layernorm, Groupnorm, MLP, GLU, Crossformer, SinusodialEmbedding, Identity
from toolkit import *
from dataloader import *


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
        self.mlp = GLU(nin, bias=bias, key=next(key))
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

    def __init__(self, nin:int, non:int, time:int, blocks:int=3, heads:int=8, groups:int=8, up=True, attention=False, bias=True, key=None):
        key = RNG(key)
        self.resnets = [Resnet(nin+nin, nin+nin, time, groups=groups, bias=bias, key=next(key)) for _ in range(blocks)]
        self.attentions = [SelfAttention(nin+nin, heads=heads, bias=bias, key=next(key)) if attention else Identity() for _ in range(blocks)]
        self.upsample = Upsample(nin+nin, non, scale=(2 if up else 1), bias=bias, key=next(key))

    def __call__(self, x, r, t, key=None):
        key = RNG(key)
        x = rearrange([x,r], 'n h w c -> h w (n c)')
        for resnet, attention in zip(self.resnets, self.attentions):
            x = resnet(x, t, key=next(key))
            x = attention(x, key=next(key))
        x = self.upsample(x, key=next(key))
        return x

class Downstage(Module):
    downsample:Downsample
    resnets:list

    def __init__(self, nin:int, non:int, time:int, blocks:int=3, heads:int=8, groups:int=8, down=True, attention=False, bias=True, key=None):
        key = RNG(key)
        self.downsample = Downsample(nin, non, scale=(2 if down else 1), bias=bias, key=next(key))
        self.resnets = [Resnet(non, non, time, groups=groups, bias=bias, key=next(key)) for _ in range(blocks)]
        self.attentions = [SelfAttention(non, heads=heads, bias=bias, key=next(key)) if attention else Identity() for _ in range(blocks)]

    def attend(self, module, x, key=None):
        h, w, c = x.shape
        x = rearrange(x, 'h w c -> (h w) c')
        x = module(x, key=key)
        x = rearrange(x, '(h w) c -> h w c', h=h, w=w)
        return x

    def __call__(self, x, t, key=None):
        key = RNG(key)
        x = self.downsample(x, key=next(key))
        for resnet, attn in zip(self.resnets, self.attentions): 
            x = resnet(x, t, key=next(key))
            x = self.attend(attn, x, key=next(key))

        return x

class Fusion(Module):
    middle:Crossformer

    def __init__(self, features:int, heads:int=16, bias=True, key=None):
        key = RNG(key)
        self.middle = SelfAttention(features, heads=heads, bias=bias, key=next(key))

    def __call__(self, x, key=None):
        h,w,c = x.shape
        x = rearrange(x, 'h w c -> (h w) c')
        x = self.middle(x, key=key)
        x = rearrange(x, '(h w) c -> h w c', h=h//2, w=w//2)

        return x

class UNet(Module):
    tpe:SinusodialEmbedding
    encoder:list
    fusion:Fusion
    decoder:list

    def __init__(self, features:int=128, channels:int=3, time:int=256, blocks:int=3, heads=8, bias=True, key=None):
        key = RNG(key)
        
        self.tpe = SinusodialEmbedding(time, key=next(key))
        self.temb = GLU(time, time, bias=bias, key=next(key))
        
        self.encoder = [
            Downstage(channels, features, time=time, heads=heads, down=False, blocks=blocks, bias=bias, key=next(key)), # 64 -> 64
            Downstage(features, 2 * features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)), # 64 -> 32
            Downstage(2 * features, 3 * features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)), # 32 -> 16
            Downstage(3 * features, 4 * features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)) # 16 -> 8
        ]

        self.fusion = Fusion(4 * features, heads=heads, bias=bias, key=next(key))

        self.decoder = [
            Upstage(features, channels, time=time, heads=heads, up=False, attention=True, blocks=blocks, bias=bias, key=next(key)), # 64 -> 64
            Upstage(2 * features, features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)), # 32 -> 64
            Upstage(3 * features, 2 * features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)), # 16 -> 32
            Upstage(4 * features, 3 * features, time=time, heads=heads, attention=True, blocks=blocks, bias=bias, key=next(key)) # 8 -> 16
        ]
        

    @forward
    def __call__(self, x, t, ctx=None, key=None):
        key = RNG(key)

        hiddens = []
        time = self.tpe(t.astype(x.dtype))
        time = self.temb(time)
        
        for idx, stage in enumerate(self.encoder): 
            hiddens.append(x := stage(x, time, key=next(key)))

        x = self.fusion(x, ctx=ctx, key=next(key))

        for idx, stage in reversed(list(enumerate(self.decoder))):
            x = stage(x, hiddens[idx], time, key=next(key))

        return x


def scalings(sig, sigdata):
    totvar = sig ** 2 + sigdata ** 2
    return sigdata ** 2 / totvar, sig* sigdata / totvar.sqrt(), 1 / totvar.sqrt()

def noisify(x, sigdata, key=None):
    key = RNG(key)
    sig = jr.normal(next(key), [len(x)], dtype=x.dtype) * 1.2 - 1.2
    sig = rearrange(jnp.exp(sig), 'b -> b () () ()')
    noise = jr.normal(next(key), x.shape, dtype=x.dtype)

    cskip, cout, cin = scalings(sig, sigdata)
    noised = x + noise * sig
    target = (x - cskip * noised) / cout
    return (noised * cin, sig.squeeze()), target

import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image

@click.command()
@click.option("--size", default=64, type=int)
@click.option("--steps", default=1000042, type=int)
@click.option("--warmup", default=10000)
@click.option("--lr", type=float, default=6e-6)
@click.option("--batch", default=4, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--features", default=128, type=int)
@click.option("--blocks", default=3, type=int)
@click.option("--channels", type=int, default=3)
@click.option("--heads", type=int, default=8)
@click.option("--bias", type=bool, default=True)
@click.option("--precision", type=str, default="half")
def train(**cfg):
    wandb.init(project = "diffusion", config = cfg)
    folder = Path(f"diffusion/{wandb.run.id}")
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

    grads = partial(gradients, precision=cfg["precision"])    
    D = UNet(features=cfg["features"], channels=cfg["channels"], time=cfg["time"], blocks=cfg["blocks"], heads=cfg["heads"], key=next(key))

    optimisers = optax.adabelief(cfg["lr"])
    states = optimisers.init(D)

    D, states = replicate(D), replicate(states)

    @ddp
    def step(D, images, states, key=None):
        key = RNG(key)
        (noised, sig), target = noisify(images, 0.5, key=next(key))

        @grads
        def loss(D):
            denoised = D(noised, sig, key=next(key))
            loss = (denoised - target) ** 2
            return loss.mean(), {}

        (l, metrics), gradients = loss(D)
        updates, states = optimisers.update(gradients, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics

    batch = next(loader)
    for idx in tqdm(range(cfg["steps"])):
        D, states, loss, metrics = step(D, batch['images'], states, key=next(key))
        wandb.log({ "loss":onp.mean(loss) })

    wandb.finish()

if __name__ == "__main__":
    train()