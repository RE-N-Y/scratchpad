import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Integer, Array

import math
import optax
import numpy as onp
import equinox
from equinox import nn, static_field as buffer, Module
from einops import rearrange, reduce, repeat, pack
from layers import Convolution, Activation, Sequential, Projection, SelfAttention, CrossAttention, Layernorm, Groupnorm, MLP, GLU, Crossformer, SinusodialEmbedding, Identity
from toolkit import *
from dataloader import *

load = equinox.tree_deserialise_leaves
save = equinox.tree_serialise_leaves

class Imageformer(Module):
    attention:SelfAttention
    mlp:GLU
    prenorm:Layernorm
    postnorm:Layernorm

    def __init__(self, features:int, heads:int=12, bias=False, dropout:float=0, key=None):
        key = RNG(key)
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.mlp = GLU(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x, key=None):
        key = RNG(key)
        x = self.attention(self.prenorm(x), key=next(key)) + x
        x = self.mlp(self.postnorm(x), key=next(key)) + x
        
        return x

class Denoiser(Module):
    tpe:SinusodialEmbedding
    temb:GLU
    wpe:jnp.ndarray
    inputs:Convolution
    encoder:list
    decoder:list
    connections:list
    layernorm:Layernorm
    mlp:Sequential
    outputs:Convolution
    patch:int = buffer()
    ratio:int = buffer()

    def __init__(self, features:int=768, patch:int=4, depth:int=12, heads:int=12, size:int=64, bias=True, key=None):
        key = RNG(key)
        
        self.patch = patch
        self.ratio = size // patch
        self.tpe = SinusodialEmbedding(features, key=next(key))
        self.temb = GLU(features, bias=bias, key=next(key))
        self.wpe = jnp.zeros(((size // patch) ** 2, features))

        self.inputs = Convolution(3, features, patch, stride=patch, bias=bias, key=next(key))
        self.encoder = [Imageformer(features, heads=heads, bias=bias, key=next(key)) for _ in range(depth)]
        self.decoder = [Imageformer(features, heads=heads, bias=bias, key=next(key)) for _ in range(depth)]
        self.connections = [Projection(2 * features, features, bias=bias, key=next(key)) for _ in range(depth)]
        self.layernorm = Layernorm([features])
        self.mlp = Sequential([Layernorm([features]), MLP(features, activation="tanh", bias=bias, key=next(key))])
        self.outputs = Convolution(features, 3 * patch * patch, kernel=1, bias=bias, key=next(key))

    @forward
    def __call__(self, x, t, key=None):
        key = RNG(key)

        T = 1
        t = self.tpe(t.astype(x.dtype))
        t = self.temb(t, key=next(key))

        t = rearrange(t, 't -> () t')
        x = rearrange(self.inputs(x), 'h w c -> (h w) c')
        
        x = x + self.wpe
        x, _ = pack([x, t], '* d')

        hiddens = []
        for layer in self.encoder:
            x = layer(x, key=next(key))
            hiddens.append(x)

        for layer, connection, hidden in zip(self.decoder, self.connections, reversed(hiddens)):
            x, _ = pack([x, hidden], 'n *')
            x = connection(x, key=next(key))
            x = layer(x, key=next(key))

        x = self.layernorm(x[T:])
        x = self.mlp(x, key=next(key))
        
        x = rearrange(x, '(h w) c -> h w c', h=self.ratio, w=self.ratio)
        x = rearrange(self.outputs(x), 'h w (hr wr c) -> (h hr) (w wr) c', hr=self.patch, wr=self.patch)

        return x
        
    

def t2i(x:Float[Array, "b h w c"]) -> onp.ndarray:
    x = x.clip(-1, 1)
    x = x * 0.5 + 0.5
    x = onp.asarray(x * 255, dtype=onp.uint8)
    return x

def scalings(sig, sigdata):
    totvar = sig ** 2 + sigdata ** 2
    return sigdata ** 2 / totvar, sig * sigdata / jnp.sqrt(totvar), 1 / jnp.sqrt(totvar)

def noisify(x, sigdata, key=None):
    key = RNG(key)
    sig = jr.normal(next(key), [len(x)], dtype=x.dtype) * 1.2 - 1.2
    sig = rearrange(jnp.exp(sig), 'b -> b () () ()')
    noise = jr.normal(next(key), x.shape, dtype=x.dtype)

    cskip, cout, cin = scalings(sig, sigdata)
    noised = x + noise * sig
    target = (x - cskip * noised) / cout
    
    return (noised * cin, sig.squeeze()), target

def sigmas(n, smin=0.01, smax=80., rho=7., key=None):
    ramp = jnp.linspace(0, 1, n)
    minrho = smin ** (1 / rho)
    maxrho = smax ** (1 / rho)
    sigs = (maxrho + ramp * (minrho - maxrho)) ** rho

    return jnp.concatenate([sigs, jnp.array([0.])])

def denoise(model, x, sig, sigdata=0.56, key=None):
    key = jr.split(key, len(x))
    sig = jnp.repeat(sig, len(x))
    sig = rearrange(sig, 'b -> b () () ()')
    cskip, cout, cin = scalings(sig, sigdata)
    
    return model(x * cin, sig.squeeze(), key) * cout + x * cskip

def heun(x, sigs, i, model, sigdata=0.56, churn=0., tmin=0., tmax=jnp.inf, noise=1., key=None):
    key = RNG(key)
    n = len(sigs)
    sig, nsig = sigs[i], sigs[i+1]

    gamma = min(churn/(n-1), 2 ** 0.5 - 1) if tmin <= sig <= tmax else 0.
    eps = jr.normal(next(key), x.shape, dtype=x.dtype) * noise
    shat = sig * (gamma+1)
    
    if gamma > 0: x = x + eps * (shat**2-sig**2)**0.5
    
    denoised = denoise(model, x, sig, sigdata=sigdata, key=next(key))
    d = (x - denoised) / sig
    dt = nsig - shat
    nx = x + d * dt
    
    if nsig == 0: return nx
    
    ndenoised = denoise(model, nx, nsig, key=next(key))
    nd = (nx - ndenoised) / nsig
    prime = (d + nd) / 2

    return x + prime * dt

def euler(x, sigs, i, model, sigdata=0.56, key=None):
    sig, nsig = sigs[i], sigs[i+1]
    denoised = denoise(model, x, sig, sigdata=sigdata, key=key)
    return x + (x - denoised) / sig * (nsig - sig)

def generate(model, num=16, steps=96, smax=80, sigdata=0.56, key=None):
    key = RNG(key)
    x = jr.normal(next(key), (num, 64, 64, 3)) * smax
    sigs = sigmas(steps, smax=smax)

    for i in tqdm(range(len(sigs)-1)):
        x = heun(x, sigs, i, model, sigdata=sigdata, key=next(key))
    return x

import math
import wandb
import click
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image

@click.command()
@click.option("--size", default=64, type=int)
@click.option("--steps", default=64, type=int)
@click.option("--warmup", default=8)
@click.option("--cooldown", default=64)
@click.option("--lr", type=float, default=3e-4)
@click.option("--batch", default=128, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--features", default=1024, type=int)
@click.option("--patch", default=2, type=int)
@click.option("--depth", default=12, type=int)
@click.option("--channels", type=int, default=3)
@click.option("--heads", type=int, default=16)
@click.option("--bias", type=bool, default=True)
@click.option("--workers", type=int, default=32)
@click.option("--precision", type=str, default="half")
@click.option("--checkpoint", type=Path)
def train(**cfg):
    wandb.init(project = "diffusion", config = cfg)
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

    loader, length = dataloader(
        "hub://reny/animefaces",
        tensors=["images"],
        batch_size=cfg["batch"],
        transform={"images":transform},
        decode_method={"images":"numpy"},
        num_workers=cfg["workers"],
        buffer_size=16384,
        shuffle=True
    )

    grads = partial(gradients, precision=cfg["precision"])    
    D = Denoiser(features=cfg["features"], depth=cfg["depth"], heads=cfg["heads"], patch=cfg["patch"], key=next(key))

    cfg["warmup"], cfg["steps"], cfg["cooldown"] = cfg["warmup"] * length, cfg["steps"] * length, cfg["cooldown"] * length
    grads = partial(gradients, precision=cfg["precision"])
    lr = optax.warmup_cosine_decay_schedule(0, cfg["lr"], cfg["warmup"], cfg["steps"], cfg["cooldown"])
    optimisers = optax.lion(lr, b1=0.95, b2=0.98, weight_decay=0.1)
    states = optimisers.init(parameters(D))
    
    D, states = replicate(D), replicate(states)

    @ddp
    def step(D, images, states, key=None):
        key = RNG(key)

        @grads
        def loss(D, images):
            (noised, sig), target = noisify(images, 0.56, key=next(key))
            denoised = D(noised, sig, jr.split(next(key), len(noised)))
            loss = (denoised - target) ** 2
            return loss.mean(), {}

        (l, metrics), gradients = loss(D, images)
        updates, states = optimisers.update(gradients, states, D)
        D = equinox.apply_updates(D, updates)

        return D, states, l, metrics

    for idx in tqdm(range(cfg["steps"])):
        batch = next(loader)
        images = rearrange(batch["images"], 'b c h w -> b h w c')
        D, states, loss, metrics = step(D, images, states, dsplit(next(key)))
        wandb.log({ "loss":onp.mean(loss) })

        if idx % 4096 == 0:
            (noised, sig), target = noisify(images[:4], 0.56, key=next(key))
            denoised = unreplicate(D)(noised, sig, jr.split(next(key), len(noised)))
            samples = generate(unreplicate(D), sigdata=0.56, key=next(key))         
    
            wandb.log({ 
                "noised":[wandb.Image(i) for i in t2i(noised)], 
                "denoised":[wandb.Image(i) for i in t2i(denoised)],
                "target":[wandb.Image(i) for i in t2i(target)],
                "samples":[wandb.Image(i) for i in t2i(samples)]
            }, commit=False)

            save(folder / "D.weight", unreplicate(D))
            save(folder / "states.ckpt", unreplicate(states))

    wandb.finish()

if __name__ == "__main__":
    train()
