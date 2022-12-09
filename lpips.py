import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange, reduce, repeat
from equinox import nn, static_field, Module
from equinox import tree_deserialise_leaves as load
from .layers import Activation, Convolution, Maxpool, normalise
from .toolkit import RNG, forward
from pathlib import Path

class VGGFeatures(Module):
    features:list

    def __init__(self, key=None):
        key = RNG(key)
        self.features = [
            Convolution(3, 64, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(64, 64, 3, padding=1, key=next(key)), Activation("relu"),
            Maxpool((2,2)),
            Convolution(64, 128, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(128, 128, 3, padding=1, key=next(key)), Activation("relu"),
            Maxpool((2,2)),
            Convolution(128, 256, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(256, 256, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(256, 256, 3, padding=1, key=next(key)), Activation("relu"),
            Maxpool((2,2)),
            Convolution(256, 512, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(512, 512, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(512, 512, 3, padding=1, key=next(key)), Activation("relu"),
            Maxpool((2,2)),
            Convolution(512, 512, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(512, 512, 3, padding=1, key=next(key)), Activation("relu"),
            Convolution(512, 512, 3, padding=1, key=next(key)), Activation("relu")
        ]

    def __call__(self, x, slices = [-1], key=None):
        features = []
        for block in self.features:
            x = block(x)
            features.append(x)

        return x, [features[idx] for idx in slices]

class LPIPS(Module):
    backbone:Module
    linears:list
    slices:list = static_field()
    mean:jnp.ndarray = static_field()
    std:jnp.ndarray = static_field()

    def __init__(self, slices = [3, 8, 15, 22, 29], features = [64, 128, 256, 512, 512], key=None):
        key = RNG(key)
        self.slices = slices
        self.backbone = VGGFeatures(key=next(key))
        self.mean, self.std = jnp.array([-.030,-.088,-.188]), jnp.array([.458,.448,.450])
        self.linears = [ Convolution(nin, 1, 1, bias=False, key=next(key)) for nin in features ]

    @classmethod
    def load(cls, path:Path):
        key = jr.PRNGKey(42)
        return load(path, cls(key=key))

    @forward
    def __call__(self, x, y, key=None):
        features = {}
        difference = 0.

        # scale x, y
        mean, std = self.mean.astype(x.dtype), self.std.astype(y.dtype)
        x, y = (x - mean) / std, (y - mean) / std

        _, features['x'] = self.backbone(x, self.slices)
        _, features['y'] = self.backbone(y, self.slices)

        for idx, (x, y) in enumerate(zip(features['x'], features['y'])):
            d = normalise(x) - normalise(y)
            d = reduce(self.linears[idx](d ** 2), 'h w c -> 1 1 c', 'mean')
            difference += d

        return difference
    