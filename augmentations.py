import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as onp
from equinox import nn, static_field, Module
from einops import rearrange, reduce, repeat
from toolkit import RNG, forward

def translation_matrix(tx, ty):
    return jnp.array([
        [1., 0., tx],
        [0., 1., ty],
        [0., 0., 1.],
    ])

def rotation_matrix(radians):
    rx, ry = jnp.cos(radians), jnp.sin(radians)
    return jnp.array([
        [  rx,  ry,  0. ],
        [ -ry,  rx,  0. ],
        [  0.,  0.,  1. ],
    ])


def shear_matrix(radians):
    shear = jnp.tan(radians)
    return jnp.array([
        [1., shear,  0.],
        [0.,    1.,  0.],
        [0.,    0.,  1.],
    ])

def scale_matrix(s):
    return jnp.array([
        [ s,  0,  0],
        [ 0,  s,  0],
        [ 0., 0,  1],
    ])



def apply_affine(image, matrix, method="linear", mode="reflect"):
    H, W, C, dtype = image.shape, image.dtype
    samplers = { "nearest": 0, "linear": 1 }

    y, x = jnp.linspace(-H/2, H/2, num=H) , jnp.linspace(-W/2, W/2, num=W)
    v, c = jnp.ones((1)), jnp.arange(3)
    coordinates = jnp.stack(jnp.meshgrid(y,x,v,c, indexing="ij"), axis=-1)

    # [M, M, M, 0] >> height
    # [M, M, M, 0] >> width
    # [M, M, M, 0] >> depth
    # [0, 0, 0, 1] >> channels

    matrix = matrix.astype(dtype)
    transform = jnp.eye(4, dtype=dtype)
    transform = transform.at[:-1,:-1].set(matrix)
    coordinates = coordinates @ transform

    coordinates += jnp.array([H/2, W/2, 0, 0])
    coordinates = jnp.delete(coordinates, -2, axis=-1) # delete "v" from 'h w v c yx(v)c'
    coordinates = rearrange(coordinates, 'h w v c yxc -> yxc (h w v c)')

    out = jsp.ndimage.map_coordinates(image, coordinates, order=samplers[method], mode=mode)
    out = rearrange(out, '(h w c) -> h w c', h=H, w=W, c=C)

    return out


def identity(x, *_, **__): return x

class CoinFlip(Module):
    augment : Module
    p : float = static_field()

    def __init__(self, augment, p=0.5):
        self.augment = augment
        self.p = p

    @forward
    def __call__(self, image, key=None):
        augment = lambda x, key : self.augment(x, key=key)
        return jax.lax.cond(jr.bernoulli(key, p=self.p), augment, identity, image, key)


class RandomBrightness(Module):
    def __call__(self, x, key=None):
        x += jr.uniform(key, dtype=x.dtype) - 0.5
        return x

class RandomSaturation(Module):
    def __call__(self, x, key=None):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        x = (x - mean) * (jr.uniform(key, dtype=x.dtype) * 2) + mean
        return x

class RandomContrast(Module):
    def __call__(self, x, key=None):
        mean = jnp.mean(x, keepdims=True)
        x = (x - mean) * (jr.uniform(key, dtype=x.dtype) + 0.5) + mean
        return x

class RandomAffine(Module):
    scale = 1.15
    translate = 0.15
    rotate:float = onp.pi / 12
    shear:float = 0.

    def __call__(self, image, key=None):
        key = RNG(key)
        h, w, c, dtype = image.shape, image.dtype
        uniform = lambda minval, maxval : jr.uniform(next(key), minval=minval, maxval=maxval, dtype=dtype)

        scale = uniform(minval=1.0, maxval=self.scale)
        tx = uniform(minval=-self.translate, maxval=self.translate) * w
        ty = uniform(minval=-self.translate, maxval=self.translate) * h

        rotation = uniform(minval=-1, maxval=1) * self.rotate
        shear = uniform(minval=-1, maxval=1) * self.shear

        matrix = rotation_matrix(rotation) @ \
                 scale_matrix(scale) @ \
                 shear_matrix(shear) @ \
                 translation_matrix(tx,ty)

        out = apply_affine(image, matrix)

        return out

class RandomHorizontalFlip(Module):
    p: float = 0.5

    def __call__(self, x, key=None):
        flip = jr.bernoulli(key, p=self.p)
        return jax.lax.cond(flip, lambda x: jnp.flip(x, axis=-1), identity, x)

class RandomVerticalFlip(Module):
    p: float = 0.5

    def __call__(self, x, key=None):
        flip = jr.bernoulli(key, p=self.p)
        return jax.lax.cond(flip, lambda x: jnp.flip(x, axis=-2), identity, x)

class RandomCutout(Module):
    ratio:float = 0.2

    def __call__(self, x, key=None):
        key = RNG(key)
        h, w, c, dtype = x.shape, x.dtype

        cut_x, cut_y = int(w * self.ratio + 0.5), int(h * self.ratio + 0.5)
        offset_x = jr.randint(next(key), (1, 1), 0, w + (1 - cut_x % 2))
        offset_y = jr.randint(next(key), (1, 1), 0, h + (1 - cut_y % 2))
        grid_y, grid_x = jnp.meshgrid(
            jnp.arange(cut_y), jnp.arange(cut_x), indexing="ij"
        )

        grid_x = jnp.clip(grid_x + offset_x - cut_x // 2, 0, w-1)
        grid_y = jnp.clip(grid_y + offset_y - cut_y // 2, 0, h-1)

        mask = jnp.ones((h,w), dtype=dtype)
        mask = mask.at[grid_y, grid_x].set(0)
        x *= rearrange(mask, 'h w -> h w 1')

        return x