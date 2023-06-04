import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer

import math
import numpy as onp
from einops import rearrange, repeat, reduce, unpack
from equinox import nn, static_field as buffer, Module
from toolkit import RNG
from functools import partial

pair = lambda t : (t,t)
exists = lambda t : t is not None
lecun = jax.nn.initializers.lecun_uniform()
xavier = jax.nn.initializers.xavier_uniform()

ACTS = { 
    "gelu":jax.nn.gelu, 
    "egelu":partial(jax.nn.gelu, approximate=False), 
    "agelu":lambda x : x * jax.nn.sigmoid(1.702 * x),
    "ngelu":lambda x : .5 * x * (1. + jax.lax.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 + x ** 3))),
    "swish":jax.nn.silu,
    "tanh":jax.nn.tanh,
    "relu":jax.nn.relu 
}

FloatArray = Float[Array, "..."]
IntArray = Integer[Array, "..."]

class Sequential(Module):
    layers:list

    def __init__(self, layers:list, key=None):
        self.layers = layers

    def __call__(self, x, key=None):
        rng = RNG(key)
        for layer in self.layers : x = layer(x, key=next(rng) if exists(key) else None)
        return x

class Activation(Module):
    function:str = buffer()
    def __init__(self, function:str, key=None):
        self.function = function
    def __call__(self, x, key=None):
        return ACTS[self.function](x)

def convolve(x:Float[Array, "b h w c"], w:Float[Array, "h w i o"], stride:int=1, padding=0, groups=1, format=("NHWC", "HWIO", "NHWC")):
    return jax.lax.conv_general_dilated(x, w, pair(stride), pair(pair(padding)), feature_group_count=groups, dimension_numbers=format)

def normalise(x, axis=-1, eps=1e-12):
    return x * jax.lax.rsqrt(jnp.sum(x ** 2, axis=axis, keepdims=True) + eps)

class Convolution(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    kernel:int = buffer()
    stride:int = buffer()
    padding:int = buffer()
    groups:int = buffer()


    def __init__(self, nin, non, kernel:int=3, stride:int=1, padding:int=0, groups:int=1, bias=True, key=None):
        self.weight = lecun(key, (kernel, kernel, nin // groups, non))
        self.bias = jnp.zeros((non,)) if bias else None
        self.kernel, self.stride, self.padding, self.groups = kernel, stride, padding, groups

    def __call__(self, x:Float[Array, "h w c"], key=None):
        x = rearrange(x, 'h w c -> 1 h w c')
        x = convolve(x, self.weight, self.stride, self.padding, groups=self.groups)
        if self.bias is not None: x = x + self.bias
        x = rearrange(x, '1 h w c -> h w c')
        return x

class Maxpool(Module):
    window:tuple = buffer()
    def __init__(self, window): self.window = window
    def __call__(self, x:Float[Array, "h w c"], key=None): 
        kh, kw = self.window
        return reduce(x, '(h kh) (w kw) c -> h w c', 'max', kh=kh, kw=kw)

class Embedding(Module):
    pages:int = buffer()
    weight:jnp.ndarray

    def __init__(self, pages:int, features:int, weight=None, key=None):
        self.pages = pages
        self.weight = jr.normal(key, (pages, features)) if weight is None else weight

    def __call__(self, idxes:IntArray):
        idxes = jax.nn.one_hot(idxes, self.pages, dtype=self.weight.dtype)
        return idxes @ self.weight

class Projection(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray

    def __init__(self, nin, non, bias=True, key=None):
        self.weight = lecun(key, (nin, non))
        self.bias = jnp.zeros((non,)) if bias else None

    def __call__(self, x:FloatArray, key=None):
        x = x @ self.weight
        if self.bias is not None: x = x + self.bias
        return x

class Layernorm(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    eps:float

    def __init__(self, shape, affine=True, eps=1e-12, key=None):
        self.weight = jnp.ones(shape) if affine else None
        self.bias = jnp.zeros(shape) if affine else None
        self.eps = eps

    def __call__(self, x:Array, key=None):
        c = x - x.mean(axis=-1, keepdims=True)
        s = (c ** 2).mean(axis=-1, keepdims=True)
        x = c * jax.lax.rsqrt(s + self.eps)
        if self.weight is not None: x = x * self.weight
        if self.bias is not None: x = x + self.bias

        return x

class Groupnorm(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    groups:int
    eps:float

    def __init__(self, groups:int, features:int, affine=True, eps=1e-12, key=None):
        self.groups = groups
        self.weight = jnp.ones([features]) if affine else None
        self.bias = jnp.zeros([features]) if affine else None
        self.eps = eps

    def __call__(self, x:Array, key=None):
        x = rearrange(x, '... (c g) -> ... c g', g=self.groups)
        c = x - reduce(x, '... g -> g', 'mean')
        s = reduce(c ** 2, '... g -> g', 'mean')
        x = c * jax.lax.rsqrt(s + self.eps)
        x = rearrange(x, '... c g -> ... (c g)')

        if self.weight is not None: x = x * self.weight
        if self.bias is not None: x = x + self.bias

        return x

class SelfAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    dropout:nn.Dropout
    causal:bool = buffer()
    heads:int = buffer()
    features:int = buffer()
    scale:float = buffer()

    def __init__(self, features:int, heads:int=12, dropout:float=0, causal=False, bias=True, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = math.sqrt(features // heads)
        self.causal = causal

        self.query = Projection(features, features, bias=bias, key=next(key))
        self.key = Projection(features, features, bias=bias, key=next(key))
        self.value = Projection(features, features, bias=bias, key=next(key))
        self.out = Projection(features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        

    def mask(self, attention):
        _,m,n = attention.shape
        dinfo = jnp.finfo(attention.dtype)
        mask = jnp.ones((m,n), dtype=int)
        attention = jnp.where(jnp.tril(mask == 1), attention, dinfo.min)

        return attention

    def __call__(self, x:Float[Array, "n d"], key=None):
        akey, okey = jr.split(key)
        q,k,v = self.query(x), self.key(x), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'n (h d) -> h n d', h=self.heads), (q,k,v))
        k = rearrange(k, 'h n d -> h d n')

        attention = q @ k / self.scale # [h m d] @ [h d n] = [h m n]
        if self.causal: attention = self.mask(attention)  
        attention = self.dropout(jax.nn.softmax(attention), key=akey)

        outputs = rearrange(attention @ v, 'h n d -> n (h d)')
        outputs = self.dropout(self.out(outputs), key=okey)

        return outputs


class ParallelAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    qnorm:Layernorm
    knorm:Layernorm
    dropout:nn.Dropout
    causal:bool = buffer()
    heads:int = buffer()
    features:int = buffer()
    scale:float = buffer()

    def __init__(self, features:int, heads:int=12, dropout:float=0, causal=False, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = math.sqrt(features // heads)
        self.causal = causal

        self.query = Projection(features, features, bias=False, key=next(key))
        self.qnorm = Layernorm([features])
        self.key = Projection(features, features, bias=False, key=next(key))
        self.knorm = Layernorm([features])
        self.value = Projection(features, features, bias=False, key=next(key))
        self.out = Projection(features, features, bias=True, key=next(key))
        self.dropout = nn.Dropout(dropout)
        

    def __call__(self, x:Float[Array, "n d"], key=None):
        akey, okey = jr.split(key)
        q,k,v = self.qnorm(self.query(x)), self.knorm(self.key(x)), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'n (h d) -> h n d', h=self.heads), (q,k,v))
        k = rearrange(k, 'h n d -> h d n')

        attention = q @ k / self.scale # [h m d] @ [h d n] = [h m n]
        attention = self.dropout(jax.nn.softmax(attention), key=akey)

        outputs = rearrange(attention @ v, 'h n d -> n (h d)')
        outputs = self.dropout(self.out(outputs), key=okey)

        return outputs

class CrossAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    dropout:nn.Dropout
    heads:int = buffer()
    features:int = buffer()
    scale:float = buffer()

    def __init__(self, features:int, context:int,  heads:int=12, dropout:float = 0, bias=False, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = math.sqrt(features // heads)

        self.query = Projection(features, features, bias=bias, key=next(key))
        self.key = Projection(context, features, bias=bias, key=next(key))
        self.value = Projection(context, features, bias=bias, key=next(key))
        self.out = Projection(features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x:Float[Array, "n d"], ctx:Float[Array, "m c"], key=None):
        akey, okey = jr.split(key)
        q,k,v = self.query(x), self.key(ctx), self.value(ctx)
        q,k,v = map(lambda x : rearrange(x, 'n (h d) -> h n d', h=self.heads), (q,k,v))

        k = rearrange(k, 'h n d -> h d n')
        attention = q @ k / self.scale # [h m d] @ [h d n] = [h m n]

        attention = self.dropout(jax.nn.softmax(attention), key=akey)
        outputs = rearrange(attention @ v, 'h n d -> n (h d)')
        outputs = self.dropout(self.out(outputs), key=okey)

        return outputs

class MLP(Module):
    input:Projection
    output:Projection
    dropout:nn.Dropout
    activation:str = buffer()

    def __init__(self, features:int, activation:str="gelu", dropout:float=0, bias=True, key=None):
        key = RNG(key)

        self.input = Projection(features, 4 * features, bias=bias, key=next(key))
        self.output = Projection(4 * features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.activation = Activation(activation)

    def __call__(self, x:Float[Array, "n d"], key=None):
        ikey, okey = jr.split(key)

        x = self.input(x)
        x = self.dropout(self.activation(x), key = ikey)
        x = self.dropout(self.output(x), key = okey)

        return x


class FocalModulation(Module):
    f:Projection
    h:Convolution
    layers:list
    projection:Projection
    dropout:nn.Dropout
    level:int = buffer()
    kernels:list = buffer()
    
    def __init__(self, features:int, window:int=3, level:int=3, factor=2, bias=True, dropout=0., key=None):
        key = RNG(key)

        last = 1
        self.level = level
        self.f = Projection(features, 2 * features + (level + last), bias=bias, key=next(key))
        self.h = Convolution(features, features, 1, stride=1, bias=bias, key=next(key))
        
        self.kernels = [factor * k + window for k in range(level)]
        self.layers = [ Convolution(features, features, kernel, padding=kernel//2, groups=features, bias=False, key=next(key)) for kernel in self.kernels]
        self.projection = Projection(features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x, key=None):
        h, w, c = x.shape
        
        x = self.f(x)
        query, ctx, gates = unpack(x, [[c], [c+c]], 'h w *')
        
        aggregation = 0
        for idx, layer in enumerate(self.layers):
            ctx = jax.nn.gelu(layer(ctx))
            aggregation = aggregation + ctx * gates[idx]    
        aggregation = jax.nn.gelu(reduce(ctx, 'h w c -> 1 1 c', 'mean')) * gates[self.level]
        
        out = query * self.h(aggregation)
        out = self.dropout(self.projection(out), key=key)
        
        return out
    
class Selformer(Module):
    attention:SelfAttention
    mlp:MLP
    prenorm:Layernorm
    postnorm:Layernorm

    def __init__(self, features:int, heads:int=12, dropout:float=0, causal=False, bias=True, key=None):
        key = RNG(key)
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, causal=causal, bias=bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], key=None):
        akey, okey = jr.split(key)
        x = self.attention(self.prenorm(x), key=akey) + x
        x = self.mlp(self.postnorm(x), key=okey) + x

        return x

class Crossformer(Module):
    mlp:MLP
    attention:CrossAttention
    qnorm:Layernorm
    kvnorm:Layernorm
    postnorm:Layernorm

    def __init__(self, features:int, context:int, heads:int=12, dropout:float=0, bias=True, key=None):
        key = RNG(key)
        self.qnorm, self.kvnorm, self.postnorm = Layernorm([features]), Layernorm([context]), Layernorm([features])
        self.attention = CrossAttention(features, context, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], ctx:Float[Array, "m c"], key=None):
        akey, okey = jr.split(key)
        x = self.attention(self.qnorm(x), self.kvnorm(ctx), key=akey) + x
        x = self.mlp(self.postnorm(x), key=okey) + x

        return x


class FourierEmbedding(Module):
    embedding:jnp.ndarray
    bands:int
    resolutions:tuple
    coordinates:bool

    def __init__(self, bands:int, resolutions:tuple, coordinates=True, key=None):
        self.embedding = self.fpe(bands, resolutions, coordinates=coordinates)

    def __call__(self, key=None):
        return self.embedding

    def coordinates(self, *resolutions):
        linspace = lambda num : jnp.linspace(-1, 1, num=num, endpoint=True)
        resolutions = [ linspace(r) for r in resolutions]
        grid = jnp.meshgrid(*resolutions, indexing='ij')
        positions = jnp.stack(grid, axis=-1)
        positions = rearrange(positions, '... i -> (...) i')

        return positions

    # taken from deepmind's perceiver implementation
    def fpe(self, bands=32, resolutions=(256,256), coordinates=True):
        # build linear positions -1 ~ +1 of shape [256 * 256, coordinates]
        coordinates = coordinates(*resolutions)
        # Nyquist frequency at the target resolution:
        frequencies = jnp.stack([jnp.linspace(1, r / 2, num=bands, endpoint=True) for r in resolutions])

        # Get frequency bands for each spatial dimension.
        # Output is size [n, d * bands]
        features = coordinates[:, :, None] * frequencies[None, :, :]
        features = rearrange(features, 'n ... -> n (...)')
        features = jnp.concatenate([jnp.sin(jnp.pi * features), jnp.cos(jnp.pi * features)], axis=-1)
        
        # Concatenate the raw input coordinates.
        if coordinates: features = jnp.concatenate([coordinates, features], axis=-1)
        
        return features

class SinusodialEmbedding(Module):
    features:int
    base:int

    def __init__(self, features:int, base:int=16384, key=None):
        self.features = features
        self.base = base

    def __call__(self, time, key=None):
        half = self.features // 2
        frequencies = math.log(self.base) / (half - 1)
        coordinates = jnp.arange(half, dtype=time.dtype)
        frequencies = jnp.exp(coordinates * -frequencies)
        frequencies = time * frequencies
        embeddings = jnp.concatenate((jnp.sin(frequencies), jnp.cos(frequencies)))

        return embeddings