import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Integer

import numpy as onp
from einops import rearrange, repeat, reduce
from equinox import nn, static_field, Module
from .toolkit import RNG

pair = lambda t : (t,t)
lecun = jax.nn.initializers.lecun_normal()
xavier = jax.nn.initializers.xavier_normal()
ACTS = { "gelu":jax.nn.gelu, "tanh":jax.nn.tanh, "relu":jax.nn.relu }

FloatArray = Float[Array, "..."]
IntArray = Integer[Array, "..."]

class Activation(Module):
    function:str = static_field()
    def __init__(self, function:str, key=None):
        self.function = function
    def __call__(self, x, key=None):
        return ACTS[self.function](x)

def convolve(x:Float[Array, "b h w c"], w:Float[Array, "h w i o"], stride:int=1, padding=0, groups=1, format=("NHWC", "HWIO", "NHWC")):
    return jax.lax.conv_general_dilated(x, w, pair(stride), pair(pair(padding)), feature_group_count=groups, dimension_numbers=format)

def normalise(x:Float[Array, "... c"], eps=1e-12):
    return x * jax.lax.rsqrt(reduce(x ** 2, '... c -> ... 1', 'sum') + eps)

class Convolution(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    kernel:int = static_field()
    stride:int = static_field()
    padding:int = static_field()
    groups:int = static_field()


    def __init__(self, nin, non, kernel:int=3, stride:int=1, padding:int=0, groups:int=1, use_bias=True, key=None):
        self.weight = lecun(key, (kernel, kernel, nin, non))
        self.bias = jnp.zeros((non,)) if use_bias else None
        self.kernel, self.stride, self.padding, self.groups = kernel, stride, padding, groups

    def __call__(self, x:Float[Array, "h w c"], key=None):
        x = rearrange(x, 'h w c -> 1 h w c')
        x = convolve(x, self.weight, self.stride, self.padding, groups=self.groups)
        if self.bias is not None: x = x + self.bias
        x = rearrange(x, '1 h w c -> h w c')
        return x

class Maxpool(Module):
    window:tuple = static_field()
    def __init__(self, window): self.window = window
    def __call__(self, x:Float[Array, "h w c"], key=None): 
        kh, kw = self.window
        return reduce(x, '(h kh) (w kw) c -> h w c', kh=kh, kw=kw)

class Embedding(Module):
    pages:int = static_field()
    weight:jnp.ndarray

    def __init__(self, pages:int, features:int, key=None):
        self.pages = pages
        self.weight = jr.normal(key, (pages, features))

    def __call__(self, idxes:IntArray):
        return jax.nn.one_hot(idxes, self.pages) @ self.weight

class Projection(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray

    def __init__(self, nin, non, use_bias=True, scale=.02, key=None):
        self.weight = jr.normal(key, (nin, non)) * scale
        self.bias = jnp.zeros((non,)) if use_bias else None

    def __call__(self, x:FloatArray, key=None):
        x = x @ self.weight
        if self.bias is not None: x = x + self.bias
        return x

class Layernorm(Module):
    weight:jnp.ndarray
    bias:jnp.ndarray
    eps:float = static_field()

    def __init__(self, shape, affine=True, eps=1e-12, key=None):
        self.weight = jnp.ones(shape) if affine else None
        self.bias = jnp.zeros(shape) if affine else None
        self.eps = eps

    def __call__(self, x:FloatArray, key=None):
        c = x - x.mean(axis=-1, keepdims=True)
        s = (c ** 2).mean(axis=-1, keepdims=True)
        x = c * jax.lax.rsqrt(s + self.eps)
        if self.weight is not None: x = x * self.weight
        if self.bias is not None: x = x + self.bias

        return x

class SelfAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    dropout:nn.Dropout
    heads:int = static_field()
    features:int = static_field()
    scale:float = static_field()

    def __init__(self, features:int, heads:int=12, dropout:float=0, use_bias=True, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = onp.sqrt(features // heads)

        self.query = Projection(features, features, use_bias=use_bias, key=next(key))
        self.key = Projection(features, features, use_bias=use_bias, key=next(key))
        self.value = Projection(features, features, use_bias=use_bias, key=next(key))
        self.out = Projection(features, features, use_bias=use_bias, key=next(key))
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x:Float[Array, "n d"], key=None):
        akey, okey = jr.split(key)
        q,k,v = self.query(x), self.key(x), self.value(x)
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
    heads:int = static_field()
    features:int = static_field()
    scale:float = static_field()

    def __init__(self, features:int, context:int,  heads:int=12, dropout:float = 0, use_bias=False, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = onp.sqrt(features // heads)

        self.query = Projection(features, features, use_bias=use_bias, key=next(key))
        self.key = Projection(context, features, use_bias=use_bias, key=next(key))
        self.value = Projection(context, features, use_bias=use_bias, key=next(key))
        self.out = Projection(features, features, use_bias=use_bias, key=next(key))
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
    activation:str = static_field()

    def __init__(self, features:int, activation:str="gelu", dropout:float=0, use_bias=True, key=None):
        key = RNG(key)

        self.input = Projection(features, 4 * features, use_bias=use_bias, key=next(key))
        self.output = Projection(4 * features, features, use_bias=use_bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def __call__(self, x:Float[Array, "n d"], key=None):
        ikey, okey = jr.split(key)

        x = self.input(x)
        x = self.dropout(ACTS[self.activation](x), key = ikey)
        x = self.dropout(self.output(x), key = okey)

        return x

class Selformer(Module):
    attention:SelfAttention
    mlp:MLP
    prenorm:Layernorm
    postnorm:Layernorm

    def __init__(self, features:int, heads:int=12, dropout:float=0, use_bias=True, key=None):
        key = RNG(key)
        self.prenorm, self.postnorm = Layernorm([features]), Layernorm([features])
        self.attention = SelfAttention(features, heads=heads, dropout=dropout, use_bias=use_bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, use_bias=use_bias, key=next(key))

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

    def __init__(self, features:int, context:int, heads:int=12, dropout:float=0, use_bias=True, key=None):
        key = RNG(key)
        self.qnorm, self.kvnorm, self.postnorm = Layernorm([features]), Layernorm([context]), Layernorm([features])
        self.attention = CrossAttention(features, context, heads=heads, dropout=dropout, use_bias=use_bias, key=next(key))
        self.mlp = MLP(features, dropout=dropout, use_bias=use_bias, key=next(key))

    def __call__(self, x:Float[Array, "n d"], ctx:Float[Array, "m c"], key=None):
        akey, okey = jr.split(key)
        x = self.attention(self.qnorm(x), self.kvnorm(ctx), key=akey) + x
        x = self.mlp(self.postnorm(x), key=okey) + x

        return x