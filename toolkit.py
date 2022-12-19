import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox
from equinox import Module
from equinox import filter_vmap as vmap, filter_pmap as pmap, filter_value_and_grad as value_and_grad
from einops import rearrange
from functools import partial

def RNG(old):
    while True:
        new, old = jr.split(old)
        yield new

def ema(last, current, decay:float=.99):
    if jnp.isnan(last) : return current
    return decay * last + (1 - decay) * current

def forward(f, static=["self"]):
    batch = partial(vmap, kwargs={ name:None for name in static })

    def inner(self, x, *args, **kwargs):
        b, *_ = x.shape
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key,b)
        out = batch(f)(self, x, *args, **kwargs)

        return out

    return inner

def batch(f):
    def inner(x, *args, **kwargs):
        n, *_ = x.shape
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key, num=n)
        return vmap(f)(x, *args, **kwargs)
    return inner

def ddp(f):
    """
    Run DDP on optimisation step function of signature 'step(model, x, ...)'
    """
    parallel = partial(pmap, axis_name="devices")
    devices = jax.local_devices()

    def inner(model, x, *args, **kwargs):
        x = rearrange(x, '(n b) ... -> n b ...', n=len(devices))
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key, num=len(devices))
        return parallel(f)(model, x, *args, **kwargs)

    return inner

def gradients(f):
    delta = partial(value_and_grad, has_aux=True)
    def inner(*args, **kwargs):
        (values, metrics), g = delta(f)(*args, **kwargs)
        return (values, metrics), jax.lax.pmean(g, axis_name="devices")
    return inner

def replicate(model):
    dynamic, static = equinox.partition(model, equinox.is_array)
    pdynamic = jax.device_put_replicated(dynamic, jax.local_devices())
    pmodel = equinox.combine(pdynamic, static)

    return pmodel

def unreplicate(pmodel):
    pdynamic, static = equinox.partition(pmodel, equinox.is_array)
    dynamic = jax.tree_util.tree_map(lambda x : x[0], pdynamic)
    model = equinox.combine(dynamic, static)

    return model

def parameterise(model, filter=equinox.is_inexact_array):
    params, static = equinox.partition(model, filter)

    def apply(params, *args, **kwargs):
        model = equinox.combine(params, static)
        return model(*args, **kwargs)

    return params, apply

def parameters(model, filter=equinox.is_inexact_array):
    return equinox.filter(model, filter)

def cast(dtype):
    def convert(model):
        filter = equinox.is_inexact_array
        dynamic, static = equinox.partition(model, filter)
        dynamic = jtu.tree_map(lambda t : t.astype(dtype), dynamic, is_leaf=filter)
        return equinox.combine(dynamic, static)
    return convert