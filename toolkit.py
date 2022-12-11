import jax
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
        out = vmap(f)(x, *args, **kwargs)
        return out
    return inner

def ddp(f):
    parallel = partial(pmap, axis_name="devices")
    devices = jax.local_devices()

    def inner(x, *args, **kwargs):
        x = rearrange(x, '(n b) ... -> n b', n=len(devices))
        if (key := kwargs.get("key")) is not None:
            kwargs["key"] = jr.split(key, num=len(devices))
        out = parallel(f)(x, *args, **kwargs)
        return out

    return inner
        

def gradients(f):
    delta = partial(value_and_grad, has_aux=True)
    def inner(*args, **kwargs):
        (values, metrics), gradients = delta(f)(*args, **kwargs)
        return (values, metrics), jax.lax.pmean(gradients, axis_name="devices")
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

def parameterise(model, filter=equinox.is_array, return_apply=False):
    params, static = equinox.partition(model, filter)

    def apply(params, *args, **kwargs):
        model = equinox.combine(params, static)
        return model(*args, **kwargs)

    return params, apply if return_apply else params

def cast(dtype):
    def convert(model):
        model = jtu.tree_map(lambda t : t.astype(dtype), is_leaf=equinox.is_inexact_array)
        return model
    return convert