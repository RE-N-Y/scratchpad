import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox
from equinox import filter_vmap as vmap, filter_pmap as pmap, filter_value_and_grad as value_and_grad
from einops import rearrange
from functools import partial
from operator import attrgetter

def RNG(old):
    while True:
        new, old = jr.split(old)
        yield new

default = lambda t, backup : backup if t is None else t
forward = partial(vmap, in_axes={"self":None})
batch = vmap

def ddp(f):
    """
    Run DDP on optimisation step function of signature 'step(model, x, ...)'
    """
    parallel = partial(pmap, axis_name="devices")
    devices = jax.local_devices()

    def inner(model, x, *args):
        x = rearrange(x, '(n b) ... -> n b ...', n=len(devices))
        return parallel(f)(model, x, *args)

    return inner

def gradients(function, precision="single"):
    """
    Computes gradient of a function with respective to its first input.
    It also handles AMP by automatically converting model weights / gradients to appoperiate precision
    Apply on a pure function to ensure full performance / minimal pain
    """
    delta = partial(value_and_grad, has_aux=True)
    identity = lambda t : t
    frtype = half if precision == "half" else identity
    bwtype = single if precision == "half" else identity

    def inner(*args, **kwargs):
        (values, metrics), g = delta(function)(*frtype(args), **frtype(kwargs))
        return (values, metrics), jax.lax.pmean(bwtype(g), axis_name="devices")
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
    """Casts any pytree into a specific dtype"""
    def convert(model):
        filter = equinox.is_inexact_array
        dynamic, static = equinox.partition(model, filter)
        dynamic = jtu.tree_map(lambda t : t.astype(dtype), dynamic, is_leaf=filter)
        return equinox.combine(dynamic, static)
    return convert

half = cast(jnp.bfloat16)
single = cast(jnp.float32)

def initialise(model, target, locations, surgery):
    where = lambda model : jtu.tree_leaves(equinox.filter(model, target, is_leaf=target), is_leaf=target)
    replace_fn = lambda m : equinox.tree_at(lambda m : [ attrgetter(l)(m) for l in locations], m, surgery(m))
    model = equinox.tree_at(where, model, replace_fn=replace_fn)

    return model