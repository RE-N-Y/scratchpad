import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

import equinox
from equinox import nn, static_field, Module
from layers import Activation, Embedding, Layernorm, Convolution, Projection, normalise
from einops import rearrange, reduce, repeat
from toolkit import RNG, forward

import math
import gdown
import numpy as onp
from pathlib import Path

KEY = jr.PRNGKey(42)
CFG = { 
    "base":{ 
        "id":"1F_f_T0mdxtjkmiKmSPQKSQ7EcjnDN1hr", 
        "net":{ 
            "vision":{ "length":50, "features":768, "heads":12, "depth":12, "size":224, "patch":32, "bias":True, "eps":1e-5, "key":KEY }, 
            "text":{ "length":77, "vocab":49408, "features":512, "heads":8, "depth":12, "eps":1e-5, "bias":True, "key":KEY }, 
            "common":{ "features":512, "key":KEY }
        } 
    }, 
    "large":{ 
        "id":"174oJZbluLI9kjq79Lvojz6zQxotn7a4_", 
        "net":{ 
            "vision":{ "length":50, "features":1024, "heads":16, "depth":24, "size":224, "patch":14, "bias":True, "eps":1e-5, "key":KEY }, 
            "text":{ "length":77, "vocab":49408, "features":768, "heads":12, "depth":12, "eps":1e-5, "bias":True, "key":KEY }, 
            "common":{ "features":768, "key":KEY }
        } 
    } 
}

class PatchEmbeddings(Module):
    projection:Convolution

    def __init__(self, channels:int, features:int, patch:int, key = None):
        self.projection = Convolution(channels, features, patch, stride=patch, bias=False, key=key)

    def __call__(self, x):
        x = self.projection(x)
        x = rearrange(x, 'h w c -> (h w) c')
        return x

class TextEmbeddings(Module):
    wte:nn.Embedding
    wpe:nn.Embedding
    positions:jnp.ndarray = static_field()

    def __init__(self, vocab, length, features:int=768, key=None):
        key = RNG(key)
        self.wte = nn.Embedding(vocab, features, key=next(key))
        self.wpe = nn.Embedding(length, features, key=next(key))
        self.positions = jnp.arange(length)

    def __call__(self, tokens, positions=None, key=None):
        wte = self.wte(tokens)
        positions = self.positions[:len(tokens)] if positions is None else positions
        wpe = self.wpe(positions)

        return wte + wpe
    

class ViTEmbeddings(Module):
    cls:jnp.ndarray
    wpe:jnp.ndarray
    patchify:nn.Dropout
    dropout:nn.Dropout
    size:int = static_field()
    patch:int = static_field()

    def __init__(self, channels:int, features:int, size:int=256, patch:int=16, dropout=0, key=None):
        cls = 1
        self.size, self.patch = size, patch
        length = (size // patch) * (size // patch)
        
        self.cls = jnp.zeros((cls, features))
        self.patchify  = PatchEmbeddings(channels, features, patch, key=key)
        self.wpe = jnp.zeros((cls + length, features))
        self.dropout = nn.Dropout(dropout)

    def __call__(self, image, key=None):
        embeddings = self.patchify(image)
        embeddings = jnp.concatenate((self.cls, embeddings))
        embeddings += self.wpe
        embeddings = self.dropout(embeddings, key=key)

        return embeddings

class CLIPAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    dropout:nn.Dropout
    length:int = static_field()
    heads:int = static_field()
    features:int = static_field()
    scale:float = static_field()

    def __init__(self, length:int, features:int, heads:int = 12, dropout:float = 0, bias = False, key = None):
        key = RNG(key)

        self.length = length
        self.heads = heads
        self.features = features
        self.scale = onp.sqrt(features // heads)

        self.query = Projection(features, features, bias=bias, key=next(key))
        self.key = Projection(features, features, bias=bias, key=next(key))
        self.value = Projection(features, features, bias=bias, key=next(key))
        self.out = Projection(features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)

    def mask(self, attention:Float[Array,'head src ctx'], mask:Float[Array, 'src ctx']):
        finfo = jnp.finfo(attention.dtype)
        masked = jnp.where(mask, attention, finfo.min)
        return masked


    def __call__(self, x, mask=None, key=None):
        akey, okey = jr.split(key)
        q,k,v = self.query(x), self.key(x), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'n (h d) -> h n d', h=self.heads), (q,k,v))

        k = rearrange(k, 'h n d -> h d n')
        attention = q @ k / self.scale # [h m d] @ [h d n] = [h m n]

        if mask is not None: attention = self.mask(attention, mask)
        attention = self.dropout(jax.nn.softmax(attention), key=akey)

        outputs = rearrange(attention @ v, 'h n d -> n (h d)')
        outputs = self.dropout(self.out(outputs), key=okey)

        return outputs

class CLIPMLP(Module):
    input:Projection
    output:Projection
    dropout:nn.Dropout
    activation:Activation

    def __init__(self, features:int, activation:str="agelu", dropout:float=0, bias=True, key=None):
        key = RNG(key)
        
        self.input = Projection(features, 4 * features, bias=bias, key=next(key))
        self.output = Projection(4 * features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.activation = Activation(activation)

    def __call__(self, x, key=None):
        ikey, okey = jr.split(key)

        x = self.input(x)
        x = self.dropout(self.activation(x), key = ikey)
        x = self.dropout(self.output(x), key = okey)

        return x

class CLIPBlock(Module):
    attention:CLIPAttention
    mlp:CLIPMLP
    prenorm:Layernorm
    postnorm:Layernorm

    def __init__(self, length:int, features:int, heads:int=12, dropout:float=0, eps=1e-5, bias=True, key=None):
        key = RNG(key)
        self.prenorm = Layernorm(features, eps=eps)
        self.attention = CLIPAttention(length, features, heads=heads, dropout=dropout, bias=bias, key=next(key))
        self.postnorm = Layernorm(features, eps=eps)
        self.mlp = CLIPMLP(features, dropout=dropout, bias=bias, key=next(key))

    def __call__(self, x, mask=None, key=None):
        akey, okey = jr.split(key)
        x = self.attention(self.prenorm(x), mask=mask, key=akey) + x
        x = self.mlp(self.postnorm(x), key=okey) + x
        return x

class VisionTransformer(Module):
    embeddings:ViTEmbeddings
    encoder:list
    prenorm:Layernorm
    postnorm:Layernorm
    features:int = static_field()

    def __init__(self,
        length:int,
        features:int=768,
        channels:int=3,
        size:int=256,
        patch:int=16,
        depth:int=12,
        heads:int=12,
        bias=True,
        eps:float=1e-5,
        dropout=0.,
        key=None
    ):
        key = RNG(key)

        self.features = features
        self.prenorm = Layernorm(features, eps=eps)
        self.postnorm = Layernorm(features, eps=eps)
        self.embeddings = ViTEmbeddings(channels, features, size=size, patch=patch, dropout=dropout, key=next(key))
        self.encoder = [
            CLIPBlock(length, features, heads=heads, bias=bias, dropout=dropout, key=next(key))
            for _ in range(depth)
        ]

    @forward
    def __call__(self, image, mask=None, key=None):
        key = RNG(key)
        embeddings = self.embeddings(image, key=next(key))
        hiddens = self.prenorm(embeddings)
        for block in self.encoder:
            hiddens = block(hiddens, mask=mask, key=next(key))
        summary = self.postnorm(hiddens[0])

        return hiddens, summary

class TextTransformer(Module):
    embeddings:TextEmbeddings
    encoder:list
    layernorm:Layernorm
    features:int = static_field()

    # [BOS] 0 [EOS] 2 [PAD] 1
    def __init__(self,
        length:int,
        vocab:int,
        features:int=768,
        depth:int=12,
        heads:int=12,
        bias=True,
        eps=1e-5,
        dropout=0.,
        key=None
    ):
        key = RNG(key)

        self.features = features
        self.embeddings = TextEmbeddings(vocab, length, features=features, key=next(key))
        self.layernorm = Layernorm(features, eps=eps)
        self.encoder = [
            CLIPBlock(length, features, heads=heads, bias=bias, dropout=dropout, key=next(key))
            for _ in range(depth)
        ]

    @forward
    def __call__(self, tokens, mask=None, key=None):
        key = RNG(key)
        hiddens = self.embeddings(tokens)
        for block in self.encoder:
            hiddens = block(hiddens, mask=mask, key=next(key))
        hiddens = self.layernorm(hiddens)
        summary = hiddens[tokens.argmax()]

        return hiddens, summary

class CLIP(Module):
    text:TextTransformer
    vision:VisionTransformer
    scale:jnp.ndarray
    tout:Projection
    vout:Projection

    def __init__(self, text, vision, features, key=None):
        key = RNG(key)
        self.text = text
        self.vision = vision
        self.tout = Projection(text.features, features, bias=False, key=next(key))
        self.vout = Projection(vision.features, features, bias=False, key=next(key))
        self.scale = jnp.ones([])


    @classmethod
    def load(cls, name="base"):
        file = Path.home() / f".cache/research/clip-{name}.weight"
        if not file.exists(): gdown.download(id=CFG[name]["id"], output=str(file))

        vision = VisionTransformer(**CFG[name]["net"]["vision"])
        text = TextTransformer(**CFG[name]["net"]["text"])
        model = cls(text=text, vision=vision, **CFG[name]["net"]["common"])
        model = equinox.tree_deserialise_leaves(file, model)

        return model

    def __call__(self, tokens, images, tmask=None, imask=None, key=None):
        tkey, ikey = jr.split(key)
        _, temb = self.text(tokens, mask=tmask, key=tkey)
        _, vemb = self.vision(images, mask=imask, key=ikey)

        text, vision = self.tout(temb), self.vout(vemb)
        text, vision = normalise(text), normalise(vision)

        scale = jnp.exp(self.scale)
        logit = text @ vision.T * scale

        return logit, temb, vemb