import jax
import jax.numpy as jnp
import jax.random as jr
import equinox
from equinox import nn, static_field, Module
from layers import Activation, Embedding, Layernorm, Projection
from einops import rearrange, reduce, repeat
from toolkit import forward, RNG

import math
import gdown
import numpy as onp
from pathlib import Path

class BERTEmbeddings(Module):
    wte:Embedding
    wpe:Embedding
    tte:Embedding
    dropout:nn.Dropout
    layernorm:Layernorm
    positions:jnp.ndarray = static_field()

    def __init__(self, vocab, length:int=512, types:int=2, features:int=768, dropout:float=0, eps=1e-12, key=None):
        key = RNG(key)
        self.wte = Embedding(vocab, features, key=next(key))
        self.wpe = Embedding(length, features, key=next(key))
        self.tte = Embedding(types, features, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = Layernorm(features, eps=eps)
        self.positions = jnp.arange(length)

    def __call__(self, tokens, positions=None, types=None, key=None):
        positions = self.positions[:len(tokens)] if positions is None else positions
        types = jnp.zeros((len(tokens),), dtype=int) if types is None else types

        embeddings = self.wte(tokens) + self.wpe(positions) + self.tte(types)
        embeddings = self.dropout(self.layernorm(embeddings))

        return embeddings

class BERTAttention(Module):
    query:Projection
    key:Projection
    value:Projection
    out:Projection
    layernorm:Layernorm
    dropout:nn.Dropout
    heads:int = static_field()
    features:int = static_field()
    scale:float = static_field()

    def __init__(self, features:int, heads:int=12, dropout:float=0, bias=False, eps=1e-12, key=None):
        key = RNG(key)

        self.heads = heads
        self.features = features
        self.scale = onp.sqrt(features // heads)
        
        self.query = Projection(features, features, bias=bias, key=next(key))
        self.key = Projection(features, features, bias=bias, key=next(key))
        self.value = Projection(features, features, bias=bias, key=next(key))
        self.out = Projection(features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = Layernorm(features, eps=eps)

    def __call__(self, x, mask=None, key=None):
        key = RNG(key)
        q,k,v = self.query(x), self.key(x), self.value(x)
        q,k,v = map(lambda x : rearrange(x, 'n (h d) -> h n d', h=self.heads), (q,k,v))

        k = rearrange(k, 'h n d -> h d n')
        attention = q @ k / self.scale # [h m d] @ [h d n] = [h m n]

        if mask is not None:
            finfo = jnp.finfo(attention.dtype)
            attention = jnp.where(mask, attention, finfo.min)

        attention = self.dropout(jax.nn.softmax(attention), key=next(key))
        outputs = rearrange(attention @ v, 'h n d -> n (h d)')
        outputs = self.layernorm(self.dropout(self.out(outputs), key=next(key)) + x)

        return outputs

class BERTMLP(Module):
    input:Projection
    output:Projection
    dropout:nn.Dropout
    layernorm:Layernorm
    activation:Activation

    def __init__(self, features:int, activation:str="egelu", dropout:float=0, bias=True, eps=1e-12, key=None):
        key = RNG(key)
        
        self.input = Projection(features, 4 * features, bias=bias, key=next(key))
        self.output = Projection(4 * features, features, bias=bias, key=next(key))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = Layernorm(features, eps=eps)
        self.activation = Activation(activation)

    def __call__(self, x, key=None):
        ikey, okey = jr.split(key)

        out = self.input(x)
        out = self.dropout(self.activation(out), key = ikey)
        out = self.layernorm(self.dropout(self.output(out), key = okey) + x)

        return out

class BERTBlock(Module):
    attention:BERTAttention
    mlp:BERTMLP

    def __init__(self, features:int, heads:int=12, dropout:float=0, bias=True, eps=1e-12, key=None):
        key = RNG(key)
        self.attention = BERTAttention(features, heads=heads, dropout=dropout, bias=bias, eps=eps, key=next(key))
        self.mlp = BERTMLP(features, dropout=dropout, bias=bias, eps=eps, key=next(key))

    def __call__(self, x, mask=None, key=None):
        key = RNG(key)
        x = self.attention(x, mask=mask, key=next(key))
        x = self.mlp(x, key=next(key))

        return x
    

class BERT(Module):
    embeddings:BERTEmbeddings
    encoder:list
    features:int = static_field()

    # [BOS] 0 [EOS] 2 [PAD] 1
    def __init__(self,
        length:int,
        vocab:int,
        types:int=2,
        features:int=768,
        depth:int=12,
        heads:int=12,
        bias=True,
        dropout=0.,
        eps=1e-12,
        key=None
    ):
        key = RNG(key)

        self.features = features
        self.embeddings = BERTEmbeddings(vocab, length, types=types, features=features, dropout=dropout, eps=eps, key=next(key))
        self.encoder = [
            BERTBlock(features,  heads=heads, bias=bias, dropout=dropout, eps=eps, key=next(key))
            for _ in range(depth)
        ]

    @classmethod
    def load(cls, name="base"):
        key = jr.PRNGKey(42)
        cache = Path("~/.cache/research")
        cfg = {
            "base": { 
                "id":"1T8XSTpch7aLzfcoWu2nugTeLQWqpZ3Lu", 
                "net":{ "length":512, "vocab":30522, "features":768, "heads":12, "depth":12, "eps":1e-12, "key":key } 
            }, 
            "large": { 
                "id":"1dc4nFEsshr0QarWPIb5fSWzONGmbDZsC", 
                "net":{ "length":512, "vocab":30522, "features":1024, "heads":16, "depth":24, "eps":1e-12, "key":key } 
            }
        }
        
        file = gdown.download(id=cfg[name]["id"], output=cache)
        model = equinox.tree_deserialise_leaves(file, cls(**cfg[name]["net"]))

        return model

    @forward
    def __call__(self, tokens, mask=None, key=None):
        key = RNG(key)
        hiddens = self.embeddings(tokens)
        for block in self.encoder: hiddens = block(hiddens, mask=mask, key=next(key))
        
        return hiddens