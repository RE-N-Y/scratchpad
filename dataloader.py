import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import jax
import numpy as onp
from einops import rearrange

import deeplake
from PIL import Image
import pickle as pkl
from pathlib import Path

# ImageNet [0.485, 0.456, 0.406] [0.229, 0.224, 0.225]
# Animefaces [0.00268127, 0.00241666, 0.002342] [1.2911796 , 1.2961912 , 1.25518782]

def cycle(loader, tensors=[]):
    while True:
        for batch in loader:
            batch = { t : jax.numpy.array(batch[t]) for t in tensors }
            yield batch


def dataloader(name:str, tensors=[], transform=[], batch_size:int=4, decode_method={}, shuffle=True, num_workers=4):
    ds = deeplake.load(name)
    loader = ds.pytorch(
        tensors=tensors, 
        transform=transform, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        decode_method=decode_method,
        num_workers=num_workers
    )

    loader = cycle(loader, tensors=tensors)

    return loader


