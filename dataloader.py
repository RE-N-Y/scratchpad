import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import jax
import numpy as onp
from einops import rearrange
from PIL import Image
import pickle as pkl
from pathlib import Path

# ImageNet [0.485, 0.456, 0.406] [0.229, 0.224, 0.225]
# Animefaces [0.00268127, 0.00241666, 0.002342] [1.2911796 , 1.2961912 , 1.25518782]

class ImageDataset(Dataset):
    def __init__(self, folder:Path, augmentations=[]):
        self.images = list(folder.glob("**/*.jpg"))
        self.transform = T.Compose([*augmentations, T.ToTensor()])
        self.mean = torch.Tensor([.5,.5,.5]) # torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([.5,.5,.5]) # torch.Tensor([0.229, 0.224, 0.225])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx:int):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        image = rearrange(image, 'c h w -> h w c')
        image = (image - self.mean) / self.std

        return image

    def tensor_to_image(self, tensor):
        tensor = onp.array(tensor)
        tensor = tensor * self.std.numpy() + self.mean.numpy()
        tensor = tensor * 255
        tensor = onp.rint(tensor).clip(0,255).astype(onp.uint8)

        return Image.fromarray(tensor)

class MemoryDataset(Dataset):
    def __init__(self, file:Path):
        with open(file, "rb") as f:
            self.data = pkl.load(f)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx:int):
        return self.data[idx]

def cycle(loader):
    while True:
        for batch in loader:
            jrray = jax.numpy.array(batch)
            yield jrray

def dataloader(dataset, batch, shuffle=True, num_workers=4):
    loader = DataLoader(dataset, batch, shuffle=shuffle, num_workers=num_workers, drop_last=True, persistent_workers=False)
    loader = cycle(loader)
    return loader


