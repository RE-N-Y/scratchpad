import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import jax
import pickle as pkl
from pathlib import Path
from PIL import Image
from einops import reduce

class ImageDataset(Dataset):
    def __init__(self, folder:Path, augmentation=[], size=256):
        self.images = list(folder.glob("**/*.jpg"))
        self.transform = T.Compose([T.Resize(size), T.ToTensor(), *augmentation])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx:int):
        image = Image.open(self.images[idx])
        image = self.transform(image)

        return image

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


