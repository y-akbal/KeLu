import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms.v2 import RandAugment
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision

from jax.tree_util import tree_map
from torch.utils import data
import numpy as np

"""
def numpy_collate(batch:
  return tree_map(np.asarray, data.default_collate(batch))
"""

train_transforms = v2.Compose([
            v2.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(),
            v2.ColorJitter(0.1,0.1,0.1),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
            v2.RandomErasing(0.25),
    ])

test_transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    ])



def return_dataset():


    training_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform= train_transforms
    )   

    test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform = test_transform
    )


    ## Cut-Mix up write here the the result will be damn easy!!!
    numpy_collate = lambda batch: tree_map(np.asarray, data.default_collate(batch))
    
        
    train_data_loader = torch.utils.data.DataLoader(training_data,
                                          batch_size=256,
                                          shuffle=True,
                                          num_workers = 12,
                                          collate_fn=numpy_collate)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=256,
                                          shuffle=False,
                                          num_workers = 12,
                                          collate_fn=numpy_collate)

    return train_data_loader, test_data_loader

"""
train, test = return_dataset()

for i, (x,y) in enumerate(train):
    print(x.mean([1,2,3]),y.shape, i)
"""