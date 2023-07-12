
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                       transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-10.0,10.0)),
                                       transforms.RandomHorizontalFlip(),
                                       #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
                                       ])


