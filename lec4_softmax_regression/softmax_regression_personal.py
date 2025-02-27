# -*- coding: utf-8 -*-

from typing import Any
import torch
import torchvision
from torch import nn, optim
import lightning as L
from torch.nn import functional as F
from torch.utils.data import DataLoader

import os
import sys
sys.path.append(os.path.join(os.path.curdir, '..'))

from util import plot_util

class CustomSoftmaxRegression(L.LightningModule):

    def __init__(self, num_inputs, num_outputs, lr):
        super().__init__()
        self.lr = lr

        self.train_loss_epoches = {}
        self.val_loss_epoches = {}

        self.net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss = F.cross_entropy(self(*batch[:-1]), batch[-1], reduction='mean')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = F.cross_entropy(self(*batch[:-1]), batch[-1], reduction='mean')
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), self.lr)
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss').item()
        self.train_loss_epoches[self.trainer.current_epoch] = avg_loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('val_loss').item()
        self.val_loss_epoches[self.trainer.current_epoch] = avg_loss
    

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=transform)
val_dataset = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)

print(f'train batch num:{len(train_loader)}, val batch num:{len(val_loader)}')

model = CustomSoftmaxRegression(28*28, 10, 0.1)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(model.train_loss_epoches.keys(), model.train_loss_epoches.values(), label="train_loss")
plt.plot(model.val_loss_epoches.keys(), model.val_loss_epoches.values(), label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training and validation loss")
plt.legend()
plt.grid(True)
plt.show()
