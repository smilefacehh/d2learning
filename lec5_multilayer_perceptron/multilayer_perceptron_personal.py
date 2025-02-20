# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.curdir, '..'))

import torch
from torch import nn, optim
import lightning as L
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader

class MLP(L.LightningModule):

    def __init__(self, num_inputs, num_hiddens, num_outputs, lr):
        super().__init__()

        self.train_loss_epoches = {}
        self.val_loss_epoches = {}

        self.lr = lr
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(num_inputs, num_hiddens), 
            nn.ReLU(), 
            # nn.Dropout(0.1),
            nn.Linear(num_hiddens, num_outputs))

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), self.lr)
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss').item()
        self.train_loss_epoches[self.trainer.current_epoch] = avg_loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('val_loss').item()        
        self.val_loss_epoches[self.trainer.current_epoch] = avg_loss


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_loader = DataLoader(
    torchvision.datasets.FashionMNIST('../data', train=True, transform=transform),
    batch_size=256, shuffle=True, num_workers=8
)
val_loader = DataLoader(
    torchvision.datasets.FashionMNIST('../data', train=False, transform=transform),
    batch_size=256, shuffle=False, num_workers=8
)

model = MLP(28*28, 256, 10, lr=0.1)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=1)
trainer.fit(model, train_loader, val_loader)

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
