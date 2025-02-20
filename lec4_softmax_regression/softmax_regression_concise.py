# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

import os
import sys
sys.path.append(os.path.join(os.path.curdir, '..'))


from d2l import torch as d2l

class SoftmaxRegression(d2l.Classifier):

    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        # lazy linear在第一次执行forward的时候根据输入x推到num_inputs，无需手动指定
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))


    def forward(self, x):
        return self.net(x)
    
    def loss(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1, ))
        return F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')
    
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

d2l.plt.show()