# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.curdir, '..'))

import torch
from torch import nn
from d2l import torch as d2l

class MLP(d2l.Classifier):

    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        # 函数参数保存为成员变量
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens), nn.ReLU(), nn.LazyLinear(num_outputs))

model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

d2l.plt.show()