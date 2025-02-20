# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.curdir, '..'))

import torch
from torch import nn
from d2l import torch as d2l

class MLP(d2l.Classifier):

    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        # 函数参数保存为成员变量
        self.save_hyperparameters()
        self.w1 = nn.Parameter(torch.normal(0, sigma, size=(num_inputs, num_hiddens)))
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.w2 = nn.Parameter(torch.normal(0, sigma, size=(num_hiddens, num_outputs)))
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
        self.lr = lr

    def relu(self, x):
        return torch.max(x, torch.zeros_like(x))

    def forward(self, x):
        x = x.reshape((-1, self.num_inputs))
        x_hidden = self.relu(x @ self.w1 + self.b1)
        x_output = x_hidden @ self.w2  + self.b2
        return x_output
    
model = MLP(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

d2l.plt.show()