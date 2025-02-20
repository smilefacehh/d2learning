# -*- coding: utf-8 -*-

import torch
from d2l import torch as d2l

class SoftmaxRegressionScratch(d2l.Classifier):
    """FasionMNIST softmax分类器
    图像尺寸：28*28=784
    类别：10
    """
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.1):
        super().__init__()
        self.save_hyperparameters()
        # 784*10 权重高斯随机初始化
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        # 10 bias初始化为0
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
    
    def softmax(self, x):
        x_exp = torch.exp(x)
        partition = x_exp.sum(1, keepdim=True) # x: [N, 10] 表示N个样本，10个类别做softmax
        return x_exp / partition
    
    def forward(self, x):
        x = x.reshape((-1, self.W.shape[0]))
        return self.softmax(torch.matmul(x, self.W) + self.b)
    
    def cross_entropy(self, y_hat, y):
        return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

    def loss(self, y_hat, y):
        return self.cross_entropy(y_hat, y)

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(28*28, 10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

d2l.plt.show()

x, y = next(iter(data.val_dataloader()))
preds = model(x).argmax(axis=1)
print(preds.shape)

wrong = preds.type(y.dtype) != y
x, y, preds = x[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([x, y], labels=labels)
d2l.plt.show()
