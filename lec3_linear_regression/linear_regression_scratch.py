# -*- coding: utf-8 -*-
import torch
from d2l import torch as d2l

class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegressionScratch(d2l.Module):
    """ 线性回归实现 y = xw + b

    w: [2, 1]，等价于 y = w1*x1 + w2*x2 + b
    x: 如果是一个批量，维度就是 [BN, 2]
    y: [BN, 1]
    b: [BN, 1]
    """
     

    def __init__(self, num_inputs, lr, sigma=0.01):
        """sumary_line
        
        Keyword arguments:
        :param num_inputs: feature的数量，或者说参数个数
        :param lr: learning rate
        :param sigma: 高斯随机初始化权重的标准差
        """
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    
    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()
    
    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)
    
class Trainer(d2l.Trainer):

    def __init__(self, max_epochs):
        super().__init__(max_epochs=max_epochs)

    def prepare_batch(self, batch):
        return batch
    
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        
        if self.val_dataloader is None:
            return
        
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
        

if __name__ == '__main__':
    model = LinearRegressionScratch(2, lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)

    d2l.plt.show()

    with torch.no_grad():
        print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
        print(f'error in estimating b: {data.b - model.b}')