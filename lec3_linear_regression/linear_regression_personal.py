# -*- coding: utf-8 -*-

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


class CustomDataset(Dataset):
    """数据集的定义
    1）原始数据构造x、y
    2）实现__len__、__getitem__
    """
    def __init__(self, w, b, noise=0.01, num_samples=1000):
        self.x = torch.randn(num_samples, len(w))
        noise = torch.randn(num_samples, 1) * noise
        self.y = self.x @ torch.reshape(w, (-1, 1)) + b + noise

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class CustomDataModule(L.LightningDataModule):
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.w = w
        self.b = b
        self.noise = noise
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.w, self.b, self.noise, self.num_train)
        self.val_dataset = CustomDataset(self.w, self.b, self.noise, self.num_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class LinearRegression(L.LightningModule):

    def __init__(self, lr):
        """初始化

        1）超参数
        2）模型
        """
        super().__init__()

        self.lr = lr
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        """训练过程
        
        1）输入数据，模型执行
        2）取出y，也就是GT，计算loss
        """
        x, y = batch
        y_hat = self(x) # 这里调用forwar函数
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) 
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        """指定优化器""" 
        return optim.SGD(self.parameters(), lr=self.lr)
    
    def get_w_b(self):
        return self.linear.weight.data, self.linear.bias.data
    

def train():
    model = LinearRegression(lr=0.03)
    data_module = CustomDataModule(torch.tensor([2, -3.4]), 4.2, batch_size=32)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)

    w, b = model.get_w_b()
    print(f'w err:{data_module.w - w.reshape(data_module.w.shape)}')
    print(f'b err:{data_module.b - b}')

def infer():
    checkpoint = './lightning_logs/version_0/checkpoints/epoch=2-step=96.ckpt'
    model = LinearRegression.load_from_checkpoint(checkpoint, lr=0.03)
    x = torch.randn(2, 2, device=model.device)
    y = model(x)

    w, b = model.get_w_b()
    print(f'w:{w} b:{b}')
    print('x:\n', x)
    print('y:\n', y)

train()

