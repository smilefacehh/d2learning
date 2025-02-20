import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义 MLP 模型
class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# 自定义回调，用于记录每个 epoch 的损失
class LossLoggingCallback(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # 记录每个 epoch 的验证损失
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        # 记录每个 epoch 的训练损失
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())


# 加载 FashionMNIST 数据集
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader


# 训练模型
def train_model():
    train_loader, val_loader = load_data(batch_size=256)
    model = MLP()
    loss_callback = LossLoggingCallback()
    trainer = pl.Trainer(max_epochs=10, callbacks=[loss_callback], accelerator='gpu', devices=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return loss_callback


# 可视化损失
def plot_losses(loss_callback):
    train_losses = loss_callback.train_losses
    val_losses = loss_callback.val_losses

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 主函数
if __name__ == "__main__":
    loss_callback = train_model()
    plot_losses(loss_callback)