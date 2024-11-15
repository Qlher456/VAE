import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 超参数
latent_dim = 100  # 潜在空间维度
hidden_dim = 64   # 隐藏层维度
num_epochs = 1000
lr = 0.0001
early_stop_tolerance = 10  # 早停容忍次数
batch_size = 128

# 创建Image文件夹
os.makedirs('Image', exist_ok=True)

# VAE的编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(hidden_dim * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.main(x).view(-1, hidden_dim * 8 * 4 * 4)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# VAE的解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 8 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z).view(-1, hidden_dim * 8, 4, 4)
        return self.main(z)

# VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # 计算重构误差
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    # 计算KL散度并加权
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    return recon_loss + kl_div

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 LFW 数据集
    dataset = datasets.LFWPeople(root='data/', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化VAE模型
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # 损失值列表
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    # 训练VAE
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            vae.train()
            images = data[0].to(device)
            optimizer.zero_grad()

            # 前向传播与损失计算
            recon_images, mu, logvar = vae(images)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 生成图像每100次保存一次
            if i % 100 == 0:
                vae.eval()
                with torch.no_grad():
                    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
                    generated = vae.decoder(fixed_noise).cpu()
                    plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.imshow(np.transpose(torchvision.utils.make_grid(generated, normalize=True), (1, 2, 0)))
                    plt.savefig(f'Image/Epoch-{epoch + 1}_Step-{i}.png')
                    plt.close()

        # 记录和输出损失
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        # 早停条件
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(vae.state_dict(), 'vae_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_tolerance:
                print("Early stopping")
                break

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("VAE Loss During Training")
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Image/vae_loss_plot.png')
    # plt.show()
