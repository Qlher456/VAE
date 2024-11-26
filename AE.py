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
input_dim = 3 * 64 * 64  # 输入维度 (3通道64x64图像)
latent_dim = 100         # 潜在空间维度
num_epochs = 1000
lr = 0.0001
early_stop_tolerance = 10  # 早停容忍次数
batch_size = 128

# 创建Image文件夹
os.makedirs('Epoch1000', exist_ok=True)

# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)

# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 3, 64, 64)  # 恢复为图像格式

# 定义AE模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载LFW数据集
    dataset = datasets.LFWPeople(root='data/', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化AE模型
    autoencoder = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 损失值列表
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    # 训练AE
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            autoencoder.train()
            images = data[0].to(device)
            optimizer.zero_grad()

            # 前向传播与损失计算
            reconstructed = autoencoder(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 生成图像每100次保存一次
            if i % 100 == 0:
                autoencoder.eval()
                with torch.no_grad():
                    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
                    generated = autoencoder.decoder(fixed_noise).cpu()
                    plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.imshow(np.transpose(
                        torchvision.utils.make_grid(generated, normalize=True), (1, 2, 0)))
                    plt.savefig(f'Epoch1000/Epoch-{epoch + 1}_Step-{i}.png')
                    plt.close()

        # 记录和输出损失
        epoch_loss /= len(dataloader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        # 早停条件
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), 'AE_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_tolerance:
                print("Early stopping")
                break

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.title("AE Loss During Training")
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Epoch1000/ae_loss_plot.png')
    # plt.show()

    # 从latent space生成新图像
    latent_vector = torch.randn(1, latent_dim).to(device)
    autoencoder.eval()
    with torch.no_grad():
        generated_image = autoencoder.decoder(latent_vector).cpu().squeeze(0)
        plt.imshow(np.transpose((generated_image + 1) / 2, (1, 2, 0)))  # 反归一化显示
        plt.axis("off")
        plt.savefig('Epoch1000/generated_image.png')
