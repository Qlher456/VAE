import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 定义超参数
latent_dim = 100  # 潜在空间维度
hidden_dim = 64   # 隐藏层维度

# 定义VAE模型结构
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

# 加载模型并生成图像
def generate_single_image(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化并加载模型
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()

    # 生成潜在空间的随机噪声
    with torch.no_grad():
        random_noise = torch.randn(1, latent_dim, device=device)
        generated_image = vae.decoder(random_noise).cpu()

    # 可视化生成的图像
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(torchvision.utils.make_grid(generated_image, normalize=True), (1, 2, 0)))
    plt.show()

# 调用生成图像的函数
generate_single_image('vae_model.pth')
