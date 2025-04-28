import os
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. CNN分类器
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_flatten_params(self):
        return torch.cat([p.flatten() for p in self.parameters()])

# 2. 高维自编码器
class ParameterAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256, hidden_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, noise=True):
        if noise:
            x = x + torch.randn_like(x)*0.001
        z = self.encoder(x)
        if noise:
            z = z + torch.randn_like(z)*0.05
        return self.decoder(z)

# 3. 扩散模型（无内部EMA）
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim=256, timesteps=500, hidden_dim=1024):
        super().__init__()
        self.timesteps = timesteps
        self.time_embed = nn.Sequential(
            nn.Embedding(timesteps, 128),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + 128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.betas = self.cosine_beta_schedule(timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(self, z, t):
        t_embed = self.time_embed(t)
        return self.noise_predictor(torch.cat([z, t_embed], 1))

    def train_step(self, z0):
        t = torch.randint(0, self.timesteps, (z0.size(0),), device=z0.device)
        alpha_bar = self.alpha_bars.to(z0.device)[t].unsqueeze(1)
        noise = torch.randn_like(z0)
        zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
        pred_noise = self(zt, t)
        return torch.mean((noise - pred_noise)**2)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)

# 4. 完整训练流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn_losses = []
    ae_losses = []
    diffusion_losses = []
    generated_accuracies = []

    num_models = 10
    params_dataset = []

    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=256, shuffle=True)

    for model_idx in range(num_models):
        print(f"\n=== Training Model {model_idx+1}/{num_models} ===")
        model = CNNClassifier().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        epochs = 10
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Loss: {avg_loss:.4f}")
            cnn_losses.append(avg_loss)

        with torch.no_grad():
            flat_params = model.get_flatten_params().cpu()
            params_dataset.append(flat_params)

        del model
        torch.cuda.empty_cache()

    params_tensor = torch.stack(params_dataset)
    param_dim = params_tensor.shape[-1]

    latent_dim = 512
    ae = ParameterAutoencoder(param_dim, latent_dim=latent_dim, hidden_dim=1024).to(device)
    ae_optim = optim.AdamW(ae.parameters(), lr=1e-4)

    print("\n=== Training Autoencoder ===")
    for epoch in range(100):
        ae.train()
        epoch_loss = 0
        for batch in params_tensor.split(8):
            batch = batch.to(device)
            ae_optim.zero_grad()
            recon = ae(batch)
            loss = torch.mean((recon - batch)**2)
            loss.backward()
            ae_optim.step()
            epoch_loss += loss.item()
        avg_ae_loss = epoch_loss / len(params_tensor.split(8))
        print(f"Epoch {epoch+1}/100 | Loss: {avg_ae_loss:.4e}")
        ae_losses.append(avg_ae_loss)

    with torch.no_grad():
        latent_z = ae.encoder(params_tensor.to(device))

    diffusion = LatentDiffusion(latent_dim=latent_dim, hidden_dim=1024).to(device)
    ema_model = LatentDiffusion(latent_dim=latent_dim, hidden_dim=1024).to(device)
    ema_model.load_state_dict(diffusion.state_dict())
    diff_optim = optim.AdamW(diffusion.parameters(), lr=5e-4)

    print("\n=== Training Diffusion Model ===")
    best_loss = float('inf')
    for step in tqdm(range(20000), desc="Diffusion Training"):
        diff_optim.zero_grad()
        loss = diffusion.train_step(latent_z)
        loss.backward()
        diff_optim.step()
        update_ema(ema_model, diffusion)
        if loss < best_loss:
            best_loss = loss.item()
        if (step+1) % 2000 == 0:
            print(f"Step {step+1} | Loss: {loss.item():.4e} | Best: {best_loss:.4e}")
            diffusion_losses.append(loss.item())

    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=512)

    print("\n=== Generating New Models (DDIM) ===")
    num_generate = 20
    for gen_id in range(num_generate):
        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            ddim_steps = 50
            for i in range(ddim_steps):
                t = torch.full((1,), diffusion.timesteps - 1 - i * diffusion.timesteps // ddim_steps, device=device, dtype=torch.long)
                pred_noise = ema_model(z, t)
                alpha = diffusion.alphas.to(t.device)[t]
                alpha_bar = diffusion.alpha_bars.to(t.device)[t]
                z = (z - (1 - alpha)/torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)

            new_params = ae.decoder(z).squeeze()

        new_model = CNNClassifier().to(device)
        state_dict = new_model.state_dict()

        ptr = 0
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                num = param.numel()
                state_dict[name] = new_params[ptr:ptr+num].view_as(param)
                ptr += num

        new_model.load_state_dict(state_dict)

        new_model.eval()
        correct = 0
        for x, y in tqdm(test_loader, desc=f"Testing Model {gen_id+1}"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = new_model(x).argmax(1)
                correct += (pred == y).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(f"Generated Model {gen_id+1} | Test Accuracy: {accuracy:.2%}")
        generated_accuracies.append(accuracy)

    # === 绘制训练曲线并保存 ===
    if not os.path.exists('output/plots'):
        os.makedirs('output/plots')

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(cnn_losses, marker='o')
    plt.title('CNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(ae_losses, marker='o')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(range(2000, 20001, 2000), diffusion_losses, marker='o')
    plt.title('Diffusion Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 4)
    plt.plot(generated_accuracies, marker='o')
    plt.title('Generated CNN Test Accuracy')
    plt.xlabel('Model ID')
    plt.ylabel('Accuracy')

    plt.tight_layout()

    # 加上当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"output/plots/training_curves_{timestamp}.png"
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.show()


    # === 保存所有训练指标到CSV ===
    metrics = {
        'cnn_losses': cnn_losses,
        'ae_losses': ae_losses,
        'diffusion_losses': diffusion_losses,
        'generated_accuracies': generated_accuracies
    }

    # 为了对齐，先找最大长度
    max_len = max(len(v) for v in metrics.values())

    # 补齐空位，让不同长度的list对齐
    for key in metrics:
        if len(metrics[key]) < max_len:
            metrics[key] += [None] * (max_len - len(metrics[key]))

    df = pd.DataFrame(metrics)
    csv_save_path = f"output/plots/training_metrics_{timestamp}.csv"
    df.to_csv(csv_save_path, index=False)
    print(f"Training metrics saved to {csv_save_path}")


if __name__ == "__main__":
    main()
