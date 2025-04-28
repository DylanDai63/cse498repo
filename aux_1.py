import torch
import torch.nn as nn

# CNN Classifier
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

# Parameter Autoencoder
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

# Latent Diffusion Model
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

# Exponential Moving Average Update
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1-decay)
