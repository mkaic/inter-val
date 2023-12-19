import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple

# Adapted from https://github.com/AntixK/PyTorch-VAE


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        input_size: int = 64,
        hidden_dims: List = None,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.bottleneck_size = input_size
        self.bottleneck_channels = self.hidden_dims[-1]
        for i in self.hidden_dims:
            self.bottleneck_size = (self.bottleneck_size + 1) // 2
        flattened_size_after_encoder = (
            self.bottleneck_size**2
        ) * self.bottleneck_channels
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(flattened_size_after_encoder, latent_dim)
        self.fc_var = nn.Linear(flattened_size_after_encoder, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, flattened_size_after_encoder)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dims[-1],
                self.hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        y_hat = self.encoder(x)
        y_hat = torch.flatten(y_hat, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(y_hat)
        log_var = self.fc_var(y_hat)

        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        z = z.view(
            -1, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size
        )
        z = self.decoder(z)
        z = self.final_layer(z)
        return z

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, y_hat, x, mu, log_var) -> Tensor:
        kullbeck_leibler_divergence_weight = 0.00025
        reconstruction_loss = F.mse_loss(y_hat, x)

        kullbeck_leibler_divergence_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = (
            reconstruction_loss
            + kullbeck_leibler_divergence_weight * kullbeck_leibler_divergence_loss
        )

        return loss
