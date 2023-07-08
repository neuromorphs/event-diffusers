"""
Implementation of Vanilla UNconditional DDPM

"""
import torch
import torch.nn as nn
from typing import Tuple

from .utils import ddpm_schedules


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class DDPM(nn.Module):
    def __init__(
        self,
        autoencoder_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.autoencoder_model = autoencoder_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements Algorithm 1 from the paper.
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using autoencoder_model.
        """

        t = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            torch.sqrt(self.alphabar_t[t, None, None, None]) * x
            + torch.sqrt(1 - self.alphabar_t[t, None, None, None]) * eps
        )

        return self.criterion(eps, self.autoencoder_model(x_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """
        Implements Algorithm 2 from the paper.
        """

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.autoencoder_model(x_i, i / self.n_T)
            x_i = (1 / torch.sqrt(self.alpha_t[i])) * (
                x_i - eps * (1 - self.alpha_t[i]) / torch.sqrt(1 - self.alphabar_t[i])
            ) + self.sqrt_beta_t[i] * z

        return x_i


class AutoEncoderModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(AutoEncoderModel, self).__init__()
        self.conv = nn.Sequential(
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. Paper uses positional embeddings.
        return self.conv(x)
