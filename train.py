from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from IPython.display import Image
from event_diffusion import DDPM, ddpm_schedules, UNet

import tonic
import torchvision

def train_frames(epochs:int=100, diffusion_steps:int=1000, lr=2e-4, device="cuda:0", batch_size=12) -> None:

    # ddpm = DDPM(autoencoder_model=AutoEncoderModel(1), betas=(1e-4, 0.02), n_T=diffusion_steps)
    ddpm = DDPM(autoencoder_model=UNet(n_channels=1, n_classes=3), betas=(1e-4, 0.02), n_T=diffusion_steps, device=device)
    ddpm.to(device)

    dataset = torch.load("frame_dataset_90.pt")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for i in range(epochs):
        ddpm.train()

        progress_bar = tqdm(dataloader)
        loss_current = None
        for x, labels in progress_bar:
            optim.zero_grad()
            x, labels = x.to(device), labels.to(device)
            loss = ddpm(x, labels)
            loss.backward()
            if loss_current is None:
                loss_current = loss.item()
            else:
                loss_current = 0.9 * loss_current + 0.1 * loss.item()
            progress_bar.set_description(f"loss: {loss_current:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(4, (1, 90, 90), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./images/ddpm_sample_{i}.png")
            display(Image(f"./images/ddpm_sample_{i}.png", width=600, height=600))

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

    display(Image(f"./images/ddpm_sample_{epochs-1}.png", width=600, height=600))


train_frames(epochs=30, diffusion_steps=500, batch_size=64) # diffusion_steps aka T