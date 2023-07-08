"""
Functionality to train a Vinalla DDPM model on event data
"""

import tonic
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from event_diffusion.model import DDPM, AutoEncoderModel

EPOCHS = 10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = tonic.transforms.Compose(
        [
            # down sample for faster training
            tonic.transforms.Downsample(spatial_factor=0.25),
            # generate frames
            tonic.transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=3),
            # tonic.transforms.ToImage(sensor_size=(32,32,2),)
        ]
    )

    dataset = tonic.datasets.DVSGesture(
        save_to="../data/event_data/", train=False, transform=transform
    )

    # Here I treated the ON and OFF as the two channels of the frame
    ddpm = DDPM(autoencoder_model=AutoEncoderModel(2), betas=(1e-4, 0.02), n_T=100)
    ddpm.to(device)

    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

    for i in range(EPOCHS):
        ddpm.train()
        progress_bar = tqdm(dataloader)
        loss_current = None
        for aa, _ in progress_bar:
            optim.zero_grad()
            # Just playing with the first frame atm
            xx = aa[:, 0, :, :, :].float()

            # Normalize the aggregated spikes to [-1,1]
            xx -= xx.min(0, keepdim=True)[0]
            xx /= xx.max(0, keepdim=True)[0]
            x = xx * 2 - 1

            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_current is None:
                loss_current = loss.item()
            else:
                loss_current = 0.9 * loss_current + 0.1 * loss.item()
            progress_bar.set_description(f"loss: {loss_current:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (2, 32, 32), device)
            grid = make_grid((xh[:, 1] - xh[:, 0]).unsqueeze(axis=1), nrow=4)
            save_image(grid, f"./ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


if __name__ == "__main__":
    main()
