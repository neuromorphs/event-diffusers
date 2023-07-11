import glob

import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image

from event_diffusion.train.config import TrainingConfig
from event_diffusion.train.data import butterfly_dataset, davis_dataset, gesture_dataset
from event_diffusion.train.loop import train_loop
from event_diffusion.train.model import model

dataset = gesture_dataset

sample_image = dataset[0]["data"]
print("Input shape:", torch.Tensor(sample_image.unsqueeze(0)).shape)
print(
    "Output shape:",
    model(torch.Tensor(sample_image.unsqueeze(0)), timestep=0).sample.shape,
)

config = TrainingConfig()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
