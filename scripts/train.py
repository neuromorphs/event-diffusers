import glob

import datasets
import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image

from event_diffusion.train.config import TrainingConfig
from event_diffusion.train.data import butterfly_dataset, davis_dataset, gesture_dataset, mnist_dataset
from event_diffusion.train.embed import EmbedFC
from event_diffusion.train.loop import train_loop
from event_diffusion.train.model import condition_model, model

config = TrainingConfig()


if config.dataset == "gesture":
    dataset = gesture_dataset
    num_classes = 11
elif config.dataset == "butterfly":
    dataset = butterfly_dataset
    num_classes = 10
elif config.dataset == "mnist":
    dataset = mnist_dataset
    num_classes = 10
elif config.dataset == "davis":
    dataset = davis_dataset
else:
    raise ValueError(f"Unknown dataset {config.dataset}")

if config.debug:
    dataset = datasets.Dataset.from_dict(dataset[:16]).with_format("torch")

embed_model = EmbedFC(num_classes, 1280)

sample_image = dataset[0]["data"]
sample_label = dataset[0]["label"]

emb = embed_model(sample_label.unsqueeze(0)).unsqueeze(0)

print("Input shape:", torch.Tensor(sample_image.unsqueeze(0)).shape)
print(
    "Output shape:",
    condition_model(
        torch.Tensor(sample_image.unsqueeze(0)),
        encoder_hidden_states=emb,
        # added_cond_kwargs={"image_embeds": emb},
        timestep=0,
    ).sample.shape,
)


noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = config.lr_scheduler(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
    **config.lr_kwargs,
)

train_loop(
    config,
    condition_model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    embed_model,
)
