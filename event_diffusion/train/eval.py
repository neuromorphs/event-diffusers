import os

import torch
from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size

    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    if config.conditional:
        images = pipeline(
            batch_size=config.eval_batch_size,
            num_classes=config.num_classes,
            generator=torch.manual_seed(config.seed),
            label=config.eval_label,
            num_inference_steps=config.num_eval_inference_steps,
        ).images
    else:
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=config.num_eval_inference_steps,
        ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
