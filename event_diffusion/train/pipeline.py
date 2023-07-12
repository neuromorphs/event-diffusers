from typing import List, Optional, Tuple, Union

import torch
from diffusers import DDPMPipeline, ImagePipelineOutput
from diffusers.utils import randn_tensor

from .embed import embed


class DDPMConditionalPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, embed_model, embed_device):
        super().__init__(unet, scheduler)
        self.embed_model = embed_model
        self.embed_device = embed_device

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        label: int = 0,
        num_classes: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        if self.embed_device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.embed_device)
        else:
            image = randn_tensor(
                image_shape, generator=generator, device=self.embed_device
            )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if num_classes is None:
                emb = None
            else:
                label_oh = torch.squeeze(
                    torch.nn.functional.one_hot(torch.as_tensor([label]), num_classes=num_classes)
                )
                label_oh = label_oh.repeat(batch_size, 1)
                emb = embed(self.embed_model, label_oh, self.embed_device)
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states=emb).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
