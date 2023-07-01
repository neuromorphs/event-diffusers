#!bin/bash

mkdir imagenet_uncond
cd imagenet_uncond
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

pip install -e git+https://github.com/openai/guided-diffusion#egg=guided_diffusion