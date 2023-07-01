# event-diffusers

To run an extremely minimalistic implementation of DDPM on Event-Based Data (DVS-Gesture):

Install the requirements:
```
pip install -r requirements.txt
```
To train the model:
```
python scripts/train_uncond_ddpm.py
```

For finetuning based on a [`pretrained Unconditional Imagenet Diffusion Model`](https://github.com/openai/guided-diffusion):
```
scripts/install_diffusion_pretrained.sh
```

To train on DVS-Gesture datasets:

*IN PROGRESS*