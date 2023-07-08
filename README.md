# event-diffusers

To run an extremely minimalistic implementation of DDPM on Event-Based Data (DVS-Gesture):

Install the requirements:

```shell
pip install -r requirements.txt
```

Or, using [poetry](https://python-poetry.org/):

```shell
poetry install
```

If you need to install poetry, make sure you are using a python version >= 3.10 and run:

```shell
curl -sSL https://install.python-poetry.org | python -
```

Add poetry to your path:

```shell
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
```

Or if you're using `zsh`:

```shell
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.zshrc
```

To train the model:

```shell
python scripts/train_uncond_ddpm.py
```

For finetuning based on a [`pretrained Unconditional Imagenet Diffusion Model`](https://github.com/openai/guided-diffusion):

```shell
scripts/install_diffusion_pretrained.sh
```

To train on DVS-Gesture datasets:

*IN PROGRESS*