"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from utils import *
from modules import UNet_conditional, EMA


config = SimpleNamespace(
    debug = True,
    run_name = "DDPM_conditional",
    epochs = 100,
    noise_steps=500,
    seed = 42,
    batch_size = 20,
    img_size = 64,
    num_classes = 11,
    dataset_path = get_dvs_gesture(), # get_cifar(img_size=64),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 1,
    num_workers=10,
    c_in = 3,
    c_out = 3,
    lr = 5e-3,
    gradient_accumulation_steps = 1,
    mixed_precision = "no",
    output_dir = "conditional-gesture-1",
)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        self.alpha_hat = self.alpha_hat.to(self.accelerator.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    # @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        self.alpha = self.alpha.to(self.accelerator.device)
        self.beta = self.beta.to(self.accelerator.device)
        model = self.ema_model if use_ema else self.model
        model = self.accelerator.unwrap_model(model)
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        # with torch.inference_mode():
        x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.accelerator.device)
        for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
            t = (torch.ones(n) * i).long().to(self.accelerator.device)
            predicted_noise = model(x, t, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            noise = noise.to(self.accelerator.device)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        # self.ema.step_ema(self.ema_model, self.model)

    def one_epoch(self, train=True, global_step=0):

        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            if self.parsed_args.debug:
                if i > 10: break
            with self.accelerator.accumulate(self.model):
                # with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.accelerator.device)
                labels = labels.to(self.accelerator.device)
                t = self.sample_timesteps(images.shape[0]).to(self.accelerator.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss.item()
                if train:
                    self.train_step(loss)
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    self.accelerator.log(logs, step=global_step)
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
                    global_step += 1
                pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss / len(self.train_dataloader), global_step

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.accelerator.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        # self.ema = EMA(0.995)
        # self.scaler = torch.cuda.amp.GradScaler()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.output_dir, "logs"),
            kwargs_handlers=[ddp_kwargs],
        )
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
            self.accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )

    def fit(self, args):
        self.args = parse_args
        self.parsed_args = args
        global_step = 0
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _, global_step = self.one_epoch(train=True, global_step=global_step)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0:
                self.log_images()

        # save model
        self.save_model(run_name=args.run_name, epoch=epoch)




def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, c_in=config.c_in, c_out=config.c_out)
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)
