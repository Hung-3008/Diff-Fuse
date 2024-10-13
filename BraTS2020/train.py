import numpy as np
import random
import os
import argparse
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from collections import OrderedDict

from dataset.brats_data_utils_multi_label import get_loader_brats
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

# Set seed for reproducibility
set_determinism(123)


def set_seed(seed):
    torch.manual_deterministic = True
    torch.manual_benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


set_seed(123)

number_modality = 4
number_targets = 3  # WT, TC, ET


class FusionDiff(nn.Module):
    def __init__(self, number_modality, number_targets) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            3, number_modality, number_targets, [64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            3,
            number_targets,
            number_targets,
            [64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [50]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(
                self.model,
                (1, number_targets, 96, 96, 96),
                model_kwargs={"image": image, "embeddings": embeddings},
            )
            sample_out = sample_out["pred_xstart"]
            return sample_out


class BraTSTrainer(Trainer):
    def __init__(
        self,
        env_type,
        max_epochs,
        batch_size,
        device="cpu",
        val_every=1,
        num_gpus=1,
        logdir="./logs/",
        master_ip="localhost",
        master_port=17750,
        training_script="train.py",
        checkpoint_path=None,
    ):
        super().__init__(
            env_type,
            max_epochs,
            batch_size,
            device,
            val_every,
            num_gpus,
            logdir,
            master_ip,
            master_port,
            training_script,
        )
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25
        )
        self.model = FusionDiff(number_modality=4, number_targets=3)
        self.model.to(self.device)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=30, max_epochs=max_epochs
        )

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.start_epoch = 0  # Initialize start_epoch
        self.global_step = 0  # Initialize global_step

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Handle multiple GPUs if specified
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle DataParallel loading if necessary
            state_dict = checkpoint["model_state_dict"]
            if self.num_gpus > 1:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if not k.startswith("module."):
                        k = "module." + k
                    new_state_dict[k] = v
                state_dict = new_state_dict
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
            self.best_mean_dice = checkpoint["best_mean_dice"]
            print(
                f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}"
            )
            # Move optimizer states to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path):
        state = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_mean_dice": self.best_mean_dice,
        }
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = x_start * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(
            x=x_t, step=t, image=image, pred_type="denoise"
        )

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss.item(), step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)
        label = label.float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        self.model.eval()
        with torch.no_grad():
            output = self.window_infer(
                image, self.model, pred_type="ddim_sample"
            )
            output = torch.sigmoid(output)
            output = (output > 0.5).float().cpu().numpy()
            target = label.cpu().numpy()
            # Whole Tumor (WT)
            o = output[:, 1]
            t = target[:, 1]
            wt = dice(o, t)
            # Tumor Core (TC)
            o = output[:, 0]
            t = target[:, 0]
            tc = dice(o, t)
            # Enhancing Tumor (ET)
            o = output[:, 2]
            t = target[:, 2]
            et = dice(o, t)
        self.model.train()
        return [wt, tc, et]

    def validation_end(self, val_outputs):
        wt_list, tc_list, et_list = zip(*val_outputs)
        wt = np.mean(wt_list)
        tc = np.mean(tc_list)
        et = np.mean(et_list)

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            best_model_path = os.path.join(
                self.logdir, "model", f"best_model_{mean_dice:.4f}.pt"
            )
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")

        final_model_path = os.path.join(
            self.logdir, "model", f"final_model_{mean_dice:.4f}.pt"
        )
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.logdir, "model", f"checkpoint_epoch_{self.epoch}.pt"
        )
        self.save_checkpoint(checkpoint_path)

        print(
            f"wt is {wt:.4f}, tc is {tc:.4f}, et is {et:.4f}, mean_dice is {mean_dice:.4f}"
        )

    def train(self, train_dataset, val_dataset):
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch = epoch
            self.model.train()
            for batch_idx, batch in enumerate(train_dataset):
                self.global_step += 1
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch}/{self.max_epochs}], Step [{batch_idx}/{len(train_dataset)}], Loss: {loss.item():.4f}"
                    )

            if (epoch + 1) % self.val_every == 0:
                val_outputs = []
                for val_batch in val_dataset:
                    val_output = self.validation_step(val_batch)
                    val_outputs.append(val_output)
                self.validation_end(val_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a FusionDiff model for BraTS dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/brats2020/MICCAI_BraTS2020_TrainingData/",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs_brats/diffusion_seg_all_loss_embed/",
        help="Directory to save logs and models",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=300, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="Validation frequency (in epochs)",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--env", type=str, default="pytorch", help="Environment type"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to resume training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from the checkpoint",
    )

    args = parser.parse_args()

    model_save_path = os.path.join(args.logdir, "model")
    os.makedirs(model_save_path, exist_ok=True)

    train_ds, val_ds, test_ds = get_loader_brats(
        data_dir=args.data_dir, batch_size=args.batch_size, fold=0
    )

    trainer = BraTSTrainer(
        env_type=args.env,
        max_epochs=args.max_epoch,
        batch_size=args.batch_size,
        device=args.device,
        logdir=args.logdir,
        val_every=args.val_every,
        num_gpus=args.num_gpus,
        master_port=17751,
        training_script=__file__,
        checkpoint_path=args.checkpoint_path if args.resume else None,
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
