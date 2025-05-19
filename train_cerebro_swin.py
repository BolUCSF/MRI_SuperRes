import os, json, deepspeed, wandb, torch, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms, data
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from tqdm import tqdm
from monai.losses.ssim_loss import SSIMLoss
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import Dataset
from monai.networks.nets.swin_unetr import SwinUNETR


class MRIDataset(Dataset):
    def downsample_mri_kspace(self, mri_image, downsampling_factor):
        """
        Downsamples an MRI image using k-space zero-filling.
        """
        # Get the image data and affine transformation
        data = mri_image[0]

        # Get the spatial dimensions
        spatial_dims = data.shape[: len(downsampling_factor)]
        num_spatial_dims = len(spatial_dims)

        # Check if the downsampling factor is valid
        if len(downsampling_factor) != num_spatial_dims:
            raise ValueError(
                f"Downsampling factor length ({len(downsampling_factor)}) must match the number of spatial dimensions ({num_spatial_dims})."
            )
        for factor in downsampling_factor:
            if not isinstance(factor, int) or factor < 1:
                raise ValueError("Downsampling factors must be positive integers.")

        # Perform k-space transform
        k_space = np.fft.fftn(data, axes=range(num_spatial_dims))
        k_space_shifted = np.fft.fftshift(k_space, axes=range(num_spatial_dims))

        # Create a new k-space array with zero-filling
        new_k_space_shape = list(k_space_shifted.shape)

        # Determine the central portion to keep in k-space
        start_indices = []
        end_indices = []
        for i in range(num_spatial_dims):
            center = spatial_dims[i] // 2
            half_kept = (spatial_dims[i] // downsampling_factor[i]) // 2
            start_indices.append(center - half_kept)
            end_indices.append(
                center + (spatial_dims[i] // downsampling_factor[i]) - half_kept
            )
            new_k_space_shape[i] = spatial_dims[i] // downsampling_factor[i]

        # Place the central portion of the original k-space into the new (larger) array
        slices = tuple(
            slice(start, end) for start, end in zip(start_indices, end_indices)
        )
        # downsampled_k_space_shifted[slices] = k_space_shifted[slices]
        downsampled_k_space_shifted = k_space_shifted[slices]

        # Inverse k-space transform to get the downsampled image
        downsampled_k_space = np.fft.ifftshift(
            downsampled_k_space_shifted, axes=range(num_spatial_dims)
        )
        downsampled_data = np.fft.ifftn(
            downsampled_k_space, axes=range(num_spatial_dims)
        ).real
        downsampled_data = downsampled_data[
            np.newaxis, ...
        ]  # Add a new axis to match the original shape

        return downsampled_data

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.resample_factors = [
            (1, 1, 3),
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 6),
            (1, 3, 1),
            (1, 4, 1),
            (1, 5, 1),
            (1, 6, 1),
            (3, 1, 1),
            (4, 1, 1),
            (5, 1, 1),
            (6, 1, 1),
        ]
        self.train_transform_1 = transforms.Compose(
            [
                transforms.LoadImaged(keys=["file_name"]),
                transforms.CopyItemsd(keys=["file_name"], names=["image"]),
                transforms.DeleteItemsd(keys=["file_name"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.EnsureTyped(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True
                ),
                transforms.Spacingd(keys=["image"], pixdim=(1, 1, 1), mode=3),
                transforms.ResizeWithPadOrCropd(
                    keys=["image"], spatial_size=(240, 240, 180)
                ),
                transforms.CopyItemsd(
                    keys=["image"], names=["hi_res", "low_res"], times=2
                ),
                transforms.DeleteItemsd(keys=["image"]),
            ]
        )
        self.train_transform_2 = transforms.Compose(
            [
                transforms.Resized(
                    keys=["low_res"], spatial_size=(240, 240, 180), mode="nearest"
                ),
                transforms.CropForegroundd(
                    keys=["hi_res", "low_res"], source_key="hi_res", allow_smaller=False
                ),
                transforms.RandSpatialCropd(
                    keys=["hi_res", "low_res"], roi_size=(64, 64, 64), random_size=False
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = self.image_paths[idx]
        transformed_data = self.train_transform_1(data)
        transformed_data["low_res"] = self.downsample_mri_kspace(
            transformed_data["low_res"], random.choice(self.resample_factors)
        )
        transformed_data = self.train_transform_2(transformed_data)
        return transformed_data


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "3224"
set_determinism(42)

train_batchsize = 18
epochs = 200
enable_gan = True

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
if rank == 0:
    wandb.init(
        project="SuperRes_A6000_SwinUNETR",
        config={
            "architecture": "SwinUNETR",
            "dataset": "cerebro",
            "epochs": epochs,
            "Gan": enable_gan,
        },
    )

checkpoint_path = "./deepspeed_checkpoint_swin"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

with open("json/train_files4.json") as f:
    train_files = json.load(f)
with open("json/val_files4.json") as f:
    validation_files = json.load(f)

train_ds = MRIDataset(train_files)
valid_ds = MRIDataset(validation_files)

sampler_train = DistributedSampler(
    train_ds, shuffle=True, num_replicas=world_size, rank=rank
)
train_loader = DataLoader(
    train_ds,
    batch_size=train_batchsize,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    sampler=sampler_train,
    collate_fn=None,
)
sampler_val = DistributedSampler(
    valid_ds, shuffle=False, num_replicas=world_size, rank=rank
)
val_loader = DataLoader(
    valid_ds,
    batch_size=train_batchsize,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    sampler=sampler_val,
    collate_fn=None,
)

model = SwinUNETR(img_size=64, in_channels=1, out_channels=1, feature_size=48)
store_dict = torch.load("/working/Project/Sup_Res/checkpoint_superformer/pytorch_model.bin")
model.load_state_dict(store_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1000, T_mult=2, eta_min=1e-6
)
adv_loss = PatchAdversarialLoss(criterion="least_squares")
discriminator = PatchDiscriminator(
    spatial_dims=3, in_channels=1, num_layers_d=3, num_channels=16
)
discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-5)

discriminator_engine, discriminator_optimizer, _, _ = deepspeed.initialize(
    model=discriminator,
    model_parameters=discriminator.parameters(),
    optimizer=optimizer_d,
    config="deepspeed_json/deepspeed_swin.json",
)

model_engine, model_optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config="deepspeed_json/deepspeed_swin.json",
)
model_engine.load_checkpoint('./deepspeed_checkpoint_swin/Swin_latest',tag='latest_step')
device = model_engine.device


def reconstruction_loss(output, target):
    return F.mse_loss(output, target)


perceptual_loss = PerceptualLoss(3)
ssim_loss = SSIMLoss(3)
perceptual_loss.to(device)


def loss_all(output, target):
    alpha = 1
    beta = 1
    sigma = 1
    recon = reconstruction_loss(output, target)
    recon *= alpha
    ssim = ssim_loss(output, target)
    ssim *= beta
    perc = perceptual_loss(output, target)
    perc *= sigma
    loss = recon + perc + ssim
    return loss, recon, perc, ssim


train_step = 0
val_step = 0
for epoch in range(epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
    model_engine.train()
    progress_bar.set_description(f"Epoch {epoch}")
    sampler_train.set_epoch(epoch)
    for step, batch_data in progress_bar:
        image = batch_data["hi_res"].to(torch.bfloat16).to(device)
        low_res = batch_data["low_res"].to(torch.bfloat16).to(device)
        model_optimizer.zero_grad()

        output = model_engine(low_res)
        loss, recon, perc, ssim = loss_all(output, image)

        if enable_gan:
            discriminator_optimizer.zero_grad()
            logits_real = discriminator_engine(image.contiguous().detach())[-1]
            loss_d_real = adv_loss(
                logits_real, target_is_real=True, for_discriminator=True
            )
            logits_fake = discriminator_engine(output.contiguous().detach())[-1]
            loss_d_fake = adv_loss(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            discriminator_engine.backward(loss_d_fake + loss_d_real)

            discriminator_engine.eval()
            with torch.no_grad():
                logits_fake = discriminator_engine(output.contiguous().detach())[-1]
            discriminator_engine.train()
            generator_loss = adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss += generator_loss

        model_engine.backward(loss)
        model_engine.step()
        if enable_gan:
            progress_bar.set_postfix(
                {
                    "gen": generator_loss.item(),
                    "loss": loss.item(),
                    "recon": recon.item(),
                    "perc": perc.item(),
                    "ssim": ssim.item(),
                }
            )
            if rank == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "gen": generator_loss.item(),
                        "recon": recon.item(),
                        "perc": perc.item(),
                        "ssim": ssim.item(),
                        "lr": model_optimizer.param_groups[0]["lr"],
                        "train_step": train_step,
                        "epoch": epoch,
                    }
                )
        else:
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "recon": recon.item(),
                    "perc": perc.item(),
                    "ssim": ssim.item(),
                }
            )
            if rank == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "recon": recon.item(),
                        "perc": perc.item(),
                        "ssim": ssim.item(),
                        "lr": model_optimizer.param_groups[0]["lr"],
                        "train_step": train_step,
                        "epoch": epoch,
                    }
                )
        train_step += 1
    val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=150)
    val_progress_bar.set_description(f"Validation Epoch {epoch}")
    for step, batch_data in val_progress_bar:
        model_engine.eval()
        with torch.no_grad():
            image = batch_data["hi_res"].to(torch.bfloat16).to(device)
            low_res = batch_data["low_res"].to(torch.bfloat16).to(device)
            output = model_engine(low_res)
            loss, recon, perc, ssim = loss_all(output, image)
            val_progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "recon": recon.item(),
                    "perc": perc.item(),
                    "ssim": ssim.item(),
                }
            )
            if rank == 0:
                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "val_recon": recon.item(),
                        "val_perc": perc.item(),
                        "val_ssim": ssim.item(),
                        "lr": model_optimizer.param_groups[0]["lr"],
                        "val_step": val_step,
                    }
                )
                if step == 2:
                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(
                        low_res[0, 0, :, 32, :].float().cpu().numpy(), cmap="gray"
                    )
                    ax[1].imshow(
                        output[0, 0, :, 32, :].float().cpu().numpy(), cmap="gray"
                    )
                    ax[2].imshow(
                        image[0, 0, :, 32, :].float().cpu().numpy(), cmap="gray"
                    )
                    wandb.log({"hi_res": fig, "epoch": epoch})
        val_step += 1
    if epoch % 50 == 0:
        save_dir = f"./{checkpoint_path}/Swin_{epoch}"
        model_engine.save_checkpoint(save_dir)
    save_dir = f"./{checkpoint_path}/Swin_latest"
    model_engine.save_checkpoint(save_dir, tag="latest_step")
wandb.finish()

# OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=4,5,6,7 taskset -c 31-47 deepspeed train_cerebro_swin.py
