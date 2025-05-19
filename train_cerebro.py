import os, json, deepspeed, wandb, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms, data
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from tqdm import tqdm
from monai.losses.ssim_loss import SSIMLoss
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import  PatchDiscriminator
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import Dataset
from monai.networks.nets.swin_unetr import SwinUNETR

class MRIDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.train_transform = transforms.Compose([
            transforms.LoadImaged(keys=["file_name"]),
            transforms.CopyItemsd(keys=['file_name'], names=['image']),
            transforms.DeleteItemsd(keys=['file_name']),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.EnsureTyped(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1,clip=True ),
            transforms.Spacingd(keys=['image'],pixdim=(1,1,1),mode=3),
            transforms.ResizeWithPadOrCropd(keys=['image'],spatial_size=(240,240,155)),
            transforms.CopyItemsd(keys=['image'], names=['hi_res','low_res'], times=2),
            transforms.DeleteItemsd(keys=['image']),
            transforms.OneOf([
            transforms.Lambdad(keys=["low_res"], func=lambda x: x[:, :, :, ::3]),
            transforms.Lambdad(keys=["low_res"], func=lambda x: x[:, :, :, ::5]),
            ]),
            transforms.Resized(keys=["low_res"], spatial_size=(240,240,155), mode="bilinear"),
            transforms.CropForegroundd(keys=['hi_res','low_res'], source_key="hi_res",allow_smaller=False),
            transforms.RandSpatialCropd(keys=['hi_res','low_res'], roi_size=(64,64,64), random_size=False),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        data = self.image_paths[idx]
        transformed_data = self.train_transform(data)
        return transformed_data


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '3224'
set_determinism(42)

train_batchsize = 18
epochs = 200
enable_gan = True

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
if rank == 0:
    wandb.init(
        project="SuperRes_A6000",
        config={
        "architecture": "SwinUNETR",
        "dataset": "cerebro",
        "epochs": epochs,
        "Gan": enable_gan,
        }
    )

checkpoint_path = 'deepspeed_checkpoint'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

with open('json/train_list.json') as f:
    train_files = json.load(f)
with open('json/val_list.json') as f:
    validation_files = json.load(f)

train_ds = MRIDataset(train_files)
valid_ds = MRIDataset(validation_files)

sampler_train = DistributedSampler(train_ds, shuffle=True, num_replicas=world_size, rank=rank)
train_loader = DataLoader(train_ds, batch_size=train_batchsize, shuffle=False, num_workers=8, persistent_workers=True, sampler=sampler_train, collate_fn=None)
sampler_val = DistributedSampler(valid_ds, shuffle=False, num_replicas=world_size, rank=rank)
val_loader = DataLoader(valid_ds, batch_size=train_batchsize, shuffle=False, num_workers=8, persistent_workers=True, sampler=sampler_val, collate_fn=None)

model = SwinUNETR(img_size=64,in_channels=1,out_channels=1,feature_size=48)
# store_dict = model.state_dict()
# model_dict = torch.load('pretrained_model/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')["state_dict"]
# for key in model_dict.keys():
#     if "out" not in key:
#         store_dict[key].copy_(model_dict[key])
store_dict = torch.load('/data/Sup_Res/deepspeed_checkpoint/siglip_latest/siglip_latest.pt')
model.load_state_dict(store_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-6)
adv_loss = PatchAdversarialLoss(criterion="least_squares")
discriminator = PatchDiscriminator(spatial_dims=3, in_channels=1, num_layers_d=3, num_channels=36)
discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-5)

discriminator_engine, discriminator_optimizer, _, _ = deepspeed.initialize(
    model=discriminator,
    model_parameters=discriminator.parameters(),
    optimizer=optimizer_d,
    config="deepspeed_json/deepspeed_low.json"
)

model_engine, model_optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config="deepspeed_json/deepspeed_low.json",
)
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

        output = model(low_res)
        loss, recon, perc, ssim = loss_all(output, image)

        if enable_gan:
            discriminator_optimizer.zero_grad()
            logits_real = discriminator_engine(image.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            logits_fake = discriminator_engine(output.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            discriminator_engine.backward(loss_d_fake+loss_d_real)

            discriminator_engine.eval()
            with torch.no_grad():
                logits_fake = discriminator_engine(output.contiguous().detach())[-1]
            discriminator_engine.train()
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss += generator_loss

        model_engine.backward(loss)
        model_engine.step()
        if enable_gan:
            progress_bar.set_postfix(
                    {
                        "gen": generator_loss.item(),
                        "loss": loss.item(),
                        'recon': recon.item(),
                        'perc': perc.item(),
                        'ssim': ssim.item(),
                    }
                )
            if rank == 0:
                wandb.log({
                    "loss": loss.item(),
                    "gen": generator_loss.item(),
                    'recon': recon.item(),
                    'perc': perc.item(),
                    'ssim': ssim.item(),
                    'lr': model_optimizer.param_groups[0]['lr'],
                    'train_step': train_step,
                    'epoch': epoch,
                })
        else:
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    'recon': recon.item(),
                    'perc': perc.item(),
                    'ssim': ssim.item(),
                }
            )
            if rank == 0:
                wandb.log({
                    "loss": loss.item(),
                    'recon': recon.item(),
                    'perc': perc.item(),
                    'ssim': ssim.item(),
                    'lr': model_optimizer.param_groups[0]['lr'],
                    'train_step': train_step,
                    'epoch': epoch,
                })
        train_step += 1
    val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=150)
    val_progress_bar.set_description(f"Validation Epoch {epoch}")
    for step, batch_data in val_progress_bar:
        model_engine.eval()
        with torch.no_grad():
            image = batch_data["hi_res"].to(torch.bfloat16).to(device)
            low_res = batch_data["low_res"].to(torch.bfloat16).to(device)
            output = model(low_res)
            loss, recon, perc, ssim = loss_all(output, image)
            val_progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    'recon': recon.item(),
                    'perc': perc.item(),
                    'ssim': ssim.item(),
                }
            )
            if rank == 0:
                wandb.log({
                    "val_loss": loss.item(),
                    'val_recon': recon.item(),
                    'val_perc': perc.item(),
                    'val_ssim': ssim.item(),
                    'lr': model_optimizer.param_groups[0]['lr'],
                    'val_step': val_step,
                })
                if step == 2:
                    fig, ax = plt.subplots(1,2)
                    ax[0].imshow(output[0,0,:,32,:].float().cpu().numpy(),cmap='gray')
                    ax[1].imshow(image[0,0,:,32,:].float().cpu().numpy(),cmap='gray')
                    wandb.log({"hi_res": fig,"epoch":epoch})
        val_step += 1
    if epoch%50 == 0:
        vae_save_dir = f"./{checkpoint_path}/siglip_{epoch}"
        model_engine.save_checkpoint(vae_save_dir)
    vae_save_dir = f"./{checkpoint_path}/siglip_latest"
    model_engine.save_checkpoint(vae_save_dir, tag = 'latest_step')
wandb.finish()