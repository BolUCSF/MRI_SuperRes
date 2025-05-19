import os, json, torch, random
import numpy as np
from monai import transforms
from monai.data import DataLoader, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import sliding_window_inference
from functools import partial
import matplotlib.pyplot as plt
from models.SuperFormer import SuperFormer
import nibabel as nib

infer_transform = transforms.Compose([
    
    transforms.CopyItemsd(keys=['path'], names=['image']),
    transforms.LoadImaged(keys=['image']),
    # transforms.DeleteItemsd(keys=['path']),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.EnsureTyped(keys=["image"]),
    transforms.Orientationd(keys=["image"], axcodes="RAS"),
    transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1,clip=True ),
    transforms.Spacingd(keys=['image'],pixdim=(1,1,1),mode="nearest"),
    transforms.ResizeWithPadOrCropd(keys=['image'],spatial_size=(240,240,160)),
    transforms.CopyItemsd(keys=['image'], names=['hi_res','low_res'], times=2),
    transforms.DeleteItemsd(keys=['image']),])

model = SuperFormer(upscale=1,
                   patch_size = 2,
                   in_chans=1,
                   img_size=64,
                   window_size=8,
                   img_range=1.0,
                   depths=[6, 6, 6],
                   embed_dim=240,
                   num_heads=[6, 6, 6],
                   mlp_ratio=2,
                   upsampler=None,
                   resi_connection="1conv",
                   ape=False, 
                   rpb=True,
                   output_type = "direct",
                   num_feat = 126)

def main(args):
    image_path = args.image_path
    device_id = args.device_id
    overlap_ratio = args.overlap_ratio
    num_upsample = args.num_upsample
    output_path = args.output_path

    image_path = {'path':image_path}
    image_data = infer_transform(image_path)
    model.load_state_dict(torch.load('/working/Project/Sup_Res/checkpoint_superformer/pytorch_model.bin',weights_only=True))
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    print(f"Device cuda:{device_id} — Free: {free_mem / 1024**2:.2f} MB, Total: {total_mem / 1024**2:.2f} MB")
    avail_mem = free_mem / 1024**2
    batch_size = np.floor((avail_mem-1200) / 1400).astype(int)
    print(f"Batch size: {batch_size}")
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[64, 64, 64],
        sw_batch_size=batch_size,
        predictor=model,
        overlap=overlap_ratio,
    )
    with torch.no_grad():
        with torch.autocast(device_type="cuda" , dtype=torch.bfloat16):
            path = image_data["path"]
            image = image_data["hi_res"].unsqueeze(0).to(device)
            low_res = image_data["low_res"].unsqueeze(0).to(device)
            for i in range(num_upsample):
                low_res = model_inferer(low_res)
            recon = low_res.squeeze().cpu().float().numpy()
            high_res = image.squeeze().cpu().float().numpy()
            basename = os.path.basename(path)
            raw_nib = nib.load(path)
            spacing = raw_nib.header['pixdim'][1:4]
            spacing[2] = 1.0
            nib_size = raw_nib.shape
            nib_size = list(nib_size)
            high_size = high_res.shape
            nib_size[2] = high_size[2]
            affine = np.eye(4)
            affine[:3,:3] = np.diag(spacing)
            transfrom_infer = transforms.Compose([
                transforms.EnsureChannelFirst(channel_dim=0),
                transforms.Spacing(pixdim=spacing,mode='bilinear'),
                transforms.ResizeWithPadOrCrop(spatial_size=nib_size),
                ])
            recon_image = transfrom_infer(recon[np.newaxis,:,:,:].copy())[0].numpy()
            high_res_image = transfrom_infer(high_res[np.newaxis,:,:,:].copy())[0].numpy()
            recon_nib = nib.Nifti1Image(recon_image.astype(np.float32),affine)
            high_res_nib = nib.Nifti1Image(high_res_image.astype(np.float32),affine)
            if output_path is None:
                output_path = os.path.dirname(path)
            else:
                os.makedirs(output_path,exist_ok=True)
            nib.save(high_res_nib,os.path.join(output_path, basename.replace('.nii','_resample.nii')))
            nib.save(recon_nib,os.path.join(output_path, basename.replace('.nii','_recon.nii')))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing arguments")

    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input NIfTI image")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save output result (default: None)")
    parser.add_argument("--device_id", type=int, default=0,
                        help="CUDA device ID to use")
    parser.add_argument("--overlap_ratio", type=float, default=0.6,
                        help="Overlap ratio for patch-based processing")
    parser.add_argument("--num_upsample", type=int, default=2,
                        help="Number of upsampling steps")

    args = parser.parse_args()
    main(args)