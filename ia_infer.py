import os
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd,
    Orientationd, ScaleIntensityd, ToTensord
)
from monai.data import Dataset, DataLoader

def run_inference(volume_path, output_path):
    transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    data_dicts = [{"image": volume_path}]
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Simula carregamento de pesos reais
    # model.load_state_dict(torch.load("checkpoint_headneck.pth"))

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            mask_nifti = nib.Nifti1Image(preds[0], affine=np.eye(4))
            nib.save(mask_nifti, output_path)
