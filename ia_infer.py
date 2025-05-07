import os
import zipfile
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImageD, AddChannelD, SpacingD,
    OrientationD, ScaleIntensityD, ToTensorD
)
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader

def inferir_cbct(input_folder: str, output_folder: str):
    # Extrai os DICOMs do ZIP
    zip_files = [f for f in os.listdir(input_folder) if f.endswith(".zip")]
    if not zip_files:
        return "Nenhum arquivo .zip encontrado no input."

    dicom_zip_path = os.path.join(input_folder, zip_files[0])
    dicom_extracted_path = os.path.join(input_folder, "dicoms")
    os.makedirs(dicom_extracted_path, exist_ok=True)

    with zipfile.ZipFile(dicom_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dicom_extracted_path)

    # Carrega volume DICOM como NIfTI
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_extracted_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    affine = np.eye(4)
    nifti_path = os.path.join(output_folder, "volume.nii.gz")
    nib.save(nib.Nifti1Image(image_array, affine), nifti_path)

    # Define as transformações
    transforms = Compose([
        LoadImageD(keys=["image"]),
        AddChannelD(keys=["image"]),
        SpacingD(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        OrientationD(keys=["image"], axcodes="RAS"),
        ScaleIntensityD(keys=["image"]),
        ToTensorD(keys=["image"])
    ])

    # Prepara o dataset
    data_dicts = [{"image": nifti_path}]
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=1)

    # Modelo MONAI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    checkpoint = torch.load("checkpoint_headneck.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Inferência
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1).cpu().numpy()

            # Aqui assumimos que classe 1 é a lesão
            volume = np.sum(outputs == 1)
            achados_path = os.path.join(output_folder, "achados.txt")
            with open(achados_path, "w") as f:
                if volume > 0:
                    f.write("Achado automático: Volume segmentado detectado (classe 1).\n")
                    f.write(f"Volume aproximado da lesão: {volume} voxels.\n")
                else:
                    f.write("Nenhum achado relevante foi detectado.\n")

            break

    return "Inferência MONAI concluída."
