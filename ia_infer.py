import nibabel as nib
import numpy as np

def run_inference(input_path: str, output_path: str):
    img = nib.load(input_path)
    data = img.get_fdata()
    mask = (data > np.percentile(data, 95)).astype(np.uint8)
    mask_nifti = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(mask_nifti, output_path)
