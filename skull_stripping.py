import os
from pathlib import Path
from nilearn import image, masking, plotting
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter


input_folder = Path(r"C:\Users\heman\PycharmProjects\PythonProject2\data\extracted_nii")
output_folder = Path(r"C:\Users\heman\PycharmProjects\PythonProject2\SkullStripped_NII")
output_folder.mkdir(parents=True, exist_ok=True)


for nii_path in input_folder.glob("*.nii*"):
    try:

        anat_img = image.load_img(str(nii_path))
        anat_data = anat_img.get_fdata()


        raw_mask = masking.compute_brain_mask(anat_img)
        cleaned_mask = binary_opening(raw_mask.get_fdata(), structure=np.ones((3, 3, 3)))
        cleaned_mask = binary_closing(cleaned_mask, structure=np.ones((3, 3, 3)))
        smoothed_mask = gaussian_filter(cleaned_mask.astype(float), sigma=1)
        final_mask = (smoothed_mask > 0.5).astype(int)


        stripped_data = anat_data * final_mask
        stripped_img = nib.Nifti1Image(stripped_data, affine=anat_img.affine, header=anat_img.header)


        filename_parts = nii_path.name.split("_")
        subject_id = filename_parts[0]  # e.g., "sub-28675"

        output_path = output_folder / f"{subject_id}.nii"
        nib.save(stripped_img, str(output_path))

        print(f"Saved: {output_path.name}")

    except Exception as e:
        print(f"❌ Failed to process {nii_path.name}: {e}")

print("\n All files processed.")
