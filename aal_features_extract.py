import os
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker


nii_folder = r"C:\Users\heman\PycharmProjects\PythonProject2\SkullStripped_NII"  # Folder containing sub-xxxxx.nii.gz files
dx_csv_path = r"C:\Users\heman\PycharmProjects\PythonProject2\ABIDEII_Composite_Phenotypic (1).csv"  # CSV with SUB_ID and DX_GROUP
output_csv_path = r"C:\Users\heman\PycharmProjects\PythonProject2\aal_features_with_dx.csv"  # Output CSV path

dx_df = pd.read_csv(dx_csv_path, encoding='latin1')
dx_df['SUB_ID'] = dx_df['SUB_ID'].astype(str)


aal_atlas = datasets.fetch_atlas_aal()
aal_labels_img = aal_atlas.maps
region_labels = aal_atlas.labels


masker = NiftiLabelsMasker(labels_img=aal_labels_img, standardize=True, memory='nilearn_cache')


all_subjects_data = []

for file in os.listdir(nii_folder):
    if file.endswith(".nii") and file.startswith("sub-"):
        subject_file = os.path.join(nii_folder, file)
        subject_id = file.split("-")[1].split(".")[0]  # Extract 28675 from 'sub-28675.nii'

        try:
            img = nib.load(subject_file)
            time_series = masker.fit_transform(img)
            mean_signal = time_series.mean(axis=0)  # 116 region signals

            # Find DX_GROUP
            dx_row = dx_df[dx_df['SUB_ID'] == subject_id]
            if dx_row.empty:
                print(f"DX_GROUP not found for {subject_id}. Skipping.")
                continue
            dx_group = dx_row.iloc[0]['DX_GROUP']

            subject_features = {
                'SUB_ID': subject_id,
                'DX_GROUP': dx_group
            }
            for i, label in enumerate(region_labels):
                subject_features[label] = mean_signal[i]

            all_subjects_data.append(subject_features)

        except Exception as e:
            print(f"Error processing {file}: {e}")


df = pd.DataFrame(all_subjects_data)
df.to_csv(output_csv_path, index=False)
print("Feature extraction complete. Output saved to:", output_csv_path)
