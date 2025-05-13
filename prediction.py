import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from nilearn import datasets, image, plotting
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler
import nibabel as nib

classifier = joblib.load("rf_autism_model.pkl")


scaler = joblib.load("scaler.pkl")

feature_names = joblib.load("feature_names.pkl")


target_rois = [
    "Cingulum_Post_R", "Cerebelum_Crus2_R", "Frontal_Inf_Oper_L", "Cingulum_Mid_R",
    "Precuneus_R", "Postcentral_R", "Thalamus_L", "Precuneus_L", "Cerebelum_Crus2_L",
    "Supp_Motor_Area_L", "Cingulum_Post_L", "Frontal_Inf_Oper_R", "Heschl_L",
    "Cerebelum_6_L", "Precentral_L", "Precentral_R", "Thalamus_R", "Parietal_Inf_R"
]


nii_file_path = r"C:\Users\heman\PycharmProjects\PythonProject2\SkullStripped_NII\sub-28698.nii"  # Replace with your actual file path


aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
atlas_filename = aal_atlas.maps
atlas_labels = aal_atlas.labels

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
masker.fit()


subject_img = image.load_img(nii_file_path)


time_series = masker.transform(subject_img)


aal_features_df = pd.DataFrame(time_series, columns=atlas_labels)


missing_rois = [roi for roi in target_rois if roi not in aal_features_df.columns]
if missing_rois:
    raise ValueError(f"The following ROIs are missing in the extracted features: {missing_rois}")


selected_features = aal_features_df[target_rois]


scaled_features = scaler.transform(selected_features)


prediction = classifier.predict(scaled_features)
prediction_proba = classifier.predict_proba(scaled_features)


print(f"Predicted Class: {'Autism' if prediction[0] == 1 else 'No Autism'}")
print(f"Prediction Probability: {prediction_proba[0][prediction[0]]:.2f}")


explainer = shap.Explainer(classifier)


shap_values = explainer(scaled_features)


shap.summary_plot(shap_values, features=scaled_features, feature_names=target_rois, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()
print("SHAP summary plot saved as 'shap_summary_plot.png'.")


shap_values_autism = shap_values.values[..., 1]

mean_shap_values = np.abs(shap_values_autism).mean(axis=0)


shap_importance = pd.Series(mean_shap_values, index=target_rois)


top_rois = shap_importance.sort_values(ascending=False).head(5)
print("\nTop 5 Contributing ROIs:")
print(top_rois.to_string())


top_roi_names = top_rois.index.tolist()


atlas_img = image.load_img(atlas_filename)


roi_indices = []
for roi in top_roi_names:
    if roi in atlas_labels:
        idx = atlas_labels.index(roi) + 1  # +1 because atlas indices start at 1
        roi_indices.append(idx)
        print(f"ROI '{roi}' mapped to atlas index {idx}")
    else:
        print(f"ROI '{roi}' not found in atlas labels.")

