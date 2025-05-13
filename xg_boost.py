import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("filtered_dataset.csv")
X = df.drop(columns=['DX_GROUP'])
y = df['DX_GROUP'].map({1: 0, 2: 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Autism", "Autism"],
            yticklabels=["No Autism", "Autism"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()
plt.savefig("xgboost_confusion_matrix.png")
plt.show()

explainer = shap.Explainer(model)
shap_values = explainer(X_train)

shap.summary_plot(shap_values, features=X_train, feature_names=X.columns.tolist(), show=False)
plt.title("SHAP Summary Plot - XGBoost")
plt.tight_layout()
plt.savefig("xgboost_shap_summary.png")
plt.show()
