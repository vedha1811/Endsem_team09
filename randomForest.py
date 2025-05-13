import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv(r"C:\Users\heman\PycharmProjects\PythonProject2\filtered_dataset.csv")
X = df.drop(columns=["DX_GROUP"])
y = df['DX_GROUP'].map({1: 0, 2: 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
print("5-Fold Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy: {:.4f}".format(cv_scores.mean()))


y_pred = classifier.predict(X_test)
print("\nTest Set Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision: {:.4f}".format(precision))
print("Recall:    {:.4f}".format(recall))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Autism", "Autism"], yticklabels=["No Autism", "Autism"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
