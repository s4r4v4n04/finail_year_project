# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv("../newdataset/DDos.pcap_ISCX.csv")

print("Dataset shape:", df.shape)
print(df.head())
df.loc[47750:,'Label']='BENIGN'

# ================================
# 3. SPLIT FEATURES & LABEL
# ================================
X = df.drop("Label", axis=1)
y = df["Label"]

# FIX BAD VALUES
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)


# ================================
# 4. LABEL ENCODING
# ================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:", le.classes_)


# ================================
# 5. FEATURE SCALING (VERY IMPORTANT FOR SVM)
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ================================
# 6. TRAIN-TEST SPLIT (STC)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.45,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ================================
# 7. SVM MODEL
# ================================
svm_model = SVC(
    kernel='rbf',       # Best for non-linear IDS data
    C=1.0,
    gamma='scale',
    random_state=42
)


# ================================
# 8. TRAIN MODEL
# ================================
svm_model.fit(X_train, y_train)


# ================================
# 9. PREDICTION & ACCURACY
# ================================
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", accuracy)


# ================================
# 10. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


# ================================
# 11. CONFUSION MATRIX PLOT
# ================================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()


# ================================
# 12. CLASSIFICATION REPORT
# ================================
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))
