# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv("../newdataset/DDos.pcap_ISCX.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Relabel rows from 47750 to end as BENIGN
df.loc[47750:, 'Label'] = 'BENIGN'


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
# 5. TRAIN-TEST SPLIT (STC)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.45,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ================================
# 6. DECISION TREE MODEL
# ================================
dt_model = DecisionTreeClassifier(
    criterion='gini',       # or 'entropy'
    max_depth=None,         # try 10, 20 for pruning
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# ================================
# 7. TRAIN MODEL
# ================================
dt_model.fit(X_train, y_train)


# ================================
# 8. PREDICTION & ACCURACY
# ================================
y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", accuracy)


# ================================
# 9. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


# ================================
# 10. CONFUSION MATRIX PLOT
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
plt.title("Decision Tree Confusion Matrix")
plt.show()


# ================================
# 11. CLASSIFICATION REPORT
# ================================
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))
