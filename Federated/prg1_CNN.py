# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ================================
# 2. LOAD DATASET
# ================================
# Change filename if needed
df = pd.read_csv("../newdataset/DDos.pcap_ISCX.csv")

print("Dataset shape:", df.shape)
print(df.head())


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

# Convert to categorical for CNN
y_cat = to_categorical(y_encoded)

print("Classes:", le.classes_)


# ================================
# 5. FEATURE SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ================================
# 6. RESHAPE FOR CNN (3D)
# ================================
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)


# ================================
# 7. TRAIN-TEST SPLIT (STC)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ================================
# 8. CNN MODEL
# ================================
model = Sequential()

model.add(Conv1D(64, kernel_size=3, activation='relu',
                 input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(y_cat.shape[1], activation='softmax'))

model.summary()


# ================================
# 9. COMPILE MODEL
# ================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ================================
# 10. TRAIN MODEL
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ================================
# 11. PREDICTION & ACCURACY
# ================================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print("\nFinal Accuracy:", accuracy)


# ================================
# 12. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)


# ================================
# 13. CONFUSION MATRIX PLOT
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
plt.title("Confusion Matrix")
plt.show()


# ================================
# 14. TRAINING ACCURACY & LOSS PLOTS
# ================================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()


# ================================
# 15. CLASSIFICATION REPORT
# ================================
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=le.classes_
))
# ================================
# 11. SAVE MODEL
# ================================
model.save("cnn_ddos_model.h5")
print("Model saved as cnn_ddos_model.h5")