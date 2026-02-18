# ================================
# 1. Imports
# ================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("../newdataset/train_test_network.csv")
df = df.dropna()
df = df.iloc[:100001, :]
print("Original shape:", df.shape)

# ================================
# 3. Encode Categorical Columns
# ================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col], _ = pd.factorize(df[col])

# ================================
# 4. Split Features & Target
# ================================
X = df.drop("Label", axis=1)
y = df["Label"]

# ================================
# 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 6. Feature Scaling
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 7. Reshape for CNN
# ================================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================================
# 8. Encode Labels
# ================================
num_classes = len(np.unique(y))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# ================================
# 9. Model Builder Function
# ================================
def build_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu',
               input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================================
# 10. Trimmed Mean Aggregation
# ================================
def trimmed_mean_aggregation(weight_list, trim_ratio=0.2):
    aggregated_weights = []
    num_models = len(weight_list)
    trim_k = int(trim_ratio * num_models)

    for layer_weights in zip(*weight_list):
        layer_stack = np.array(layer_weights)
        sorted_weights = np.sort(layer_stack, axis=0)

        trimmed = sorted_weights[trim_k:num_models - trim_k]
        aggregated_layer = np.mean(trimmed, axis=0)

        aggregated_weights.append(aggregated_layer)

    return aggregated_weights

# ================================
# 11. Multi-Round Training
# ================================
num_rounds = 5
accuracy_scores = []
all_weights = []

for r in range(num_rounds):
    print(f"\n========== ROUND {r+1}/{num_rounds} ==========")

    K.clear_session()
    model = build_model()

    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Round {r+1} Accuracy: {test_acc:.4f}")

    accuracy_scores.append(test_acc)
    all_weights.append(model.get_weights())

# ================================
# 12. Apply Trimmed Mean
# ================================
print("\nApplying Trimmed Mean Aggregation...")

aggregated_weights = trimmed_mean_aggregation(all_weights, trim_ratio=0.2)

final_model = build_model()
final_model.set_weights(aggregated_weights)

final_loss, final_acc = final_model.evaluate(X_test, y_test, verbose=0)

print("\n==============================")
print("Accuracy per round:")
for i, acc in enumerate(accuracy_scores, 1):
    print(f"Round {i}: {acc:.4f}")

print(f"\nFINAL Trimmed Mean Accuracy: {final_acc:.4f}")
print("==============================")

# ================================
# 13. Predictions & Metrics
# ================================
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# ================================
# 14. Confusion Matrix
# ================================
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN + Trimmed Mean Confusion Matrix")
plt.show()

# ================================
# 15. Save Final Model
# ================================
final_model.save("network_cnn_trimmedmean_model.h5")
print("Model saved as network_cnn_trimmedmean_model.h5")
