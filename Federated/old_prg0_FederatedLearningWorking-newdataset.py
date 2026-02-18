# ================================
# 1. Imports
# ================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 2. Load Dataset
# ================================
df = pd.read_csv("../newdataset/train_test_network.csv")
"""
src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state,missed_bytes,src_pkts,src_ip_bytes,dst_pkts,dst_ip_bytes,dns_query,dns_qclass,dns_qtype,dns_rcode,dns_aa,dns_rd,dns_ra,dns_rejected,ssl_version,ssl_cipher,ssl_resumed,ssl_established,ssl_subject,ssl_issuer,http_trans_depth,http_method,http_uri,http_version,http_request_body_len,http_response_body_len,http_status_code,http_user_agent,http_orig_mime_types,http_resp_mime_types,weird_name,weird_addl,weird_notice,Label,type
192.168.1.37,4444,192.168.1.193,49178,tcp,-,290.371539,101568,2592,OTH,0,108,108064,31,3832,-,0,0,0,-,-,-,-,-,-,-,-,-,-,-,-,-,-,0,0,0,-,-,-,-,-,-,1,backdoor
192.168.1.193,49180,192.168.1.37,8080,tcp,-,0.000102,0,0,REJ,0,1,52,1,40,-,0,0,0,-,-,-,-,-,-,-,-,-,-,-,-,-,-,0,0,0,-,-,-,-,-,-,1,backdoor
"""
df=df.dropna(0)
df=df.iloc[:100001,:]
print("Original shape:", df.shape)

# ================================
# 3. Keep Required Columns ONLY
# ================================
"""
selected_columns = [
    'src_ip','src_port','dst_ip','dst_port','service',
    'duration','src_bytes','dst_bytes','conn_state',
    'missed_bytes','src_pkts','src_ip_bytes',
    'dst_pkts','dst_ip_bytes','dns_query',
    'dns_qclass','dns_qtype',
    'http_response_body_len','http_status_code',
    'Label'
]

df = df[selected_columns]
"""
print("After column filtering:", df.shape)

# ================================
# 4. Encode Categorical Columns
# ================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col], _ = pd.factorize(df[col])

# ================================
# 5. Split Features & Target
# ================================
X = df.drop("Label", axis=1)
y = df["Label"]

# ================================
# 6. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 7. Feature Scaling
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 8. Reshape for CNN (samples, features, channels)
# ================================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================================
# 9. Encode Labels (One-Hot)
# ================================
num_classes = len(np.unique(y))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# ================================
# 10. CNN Model
# ================================
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
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

model.summary()

# ================================
# 11. Train Model
# ================================
history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# ================================
# 12. Evaluate Model
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ================================
# 13. Predictions & Metrics
# ================================
y_pred = model.predict(X_test)
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
plt.title("CNN Confusion Matrix")
plt.show()

# ================================
# 15. Save Model
# ================================
model.save("network_cnn_model.h5")
print("Model saved as network_cnn_model.h5")


# ================================
# 10–12. CNN TRAINING (5 ROUNDS)
# ================================

from tensorflow.keras import backend as K

num_rounds =2
accuracy_scores = []

for round_id in range(num_rounds):
    print(f"\n========== ROUND {round_id + 1} / {num_rounds} ==========")

    # Clear previous TF graph (VERY IMPORTANT)
    K.clear_session()

    # ================================
    # Build CNN Model
    # ================================
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

    # ================================
    # Train Model
    # ================================
    model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    # ================================
    # Evaluate Model
    # ================================
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(test_acc)

    print(f"Round {round_id + 1} Test Accuracy: {test_acc:.4f}")

# ================================
# FINAL AVERAGE ACCURACY
# ================================
final_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print("\n==============================")
print("Accuracy per round:")
for i, acc in enumerate(accuracy_scores, 1):
    print(f"Round {i}: {acc:.4f}")

print(f"\nFINAL AVERAGE ACCURACY (5 rounds): {final_accuracy:.4f}")
print("==============================")

