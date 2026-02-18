# predict_last_record.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
MODEL_PATH = "seq2seq_nids.h5"                       # your trained model
SCALER_PATH = "scaler.save"                          # saved StandardScaler
CSV_PATH = "wustl-ehms-2020_with_attacks_categories.csv"  # dataset for last record
SEQ_LEN = 10                                        # must match training SEQ_LEN

# ---------------- Utilities ----------------
def load_and_clean(csv_path):
    """Keep numeric columns + label"""
    df = pd.read_csv(csv_path)
    label_col = "Label"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col not in numeric_cols:
        numeric_cols.append(label_col)
    
    df_num = df[numeric_cols].dropna().reset_index(drop=True)
    return df_num, label_col

def seq_majority_label(y_seq_pred):
    """Majority vote across timesteps"""
    votes = y_seq_pred.sum(axis=1)
    return (votes >= (y_seq_pred.shape[1] // 2 + 1)).astype(int)

# ---------------- Load Model & Scaler ----------------
def sparse_time_loss(y_true, y_pred):
    # needed if your model was trained with custom loss
    import tensorflow as tf
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(loss)

print("Loading model...")
model = load_model(MODEL_PATH, custom_objects={"sparse_time_loss": sparse_time_loss})

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

# ---------------- Load CSV & last record ----------------
print("Loading dataset and cleaning numeric columns...")
df_clean, label_col = load_and_clean(CSV_PATH)

# Features (same as used in training)
feature_cols = [c for c in df_clean.columns if c != label_col]
n_features = len(feature_cols)

# Last record
last_row = df_clean.tail(1).copy()                  # shape (1, n_features + label)
last_features = last_row[feature_cols].copy()       # shape (1, n_features)

# ---------------- Scale and reshape ----------------
row_arr = last_features.values.reshape(-1, n_features)  # shape (1, n_features)
row_scaled = scaler.transform(row_arr)                   # scaled

# Repeat row to match SEQ_LEN timesteps
X_last = np.repeat(row_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)  # shape (1, SEQ_LEN, n_features)

# ---------------- Predict ----------------
pred_probs = model.predict(X_last)
pred_seq = np.argmax(pred_probs, axis=-1)[0]  # per-timestep prediction
final_label = seq_majority_label(pred_seq.reshape(1, -1))[0]

# ---------------- Output ----------------
print("\n================ LAST RECORD PREDICTION ================\n")
print("Last Record Raw Values:")
print(last_row)
print("\nPredicted per-timestep classes:", pred_seq.tolist())
print("Final Predicted Category:", "ATTACK" if final_label == 1 else "NORMAL")
print("Attack Probability (t=0):", pred_probs[0][0][1])
print("\n========================================================\n")
