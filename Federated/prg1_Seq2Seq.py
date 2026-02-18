# seq2seq_nids.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ---- CONFIG ----
CSV_PATH = "wustl-ehms-2020_with_attacks_categories.csv"   # point to your CSV file
SEQ_LEN = 10                    # length of sequence window (tunable)
STRIDE = 1                      # slide stride between windows
BATCH_SIZE = 64
EPOCHS = 40
LATENT_DIM = 64                 # LSTM hidden size
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
NUM_CLASSES = 2                 # normal vs attack (binary)
MODEL_SAVE_PATH = "seq2seq_nids.h5"

# ---- Utilities ----
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    # Inspect columns - in your snippet:
    # remove MACs, Packet_num, Attack Category (textual) if present
    # Keep numeric / useful features and the 'Label' column at end.
    # Adjust these column names to match your file exactly if different.
    # We try to keep all numeric features automatically.
    label_col = "Label"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    # Select numeric columns only except label (we'll keep label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col not in numeric_cols:
        # label might be integer but safe to ensure label preserved
        numeric_cols.append(label_col)

    df_num = df[numeric_cols].copy()
    # Drop rows with NaNs (or handle otherwise)
    df_num = df_num.dropna().reset_index(drop=True)
    return df_num, label_col

def build_windows(df, features, label_col, seq_len=10, stride=1):
    """
    Build overlapping windows from dataframe.
    Inputs:
      df: dataframe with numeric features and label column
      features: list of feature column names (exclude label)
      label_col: label column name
    Returns:
      X: ndarray (n_windows, seq_len, n_features)
      y: ndarray (n_windows, seq_len) with integer labels per timestep
    """
    data = df[features].values
    labels = df[label_col].values
    n = data.shape[0]
    X_windows = []
    y_windows = []
    for start in range(0, n - seq_len + 1, stride):
        end = start + seq_len
        X_windows.append(data[start:end])
        y_windows.append(labels[start:end])
    X = np.stack(X_windows)
    y = np.stack(y_windows)
    return X, y

def make_binary_label(y_seq):
    # convert label values to binary:
    # assumption: in your CSV 'Label' == 0 means normal; non-zero means attack.
    # If your dataset uses strings for Attack Category, you can map accordingly.
    return (y_seq != 0).astype(int)

# ---- Load data ----
df, label_col = load_and_clean(CSV_PATH)

# pick features: all numeric except label
feature_cols = [c for c in df.columns if c != label_col]
print("Using features:", feature_cols)

# Optionally remove obviously irrelevant numeric columns
# e.g., Temp/SpO2 columns if they aren't network signals; but for now keep all numeric
# feature_cols = [c for c in feature_cols if c not in ("Packet_num", "SrcMac", "DstMac")]

# Build windows
X_raw, y_raw = build_windows(df, feature_cols, label_col, seq_len=SEQ_LEN, stride=STRIDE)
print("Windows built:", X_raw.shape, y_raw.shape)

# Binary labels per timestep (0 normal, 1 attack)
y_bin = make_binary_label(y_raw)   # shape (N, seq_len)

# flatten to fit scaler
n_windows, seq_len, n_features = X_raw.shape
X_flat = X_raw.reshape(-1, n_features)

# scale features
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)
X_scaled = X_flat_scaled.reshape(n_windows, seq_len, n_features)

# Prepare y for seq2seq classification: each timestep is class index (0/1)
y_seq = y_bin  # shape (n_windows, seq_len)

# Train/test split on windows (important: do not mix rows from same time into both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_seq, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)

print("Train/Test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# compute class weights for imbalance (per timestep flattened)
flat_y_train = y_train.flatten()
from sklearn.utils.class_weight import compute_class_weight
class_weights_vals = compute_class_weight("balanced", classes=np.unique(flat_y_train), y=flat_y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights_vals)}
print("Class weights (for flattened timesteps):", class_weights_dict)

# For Keras sample weighting per-sequence, compute weight per window by averaging labels
# windows with any attack get higher weight. Alternatively, use timestep-level loss with class weights.
# We'll use timestep-level class_weight via a custom loss wrapper.

# ---- Seq2Seq model (encoder-decoder LSTM) ----
def build_seq2seq_model(n_features, latent_dim=64, seq_len=10, num_classes=2):
    # Encoder
    encoder_inputs = layers.Input(shape=(seq_len, n_features), name="encoder_inputs")
    encoder_lstm = layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder - we'll use a RepeatVector + LSTM with initial state from encoder
    # and return sequences to produce classification per timestep.
    # Using return_sequences=True produces output for each timestep in seq_len.
    # Provide initial_state=encoder_states to initialize memory.
    decoder_inputs = layers.RepeatVector(seq_len)(state_h)   # simple context repeated
    # Alternatively, feed zeros as inputs:
    # decoder_inputs = layers.Input(shape=(seq_len, latent_dim))   # for teacher forcing variant
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, name="decoder_lstm")
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # time-distributed classification
    dense_time = layers.TimeDistributed(layers.Dense(num_classes, activation="softmax"), name="time_dist")
    outputs = dense_time(decoder_outputs)

    model = models.Model(encoder_inputs, outputs, name="seq2seq_nids")
    return model

model = build_seq2seq_model(n_features=n_features, latent_dim=LATENT_DIM, seq_len=SEQ_LEN, num_classes=NUM_CLASSES)
model.summary()

# custom loss: sparse categorical crossentropy but allow per-timestep class weights
# Keras supports sample_weight and class_weight only for per-sample class weighting, not per-timestep classes when outputs are sequences.
# We'll flatten timesteps in the loss by using sparse_categorical_crossentropy on flattened tensors.

def sparse_time_loss(y_true, y_pred):
    # y_true: (batch, seq_len)
    # y_pred: (batch, seq_len, num_classes)
    y_true_flat = tf.reshape(y_true, [-1])  # (batch * seq_len,)
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(loss)

model.compile(optimizer=optimizers.Adam(1e-3),
              loss=sparse_time_loss,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_acc")])

# Prepare y_train/y_test as integer arrays (they already are 0/1)
# Keras expects y for categorical predictions to match shape (batch, seq_len) for sparse losses we wrapped.
y_train_int = y_train.astype(np.int32)
y_test_int = y_test.astype(np.int32)

# Callbacks
cb_early = callbacks.EarlyStopping(monitor="val_sparse_acc", patience=6, restore_best_weights=True)
cb_checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_sparse_acc")

# ---- Fit ----
history = model.fit(
    X_train, y_train_int,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_early, cb_checkpoint],
    verbose=2
)

# ---- Evaluate / Inference ----
# Predict probabilities per timestep
y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
# shape: (n_windows_test, seq_len, num_classes)
y_pred_seq = np.argmax(y_pred_probs, axis=-1)  # predicted class each timestep

# Sequence-level label: majority vote across timesteps
def seq_majority_label(y_seq_pred):
    # y_seq_pred: (n_windows, seq_len)
    # returns (n_windows,) binary label
    votes = y_seq_pred.sum(axis=1)  # number of predicted attack timesteps
    return (votes >= (y_seq_pred.shape[1] // 2 + 1)).astype(int)  # majority

y_test_seq_labels = seq_majority_label(y_test_int)
y_pred_seq_labels = seq_majority_label(y_pred_seq)

print("Sequence-level classification report (majority-vote):")
print(classification_report(y_test_seq_labels, y_pred_seq_labels, digits=4))

print("Confusion matrix (seq-level):")
print(confusion_matrix(y_test_seq_labels, y_pred_seq_labels))

# Also produce timestep-level report by flattening
flat_true = y_test_int.flatten()
flat_pred = y_pred_seq.flatten()
print("Timestep-level classification report:")
print(classification_report(flat_true, flat_pred, digits=4))

# Save scaler + model for later use
import joblib
joblib.dump(scaler, "scaler.save")
model.save(MODEL_SAVE_PATH)
print("Model and scaler saved.")

# ---- Inference helper ----
def infer_on_raw_rows(df_rows):
    """
    Accept a DataFrame of raw numeric rows in chronological order,
    build windows and predict seq labels. This function scales using saved scaler.
    """
    Xr = df_rows[feature_cols].values
    # if length < SEQ_LEN, you could pad or skip
    if Xr.shape[0] < SEQ_LEN:
        raise ValueError("Need at least seq_len rows for a prediction window.")
    # For demonstration, we'll predict only first window
    Xw = Xr[:SEQ_LEN]
    Xw_scaled = scaler.transform(Xw).reshape(1, SEQ_LEN, n_features)
    probs = model.predict(Xw_scaled)
    pred_seq = np.argmax(probs, axis=-1)[0]
    seq_label = seq_majority_label(pred_seq.reshape(1, -1))[0]
    return seq_label, pred_seq, probs

# Example: use first window of test set for demonstration
demo_label, demo_pred_seq, demo_probs = infer_on_raw_rows(df.iloc[:SEQ_LEN])
print("Demo inference seq label:", demo_label)


# ============================================================
#  PREDICT CATEGORY FOR LAST RECORD OF THE DATASET
# ============================================================

print("\n================ LAST RECORD PREDICTION ================\n")

# 1. Load last full row of the original CSV → then apply SAME cleaning
df_full = pd.read_csv(CSV_PATH)

# Apply SAME cleaning used during training
df_clean, label_col = load_and_clean(CSV_PATH)

# 2. Extract last numeric-cleaned record (single row)
last_row = df_clean.tail(1).copy()        # shape → (1, n_features+label)

# 3. Separate features (remove label)
last_features = last_row[feature_cols].copy()    # shape (1, n_features)

# 4. Convert to numeric row vector → shape (n_features,)
row_arr = last_features.values.reshape(-1, n_features)

# 5. Scale using saved scaler
row_scaled = scaler.transform(row_arr)     # shape (1, n_features)

# 6. Expand to 10 timesteps to match SEQ_LEN
#    The same single row is repeated 10 times
X_last = np.repeat(row_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)
# shape = (1, 10, n_features)

# 7. Predict using the trained seq2seq model
pred_probs = model.predict(X_last)
pred_seq = np.argmax(pred_probs, axis=-1)[0]     # per-timestep prediction

# 8. Majority vote across timesteps → final label
final_label = seq_majority_label(pred_seq.reshape(1, -1))[0]

# 9. Print results
print("Last Record Raw Values:")
print(last_row)

print("\nPredicted per-timestep classes:", pred_seq.tolist())
print("Final Predicted Category:", "ATTACK" if final_label == 1 else "NORMAL")

# For convenience: probability of attack on first timestep
print("Attack Probability (t=0):", pred_probs[0][0][1])

print("\n========================================================\n")

