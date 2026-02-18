import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

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

def sparse_time_loss(y_true, y_pred):
    """Custom loss if needed"""
    import tensorflow as tf
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(loss)

# ---------------- GUI Functions ----------------
def browse_model():
    path = filedialog.askopenfilename(filetypes=[("Keras H5 Model","*.h5")])
    if path:
        model_entry.delete(0, tk.END)
        model_entry.insert(0, path)

def browse_csv():
    path = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
    if path:
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, path)

def predict_last_record():
    try:
        model_path = model_entry.get()
        csv_path = csv_entry.get()
        if not model_path or not csv_path:
            messagebox.showerror("Error", "Please select both model and CSV file")
            return
        
        # Load model
        model = load_model(model_path, custom_objects={"sparse_time_loss": sparse_time_loss})
        # Load scaler
        scaler = joblib.load("scaler.save")
        
        # Load CSV and clean numeric columns
        df_clean, label_col = load_and_clean(csv_path)
        feature_cols = [c for c in df_clean.columns if c != label_col]
        n_features = len(feature_cols)

        # Last record
        last_row = df_clean.tail(1).copy()
        last_features = last_row[feature_cols].copy()

        # Scale
        row_arr = last_features.values.reshape(-1, n_features)
        row_scaled = scaler.transform(row_arr)

        # Expand to SEQ_LEN timesteps (repeat single row)
        SEQ_LEN = 10
        X_last = np.repeat(row_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)

        # Predict
        pred_probs = model.predict(X_last)
        pred_seq = np.argmax(pred_probs, axis=-1)[0]
        final_label = seq_majority_label(pred_seq.reshape(1, -1))[0]

        # Show results
        msg = (
            f"Last Record Raw Values:\n{last_row}\n\n"
            f"Predicted per-timestep classes: {pred_seq.tolist()}\n"
            f"Final Predicted Category: {'ATTACK' if final_label==1 else 'NORMAL'}\n"
            f"Attack Probability (t=0): {pred_probs[0][0][1]:.4f}"
        )
        messagebox.showinfo("Prediction Result", msg)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(str(e))

# ---------------- GUI Layout ----------------
root = tk.Tk()
root.title("Seq2Seq NIDS - Last Record Predictor")
root.geometry("700x250")

# Model file
tk.Label(root, text="Select .h5 Model:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
model_entry = tk.Entry(root, width=60)
model_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_model).grid(row=0, column=2, padx=5, pady=5)

# CSV file
tk.Label(root, text="Select CSV Dataset:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
csv_entry = tk.Entry(root, width=60)
csv_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_csv).grid(row=1, column=2, padx=5, pady=5)

# Predict button
tk.Button(root, text="Predict Last Record", width=30, height=2, command=predict_last_record).grid(row=3, column=1, pady=20)

root.mainloop()
