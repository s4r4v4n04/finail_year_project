import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import time
import random

# ---------------- Config ----------------
SEQ_LEN = 10           # Sequence length used in training
PACKET_HEIGHT = 20     # Height of packet rectangles in canvas
PACKET_GAP = 5         # Gap between packets
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400
SIM_DELAY = 0.5        # seconds per packet

# ---------------- Utilities ----------------
def load_and_clean(csv_path):
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
    votes = y_seq_pred.sum(axis=1)
    return (votes >= (y_seq_pred.shape[1] // 2 + 1)).astype(int)

def sparse_time_loss(y_true, y_pred):
    import tensorflow as tf
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    return tf.reduce_mean(loss)

# ---------------- GUI ----------------
class DoSSimulator:
    def __init__(self, master):
        self.master = master
        master.title("DoS Attack Simulator - Seq2Seq NIDS")
        master.geometry("800x700")

        # Paths
        tk.Label(master, text="Model (.h5):").grid(row=0, column=0, sticky="e")
        self.model_entry = tk.Entry(master, width=60)
        self.model_entry.grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.browse_model).grid(row=0, column=2)

        tk.Label(master, text="CSV Dataset:").grid(row=1, column=0, sticky="e")
        self.csv_entry = tk.Entry(master, width=60)
        self.csv_entry.grid(row=1, column=1)
        tk.Button(master, text="Browse", command=self.browse_csv).grid(row=1, column=2)

        tk.Label(master, text="Packets to simulate:").grid(row=2, column=0, sticky="e")
        self.packet_count_entry = tk.Entry(master, width=10)
        self.packet_count_entry.insert(0, "50")
        self.packet_count_entry.grid(row=2, column=1, sticky="w")

        tk.Button(master, text="Start Simulation", command=self.start_simulation).grid(row=2, column=1, sticky="e")
        tk.Button(master, text="Stop Simulation", command=self.stop_simulation).grid(row=2, column=2)

        # Canvas for animation
        self.canvas = tk.Canvas(master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
        self.canvas.grid(row=3, column=0, columnspan=3, pady=10)

        # Log panel
        tk.Label(master, text="Simulation Log:").grid(row=4, column=0, sticky="w", pady=(10,0))
        self.log_text = scrolledtext.ScrolledText(master, width=90, height=15)
        self.log_text.grid(row=5, column=0, columnspan=3, pady=5)

        # Internal variables
        self.running = False
        self.packet_y = CANVAS_HEIGHT
        self.sim_thread = None

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras H5 Model","*.h5")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv")])
        if path:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, path)

    def start_simulation(self):
        if self.running:
            messagebox.showwarning("Warning", "Simulation already running!")
            return
        self.running = True
        self.sim_thread = threading.Thread(target=self.run_simulation)
        self.sim_thread.start()

    def stop_simulation(self):
        self.running = False

    def run_simulation(self):
        try:
            model_path = self.model_entry.get()
            csv_path = self.csv_entry.get()
            if not model_path or not csv_path:
                messagebox.showerror("Error", "Please select both model and CSV file")
                self.running = False
                return

            # Load model and scaler
            model = load_model(model_path, custom_objects={"sparse_time_loss": sparse_time_loss})
            scaler = joblib.load("scaler.save")

            # Load dataset
            df_clean, label_col = load_and_clean(csv_path)
            feature_cols = [c for c in df_clean.columns if c != label_col]
            n_features = len(feature_cols)

            # Log info
            self.log_text.insert(tk.END, f"Simulation started with {len(df_clean)} records\n")
            self.log_text.see(tk.END)

            # Prepare simulation
            packet_count = int(self.packet_count_entry.get())
            history_window = []  # last SEQ_LEN packets
            attack_count = 0
            normal_count = 0

            # Split dataset into normal vs attack rows
            attack_rows = df_clean[df_clean[label_col] != 0]
            normal_rows = df_clean[df_clean[label_col] == 0]

            for i in range(packet_count):
                if not self.running:
                    break

                # Randomly pick attack or normal row to simulate DOS pattern
                if random.random() < 0.3 and len(attack_rows) > 0:   # 30% chance attack
                    row = attack_rows.sample(1)
                else:
                    row = normal_rows.sample(1)
            
                last_features = row[feature_cols].copy()
                row_arr = last_features.values.reshape(-1, n_features)
                row_scaled = scaler.transform(row_arr)

                # Update sliding window
                history_window.append(row_scaled[0])
                if len(history_window) > SEQ_LEN:
                    history_window.pop(0)

                # Predict only if we have SEQ_LEN packets
                label_str = "WAIT"
                prob_attack = 0.0
                if len(history_window) == SEQ_LEN:
                    X_last = np.array(history_window).reshape(1, SEQ_LEN, n_features)
                    pred_probs = model.predict(X_last, verbose=0)
                    pred_seq = np.argmax(pred_probs, axis=-1)[0]
                    final_label = seq_majority_label(pred_seq.reshape(1, -1))[0]
                    prob_attack = pred_probs[0][-1][1]  # last timestep probability
                    label_str = "ATTACK" if final_label == 1 else "NORMAL"
                    if final_label == 1:
                        attack_count += 1
                    else:
                        normal_count += 1

                # Draw packet on canvas
                color = "red" if label_str=="ATTACK" else "green"
                self.draw_packet(i, color, f"{label_str} ({prob_attack:.2f})")

                # Log
                self.log_text.insert(tk.END, f"Packet {i+1}: {label_str}, Prob Attack: {prob_attack:.2f}\n")
                self.log_text.see(tk.END)

                time.sleep(SIM_DELAY)

            self.log_text.insert(tk.END, f"\nSimulation ended. ATTACK: {attack_count}, NORMAL: {normal_count}\n")
            self.log_text.see(tk.END)
            self.running = False

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.running = False

    def draw_packet(self, idx, color, label_str):
        # Shift all existing packets down
        self.canvas.move("packet", PACKET_HEIGHT + PACKET_GAP, 0)
        # Draw new packet at top
        y0 = 0
        y1 = PACKET_HEIGHT
        rect = self.canvas.create_rectangle(10, y0, CANVAS_WIDTH-10, y1, fill=color, tags="packet")
        text = self.canvas.create_text(CANVAS_WIDTH//2, y0 + PACKET_HEIGHT//2, text=label_str, fill="white", tags="packet")
        self.canvas.update()

# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    simulator = DoSSimulator(root)
    root.mainloop()
