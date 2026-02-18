import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import time
import random
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SEQ_LEN = 10
PACKET_HEIGHT = 20
PACKET_GAP = 5
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400
SIM_DELAY = 0.5

# ---------------- Utilities ----------------
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    label_col = "Label"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
        
    
    # -----------------------------
    # MAP LABELS: BENIGN -> 0, DDos -> 1
    # -----------------------------
    df[label_col] = df[label_col].map({
        "BENIGN": 0,
        "DDos": 1
    })
    
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

        # Canvas animation
        self.canvas = tk.Canvas(master, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
        self.canvas.grid(row=3, column=0, columnspan=3, pady=10)

        tk.Label(master, text="Simulation Log:").grid(row=4, column=0, sticky="w", pady=(10,0))
        self.log_text = scrolledtext.ScrolledText(master, width=90, height=15)
        self.log_text.grid(row=5, column=0, columnspan=3, pady=5)

        # Internal vars
        self.running = False
        self.sim_thread = None
        self.prob_history = []   # ---------------- ADD THIS ----------------

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

        self.prob_history = []      # reset previous history
        self.running = True
        self.sim_thread = threading.Thread(target=self.run_simulation)
        self.sim_thread.start()

    def stop_simulation(self):
        self.running = False
        self.show_last25_chart()   # ---------------- SHOW CHART ON STOP ----------------

    def show_last25_chart(self):
        if len(self.prob_history) == 0:
            messagebox.showinfo("Info", "No probability data recorded yet.")
            return

        last25 = self.prob_history[-25:]

        plt.figure(figsize=(8,5))
        plt.bar(range(len(last25)), last25)
        plt.title("Last 25 Attack Probabilities")
        plt.xlabel("Packet Index")
        plt.ylabel("Attack Probability")
        plt.ylim(0, 1)
        plt.show()

    def run_simulation(self):
        try:
            model_path = self.model_entry.get()
            csv_path = self.csv_entry.get()
            if not model_path or not csv_path:
                messagebox.showerror("Error", "Please select both model and CSV file")
                self.running = False
                return

            model = load_model(model_path, custom_objects={"sparse_time_loss": sparse_time_loss})
            scaler = joblib.load("scaler.save")

            df_clean, label_col = load_and_clean(csv_path)
            feature_cols = [c for c in df_clean.columns if c != label_col]
            n_features = len(feature_cols)

            self.log_text.insert(tk.END, f"Simulation started with {len(df_clean)} records\n")
            self.log_text.see(tk.END)

            packet_count = int(self.packet_count_entry.get())
            history_window = []
            attack_rows = df_clean[df_clean[label_col] != 0]
            normal_rows = df_clean[df_clean[label_col] == 0]

            for i in range(packet_count):
                if not self.running:
                    break

                # Random: 30% attack
                if random.random() < 0.3 and len(attack_rows) > 0:
                    row = attack_rows.sample(1)
                else:
                    row = normal_rows.sample(1)

                features = row[feature_cols].values.reshape(1, -1)
                scaled = scaler.transform(features)

                history_window.append(scaled[0])
                if len(history_window) > SEQ_LEN:
                    history_window.pop(0)

                label_str = "WAIT"
                prob_attack = 0.0

                if len(history_window) == SEQ_LEN:
                    X_last = np.array(history_window).reshape(1, SEQ_LEN, n_features)
                    pred_probs = model.predict(X_last, verbose=0)

                    pred_seq = np.argmax(pred_probs, axis=-1)[0]
                    final_label = seq_majority_label(pred_seq.reshape(1, -1))[0]
                    prob_attack = pred_probs[0][-1][1]  # probability for attack class

                    self.prob_history.append(prob_attack)

                    label_str = "ATTACK" if final_label == 1 else "NORMAL"

                color = "red" if label_str == "ATTACK" else "green"
                self.draw_packet(i, color, f"{label_str} ({prob_attack:.2f})")

                self.log_text.insert(tk.END, f"Packet {i+1}: {label_str}, Prob: {prob_attack:.2f}\n")
                self.log_text.see(tk.END)

                time.sleep(SIM_DELAY)

            self.running = False

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.running = False

    def draw_packet(self, idx, color, label_str):
        self.canvas.move("packet", PACKET_HEIGHT + PACKET_GAP, 0)
        rect = self.canvas.create_rectangle(10, 0, CANVAS_WIDTH-10, PACKET_HEIGHT,
                                            fill=color, tags="packet")
        text = self.canvas.create_text(CANVAS_WIDTH//2, PACKET_HEIGHT//2,
                                       text=label_str, fill="white",
                                       tags="packet")
        self.canvas.update()

# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    simulator = DoSSimulator(root)
    root.mainloop()
