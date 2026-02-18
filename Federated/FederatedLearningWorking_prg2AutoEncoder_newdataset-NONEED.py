# ================================
# 1. Imports
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models

# ================================
# 2. Load YOUR Dataset
# ================================
df = pd.read_csv("../newdataset/train_test_network.csv")
print("Dataset shape:", df.shape)
#df= df['src_ip', 'src_port', 'dst_ip', 'dst_port', 'service', 'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_query', 'dns_qclass', 'dns_qtype', 'http_response_body_len', 'http_status_code', 'Label','type']
import pandas as pd

# Load dataset
#df = pd.read_csv("train_test_network.csv")

# Number of samples per label
SAMPLES_PER_LABEL = 10000

# Filter dataframe: 1000 rows per label
df_balanced = (
    df.groupby("Label", group_keys=False)
      .apply(lambda x: x.sample(
          n=min(len(x), SAMPLES_PER_LABEL),
          random_state=42
      ))
      .reset_index(drop=True)
)
df=df_balanced
# Check result
# ================================
# 3. Separate Label
# ================================
label_col = "Label"
df[label_col] = df[label_col].astype(int)

# ================================
# 4. Factorize ALL categorical columns
# ================================
for col in df.columns:
    if df[col].dtype == "object":
        df[col], _ = pd.factorize(df[col])

# ================================
# 5. Ensure Non-negative for chi2
# ================================
df[df < 0] = 0

# ================================
# 6. Feature / Target split
# ================================
X = df.drop(label_col, axis=1)
y = df[label_col]

# ================================
# 7. Feature Selection (Chi-Square)
# ================================
selector = SelectKBest(score_func=chi2, k=20)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

print("Selected Features:", selected_features.tolist())

# ================================
# 8. Feature Importance
# ================================
et_model = ExtraTreesClassifier()
et_model.fit(X, y)

importances = pd.Series(et_model.feature_importances_, index=X.columns)
importances.nlargest(30).plot(kind="barh")
plt.show()

# ================================
# 9. Balance Dataset (SMOTE)
# ================================
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X, y)

y_smote = LabelEncoder().fit_transform(y_smote)
Y_smote = pd.get_dummies(y_smote).values

# ================================
# 10. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_smote, Y_smote, test_size=0.2, random_state=42
)

# ================================
# 11. Scaling
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ================================
# 12. AutoEncoder Model
# ================================
input_layer = layers.Input(shape=(X_train.shape[1], 1))

x = layers.Conv1D(64, 3, activation="relu", padding="same")(input_layer)
x = layers.MaxPooling1D(2, padding="same")(x)
x = layers.LSTM(64, activation="relu", return_sequences=False)(x)
encoded = layers.Dense(32, activation="relu")(x)

x = layers.Dense(64, activation="relu")(encoded)
x = layers.Reshape((1, 64))(x)
x = layers.LSTM(64, activation="relu", return_sequences=True)(x)
x = layers.UpSampling1D(size=X_train.shape[1])(x)
decoded = layers.Conv1D(1, 3, activation="sigmoid", padding="same")(x)

autoencoder = models.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# ================================
# 13. Train AutoEncoder
# ================================
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_test, X_test),
    verbose=1
)

# ================================
# 14. Evaluate
# ================================
reconstructed = autoencoder.predict(X_test)
mse = mean_squared_error(X_test.flatten(), reconstructed.flatten())
print("Reconstruction MSE:", mse)

# ================================
# 15. Save Model
# ================================
autoencoder.save("network_autoencoder.h5")
print("Model saved as network_autoencoder.h5")
