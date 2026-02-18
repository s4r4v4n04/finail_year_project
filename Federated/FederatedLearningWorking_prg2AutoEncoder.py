

"""# 5- Data Pre-processing"""
#'SrcMac',	'DstMac',
drop_columns = ['SrcAddr',	'DstAddr', 'Dport',	'SrcBytes',	'DstBytes',	'SrcGap',	'DstGap',	'SIntPktAct',	'DIntPktAct',
                'sMaxPktSz',	'dMaxPktSz',	'sMinPktSz',	'dMinPktSz', 'Trans',	'TotPkts',	'TotBytes',	'Loss',	'pLoss',
                'pSrcLoss',	'pDstLoss',	'Rate',		'SpO2',	'SYS',	'DIA',	'Heart_rate',	'Resp_Rate',	'ST']
df_ExtraTrees= df_dropped.drop(columns=drop_columns)
df_ExtraTrees.info()

#df_ExtraTrees.loc[df_ExtraTrees['Sport'] == 'dircproxy']
#df_ExtraTrees.loc[df_ExtraTrees['Sport'] == 'fido']
#df_ExtraTrees.loc[df_ExtraTrees['Sport'] == 'tfido']
#df1.loc[df1['Sport'] == 'fido']
#df1 = df_new.drop([7923])

df_ExtraTrees = df_ExtraTrees.drop([10633])

df_ExtraTrees['Sport'] = df_ExtraTrees['Sport'].astype(float)
df_ExtraTrees['Pulse_Rate'] = df_ExtraTrees['Pulse_Rate'].astype(float)
df_ExtraTrees['Packet_num'] = df_ExtraTrees['Packet_num'].astype(float)
df_ExtraTrees.info()

X = df_ExtraTrees.drop('Label', axis = 1)
y = df_ExtraTrees['Label']
X.shape, y.shape

print(X.head())

from imblearn.over_sampling import SMOTE
X_smote, y_smote = SMOTE().fit_resample(X, y)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y_smote)
Y_smote= pd.get_dummies(y1).values

Y_smote

Y_smote.shape

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, Y_smote, test_size=0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

"""**Univariate selection:**"""
#'SrcMac',	'DstMac',
drop_columns = ['SrcAddr',	'DstAddr', 'Sport',	'Dport', 	'Packet_num']
df_uni= df_dropped.drop(columns=drop_columns)
print(df_uni.info())

df_uni['sMaxPktSz'] = df_uni['sMaxPktSz'].astype(float)
df_uni['dMaxPktSz'] = df_uni['dMaxPktSz'].astype(float)
df_uni['sMinPktSz'] = df_uni['sMinPktSz'].astype(float)
df_uni['dMinPktSz'] = df_uni['dMinPktSz'].astype(float)
df_uni['Trans'] = df_uni['Trans'].astype(float)
df_uni['Loss'] = df_uni['Loss'].astype(float)
df_uni['SpO2'] = df_uni['SpO2'].astype(float)
df_uni['Pulse_Rate'] = df_uni['Pulse_Rate'].astype(float)
df_uni['SYS'] = df_uni['SYS'].astype(float)
df_uni['DIA'] = df_uni['DIA'].astype(float)
df_uni['Heart_rate'] = df_uni['Heart_rate'].astype(float)
df_uni['Resp_Rate'] = df_uni['Resp_Rate'].astype(float)
df_uni.info()

X_uni = df_uni.drop('Label', axis = 1)
y_uni = df_uni['Label']
X_uni.shape, y_uni.shape
print(X_uni.head())

from imblearn.over_sampling import SMOTE
X_uni_smote, y_uni_smote = SMOTE().fit_resample(X_uni, y_uni)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1_uni = encoder.fit_transform(y_uni_smote)
Y_uni_smote= pd.get_dummies(y1_uni).values

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
X_uni_train, X_uni_test, y_uni_train, y_uni_test = train_test_split(X_uni_smote, Y_uni_smote, test_size=0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaling = StandardScaler()
X_uni_train = scaling.fit_transform(X_uni_train)
X_uni_test = scaling.transform(X_uni_test)

#!pip install jax==0.0
"""# 6- Federated  Model Approach""" #5 Clients

from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Dropout, Activation, Embedding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import SGD
#opt = SGD(lr=0.0001)
#batch_size = 64

# 1. define the network
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
lstm_output_size = 34 #70
import time

# Get the start time
start_time = time.time()
tensorflow_federated_learning_average_accuracyscore=0
clientscount=1#10
epochcount=500#
#Fedarated Client Count is 10
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Convolution1D, MaxPooling1D, Flatten, UpSampling1D

X_train, X_test, y_train, y_test = train_test_split(X_smote, Y_smote, test_size=0.2, random_state = 42)
scaling = StandardScaler()
X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the input layer
input_layer = layers.Input(shape=(12, 1))

# Encoder
x = layers.Conv1D(64, 3, activation="relu", padding="same")(input_layer)
x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
x = layers.LSTM(64, activation="relu", return_sequences=False)(x)
encoded = layers.Dense(32, activation="relu")(x)  # Latent space representation

# Decoder
x = layers.Dense(64, activation="relu")(encoded)
x = layers.Reshape((1, 64))(x)  # Reshape to (1, 64) for LSTM compatibility
x = layers.LSTM(64, activation="relu", return_sequences=True)(x)
x = layers.UpSampling1D(size=12)(x)
decoded = layers.Conv1D(1, 3, activation="sigmoid", padding="same")(x)  # Output shape matches the input shape

# AutoEncoder model
autoencoder = models.Model(inputs=input_layer, outputs=decoded)

# Compile the AutoEncoder
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["accuracy"])

# Summary of the AutoEncoder
autoencoder.summary()

# Training the AutoEncoder
history = autoencoder.fit(
    X_train, X_train,  # Input and output are the same for unsupervised training
    epochs=200,
    batch_size=512,
    validation_data=(X_test, X_test),
    verbose=1
)

# Evaluate the AutoEncoder
loss, acc = autoencoder.evaluate(X_test, X_test)
print(f"Test Loss: {loss}, Test Accuracy: {acc}")


from sklearn.metrics import mean_squared_error

# Calculate MSE during training (from history object)
train_mse = history.history['loss']
val_mse = history.history['val_loss']

print("Training MSE over epochs:", train_mse)
print("Validation MSE over epochs:", val_mse)

# Reconstruct the input (predict on X_test)
reconstructed_X_test = autoencoder.predict(X_test)

# Calculate Mean Squared Error for the test set
test_mse = mean_squared_error(X_test.flatten(), reconstructed_X_test.flatten())
print(f"Test Mean Squared Error (MSE): {test_mse}")


exit()

# Encode the input to get latent space representation
encoder = Model(inputs=input_layer, outputs=encoded)
encoded_data = encoder.predict(X_test)

# Decode the latent space representation back to original dimensions
decoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-3](decoder_input)  # Reverse Dense
decoder_layer = tf.expand_dims(decoder_layer, axis=1)
decoder_layer = autoencoder.layers[-2](decoder_layer)  # Reverse LSTM
decoder_output = autoencoder.layers[-1](decoder_layer)  # Reverse Convolution
decoder = Model(inputs=decoder_input, outputs=decoder_output)

decoded_data = decoder.predict(encoded_data)

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Print a portion of the decoded data (reconstructed input)
print("Decoded Data (Reconstructed Input):")
print(decoded_data[:5])  # Print the first 5 reconstructed samples

# Convert decoded data and original test data into 1D arrays for comparison
original_flattened = np.round(X_test.flatten())
decoded_flattened = np.round(decoded_data.flatten())

# Compute the confusion matrix
conf_matrix = confusion_matrix(original_flattened, decoded_flattened)

# Print the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Optionally, print classification report for precision, recall, and F1 score
print("\nClassification Report:")
print(classification_report(original_flattened, decoded_flattened))
exit()
