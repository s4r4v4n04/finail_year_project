

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
# %matplotlib inline
import os
from google.colab import drive
drive.mount('/content/drive')

#df = pd.read_csv('/content/drive/MyDrive/GlobeCom/drive1-wus-2020.csv')

df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')
shape = df.shape
print('Dataframe shape: ', shape)
print('Number of rows: ', shape[0])
print('Number of columns: ', shape[1])
#df.shape

df.head()
df['Dur']=df['Dur']*1000
drop_columns = ['Dir', 'Flgs','Attack Category']
df_dropped = df.drop(columns=drop_columns)
df_dropped.info()

"""# 2. Null and duplicated values <a name="2"></a>

**Checking Missing Value**

Following code shows dataset does not have any missing values.
"""

df_dropped.isnull().sum()

"""**Checking duplicated values**

Now we need to check if there is any duplicated value, because it does not make any sense to have duplicate value in our analysis.
I will check shape, so the dataset does not have a duplicated value.
"""

df_dropped[df_dropped.duplicated()].shape

df_dropped.describe()

"""# 2. Exploratory Data Analysis <a name="3"></a>

**1- Number of attacks in the dataset:**
"""

df_dropped['Label'].value_counts()

"""**2- number of Source address and destination address in the dataset:**"""

df_dropped['SrcAddr'].value_counts()

df_dropped['DstAddr'].value_counts()

"""**3- How many source address suffered most attacks?**"""

IoMT_attack = df_dropped[df_dropped['Label'] == 1]

IoMT_attack['SrcAddr'].value_counts()

"""**4- How many destination address suffered most attacks?**"""

IoMT_attack['DstAddr'].value_counts()

"""**5 - Which Source port caused the most attacks?**"""

IoMT_attack['Sport'].value_counts().nlargest(10)

plt.figure(figsize=(15,10))
ax = sns.countplot(x='Sport', data=IoMT_attack, palette='CMRmap', order=IoMT_attack.Sport.value_counts().iloc[:20].index)
ax.set(xlabel='Source port', ylabel='Total Attack Packets')

"""**5 - Which destination port suffered the most attacks?**"""

IoMT_attack['Dport'].value_counts().nlargest(10)

"""**Analyzing biometric Data**

**1- Is there any relationship between high heart rate and attack?**
"""

plt.figure(figsize=(25,10))
sns.countplot(data= IoMT_attack, x='Heart_rate',hue='Label')
plt.title('Destination Port v/s Attack\n')

"""**2- What is the level of Blood oxygen during the attack?**"""
"""
plt.figure(figsize=(15,10))
sns.countplot(data= IoMT_attack, x='SpO2',hue='Label')
plt.title('Blood oxygen Level v/s Attack\n')
"""
"""**3- What is the patient's temperature  during the attack?**"""
"""
plt.figure(figsize=(25,10))
ax = sns.countplot(x='Temp', data=IoMT_attack, palette='CMRmap', order=IoMT_attack.Temp.value_counts().iloc[:10].index)
ax.set(xlabel='Temp', ylabel='Total Attack Packets')
ax.set(xlabel='Temperature (Celsius).', ylabel='Length (mean)')
"""
"""# Other information"""

plt.figure(figsize=(25,10))
ax = sns.countplot(x='Dur', data=IoMT_attack, palette='CMRmap', order=IoMT_attack.Dur.value_counts().iloc[:10].index)
ax.set(xlabel='Transmitted packet size duration', ylabel='Total Attack Packets')

df_new=df_dropped.fillna(0)

"""# 4. Feature Selection <a name="4"></a>

**1- Univariate Selection:**
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = df_new.copy()
X = data.iloc[2:,3:39]#40]  #independent columns
y = data.iloc[2:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(44,'Score'))  #print 10 best features

"""**2. Feature Importance:** You can gain the significance of each feature of your dataset by using the Model Characteristics property. Feature value gives you a score for every function of your results, the higher the score the more significant or appropriate the performance variable is. Feature importance is the built-in class that comes with Tree Based Classifiers, we will use the Extra Tree Classifier to extract the top 10 features for the dataset."""

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.show()



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

X.head()

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
df_uni.info()

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

X_uni.head()

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

"""# 6- Single Model Approach"""

from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Dropout, Activation, Embedding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import SGD
#opt = SGD(lr=0.0001)
#batch_size = 64

# 1. define the network

model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(12,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
])
#model = Sequential()
#model.add(LSTM(64,input_dim=X_train.shape[0]))
#model.add(Dropout(0.1))
#model.add(Dense(1))


#model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100,batch_size=512, validation_data=(X_test, y_test), verbose=1)#, class_weight=class_weights)

print(model.evaluate(X_test, y_test))

y_preds_dnn = model.predict(X_test)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_preds_dnn, axis=1)


print(confusion_matrix(y_test_class, y_pred_class))

print("Classification Report: \n", classification_report(y_test_class, y_pred_class))

print('Summary of the results after each epoch')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='lower right')
plt.show()


plt.plot(history.history['loss'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='upper right')
plt.show()

ax = plt.subplot()
predict_results = model.predict(X_test)
predict_results = predict_results.argmax(axis = 1)
cm = confusion_matrix(y_test_class, predict_results)

sns.heatmap(cm, annot = True, ax =ax, fmt='g'); # cmap= "YlGnBu"
ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels');
ax.set_title('Confusion Matrix');

from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Dropout, Activation, Embedding
import tensorflow_federated as tf
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import SGD
#opt = SGD(lr=0.0001)
#batch_size = 64

# 1. define the network

model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(12,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
])
#model = Sequential()
#model.add(LSTM(64,input_dim=X_train.shape[0]))
#model.add(Dropout(0.1))
#model.add(Dense(1))


#model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100,batch_size=512, validation_data=(X_test, y_test), verbose=1)#, class_weight=class_weights)

print(model.evaluate(X_test, y_test))

y_preds_dnn = model.predict(X_test)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_preds_dnn, axis=1)


print(confusion_matrix(y_test_class, y_pred_class))

print("Classification Report: \n", classification_report(y_test_class, y_pred_class))

print('Summary of the results after each epoch')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
exit()

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='lower right')
plt.show()


plt.plot(history.history['loss'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='upper right')
plt.show()

ax = plt.subplot()
predict_results = model.predict(X_test)
predict_results = predict_results.argmax(axis = 1)
cm = confusion_matrix(y_test_class, predict_results)

sns.heatmap(cm, annot = True, ax =ax, fmt='g'); # cmap= "YlGnBu"
ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels');
ax.set_title('Confusion Matrix');

"""**Univarint**"""

from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Dense, Dropout, Activation, Embedding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import SGD
#opt = SGD(lr=0.0001)
#batch_size = 64

# 1. define the network

model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(34,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
])
#model = Sequential()
#model.add(LSTM(64,input_dim=X_train.shape[0]))
#model.add(Dropout(0.1))
#model.add(Dense(1))


#model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

history = model.fit(X_uni_train, y_uni_train, epochs=10,batch_size=512, validation_data=(X_uni_test, y_uni_test), verbose=1)#, class_weight=class_weights)

print(model.evaluate(X_uni_test, y_uni_test))

y_preds_dnn = model.predict(X_uni_test)
y_test_class = np.argmax(y_uni_test, axis=1)
y_pred_class = np.argmax(y_preds_dnn, axis=1)


print(confusion_matrix(y_test_class, y_pred_class))

print("Classification Report: \n", classification_report(y_test_class, y_pred_class))

print('Summary of the results after each epoch')
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='lower right')
plt.show()


plt.plot(history.history['loss'])
from matplotlib import pyplot as plt
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Cross-Validation'], loc='upper right')
plt.show()

ax = plt.subplot()
predict_results = model.predict(X_uni_test)
predict_results = predict_results.argmax(axis = 1)
cm = confusion_matrix(y_test_class, predict_results)

sns.heatmap(cm, annot = True, ax =ax, fmt='g'); # cmap= "YlGnBu"
ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels');
ax.set_title('Confusion Matrix');

"""# 7- Federated Learning Approach"""

#!pip list -v | grep tensorflow

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# pip install sklearn
# pip install pandas
# pip install matplotlib
# pip install tensorflow
# 
# pip uninstall --yes tensorboard tb-nightly
# 
# pip install --quiet --upgrade tensorflow-federated
# pip install --quiet --upgrade nest-asyncio
# pip install --quiet --upgrade tensorboard

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#!pip install --upgrade tensorflow-federated

import tensorflow_federated as tff
print(tff.federated_computation(lambda: 'Hello World'))



# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow.keras.layers import BatchNormalization
#from keras.layers.normalization import LayerNormalization
#import tensorflow_federated as tff
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

SEED = 1337
tf.random.set_seed(SEED)

for col in  df_dropped.columns[:]:
    df_dropped[col] = pd.to_numeric(df_dropped[col], errors='coerce')#, downcast='int')

df_dropped.info()

#df_dropped['Pulse_Rate'] = df_dropped['Pulse_Rate'].astype(float)
#df_dropped['Packet_num'] = df_dropped['Packet_num'].astype(float)
#df_dropped.info()
#'SrcMac',	'DstMac',
drop_columns = ['Sport', 'SrcAddr',	'DstAddr', 'Dport',	'SrcBytes',	'DstBytes',	'SrcGap',	'DstGap',	'SIntPktAct',	'DIntPktAct',
                'sMaxPktSz',	'dMaxPktSz',	'sMinPktSz',	'dMinPktSz', 'Trans',	'TotPkts',	'TotBytes',	'Loss',	'pLoss',
                'pSrcLoss',	'pDstLoss',	'Rate',		'SpO2',	'SYS',	'DIA',	'Heart_rate',	'Resp_Rate',	'ST']
df_new= df_dropped.drop(columns=drop_columns)
df_new.info()

df_new['Pulse_Rate'] = df_new['Pulse_Rate'].astype(float)
df_new['Packet_num'] = df_new['Packet_num'].astype(float)
df_new.info()

df_new.head()





df_new.shape

#df_new.loc[df_new['Sport'] == 'dircproxy']
df1.loc[df1['Sport'] == 'fido']
df1 = df_new.drop([7923])

df1 = df_new.drop([7923])

df1.loc[df1['Sport'] == 'fido']

df_new['Pulse_Rate'].value_counts()

#Creating Client1 and Client2 spilits:
client_df1 =  df1[:len(df1.index)//2]
client_df2 =  df1[len(df1.index)//2:]

#Creating Client1 and Client2 spilits:
#client_df1 =  df_dropped[:len(df_dropped.index)//2]
#client_df2 =  df_dropped[len(df_dropped.index)//2:]



client_df1.head()

client_df1['Label'].value_counts()

client_df2['Label'].value_counts()

EPOCHS = 100
BATCH_SIZE = 64

"""Dir	Flgs	['SrcAddr',	'DstAddr', 'Dport',	'SrcBytes',	'DstBytes',	'SrcGap',	'DstGap',	'SIntPkt',	'DIntPkt', 'sMaxPktSz',	'dMaxPktSz',	'sMinPktSz',	'dMinPktSz', 'Trans',	'TotPkts',	'TotBytes',	'Loss',	'pLoss',	'pSrcLoss',	'pDstLoss',	'Rate',	'SrcMac',	'DstMac',	'SpO2',	'SYS',	'DIA',	'Heart_rate',	'Resp_Rate',	'ST']"""

def make_tf_dataset(dataframe, negative_ratio=None, batch_size=None):
    dataset = dataframe.drop(['Dur']	, axis=1)

    # Class balancing
    pos_df = dataset[dataset['Label'] == 1]
    neg_df = dataset[dataset['Label'] == 0]
    if negative_ratio:
        neg_df = neg_df.iloc[random.sample(range(0, len(neg_df)), len(pos_df)*negative_ratio), :]
    balanced_df = pd.concat([pos_df, neg_df], ignore_index=True, sort=False)

    y = balanced_df.pop('Label')

    # Dataset creation
    dataset = tf.data.Dataset.from_tensor_slices((balanced_df.values, y.to_frame().values))
    dataset = dataset.shuffle(2048, seed=SEED)
    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset

train_data, val_data = [], []
for client_data in [client_df1, client_df2]:
    train_df, val_df = train_test_split(client_data, test_size=0.1, random_state=SEED)

    # Scaling (Standardization actually hurts performance)
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_df.drop(['Label'], axis=1))
    val_features = scaler.transform(val_df.drop(['Label'], axis=1))

    train_df[train_df.columns.difference(['Label'])] = train_features
    val_df[val_df.columns.difference(['Label'])] = val_features

    # TF Datasets
    train_data.append(make_tf_dataset(train_df, negative_ratio=2, batch_size=BATCH_SIZE))
    val_data.append(make_tf_dataset(val_df, batch_size=1))

def input_spec():
    return (
        tf.TensorSpec([None,10], tf.float64),
        tf.TensorSpec([None, 1], tf.int64)
    )

def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), Precision(), Recall()])

trainer = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam()
)

state = trainer.initialize()
train_hist = []
for i in range(EPOCHS):
    state, metrics = trainer.next(state, train_data)
    train_hist.append(metrics)

    print(f"\rRun {i+1}/{EPOCHS}", end="")

evaluator = tff.learning.build_federated_evaluation(model_fn)

federated_metrics = evaluator(state.model, val_data)
federated_metrics

"""# Single Model with all Data at once (for comparison)"""

train_data = train_data[0].concatenate(train_data[1])
val_data = val_data[0].concatenate(val_data[1])

def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[BinaryAccuracy(), Precision(), Recall()],
    )

    return model

model = model_fn()
history = model.fit(train_data, epochs=EPOCHS)

test_scores = model.evaluate(val_data)
single_metrics = {
    'loss': test_scores[0],
    'binary_accuracy': test_scores[1],
    'precision': test_scores[2],
    'recall': test_scores[3]
}
single_metrics

print(f"---Single model metrics---\n{single_metrics}\n")
print(f"---Federated model metrics---\n{dict(federated_metrics)}")

"""# 8- Conclusion"""
