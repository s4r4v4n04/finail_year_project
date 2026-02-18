
# Commented out IPython magic to ensure Python compatibility.
import collections
import os
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import time

from matplotlib import pyplot as plt

import nest_asyncio
nest_asyncio.apply()

# %matplotlib inline

tff.federated_computation(lambda: 'Hello, World!')()

from google.colab import drive
drive.mount('/content/drive')
os.chdir("/content/drive/My Drive")
get_ipython().system('ls')
#This is for google colab
path = '/content/drive/My Drive' #/wustl-ehms-2020_with_attacks_categories'
tp = 'cic'      # Type: choose between 'cic' and 'custom'
tm = 180        # Timeout: choose one value from the following list [15, 30, 60, 90, 120, 180]
n_workers = 2   # #workers: choose between 2 and 5 workers (2 and 5 inclusive)

assert tp in ('cic', 'custom'), "Wrong dataset type"
assert tm in (15, 30, 60, 90, 120, 180), "Wrong time"
assert 2 <= n_workers <= 5, "At least 2 and at most 5 workers (docker containers) are required"

train_csv='wustl-ehms-2020_with_attacks_categories.csv'

df = pd.read_csv(train_csv)
pd.options.display.float_format = '{:,.0f}'.format
df['Dur']=df['Dur']*1000
df=df.fillna(0)
drop_columns = ['Dir', 'Flgs','Attack Category']
df_dropped = df.drop(columns=drop_columns)
df=df_dropped
for f in df.columns.values:
  df[f] = df[f].astype(int)
df = df[list(df.columns.values)].astype(int)
df_dropped.info()

df_train = df
df_test = df #pd.read_csv(train_csv)
print(df_train.head())

df_train.dtypes

df_train.info()

unique_labels = list(df_train.Label.astype('category').unique())
unique_codes = list(df_train.Label.astype('category').cat.codes.unique())
mapping = {unique_codes[i] : unique_labels[i] for i in range(len(unique_labels))}

mapping

df_train['Label']

df_test['Label']

train = df_train
 test = df_test

n_samples = int(df_train.shape[0] / n_workers)

assert n_samples > 0, "Each worker must be assigned at least one data point"



import numpy as np

import tensorflow as tf
#Considering y variable holds numpy array
df_train = tf.convert_to_tensor(df_train, dtype=tf.int64)
df_test = tf.convert_to_tensor(df_test, dtype=tf.int64)


n_epochs = 2
shuffle_buffer_size = df_train.shape[0]
batch_size = 250
prefetch_buffer_size = 50
input_shape = train.shape[1] - 1
output_shape = len(unique_codes)

"""def preprocess(dataframe):

    #Flatten a batch `pixels` and return the features as an `OrderedDict`.
    def map_fn(dataset):
        return collections.OrderedDict(
            x=tf.cast(dataset[:,:-1], tf.float64),
            y=tf.cast(tf.reshape(dataset[:,-1], shape=(-1, 1)), tf.int64)
        )

    return tf.data.Dataset.from_tensor_slices(dataframe).repeat(n_epochs).shuffle(
        shuffle_buffer_size).batch(batch_size).map(map_fn).prefetch(prefetch_buffer_size)

"""
#tf.data.Dataset.from_tensor_slices(train)
client_data=[]
client_data.append(train.iloc[1:10001,:]) #[preprocess(train.sample(n=n_samples)) for _ in range(n_workers)]
client_data.append(train.iloc[10001:,:] )#[preprocess(train.sample(n=n_samples)) for _ in range(n_workers)]

client_data

for i in range(n_workers):
    print(f"Worker {i+1} data contains {len(client_data[i])} training points")

# Number of examples per layer for a sample of clients
fig = plt.figure(figsize=(20, 7))
fig.suptitle('Label Counts for a Sample of Worker Data')
fig.tight_layout()

"""
for i in range(n_workers):
    m = 0
    plot_data = collections.defaultdict(list)
    for label in list(client_data[i])[0]['y'].numpy()[:,0]:
        # Append counts individually per label to make plots
        # more colorful instead of one color per plot.
        plot_data[label].append(label)
        m = max(m, len(plot_data[label]))

    n_cols = n_workers if n_workers < 5 else 5
    xlim = [0, m+(5-m%5)]
    ylim = [min(unique_codes)-1, max(unique_codes)+1]
    yticks = list(range(min(unique_codes), max(unique_codes)+1))
    yticks_labels = [mapping[k] for k in range(0, max(unique_codes)+1)]

    plt.subplot(int(n_workers / 5)+1, n_cols, i+1)
    plt.subplots_adjust(wspace=0.3)
    plt.title('Worker {}'.format(i+1))
    plt.xlabel('#points')
    plt.xlim(xlim)
    plt.ylabel('Label')
    plt.ylim(ylim)
    plt.yticks(yticks, labels=yticks_labels)

    # plot values on top of bars
    for key in plot_data:
        if len(plot_data[key]) > 0:
            plt.text(len(plot_data[key])+0.6, int(key)-0.1, str(len(plot_data[key])), ha='center')

    for j in range(min(unique_codes),max(unique_codes)+1):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[k-0.5 for k in range(min(unique_codes),max(unique_codes)+2)],
            orientation='horizontal'
        )
"""
def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(output_shape, kernel_initializer='zeros', activation='relu'),
        tf.keras.layers.Softmax(),
    ])
    return tff.learning.from_keras_model(model,
        # Note: input spec is the _batched_ shape, and includes the
        # label tensor which will be passed to the loss function. This model is
        # therefore configured to accept data _after_ it has been preprocessed.
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, input_shape], dtype=tf.float64),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)
