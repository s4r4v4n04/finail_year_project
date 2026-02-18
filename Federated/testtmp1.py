
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#!pip install --upgrade tensorflow-federated

import tensorflow_federated as tff
print(tff.federated_computation(lambda: 'Hello World'))

