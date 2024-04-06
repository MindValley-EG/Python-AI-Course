# delete this cell
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
%matplotlib inline

max_features = 10_000  # number of words to consider as features

# each review is encoded as a sequence of word indexes
# indexed by overall frequency in the dataset
# output is 0 (negative) or 1 (positive)
imdb = tf.keras.datasets.imdb.load_data(num_words=max_features)
(raw_input_train, y_train), (raw_input_test, y_test) = imdb

print(type(raw_input_train))
print(type(y_train))
print(type(raw_input_test))
print(type(y_test))

print(len(raw_input_train))
print(len(raw_input_test))

index = 3
# Convert to numpy for better printing
np.array(raw_input_train[index])

len(raw_input_train[index]), y_train[index]

maxlen = 500  # cut texts after this number of words (among top max_features most common words)
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
input_train = tf.keras.preprocessing.sequence.pad_sequences(raw_input_train, maxlen=maxlen)
input_test = tf.keras.preprocessing.sequence.pad_sequences(raw_input_test, maxlen=maxlen)

print(type(input_train))
print(type(input_test))

input_train[index]

print(input_train[index].shape)
print(sum(input_train[index]!=0))

# If the review have more than 500, it truncates from the beggining,
# check the documentation for pad_sequence for more details
raw_input_train[index][50]

from tensorflow.keras.layers import GRU, Embedding, Dense

embedding_dim = 32

model = keras.Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_shape=(maxlen,))) # encoder
model.add(GRU(units=32))
model.add(Dense(units=32, activation='relu'))                      # latent space
model.add(Dense(units=1, activation='sigmoid'))            # binary classifier as decoder

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

batch_size = 1000
epochs = 10

history = model.fit(input_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluation on the trianing split
train_loss, train_accuracy = model.evaluate(input_train, y_train, batch_size=batch_size)
train_accuracy

# Evaluation on the testing split
test_loss, test_accuracy = model.evaluate(input_test, y_test, batch_size=batch_size)
test_accuracy

# precition
model.predict(input_test[0:5])

# ground truth
y_test[0:5]


def plot_history(history, samples=10, init_phase_samples=None):
    epochs = history.params['epochs']

    acc = history.history['accuracy']

    every_sample = int(epochs / samples)
    acc = pd.DataFrame(acc).iloc[::every_sample, :]

    fig, ax = plt.subplots(figsize=(20, 5))

    ax.plot(acc, 'b', label='Training acc')
    ax.set_title('Training and validation accuracy')
    ax.legend()


plot_history(history)