import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

def plot_curve(epochs, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()

print("Loaded the plot_curve function.")

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Output example #2917 of the training set.
x_train[2917]

x_train[2917].shape

# Use false colors to visualize the array.
plt.imshow(x_train[2917], cmap="gray")

# Output row #10 of example #2917.
x_train[2917][10]

# Output pixel #16 of row #10 of example #2900.
x_train[2900][10][16]

x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
print(x_train_normalized[2900][10]) # Output a normalized row

my_learning_rate = 0.01
epochs = 10
batch_size = 4000

model = tf.keras.models.Sequential()
model.add(layers.Input(shape=(28, 28)))
model.add(layers.Flatten())
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=my_learning_rate),
              loss="crossentropy",
              metrics=['accuracy'])

model.summary()

history = model.fit(x=x_train_normalized,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True)


epochs = history.epoch
hist = pd.DataFrame(history.history)

plot_curve(epochs, hist, ["accuracy"])

model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

