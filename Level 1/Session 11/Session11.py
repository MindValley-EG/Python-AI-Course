import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=100)
images = faces.images
classes = faces.target_names
labels = faces.target

print(classes)
print(len(classes))
print(images.shape)

%matplotlib inline
import matplotlib.pyplot as plt

i = 0
plt.figure(figsize=(2, 5))
plt.imshow(images[i], cmap='gist_gray')
plt.title(classes[labels[i]])
plt.axis('off')
plt.show();

df = pd.DataFrame(labels)
df = df.apply(lambda x: classes[x])
df.value_counts().plot(kind='bar', rot=20, xlabel="Classes", ylabel="Frequency");


# A numpy array wit the length of labels
mask = np.zeros(len(labels), dtype=bool)

for target in range(5):
    # Boolean expression of which image belong to the current 'target' class
    class_images = (labels == target)
    # Returens the indeces of the class images
    indeces = np.where(class_images)[0]
    # Then we retieve the first 100 images of that class
    hundred_images = indeces[:100]
    # We then convert the value of these indeces in the array to True
    mask[hundred_images] = True

x_faces = faces.data[mask]
y_faces = faces.target[mask]
x_faces.shape

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

face_images = x_faces / 255
face_labels = to_categorical(y_faces)

x_train, x_test, y_train, y_test = train_test_split(face_images, face_labels, train_size=0.8, stratify=face_labels,
                                                    random_state=0)

from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dense(class_count, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)


def show_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()


show_history(hist)

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(class_count, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

show_history(hist)