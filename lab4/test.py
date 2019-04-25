from tensorflow import keras
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with open('train.pkl', mode='rb') as file_:
    TEST_DATA = pickle.load(file_)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

photos = np.array([np.array(x).reshape(36, 36) for x in TEST_DATA[0]])
labels = TEST_DATA[1]

point = int(len(photos)*0.9)
print(point)
train_photos = photos[:point]
train_labels = labels[:point]

test_photos = photos[point:]
test_labels = labels[point:]


print(TEST_DATA[0][0])
print(len(photos[0]))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(photos[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(36, 36)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_photos, train_labels, epochs=50)
test_loss, test_acc = model.evaluate(test_photos, test_labels)

print('Test accuracy:', test_acc)

