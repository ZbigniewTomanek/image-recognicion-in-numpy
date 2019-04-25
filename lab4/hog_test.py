from tensorflow import keras
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools import hog
from logistic_regression import prediction

with open('train.pkl', mode='rb') as file_:
    TEST_DATA = pickle.load(file_)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

photos = np.array([np.array(x).reshape(36, 36) for x in TEST_DATA[0]])
labels = TEST_DATA[1]


def dataset_test(dataset):
    point = int(len(dataset) * 0.9)
    print(point)

    train_photos = np.array(dataset[:point])
    train_labels = np.array(labels[:point])

    test_photos = np.array(dataset[point:])
    test_labels = np.array(labels[point:])

    print(train_photos.shape)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=train_photos[0].shape),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(train_photos, train_labels, epochs=30)
    test_loss, test_acc = model.evaluate(test_photos, test_labels)

    print('Test accuracy:', test_acc)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 36, 36, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 9 * 9 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_test(dataset):
    point = int(len(dataset) * 0.9)
    print(point)

    train_data = np.array(dataset[:point])
    train_labels = np.array(labels[:point])

    eval_data = np.array(dataset[point:])
    eval_labels = np.array(labels[point:])

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    mnist_classifier.train(input_fn=train_input_fn, steps=1000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def mask_value(photo, x, y, mask):
    size = len(mask)
    photo = photo[x:x + size, y:y + size]
    return sum(sum(photo * mask))


def search_frame(photo, window_size=28):
    size = len(photo)
    delta = size - window_size

    outer_kernel = np.ones((window_size, window_size))
    outer_kernel[1:window_size - 1, 1:window_size - 1] = 0

    inner_kernel = np.ones((window_size, window_size))
    inner_kernel[0] = 0
    inner_kernel[:, 0] = 0
    inner_kernel[window_size - 1] = 0
    inner_kernel[:, window_size - 1] = 0

    inner_values = []
    outer_values = []

    for x in range(delta):
        for y in range(delta):
            inner_values.append(mask_value(photo, x, y, inner_kernel))
            outer_values.append(mask_value(photo, x, y, outer_kernel))

    inner_i = np.argmax(inner_values)
    outer_i = np.argmin(outer_values)

    ix = inner_i % delta
    iy = int(inner_i / delta)

    ox = outer_i % delta
    oy = int(outer_i / delta)

    return photo[ox:ox + window_size, oy:oy + window_size]


def refactor_photos(photos):
    pass


def plot_photos(photos, number=25):
    plt.figure(figsize=(10, 10))
    for i in range(number):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(photos[i], cmap=plt.cm.binary)
    plt.show()


def train_model(photos):
    # point = int(len(photos) * 0.9)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_photos, train_labels), (_, _) = fashion_mnist.load_data()

    # train_photos = photos[:point]
    # train_labels = labels[:point]

    # test_photos = photos[point:]
    # test_labels = labels[point:]

    test_photos = photos
    test_labels = labels

    x = len(photos[0])

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x, x)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(train_photos, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_photos, test_labels)

    print('Test accuracy:', test_acc)


def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def use_regression_to_find_frame(photo, w, theta, window_size=28):
    size = len(photo)
    delta = size - window_size
    frames = []

    for x in range(delta):
        for y in range(delta):
            frames.append(hog(photo[x:x + window_size, y:y + window_size], flatten=True))

    pred = prediction(np.array(frames), w, theta)
    index = pred[::-1].argmax()

    x = index % delta
    y = int(index / delta)

    return photo[x:x + window_size, y:y + window_size]


def extract_frames_with_reg(images):
    data = load_data('regression_model.pkl')
    w = data['w']
    theta = data['theta']

    cropped = []
    i = 0
    for img in images:
        print(i)
        i += 1
        cropped.append(use_regression_to_find_frame(img, w, theta))

    cropped = np.array(cropped)

    with open('C:\\Users\\zbigi\\PycharmProjects\\msid\\lab4\\cropped_dataset.pkl', 'wb') as f:
        pickle.dump(cropped, f)

    return cropped


if __name__ == '__main__':
    # plot_photos(photos)
    # dataset = [hog(img, flatten=True) for img in photos]
    #imgs = load_data('cropped_dataset.pkl')
    #dataset_test(imgs)
    # hog_test(photos)

    cnn_test(photos)
