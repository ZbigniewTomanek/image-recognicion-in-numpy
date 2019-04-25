import numpy as np
from tools import hog
import pickle
from tensorflow import keras
from logistic_regression import (model_selection, prediction, f_measure)
from random import shuffle
import time
from hog_test import plot_photos

with open('train.pkl', mode='rb') as file_:
    TEST_DATA = pickle.load(file_)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

photos = np.array([np.array(x).reshape(36, 36) for x in TEST_DATA[0]])
labels = TEST_DATA[1]


def save_data(train_images, train_labels, val_images, val_labels):
    d = {'x_train': train_images,
         'y_train': train_labels,
         'x_val': val_images,
         'y_val': val_labels
         }
    with open('C:\\Users\\zbigi\\PycharmProjects\\msid\\lab4\\hog_data.pkl', 'wb') as f:
        pickle.dump(d, f)


def add_gaussian_noise_image(img):
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, len(img)*len(img)).reshape(len(img), len(img))
    s /= s.max()
    return np.abs(s + img)


def export_hog_dataset():
    fashion_mnist = keras.datasets.fashion_mnist

    (good_images, _), _ = fashion_mnist.load_data()
    good_images = good_images / 255

    bad_images = np.array([add_gaussian_noise_image(img) for img in good_images])
    plot_photos(bad_images)

    good_labels = np.ones(len(good_images)).reshape(len(good_images), 1)
    bad_labels = np.zeros(len(good_images)).reshape(len(good_images), 1)

    print(good_labels)

    dataset = list(zip(good_images, good_labels))
    dataset += list(zip(bad_images, bad_labels))

    shuffle(dataset)
    dataset = list(zip(*dataset))

    images = list(dataset[0])
    labels = list(dataset[1])

    point = int(len(images) * 0.9)
    images = [hog(photo, flatten=True) for photo in images]

    train_photos = images[:point]
    train_labels = labels[:point]

    test_photos = images[point:]
    test_labels = labels[point:]

    save_data(train_photos, train_labels, test_photos, test_labels)


def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def test_image(image, w, theta):
    return prediction(image, w, theta)


def learn_regression__model(data_part=500):
    eta = 0.1
    epochs = 100
    minibatch_size = 50
    lambdas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    thetas = list(np.arange(0.1, 0.9, 0.05))

    data = load_data('hog_data.pkl')

    p = int(len(data['x_train'])/data_part)
    p = 800

    x_val = np.array(data['x_val'][:p])
    y_val = np.array(data['y_val'][:p])
    x_train = np.array(data['x_train'][p:p*10])
    y_train = np.array(data['y_train'][p:p*10])
    x_test = np.array(data['x_train'][p*10:])
    y_test = np.array(data['y_train'][p*10:])

    t0 = time.time()
    w_0 = np.zeros((x_train.shape[1], 1))
    print(x_train.shape)

    print('Starting learning for ', p*10+p, 'records')
    l, t, w_computed, F = model_selection(x_train, y_train, x_val, y_val, w_0, epochs, eta, minibatch_size, lambdas,
                                          thetas)
    t1 = time.time()

    print('Learning has taken {} seconds'.format(t1-t0))

    print('Best w:', w_computed)
    print('Best theta', t)

    pred = prediction(x_test, w_computed, t)
    f_val = f_measure(y_test, pred)
    print('F measure:', f_val)

    data_to_save = {
        'w': w_computed,
        'theta': t,
        'f_val': f_val
    }

    with open('C:\\Users\\zbigi\\PycharmProjects\\msid\\lab4\\regression_model.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)


if __name__ == '__main__':
    #print(hog(photos[0], flatten=True).shape)
    #export_hog_dataset()
    learn_regression__model()
    #data = load_data('regression_model.pkl')
    #print(data['f_val'])
