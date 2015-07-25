from __future__ import absolute_import
import numpy as np
import os
import sys


def load_data():
    dirname = "Data"
    compname = "Kaggle-DirtyDocs"
    path = os.path.expanduser(os.path.join('~', dirname, compname))

    subdirname = "train"
    nb_train_samples = len([name for name in os.listdir(os.path.join(path,subdirname)) if os.path.isfile(os.path.join(path, subdirname, name))])
    nb_train_names = os.listdir(os.path.join(path,subdirname))

    subdirname = "test"
    nb_test_samples = len([name for name in os.listdir(os.path.join(path,subdirname)) if os.path.isfile(os.path.join(path, subdirname, name))])
    nb_test_names = os.listdir(os.path.join(path,subdirname))

    X_train = np.zeros((nb_train_samples, 420, 540), dtype="uint8")
    y_train = np.zeros((nb_train_samples, 420, 540), dtype="uint8")
    X_test = np.zeros((nb_test_samples, 420, 540), dtype="uint8")

    for i in range(1, nb_train_samples):
        img = misc.imread(os.path.join(path, 'train', '%s' % nb_train_names[i-1]))
        img_cleaned = misc.imread(os.path.join(path, 'train_cleaned', '%s' % nb_train_names[i-1]))
        if img.shape != (420,540):
            img = np.pad(img, ((0, 420 - img.shape[0]), (0, 540-img.shape[1])), mode='constant', constant_values = 0)
        if img_cleaned.shape != (420,540):
            img_cleaned = np.pad(img_cleaned, ((0, 420 - img_cleaned.shape[0]), (0, 540-img_cleaned.shape[1])), mode='constant', constant_values = 0)
        X_train[i-1,:,:] = img
        y_train[i-1,:,:] = img_cleaned

    for i in range(1, nb_test_samples):
        img = misc.imread(os.path.join(path, 'test', '%s' % nb_test_names[i-1]))
        if img.shape != (420,540):
            img = np.pad(img, ((0, 420 - img.shape[0]), (0, 540-img.shape[1])), mode='constant', constant_values = 0)
        X_test[i-1,:,:] = img


    return (X_train, y_train), X_test