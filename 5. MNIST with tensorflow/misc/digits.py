import os
import math
import numpy as np
import tensorflow as tf

class Digits:
    def __init__(self, data_dir, batch_size):
        # load MNIST data (if not available)
        self._data = os.path.join(data_dir, 'mnist.npz')
        self._train, self._test = tf.keras.datasets.mnist.load_data(path=self._data)
        self._batch_size = batch_size
        self._train_count = self._train[0].shape[0]
        self._size = self._train[0].shape[1] * self._train[0].shape[2]
        self._total = math.ceil((1. * self._train_count) / self._batch_size)

        self._testX = self._test[0].reshape(self._test[0].shape[0], self._size) / 255.
        self._testY = np.eye(10)[self._test[1]]

        self._trainX = self._train[0].reshape(self._train_count, self._size) / 255.
        self._trainY = self._train[1]

    def __iter__(self):
        # shuffle arrays
        p = np.random.permutation(self._trainX.shape[0])
        self._trainX = self._trainX[p]
        self._trainY = self._trainY[p]

        # reset counter
        self._current = 0

        return self

    def __next__(self):
        if self._current > self._train_count:
            raise StopIteration

        x = self._trainX[self._current : self._current + self._batch_size,:]
        y = np.eye(10)[self._trainY[self._current : self._current + self._batch_size]]

        if x.shape[0] == 0:
            raise StopIteration

        self._current += self._batch_size
        
        return x, y

    def __getitem__(self, index):
        index = 0 if index < 0 else index
        index = self._train_count - 1 if index > self._train_count else index
        x = self._trainX[index, :]
        y = self._trainY[index]

        return x, y

    @property
    def test(self):
        return self._testX, self._testY

    @property
    def total(self):
        return self._total

    def stringify(self, index):
        index = 0 if index < 0 else index
        index = self._train_count - 1 if index > self._train_count else index
        x = self._trainX[index, :]
        y = np.eye(10)[self._trainY[index]]

        s = np.sqrt(x.shape[0])

        for i in range(x.shape[0]):
            if i == 0: print("{{\r\n   ", end="")
            elif i % s == 0: print("\r\n   ", end="")
            print("{:>3}, ".format(int(x[i] * 255)), end="")

        print('\r\n}}\r\n', end="")
        print(y)


if __name__ == "__main__":
    p = os.path.abspath('..\\data')
    digits = Digits(p, 1000)

    idx = 756
    digits.stringify(idx)
    np.savetxt('v.txt', digits[idx][0]*255, delimiter=',', fmt='%d', newline=',')
    print(digits[idx][1])