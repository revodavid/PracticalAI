import os
import math
import numpy as np
import tensorflow as tf

def get_data():
    train, test = tf.keras.datasets.mnist.load_data()
    x = train[0]
    y = train[1]
    size = x.shape[1] * x.shape[2]
    return { 'X': x.reshape(x.shape[0], size) / 255. , 
             'y': train[1] }

if __name__ == "__main__":
    s = get_data()
    print(s['X'].shape, s['y'].shape)