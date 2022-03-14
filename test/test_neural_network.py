import unittest
from neural_network import NeuralNetwork
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential

class test_NeuralNetwork(unittest.TestCase):
    def test_setModel(self):
        model = Sequential()
        model.add(Flatten(input_shape=(28,28,1)))
        model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.1)))
        nn = NeuralNetwork()
        nn.set_model(model)
        self.assertTrue(hasattr(nn, 'model'))

    def test_countLayers(self):
        model = Sequential()
        model.add(Flatten(input_shape=(28,28,1)))
        model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.1)))
        model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.1)))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation=tf.nn.relu, kernel_regularizer = tf.keras.regularizers.l1(0.1)))
        model.add(Dense(10, activation=tf.nn.softmax))
        nn = NeuralNetwork()
        nn.set_model(model)
        self.assertEqual(nn.num_layers(),4)
        self.assertEqual(nn.layers,[1,2,4,5])

if __name__ == "__main__":
    unittest.main()
