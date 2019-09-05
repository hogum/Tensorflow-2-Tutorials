"""
    Auto Encoder Handwrited digits reconstruction
"""

import tensorflow as tf
import numpy as np

# The Encoder and Decoder are ``layers`` that will be components
# of the AutoEncoder


class Encoder(tf.keras.layers.Layer):
    """
        Learns the structure of the input data
    """

    def __init__(self, hidden_dim=512):
        super(Encoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=hidden_dim)

    def call(self, inputs):
        activation = self.fc1(inputs)
        output = self.output_layer(activation)

        return output


class Decoder(tf.keras.layers.Layer):
    """
        Reconstructs the fed data from its lower
        dimension to the original
    """

    def __init__(self, hidden_dim, image_size):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=image_size)

    def call(self, h_inputs):
        activation = self.fc1(h_inputs)
        x = self.output_layer(activation)

        return x


class AutoEncoder(tf.keras.models.Model):
    """
        Encoder + Decoder Model

        Parameters
        ----------
        h_dims: int
            Output dimension of hidden layers
        image_dim: int
            Ouput Image height * width
    """

    def __init__(self, h_dim, image_dim):
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim, image_dim)

    def call(self, inputs):
        h = self.encoder(inputs)
        reconstructed = self.decoder(h)

        return reconstructed
