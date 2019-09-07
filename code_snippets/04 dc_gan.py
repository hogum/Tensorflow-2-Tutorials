"""
    Generation of mnist Images using Deep Covolutional
    Generative Adversarial Network
"""

import tensorflow as tf


def load_dataset(batch_size=256):
    """
        Loads and prepares the training data
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    im_height, im_width = x_train.shape[1:]
    # combine train and test data
    train_images = x_train.concatenate(x_train, x_test, axis=0)
    train_images = train_images.reshape(
        [train_images.shape[0], im_height, im_width, 1]
    ).astype('float32')

    # Normalize images [-1, 1]
    # (xi - .5*max[x])/ .5*max[x]
    train_images = (train_images - 127.5) / 127.5
    buffer_size = train_images.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(
        buffer_size).batch(batch_size)

    return train_dataset


class Generator(tf.keras.layers.Model):
    """
        Produces an image from random noise
    """

    def __init__(self, in_shape=[100]):
        super(Generator, self).__init__()
        inputs = tf.keras.Input(shape=in_shape)

        self.x0 = tf.keras.layers.Dense(units=256*28*28, input_shape=in_shape)
        self.x1 = tf.keras.layers.BatchNormalization()
        self.x2 = tf.keras.layers.LeakyReLU()

        self.x3 = tf.keras.layers.Reshape([28, 28, 256])

        self.x4 = tf.keras.layers.Convolution2DTranspose(filters=128,
                                                         kernel_size=(5, 5),
                                                         padding='same')
        self.x5 = tf.keras.layers.BatchNormalization()
        self.x6 = tf.keras.layers.LeakyRelu()

        self.x7 = tf.keras.layers.Convolution2DTranspose(filters=64,
                                                         kernel_size=(5, 5),
                                                         strides=1,
                                                         padding='same')
        self.x8 = tf.keras.layers.BatchNormalization()
        self.x9 = tf.keras.layers.LeakyReLU()

        self.output_prediction = tf.keras.layers.Convolution2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            strides=1,
            padding='same')

        def call(self, inputs):
            """
                Forward pass
            """
            x = self.x0(inputs)

            for i in range(1, 10):
                x = getattr(self, f'x{i}')(x)

            return self.output_prediction(x)


class Discriminator(tf.keras.models.Model):
    """
        Image Classifier
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.x0 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=5,
                                         strides=2,
                                         padding='same',
                                         input_shape=[28, 28, 1])
        self.x1 = tf.keras.layers.LeakyReLU()
        self.x2 = tf.keras.layers.Dropout(0.3)

        self.x3 = tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=5,
                                         strides=2,
                                         padding='same')

        self.x4 = tf.keras.layers.LeakyReLU()
        self.x5 = tf.keras.layers.Dropout(0.3)

        self.x6 = tf.keras.layers.Flatten()
        self.output_prediction = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.x0(inputs)

        for i in range(1, 7):
            x = getattr(self, f'x{i}')(x)

        return self.output_prediction(x)
