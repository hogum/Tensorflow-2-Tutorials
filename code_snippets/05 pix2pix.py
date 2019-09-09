"""
    Pixel to pixel example
"""
import tensorflow as tf


class UpSample(tf.keras.models.Model):
    """
        UpSampling Unit for the Generator Model

        Parameters
        ----------
        filters: int
            No. of filters to app;y during convolution
        size: int
           Size of the kernel convolution window
        drop_out: float
            The rate of dropout  to apply to the model layers

    """

    def __init__(self, size, filters=128, drop_out=0.4):
        super(UpSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=size,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(
                0, 0.02)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.drop_out = tf.keras.layers.Dropot(drop_out)

    def call(self, inputs, training=False):
        """
            Forward Pass
        """
        x = self.conv1(inputs[0])
        x = self.batch_norm(x, training=training)
        x = self.drop_out(x, training=training)
        x = tf.nn.relu(x)
        output = tf.concat([x, inputs[1]], axis=-1)

        return output


class DownSample(tf.keras.models.Model):
    """
        DownSampling Model to the Generator
    """

    def __init__(self, kernel_size=5, filters=128, batch_norm=0):
        super(DownSample, self).__init__()
        self.batch_norm = None

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            kernel_initializer=tf.random_normal_initializer(
                mean=0., stddev=0.02),
            padding='same',
            use_bias=False)
        if batch_norm:
            self.batch_norm = \
                tf.keras.layers.BatchNormalization(batch_norm)
        self.output_pred = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=None):
        """
            Forward pass
        """
        x = self.conv1(inputs)
        x = self.batch_norm(x, training=training) if self.batch_norm else x
        return self.output_pred(x)


class Generator(tf.keras.layers.Layer):
    """
        Generates
    """

    def __init__(self):
        super(Generator, self).__init__()
