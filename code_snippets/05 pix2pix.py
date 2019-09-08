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
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters,
                kernel_size=size,
                strides=2,
                padding='same',
                use_bias=False,
                kernel_initializer=tf.random_normal_initializer(0, 0.02)
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
        output = tf.concat(x, inputs[1], axis=-1)

        return output





class Generator(tf.keras.layers.Layer):
    """
        Generates
    """
    def __init__(self):
        super(Generator, self).__init__()



