"""
    Pixel to pixel example
"""
import tensorflow as tf


class UpSample(tf.keras.models.Model):
    """
        UpSampling Unit for the Generator Model

        Parameters
        ----------
        size: int
           Size of the kernel convolution window
        filters: int
            No. of filters to app;y during convolution
        drop_out: float
            The rate of dropout to apply to the model layers
    """

    def __init__(self, filters=128, size=4, drop_out=0):
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

        if drop_out:
            self.drop_out = tf.keras.layers.Dropout(
                drop_out)

    def call(self, inputs, training=None):
        """
            Forward Pass
        """
        x = self.conv1(inputs[0])
        x = self.batch_norm(x, training=training) if self.batch_norm else x
        x = self.drop_out(x, training=training) if hasattr(
            self, 'drop_out') else x
        x = tf.nn.relu(x)
        output = tf.concat([x, inputs[1]], axis=-1)

        return output


class DownSample(tf.keras.models.Model):
    """
        DownSampling Model to the Generator

        Parameters
        ----------
        kernel_size: int
            Size of the convolution window
        filters: int
            Numbers of convolution filters
        batch_norm: bool
            Apply BatchNormalization
    """

    def __init__(self, kernel_size=4, filters=128, batch_norm=False):
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


class Generator(tf.keras.models.Model):
    """
        U-Net model

        Parameters
        ----------
        output_channels: int
            Number of output convolution filters
    """

    def __init__(self, output_channels=3):
        super(Generator, self).__init__()
        # Downsample
        self.down_sample0 = DownSample(
            kernel_size=64,
            filters=4, batch_norm=False)        # (None, 128, 128, 64)
        self.down_sample1 = DownSample(4, 128)  # (None, 64, 64, 128)
        self.down_sample2 = DownSample(kernel_size=256)  # (None, 32, 32, 256)
        self.down_sample3 = DownSample(kernel_size=512)  # (None, 16, 16, 512)
        self.down_sample4 = DownSample(kernel_size=512)  # (None, 8, 8, 512)
        self.down_sample5 = DownSample(kernel_size=512)  # (None, 4, 4, 512)
        self.down_sample6 = DownSample(kernel_size=512)  # (None, 2, 2, 512)
        self.down_sample7 = DownSample(kernel_size=512)  # (None, 1, 1, 512)

        # UpSample

        self.up_sample0 = UpSample(filters=512, size=4,
                                   drop_out=.5)       # (None, 2, 2, 1024)
        self.up_sample1 = UpSample(512, drop_out=.5)  # (None, 4, 4, 1024)
        self.up_sample2 = UpSample(512, drop_out=.4)  # (None, 8, 8, 1024)
        self.up_sample3 = UpSample(512)               # (None, 16, 16, 1024)

        self.up_sample4 = UpSample(256)  # (None, 32, 32, 512)
        self.up_sample5 = UpSample(128)  # (None, 64, 64, 256)
        self.up_sample6 = UpSample(64)   # (None, 128, 128, 128)

        self.output_pred = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding='same',
            activation='tanh',
            kernel_initializer=tf.random_normal_initializer(0., 0.02)
        )  # (None, 256, 256, 3)

    def call(self, inputs):
        """
            Upsamples and DownSamples the input
        """
        # input -> (None, 256, 256, 3)
        for i in range(0, 9):
            inputs = getattr(self, f'down_sample{i}')(inputs)

        for i in range(0, 7):
            inputs = getattr(self, f'up_sample{i}')(inputs)

        output = self.output_pred(inputs)  # (None, 256, 256, 3)

        return output


class Discriminator(tf.keras.models.Model):
    """
        PatchGAN
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.x_0 = DownSample(filters=64)           # (None, 128, 128, 64)
        self.x_1 = DownSample(128, batch_norm=True)  # (None, 64, 64, 128)
        self.x_2 = DownSample(256, batch_norm=True)  # (None, 32, 32, 256)

        # [-1, 32, 32, 256] -> [-1, 34, 34, 256]
        self.x_3 = tf.keras.layers.ZeroPadding2D()
        self.x_4 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=4,
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=False)  # (None, 31, 31, 512)
        self.x_5 = tf.keras.layers.BatchNormalization()
        self.x_6 = tf.keras.layers.LeakyReLU()

        # [-1, 31, 31, 512] -> [-1, 33, 33, 512]
        self.x_7 = tf.keras.layers.ZeroPadding2D()
        # (None, 30, 30, 1)
        self.output_pred = tf.keras.layers.Convolution2D(
            1, 4,
            kernel_initializer=tf.random_normal_initializer(0., 0.02)
        )

    def call(self, inputs, training=None):
        input_img, target_img = inputs

        # Concatenate input and target image
        # (None, 256, 256, channels * 2)
        x = tf.concat([input_img, target_img], axis=-1)

        # Call x on each layer
        for i in range(0, 8):
            x = getattr(self, f'x_{i}')(x)

        x = self.output_pred(x)  # (None, 30, 30, 1)

        return x


class DownSampleDiscriminator(tf.keras.models.Model):
    """
        DownSample for the Discriminator
    """

    def __init__(self, filters, kernel_size=4, batch_norm=False):
        super(DownSampleDiscriminator, self).__init__()

        self.batch_norm = None
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(
                0., 0.02),
            use_bias=False)
        if batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        activation = tf.nn.leaky_relu(x)

        return activation
