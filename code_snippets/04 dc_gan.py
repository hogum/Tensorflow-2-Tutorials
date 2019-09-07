"""
    Generation of mnist Images using Deep Covolutional
    Generative Adversarial Network
"""
import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


def load_dataset(batch_size=256):
    """
        Loads and prepares the training data
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    im_height, im_width = x_train.shape[1:]

    # combine train and test data
    train_images = np.concatenate([x_train, x_test], axis=0)
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


class Generator(tf.keras.models.Model):
    """
        Produces an image from random noise
    """

    def __init__(self, in_shape=(100, )):
        super(Generator, self).__init__()

        self.x0 = tf.keras.layers.Dense(
            units=256*28*28, use_bias=False, input_shape=in_shape)
        self.x1 = tf.keras.layers.BatchNormalization()
        self.x2 = tf.keras.layers.LeakyReLU()

        self.x3 = tf.keras.layers.Reshape((28, 28, 256))

        self.x4 = tf.keras.layers.Convolution2DTranspose(filters=256,
                                                         kernel_size=2,
                                                         strides=(1, 1),
                                                         padding='same',
                                                         use_bias=False)
        self.x5 = tf.keras.layers.BatchNormalization()
        self.x6 = tf.keras.layers.LeakyReLU()

        self.x7 = tf.keras.layers.Convolution2DTranspose(filters=64,
                                                         kernel_size=(5, 5),
                                                         strides=1,
                                                         padding='same',
                                                         use_bias=False)
        self.x8 = tf.keras.layers.BatchNormalization()
        self.x9 = tf.keras.layers.LeakyReLU()

        self.output_prediction = tf.keras.layers.Convolution2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            strides=1,
            activation='tanh',
            use_bias=False,
            padding='same')

    def call(self, inputs, training=False):
        """
            Forward pass
        """
        x = self.x0(inputs)

        for i in range(1, 10):
            x = getattr(self, f'x{i}')(x)

        return self.output_prediction(x)

    def loss_f(self, fake_imgs):
        """
            Gives the generator loss:
            (How well the Generator was able to trick the Discriminator)

            It Compares the Discriminator decisions on
            the generated images
        """
        cross_entropy = cross_entropy_loss()
        return cross_entropy(tf.ones_like(fake_imgs), fake_imgs)


class Discriminator(tf.keras.models.Model):
    """
        Image Classifier

        Returns 1 - Real images
                0 - Fake
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

    def call(self, inputs, training=False):
        x = self.x0(inputs)

        for i in range(1, 7):
            x = getattr(self, f'x{i}')(x)

        return self.output_prediction(x)

    def loss_f(self, real_images, fake_images):
        """
            Shows the Discriminator's ability to distinguish
            real images from fakes

            Involves comparing the Generator's predictions on real
            images, and fake images to an array of ones and
            zeros respectively
        """
        cross_entropy = cross_entropy_loss()
        real_loss = cross_entropy(tf.ones_like(real_images), real_images)
        fake_loss = cross_entropy(tf.ones_like(fake_images), fake_images)
        total_loss = real_loss + fake_loss

        return total_loss


class GAN():
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.generator = Generator()
        self.discriminator = Discriminator()

        optimizer = tf.keras.optimizers.Adam
        self.gen_optimizer = optimizer(lr)
        self.disc_optimizer = optimizer(lr)

        self.save_checkpoints()

    def save_checkpoints(self):
        """
            Saves model checkpoints
        """

        chkpoint_path = './checkpoints'
        self.chkpoint_prefix = os.path.join(chkpoint_path, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            discriminator=self.discriminator,
            generator=self.generator)

    @tf.function
    def train_step(self, images, batch_size, dim=100):
        """
            Computes the gradients of the Generator and
            Discriminator for each input batch
      """
        # Generate random noise
        noise = tf.random.normal([batch_size, dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(gen_images, training=True)

            gen_loss = self.generator.loss_f(fake_output)
            disc_loss = self.discriminator.loss_f(real_output, fake_output)

        gen_grads = gen_tape.gradient(
            target=gen_loss,
            sources=self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables))

    def train(self, dataset, epochs=50, batch_size=256, dim=100):
        """
            Trains the models
        """
        n_seed_examples = 32
        seed = tf.random.normal([n_seed_examples, dim])

        for epoch in range(epochs):
            for image_batch in dataset:
                self.train_step(image_batch, batch_size=batch_size)
            self.generate_images(epoch + 1, seed)

            if epoch % 10 == 0:
                self.checkpoint.save(file_prefix=self.chkpoint_prefix)

    def generate_images(self, epoch, input_imgs):
        """
            Generate images from the trained Generator
        """
        # Run in inference mode
        predictions = self.generator(input_imgs, training=False)

        plt.figure(figsize=(7, 7))
        rows, cols = 7, 7

        for i in range(len(predictions.shape[0])):
            plt.subplot(rows, cols, 1 + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig(f'gan_image_epoch_{epoch}.png')
        plt.show()


def cross_entropy_loss():
    """
       Returns an instance to compute Cross Entropy loss
    """
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)


def main():
    """
        Runs GAN
    """

    batch_size = 256
    model = GAN()
    # restore checkpoints
    # model.checkpoint.restore('./checkpoints')
    dataset = load_dataset(batch_size=batch_size)

    model.train(dataset, epochs=32, batch_size=batch_size)


if __name__ == '__main__':
    main()
