"""
    Auto Encoder Handwrited digits reconstruction
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

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
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim, image_dim)

    def call(self, inputs):
        h = self.encoder(inputs)
        reconstructed = self.decoder(h)

        return reconstructed


def loss(original_x, reconstructed, batch_size):
    """
       Computes reconstruction loss
           (loss between the original and modeled images)
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=original_x, logits=reconstructed)

    # KL divergence
    loss = tf.reduce_sum(loss) / batch_size
    return loss


def train(model, dataset, lr=1e-4, epochs=20, batch_size=128):
    """
        Fits the model
    """
    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(epochs):
        for step, x in enumerate(dataset):
            # [28, 28] -> [None, 28 * 28] Batch size dim
            x = tf.reshape(x, [-1, 28 * 28])

            with tf.GradientTape() as tape:
                # Forward pass
                fake_imgs = model(x)

                # loss
                reconstruction_loss = loss(x, fake_imgs, batch_size)
            grads = tape.gradient(target=reconstruction_loss,
                                  sources=model.trainable_variables)
            # clip gradients
            grads, _ = tf.clip_by_global_norm(grads, 15)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if not step % 25:
                print(f'epoch: [{epoch + 1}/{epochs}], step: [{step}]' +
                      f'   loss: {reconstruction_loss:.5f}'
                      )

        # Save reconstructed images of every batch
        save_images(model, x[:batch_size // 4], epoch)


grid_size = 196
image_grid = Image.new(mode='L', size=[grid_size, grid_size])


def save_images(model, x, epoch):
    """
        Saves reconstructed image plots
    """
    idx = 0

    # reconstruct images
    logits = model(x)
    output = tf.nn.sigmoid(logits)

    # [Batch_s, 784] - > [batch_s, 28, 28]
    fake_imgs = tf.reshape(output, [-1, 28, 28]).numpy() * 255

    # original images
    original_imgs = tf.reshape(x, [-1, 28, 28])

    # Concatenate reconstructed and original images
    concat_tnsr = tf.concat([original_imgs, fake_imgs], axis=0)
    images = concat_tnsr.numpy() * 255
    images = images.astype('uint8')

    for row in range(0, grid_size, 28):
        for col in range(0, grid_size, 28):
            img = images[idx]
            img = Image.fromarray(img, mode='L')
            image_grid.paste(img, (row, col))
            idx += 1

    plt.imshow(np.asarray(image_grid), cmap='gray')

    if not epoch % 20:
        plt.savefig(f'autoencoder_epoch_{epoch + 1}.png')
        print('Image saved')

    # plt.show()


def main():
    """
        Runs AutoEncoder
    """
    batch_size = 128

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # Labels not required
    x_train = np.concatenate([x_train, x_test], axis=0)

    # [70000, 28, 28]
    x_train = x_train.astype('float32') / 255.  # Use float dtype & normalize
    img_h, img_w = x_train.shape[1:]

    # Build model
    model = AutoEncoder(h_dim=128, image_dim=img_w * img_h)
    model.build(input_shape=(4, img_h * img_w))  # [n_layers, img_size]
    model.summary()

    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        x_train).shuffle(buffer_size=x_train.shape[0] * 2).batch(batch_size)

    # train
    train(model, dataset, batch_size=batch_size, epochs=20000)


if __name__ == '__main__':
    main()
