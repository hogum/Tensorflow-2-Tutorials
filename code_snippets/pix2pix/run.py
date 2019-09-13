import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from pixel2pixel import Generator, Discriminator

BATCH_SIZE = 1
IMG_WIDTH, IMG_HEIGHT = 256, 256
DATA_PATH = None


def fetch_data():
    """
        Downloads the dataset
    """
    global DATA_PATH

    file_origin = 'https://people.eecs.berkeley.edu/' + \
        '~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

    path = tf.keras.utils.get_file(fname='facades.tar.gz',
                                   origin=file_origin,
                                   extract=True)
    DATA_PATH = os.path.join(os.path.dirname(path), 'facades/')


def load_image(image_path, random_jitter=False):
    """
        Loads and preprocess an image from the given path

        Parameters
        ----------
        random_jitter: bool
            Resize the image to (286, 286) then randomly crop to (256 X 256)
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image)

    w = image.shape[1] // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    real_image = tf.cast(real_image, tf.float32)
    input_image = tf.cast(input_image, tf.float32)

    if random_jitter:  # (286, 286, 3)
        input_image, real_image = resize_image(input_image, real_image,
                                               height=286, width=286)

        input_image, real_image = crop_image(input_image, real_image)

        # Randomly flip the image horizontally -> left to right
        if tf.random_uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image, real_image = resize_image(input_image, real_image,
                                               IMG_WIDTH, IMG_HEIGHT)
    # Normalize images from [-1, 1]
    input_image = (input_image / 127.5) - 1  # [256, 256, 3]
    real_image = (real_image / 127.5) - 1  # [256, 256, 3]

    # [256, 256, 6]
    output = tf.concat([input_image, real_image], axis=2)

    return output


def resize_image(input_image, real_image, height, width):
    """
        Resizes the images to the given (h * w)
    """
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def crop_image(input_image, real_image):
    """
        Randomly crops to (256, 256, 3)
    """
    stacked_im = tf.stack([input_image, real_image], axis=0)
    cropped_im = tf.random_crop(stacked_im, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_im


def load_dataset(split):
    """
        Creates input datasets from the downloaded data
        ---------
        split: str
            train or test
    """
    data_path = tf.data.Dataset.list_files(DATA_PATH + f'/{split}/*.jpg')
    train_data = [load_image(x, random_jitter=True)
                  for x in iter(data_path)]
    train_data = tf.stack(train_data, axis=0)  # (256, 256, 3)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_data).shuffle(400).batch(BATCH_SIZE)
    return train_dataset


def discriminator_loss(real_output, generated_output):
    """
        Computes the real and generated sigmoid cross
        entropy loss
    """
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss(tf.ones_like(real_output), real_output)
    generated_loss = loss(tf.zeros_like(generated_output), generated_output)

    return real_loss + generated_loss


def generator_loss(disc_generated_output, generated_output, target):
    """
        Computes cross entropy loss of generated images
    """
    lambda_ = 100
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - generated_output))

    # gan_loss + lambda (|MAE|)
    total_loss = gan_loss + (lambda_ * l1_loss)
    return total_loss


def generated_images(model, test_input, target, epoch):
    """
        Run the model for generated output
    """
    pred = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display = test_input[0], target[0], pred[0]
    titles = 'Input image', 'Ground Truth', 'Predicted Image'

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(display[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(f'images/epoch_{epoch}.png')
    # plt.show()


def train(generator, discriminator, datasets, epochs=100):
    """
        Train the model
    """
    train_dataset, test_dataset = datasets
    steps = train_dataset.shape[0]

    for epoch in range(epochs):
        start = time.time()
        for step, inputs in enumerate(train_dataset):
            disc_loss, gen_loss = train_step(inputs, generator, discriminator)
            if not step % 100:
                print(f'epoch: [{epoch}/{epochs}], step: [{step}/{steps}], '
                      + f'  disc_loss: {disc_loss:.4f}  '
                        + f'gen_loss: {gen_loss:.4f}')

                for x in test_dataset:
                    input_image, target = tf.split(
                        x, num_or_size_splits=[3, 3], axis=3)
                    generated_images(generator, input_image, target, epoch)

        print(f'Time taken for epoch {epoch + 1}: {time.time() - start}')


def train_step(inputs, generator, discriminator, lr=2e-5):
    """
        Applies the loss gradients
    """
    gen_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    input_image, target = tf.split(inputs,
                                   num_or_size_splits=[3, 3],
                                   axis=3)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Pixel2pixel generated image
        gen_output = generator(input_image, training=True)
        # original image + real pix2pix image
        disc_real_output = discriminator([input_image, target], training=True)
        # fake pix2pix + original image
        disc_fake_output = discriminator(
            [input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_fake_output, gen_output, target)
        disc_loss = discriminator(disc_real_output, disc_fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(
        zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(
        zip(disc_grads, discriminator.trainable_variables))

    return disc_loss, gen_loss


def main():
    datasets_names = ['train', 'test']
    datasets = [load_dataset(name) for name in datasets_names]
    train(generator=Generator(),
          discriminator=Discriminator(),
          datasets=datasets,
          epochs=2000)


if __name__ == '__main__':
    main()
