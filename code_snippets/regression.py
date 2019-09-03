"""
    Linear Regression example on the Boston housing dataset
"""

import tensorflow as tf


class LinearRegression(tf.keras.models.Model):
    """
        Linear Regression Model
    """

    def __init__(self):
        super(LinearRegression, self).__init__()

        #  Create the parameter tensors
        # dimensions [13, 1] -> in, out
        self.w = self.add_variable('weight_in', [13, 1])
        self.b = self.add_variable('bias', [1])  # [Out]

    def call(self, x):
        """
            Forward Pass
        """
        # y = wx + b
        y = tf.matmul(x, self.w) + self.b

        return y


def preprocess_data():
    """
        Loads and preprocesses the dataset
    """
    data = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = data.load_data()

    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)


def train(model, train_dataset, validation_data=None, lr=1e-4,
          epochs=10, val_intv=10):
    """
        Runs the gradient descent on the Model
    """
    loss_func = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(lr)

    for epoch in range(epochs):
        for step, (X, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(X)

                # [b, 1] -> [b]
                logits = tf.squeeze(logits, axis=1)
                loss = loss_func(y, logits)
            grad = tape.gradient(
                target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # Show train and validation loss at same epoch
        if not epoch % val_intv:
            print(f'{epoch} loss: {loss.numpy()}', end=' ')

        if not epoch % val_intv and validation_data is not None:
            # Validate model
            for X, y in validation_data:
                # [b, 1]
                logits = model(X)
                # [b]
                logits = tf.squeeze(logits)

                # y_true vs y_pred
                loss = loss_func(y, logits)

                print(f'val_loss: {loss.numpy()}')


def main():
    """
        Creates and trains regression model
    """
    model = LinearRegression()
    train_data, val_data = preprocess_data()

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(128)

    train(model, train_dataset, val_dataset, epochs=1000, val_intv=100)


if __name__ == '__main__':
    main()
