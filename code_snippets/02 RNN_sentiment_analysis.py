"""
    Sentiment Analysis on the IMDB dataset
"""
import tensorflow as tf

n_words = 100000  # Unique words to use


class RNN(tf.keras.models.Model):
    """
        Recurrent Neural Network Model

        Parameters
        ----------
        units: int
            RNN Cell Output dimension
        n_layers: int
            No. of layers in the RNN model
    """

    def __init__(self, units, n_layers):
        super(RNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=n_words, output_dim=100)

        cells = [tf.keras.layers.LSTM(units, activation='relu')
                 for _ in n_layers]
        self.rnn = tf.keras.layers.RNN(cells, return_sequences=True)
        self.fc = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        """
            Forward pass
        """
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.fc(x)

        return x


def preprocess_data():
    """
        Loads and prepares the data for input
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=n_words,
        skip_words=10)

    return (x_train, y_train), (x_test, y_test)


def main():
    """
        Instanitates and trains Model
    """
    lr = 1e-4
    batch_size = 64
    n_epochs = 10
    steps = 30

    train_data, test_data = preprocess_data()

    model = RNN(units=64, n_layers=2)

    # build
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    # train
    model.fit(*train_data, batch_size=batch_size, epochs=n_epochs,
              validation_split=.15, steps_per_epoch=steps,
              verbose=2)

    # evaluate
    accuracy = model.evaluate(*test_data, batch_size=128)

    print('Accuracy: ', accuracy)


if __name__ == '__main__':
    main()
