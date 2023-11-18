import tensorflow as tf


class Inject(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print('\n 99%% accuracy stop now')
            self.model.stop_training = True


def run():
    fasion_mnist = tf.keras.datasets.mnist
    (X, Y), (x_check, y_check) = fasion_mnist.load_data()
    X, x_check = X / 255.0, x_check / 255.0
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(
                                     512, activation=tf.nn.relu),
                                 tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=10, callbacks=[Inject()])


if __name__ == '__main__':
    run()
