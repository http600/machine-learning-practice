
# from tensorflow import keras
import keras
import tensorflow as tf


class Inject(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print('\n 99%% accuracy stop now')
            self.model.stop_training = True


def run():
    fasion_mnist = keras.datasets.fashion_mnist
    (X, Y), (x_check, y_check) = fasion_mnist.load_data()
    # X, Y = X / 255.0, Y / 255.0
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=keras.optimizers.legacy.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(X, Y, epochs=50, callbacks=[Inject()])
    print(model.evaluate(x_check, y_check))


if __name__ == '__main__':
    run()
