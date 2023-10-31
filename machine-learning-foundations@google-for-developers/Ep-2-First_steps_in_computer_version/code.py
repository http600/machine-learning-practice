
import keras
import tensorflow as tf


class Inject(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print('\n 99%% accuracy stop now')
            self.model.stop_training = True


def run():
    fasion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fasion_mnist.load_data()
    train_images, train_labels = train_images / 255.0, train_labels / 255.0
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                              keras.layers.Dense(128, activation=tf.nn.relu),
                              keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=keras.optimizers.legacy.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=50, callbacks=[Inject()])
    print(model.evaluate(test_images, test_labels))


if __name__ == '__main__':
    run()
