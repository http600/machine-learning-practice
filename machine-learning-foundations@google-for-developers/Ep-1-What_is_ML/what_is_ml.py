
import keras
import numpy as np


def run():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    X = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
    Y = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
    model.fit(X, Y, epochs=500)
    print(model.predict([10]))


if __name__ == '__main__':
    run()
