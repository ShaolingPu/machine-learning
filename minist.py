import keras
from keras.datasets import mnist
import numpy as np

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28 * 28))
x_test = x_test.astype('float32') / 255


from keras import models, layers
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28, )))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(x_train, y_train, epochs = 5, batch_size = 128)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc:', test_acc )
