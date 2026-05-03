"""Train the CNN on full MNIST in CI and export to TF.js format."""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

print(f'TensorFlow {tf.__version__}')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(8, 3, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
], name='hello_cnn')

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_test, y_test),
    verbose=1,
)

tfjs.converters.save_keras_model(model, 'model')

with open('model/history.json', 'w') as f:
    json.dump({
        'acc':     [float(v) for v in history.history['accuracy']],
        'loss':    [float(v) for v in history.history['loss']],
        'val_acc': [float(v) for v in history.history['val_accuracy']],
    }, f)

final_acc = history.history['val_accuracy'][-1]
print(f'\nFinal val accuracy: {final_acc:.4f}')
