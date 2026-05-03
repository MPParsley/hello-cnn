"""Train the CNN on full MNIST and write TF.js LayersModel files directly.

We intentionally do NOT import tensorflowjs: that package transitively pulls in
jax (jax2tf) and uses removed NumPy aliases (np.object), causing import errors
across Python/NumPy/JAX version combinations. Generating the two TF.js files
(model.json + group1-shard1of1.bin) from pure Keras + NumPy is straightforward.
"""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f'TensorFlow {tf.__version__}  Keras {keras.__version__}')

os.makedirs('model', exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# ── Model ─────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(8,  3, activation='relu', name='conv2d'),
    keras.layers.MaxPooling2D(2, name='max_pooling2d'),
    keras.layers.Conv2D(16, 3, activation='relu', name='conv2d_1'),
    keras.layers.MaxPooling2D(2, name='max_pooling2d_1'),
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(64, activation='relu', name='dense'),
    keras.layers.Dense(10, activation='softmax', name='dense_1'),
], name='hello_cnn')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_test, y_test),
    verbose=1,
)

# ── Export: TF.js LayersModel format ─────────────────────────────────────────
# Collect trainable variables with TF.js-compatible names:
#   hello_cnn/conv2d/kernel:0  →  conv2d/kernel
named_weights = []
for var in model.trainable_variables:
    name = var.name.split(':')[0]                    # drop ':0'
    parts = name.split('/')
    if parts[0] == model.name:
        parts = parts[1:]                            # drop model-name prefix
    named_weights.append(('/'.join(parts), var.numpy()))

# Binary shard: all weights serialised as float32, in order.
shard = bytearray()
for _, arr in named_weights:
    shard.extend(arr.flatten().astype(np.float32).tobytes())

with open('model/group1-shard1of1.bin', 'wb') as f:
    f.write(shard)

model_json = {
    'format': 'layers-model',
    'generatedBy': f'keras {keras.__version__}',
    'convertedBy': None,
    'modelTopology': json.loads(model.to_json()),
    'weightsManifest': [{
        'paths': ['group1-shard1of1.bin'],
        'weights': [
            {'name': name, 'shape': list(arr.shape), 'dtype': 'float32'}
            for name, arr in named_weights
        ],
    }],
}

with open('model/model.json', 'w') as f:
    json.dump(model_json, f)

# ── Training history for the browser chart ────────────────────────────────────
with open('model/history.json', 'w') as f:
    json.dump({
        'acc':     [float(v) for v in history.history['accuracy']],
        'loss':    [float(v) for v in history.history['loss']],
        'val_acc': [float(v) for v in history.history['val_accuracy']],
    }, f)

print(f'\nFinal val accuracy: {history.history["val_accuracy"][-1]:.4f}')
print(f'Weights shard:      {len(shard):,} bytes')
