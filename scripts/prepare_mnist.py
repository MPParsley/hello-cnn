"""Download and subset MNIST into compact uint8 binary files for the demo."""
import os
import numpy as np
from PIL import Image

N_TRAIN, N_TEST = 10_000, 2_000

os.makedirs('data', exist_ok=True)

imgs = np.array(Image.open('/tmp/mnist_images.png').convert('L'), dtype=np.uint8)
train_imgs = imgs[:N_TRAIN]
test_imgs  = imgs[55_000 : 55_000 + N_TEST]
np.concatenate([train_imgs, test_imgs]).tofile('data/mnist_images.bin')

labels = np.frombuffer(open('/tmp/mnist_labels_uint8', 'rb').read(), dtype=np.uint8)
train_labels = labels[:N_TRAIN * 10]
test_labels  = labels[55_000 * 10 : (55_000 + N_TEST) * 10]
np.concatenate([train_labels, test_labels]).tofile('data/mnist_labels.bin')

print(f'images: {(N_TRAIN + N_TEST) * 784:,} bytes')
print(f'labels: {(N_TRAIN + N_TEST) * 10:,} bytes')
