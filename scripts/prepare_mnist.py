"""Write three MNIST subsets as raw uint8 binaries for the in-browser training option."""
import os
import numpy as np
from PIL import Image

os.makedirs('data', exist_ok=True)

imgs   = np.array(Image.open('/tmp/mnist_images.png').convert('L'), dtype=np.uint8)
labels = np.frombuffer(open('/tmp/mnist_labels_uint8', 'rb').read(), dtype=np.uint8)

# Original layout: rows 0-54999 = train, rows 55000-64999 = test.
TEST_OFFSET = 55_000

for name, n_train, n_test in [('1k', 1_000, 500), ('3k', 3_000, 500), ('10k', 10_000, 2_000)]:
    np.concatenate([imgs[:n_train], imgs[TEST_OFFSET:TEST_OFFSET+n_test]]).tofile(f'data/mnist_images_{name}.bin')
    np.concatenate([labels[:n_train*10], labels[TEST_OFFSET*10:(TEST_OFFSET+n_test)*10]]).tofile(f'data/mnist_labels_{name}.bin')
    print(f'{name}: {(n_train+n_test)*784:,} image bytes, {(n_train+n_test)*10:,} label bytes')
