'use strict';
/**
 * Train the CNN on full MNIST (60 000 examples) and save as a TF.js
 * LayersModel.  Runs in Node.js so the output is natively TF.js format —
 * no Python-to-TF.js converter and no fragile Python ML dependency chain.
 */
const tf   = require('@tensorflow/tfjs-node');
const fs   = require('fs');
const path = require('path');
const zlib = require('zlib');
const https = require('https');

const MNIST_BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const FILES = [
  'train-images-idx3-ubyte.gz',
  'train-labels-idx1-ubyte.gz',
  't10k-images-idx3-ubyte.gz',
  't10k-labels-idx1-ubyte.gz',
];

// ── Download + gunzip ─────────────────────────────────────────────────────────
function download(filename) {
  const dest = path.join('/tmp', filename.replace('.gz', ''));
  if (fs.existsSync(dest)) { console.log(`  cached  ${filename}`); return Promise.resolve(dest); }
  return new Promise((resolve, reject) => {
    console.log(`  fetch   ${MNIST_BASE + filename}`);
    https.get(MNIST_BASE + filename, res => {
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
      const out = fs.createWriteStream(dest);
      res.pipe(zlib.createGunzip()).pipe(out);
      out.on('finish', () => resolve(dest));
      out.on('error', reject);
    }).on('error', reject);
  });
}

// ── Parse IDX binary format ───────────────────────────────────────────────────
function readImages(filepath) {
  const buf = fs.readFileSync(filepath);
  if (buf.readUInt32BE(0) !== 2051) throw new Error('Bad image magic number');
  const n = buf.readUInt32BE(4), rows = buf.readUInt32BE(8), cols = buf.readUInt32BE(12);
  const data = new Float32Array(n * rows * cols);
  for (let i = 0; i < data.length; i++) data[i] = buf[16 + i] / 255;
  return tf.tensor4d(data, [n, rows, cols, 1]);
}

function readLabels(filepath) {
  const buf = fs.readFileSync(filepath);
  if (buf.readUInt32BE(0) !== 2049) throw new Error('Bad label magic number');
  const n = buf.readUInt32BE(4);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) labels[i] = buf[8 + i];
  return tf.oneHot(tf.tensor1d(labels, 'int32'), 10).toFloat();
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`TensorFlow.js ${tf.version.tfjs}`);

  console.log('\nDownloading MNIST...');
  const [trainImgPath, trainLblPath, testImgPath, testLblPath] =
    await Promise.all(FILES.map(download));

  console.log('Parsing...');
  const trainXs = readImages(trainImgPath);
  const trainYs = readLabels(trainLblPath);
  const testXs  = readImages(testImgPath);
  const testYs  = readLabels(testLblPath);
  console.log(`  train ${trainXs.shape}  test ${testXs.shape}`);

  const model = tf.sequential({
    name: 'hello_cnn',
    layers: [
      tf.layers.conv2d({ inputShape: [28,28,1], filters: 8,  kernelSize: 3, activation: 'relu', name: 'conv2d' }),
      tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: 'max_pooling2d' }),
      tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', name: 'conv2d_1' }),
      tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: 'max_pooling2d_1' }),
      tf.layers.flatten({ name: 'flatten' }),
      tf.layers.dense({ units: 64, activation: 'relu', name: 'dense' }),
      tf.layers.dense({ units: 10, activation: 'softmax', name: 'dense_1' }),
    ],
  });

  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  model.summary();

  const history = { acc: [], loss: [], val_acc: [] };

  console.log('\nTraining...');
  await model.fit(trainXs, trainYs, {
    epochs: 10,
    batchSize: 512,
    validationData: [testXs, testYs],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc  = logs.acc     ?? logs.accuracy     ?? 0;
        const vAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        history.acc.push(acc);
        history.loss.push(logs.loss);
        history.val_acc.push(vAcc);
        console.log(`  epoch ${epoch + 1}/10  loss=${logs.loss.toFixed(4)}  acc=${acc.toFixed(4)}  val_acc=${vAcc.toFixed(4)}`);
      },
    },
  });

  tf.dispose([trainXs, trainYs, testXs, testYs]);

  fs.mkdirSync('model', { recursive: true });
  await model.save('file://./model');

  fs.writeFileSync('model/history.json', JSON.stringify({ acc: history.acc, loss: history.loss, val_acc: history.val_acc }));

  console.log(`\nSaved to ./model/`);
  console.log(`Final val accuracy: ${history.val_acc.at(-1).toFixed(4)}`);
}

main().catch(err => { console.error(err); process.exit(1); });
