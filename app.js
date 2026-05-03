'use strict';

// On mobile, use a smaller dataset and batch to stay within memory limits.
const MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

const EPOCHS     = 10;
const BATCH_SIZE = MOBILE ? 64   : 512;
const TRAIN_SIZE = MOBILE ? 3000 : 10_000;
const TEST_SIZE  = MOBILE ? 500  : 2_000;
const IMAGE_SIZE = 784; // 28×28

// ─── MNIST Data Loader ────────────────────────────────────────────────────────
// CI writes a 10 000-train + 2 000-test subset as raw uint8 binaries (~9.5 MB).
// Mobile only needs the first 3 500 rows of the images file.
const MNIST_IMAGES_URL = './data/mnist_images.bin';
const MNIST_LABELS_URL = './data/mnist_labels.bin';
const MNIST_IMAGES_BYTES = (TRAIN_SIZE + TEST_SIZE) * IMAGE_SIZE;

class MnistData {
  constructor(onProgress) {
    this.trainImages = null;
    this.testImages  = null;
    this.trainLabels = null;
    this.testLabels  = null;
    this._onProgress = onProgress || (() => {});
  }

  async load() {
    const [imgBuffer, labelBuffer] = await Promise.all([
      this._stream(MNIST_IMAGES_URL, MNIST_IMAGES_BYTES),
      this._stream(MNIST_LABELS_URL, 0),
    ]);
    const N = TRAIN_SIZE + TEST_SIZE;
    const allImages = new Uint8Array(imgBuffer);
    const allLabels = new Uint8Array(labelBuffer);

    // Keep as Uint8Array — normalization happens per-batch inside tf.tidy().
    this.trainImages = allImages.subarray(0, IMAGE_SIZE * TRAIN_SIZE);
    this.testImages  = allImages.subarray(IMAGE_SIZE * TRAIN_SIZE, IMAGE_SIZE * N);
    this.trainLabels = allLabels.subarray(0, 10 * TRAIN_SIZE);
    this.testLabels  = allLabels.subarray(10 * TRAIN_SIZE, 10 * N);
  }

  async _stream(url, knownBytes) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Fetch failed: ${url} (${res.status})`);
    // Do NOT use content-length: GitHub Pages gzip-encodes responses, so
    // content-length reflects the compressed size while the reader yields
    // decompressed bytes — causing received/total to wildly exceed 100%.
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (knownBytes) this._onProgress(Math.min(received / knownBytes, 1));
    }
    // Concatenate chunks then immediately drop the chunk array so GC can reclaim it.
    const out = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
    chunks.length = 0;
    return out.buffer;
  }

  getTrainBatch(batchSize) {
    return this._getBatch(this.trainImages, this.trainLabels, TRAIN_SIZE, batchSize);
  }

  getTestData(n = 1000) {
    return this._getBatch(this.testImages, this.testLabels, TEST_SIZE, n);
  }

  _getBatch(images, labels, total, n) {
    return tf.tidy(() => {
      const start = Math.floor(Math.random() * (total - n));
      // Slice uint8, create tensor, normalise to [0,1] — only allocates for this batch.
      const xs = tf.tensor(
        images.slice(start * IMAGE_SIZE, (start + n) * IMAGE_SIZE),
        [n, 28, 28, 1], 'int32'
      ).toFloat().div(255);
      const ys = tf.tensor2d(
        labels.slice(start * 10, (start + n) * 10),
        [n, 10]
      );
      return { xs, ys };
    });
  }
}

// ─── Model ───────────────────────────────────────────────────────────────────
function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 16, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

// ─── Chart ───────────────────────────────────────────────────────────────────
const chartData = { trainLoss: [], trainAcc: [], valAcc: [] };

function drawChart() {
  const canvas = document.getElementById('chartCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
  const w = W - PAD.left - PAD.right;
  const h = H - PAD.top - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#1a1d2e';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#2a2d45';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top + (h / 5) * i;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + w, y); ctx.stroke();
    ctx.fillStyle = '#9e9eb8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((1 - i / 5).toFixed(1), PAD.left - 6, y + 4);
  }

  const n = chartData.trainLoss.length;
  if (n < 2) return;

  function plotLine(data, color, label, labelY) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = PAD.left + (i / (n - 1)) * w;
      const y = PAD.top + (1 - Math.min(v, 1)) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, PAD.left + 4, labelY);
  }

  plotLine(chartData.trainAcc,  '#6c63ff', 'Train Acc',  PAD.top + 14);
  plotLine(chartData.valAcc,    '#43e97b', 'Val Acc',    PAD.top + 28);
  plotLine(chartData.trainLoss, '#ff6584', 'Train Loss', PAD.top + 42);

  // X axis labels
  ctx.fillStyle = '#9e9eb8';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    const x = PAD.left + (i / (n - 1)) * w;
    ctx.fillText(i + 1, x, H - PAD.bottom + 16);
  }
  ctx.fillText('Epoch', PAD.left + w / 2, H - 4);
}

// ─── Training ────────────────────────────────────────────────────────────────
let model = null;
let mnist = null;

async function train() {
  const trainBtn = document.getElementById('trainBtn');
  const status   = document.getElementById('status');
  const progress = document.getElementById('progressContainer');
  const bar      = document.getElementById('progressBar');
  const label    = document.getElementById('progressLabel');
  const metrics  = document.getElementById('metrics');
  const chart    = document.getElementById('lossChart');

  trainBtn.disabled = true;
  setStatus('training', 'Loading MNIST data…');
  progress.classList.remove('hidden');
  metrics.classList.remove('hidden');
  chart.classList.remove('hidden');

  try {
    // iOS WebGL is memory-constrained; CPU backend avoids GPU allocation entirely.
    if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
      await tf.setBackend('cpu');
    }
    mnist = new MnistData((pct) => {
      bar.style.width   = (pct * 50) + '%'; // first 50% of bar = download
      label.textContent = `Downloading data… ${Math.round(pct * 100)}%`;
    });
    await mnist.load();
    bar.style.width   = '50%';
    label.textContent = 'Data ready — building model…';
    setStatus('training', 'Training…');

    model = buildModel();
    chartData.trainLoss = [];
    chartData.trainAcc  = [];
    chartData.valAcc    = [];

    // Total batches across all epochs for fine-grained progress
    const batchesPerEpoch = Math.ceil(BATCH_SIZE * 0.9 / 128);
    const totalBatches    = EPOCHS * batchesPerEpoch;
    let   batchesDone     = 0;

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
      // Update label at epoch START so the UI never looks frozen
      label.textContent = `Epoch ${epoch + 1} / ${EPOCHS}`;
      if (epoch === 0) setStatus('training', 'Compiling shaders & training…');
      await tf.nextFrame(); // yield so the browser paints before heavy work

      const batch = mnist.getTrainBatch(BATCH_SIZE);
      const history = await model.fit(batch.xs, batch.ys, {
        batchSize: 128,
        epochs: 1,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
          onBatchEnd: async () => {
            batchesDone++;
            // progress bar: 50% download + 50% training
            bar.style.width = Math.min(50 + (batchesDone / totalBatches) * 50, 100) + '%';
            await tf.nextFrame();
          },
        },
      });
      tf.dispose([batch.xs, batch.ys]);

      const acc  = history.history.acc?.[0]     ?? history.history.accuracy?.[0]     ?? 0;
      const loss = history.history.loss?.[0]    ?? 0;
      const vAcc = history.history.val_acc?.[0] ?? history.history.val_accuracy?.[0] ?? 0;

      chartData.trainAcc.push(acc);
      chartData.trainLoss.push(loss);
      chartData.valAcc.push(vAcc);
      drawChart();

      document.getElementById('metricAcc').textContent    = (acc  * 100).toFixed(1) + '%';
      document.getElementById('metricLoss').textContent   = loss.toFixed(4);
      document.getElementById('metricValAcc').textContent = (vAcc * 100).toFixed(1) + '%';
      setStatus('training', `Training… epoch ${epoch + 1} / ${EPOCHS}`);
    }

    setStatus('done', `Done — val accuracy ${(chartData.valAcc.at(-1) * 100).toFixed(1)}%`);
    document.getElementById('drawSection').classList.remove('hidden');
    document.getElementById('drawSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    console.error(err);
    setStatus('error', 'Error: ' + err.message);
    trainBtn.disabled = false;
  }
}

function setStatus(cls, msg) {
  const el = document.getElementById('status');
  el.className = 'status ' + cls;
  el.textContent = msg;
}

// ─── Drawing Canvas ───────────────────────────────────────────────────────────
function initCanvas() {
  const canvas = document.getElementById('drawCanvas');
  const ctx    = canvas.getContext('2d');
  let drawing  = false;
  let lastX = 0, lastY = 0;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / r.width;
    const scaleY = canvas.height / r.height;
    const cx = ((e.touches ? e.touches[0].clientX : e.clientX) - r.left) * scaleX;
    const cy = ((e.touches ? e.touches[0].clientY : e.clientY) - r.top)  * scaleY;
    return [cx, cy];
  }

  function start(e) {
    e.preventDefault();
    drawing = true;
    [lastX, lastY] = getPos(e);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    predict();
  }

  function draw(e) {
    e.preventDefault();
    if (!drawing) return;
    const [x, y] = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
    predict();
  }

  function stop(e) { e.preventDefault(); drawing = false; }

  canvas.addEventListener('mousedown',  start);
  canvas.addEventListener('mousemove',  draw);
  canvas.addEventListener('mouseup',    stop);
  canvas.addEventListener('mouseleave', stop);
  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove',  draw,  { passive: false });
  canvas.addEventListener('touchend',   stop,  { passive: false });

  document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearPredBars();
    document.getElementById('featureMapsSection').classList.add('hidden');
  });
}

// ─── Prediction ──────────────────────────────────────────────────────────────
function clearPredBars() {
  const container = document.getElementById('predBars');
  container.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'pred-row';
    row.innerHTML = `
      <span class="pred-digit">${i}</span>
      <div class="pred-bar-wrap"><div class="pred-bar-fill" id="bar${i}" style="width:0%"></div></div>
      <span class="pred-pct" id="pct${i}">0%</span>`;
    container.appendChild(row);
  }
}

function renderPredBars(probs) {
  const top = probs.indexOf(Math.max(...probs));
  probs.forEach((p, i) => {
    const fill = document.getElementById('bar' + i);
    const pct  = document.getElementById('pct' + i);
    fill.style.width = (p * 100).toFixed(1) + '%';
    fill.className = 'pred-bar-fill' + (i === top ? ' top' : '');
    pct.textContent = (p * 100).toFixed(0) + '%';
  });
}

let predTimeout = null;

function predict() {
  if (!model) return;
  clearTimeout(predTimeout);
  predTimeout = setTimeout(() => {
    tf.tidy(() => {
      const canvas = document.getElementById('drawCanvas');
      const raw    = tf.browser.fromPixels(canvas, 1).toFloat();
      const scaled = tf.image.resizeBilinear(raw, [28, 28]).div(255);
      const input  = scaled.reshape([1, 28, 28, 1]);
      const probs  = Array.from(model.predict(input).dataSync());
      renderPredBars(probs);
      showFeatureMaps(input);
    });
  }, 30);
}

// ─── Feature Maps ─────────────────────────────────────────────────────────────
async function showFeatureMaps(input) {
  const section = document.getElementById('featureMapsSection');
  const container = document.getElementById('featureMaps');

  const conv1 = model.layers[0];
  const subModel = tf.model({ inputs: model.inputs, outputs: conv1.output });
  const fmaps = subModel.predict(input);
  const [, fH, fW, numFilters] = fmaps.shape;
  const data = await fmaps.array();
  tf.dispose(fmaps);

  container.innerHTML = '';
  for (let f = 0; f < numFilters; f++) {
    const canvas = document.createElement('canvas');
    canvas.width = fW;
    canvas.height = fH;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(fW, fH);
    let min = Infinity, max = -Infinity;
    for (let y = 0; y < fH; y++) for (let x = 0; x < fW; x++) {
      const v = data[0][y][x][f];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    for (let y = 0; y < fH; y++) for (let x = 0; x < fW; x++) {
      const idx = (y * fW + x) * 4;
      const v   = Math.round(((data[0][y][x][f] - min) / range) * 255);
      img.data[idx]     = v;
      img.data[idx + 1] = Math.round(v * 0.4);
      img.data[idx + 2] = 255;
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    container.appendChild(canvas);
  }
  section.classList.remove('hidden');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  clearPredBars();
  initCanvas();
  document.getElementById('trainBtn').addEventListener('click', train);
});
