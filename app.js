'use strict';

const IMAGE_SIZE = 784;
const EPOCHS     = 10;
const MOBILE     = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

const TRAIN_CONFIGS = {
  '1k':  { trainSize: 1_000, testSize:   500, batchSize: 128, imageUrl: './data/mnist_images_1k.bin',  labelUrl: './data/mnist_labels_1k.bin'  },
  '3k':  { trainSize: 3_000, testSize:   500, batchSize: 256, imageUrl: './data/mnist_images_3k.bin',  labelUrl: './data/mnist_labels_3k.bin'  },
  '10k': { trainSize: 10_000, testSize: 2_000, batchSize: 512, imageUrl: './data/mnist_images_10k.bin', labelUrl: './data/mnist_labels_10k.bin' },
};

let model = null;

// ─── Shared chart ─────────────────────────────────────────────────────────────
function drawChart(canvasId, history) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 20, right: 20, bottom: 40, left: 50 };
  const w = W - PAD.left - PAD.right;
  const h = H - PAD.top - PAD.bottom;
  const n = history.acc.length;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1a1d2e';
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = '#2a2d45';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = PAD.top + (h / 5) * i;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + w, y); ctx.stroke();
    ctx.fillStyle = '#9e9eb8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((1 - i / 5).toFixed(1), PAD.left - 6, y + 4);
  }

  function plotLine(data, color, label, labelY) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = PAD.left + (i / (n - 1)) * w;
      const y = PAD.top + (1 - Math.min(v, 1)) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = color; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(label, PAD.left + 4, labelY);
  }

  plotLine(history.acc,     '#6c63ff', 'Train Acc',  PAD.top + 14);
  plotLine(history.val_acc, '#43e97b', 'Val Acc',    PAD.top + 28);
  plotLine(history.loss,    '#ff6584', 'Train Loss', PAD.top + 42);

  ctx.fillStyle = '#9e9eb8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    ctx.fillText(i + 1, PAD.left + (i / (n - 1)) * w, H - PAD.bottom + 16);
  }
  ctx.fillText('Epoch', PAD.left + w / 2, H - 4);
}

// ─── Pre-trained mode ─────────────────────────────────────────────────────────
async function runPretrained() {
  const status = document.getElementById('preStatus');
  const bar    = document.getElementById('preProgressBar');

  try {
    status.className = 'status training';
    status.textContent = 'Loading model weights…';
    bar.style.width = '30%';

    model = await tf.loadLayersModel('./model/model.json');
    bar.style.width = '70%';

    const history = await fetch('./model/history.json').then(r => r.json());
    bar.style.width = '100%';

    const finalAcc = history.val_acc.at(-1);
    status.className = 'status done';
    status.textContent = `Ready — ${(finalAcc * 100).toFixed(1)}% accuracy on MNIST test set`;

    document.getElementById('preMetrics').classList.remove('hidden');
    document.getElementById('preMetricAcc').textContent    = (history.acc.at(-1)  * 100).toFixed(1) + '%';
    document.getElementById('preMetricLoss').textContent   = history.loss.at(-1).toFixed(4);
    document.getElementById('preMetricValAcc').textContent = (finalAcc * 100).toFixed(1) + '%';

    document.getElementById('preLossChart').classList.remove('hidden');
    drawChart('preChartCanvas', history);

    showDrawSection();
  } catch (err) {
    console.error(err);
    status.className = 'status error';
    status.textContent = 'Error: ' + err.message;
  }
}

// ─── In-browser training mode ─────────────────────────────────────────────────
async function streamBinary(url, knownBytes, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${url} (${res.status})`);
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (knownBytes) onProgress(Math.min(received / knownBytes, 1));
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  chunks.length = 0;
  return out;
}

async function runTraining() {
  const cfg = TRAIN_CONFIGS[document.querySelector('input[name="trainSize"]:checked').value];
  const { trainSize, testSize, batchSize } = cfg;
  const totalImages = (trainSize + testSize) * IMAGE_SIZE;

  const status    = document.getElementById('trainStatus');
  const bar       = document.getElementById('trainProgressBar');
  const label     = document.getElementById('trainProgressLabel');
  const container = document.getElementById('trainProgressContainer');
  const metrics   = document.getElementById('trainMetrics');
  const chart     = document.getElementById('trainLossChart');
  const trainBtn  = document.getElementById('trainBtn');

  trainBtn.disabled = true;
  container.classList.remove('hidden');
  metrics.classList.remove('hidden');
  chart.classList.remove('hidden');

  const history = { acc: [], loss: [], val_acc: [] };

  try {
    status.className = 'status training';
    status.textContent = 'Downloading data…';

    const [imgBytes, lblBytes] = await Promise.all([
      streamBinary(cfg.imageUrl, totalImages, p => {
        bar.style.width   = (p * 50) + '%';
        label.textContent = `Downloading… ${Math.round(p * 100)}%`;
      }),
      streamBinary(cfg.labelUrl, 0, () => {}),
    ]);

    bar.style.width   = '50%';
    label.textContent = 'Building model…';
    status.textContent = 'Training…';
    await tf.nextFrame();

    const allImages = imgBytes;
    const allLabels = lblBytes;

    model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 8,  activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.conv2d({ kernelSize: 3, filters: 16, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const batchesPerEpoch = Math.ceil(trainSize * 0.9 / batchSize);
    const totalBatches    = EPOCHS * batchesPerEpoch;
    let   batchesDone     = 0;

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
      label.textContent = `Epoch ${epoch + 1} / ${EPOCHS}`;
      await tf.nextFrame();

      const start = Math.floor(Math.random() * (trainSize - batchSize));
      const xs = tf.tidy(() =>
        tf.tensor(allImages.slice(start * IMAGE_SIZE, (start + batchSize) * IMAGE_SIZE),
          [batchSize, 28, 28, 1], 'int32').toFloat().div(255)
      );
      const ys = tf.tidy(() =>
        tf.tensor2d(allLabels.slice(start * 10, (start + batchSize) * 10), [batchSize, 10])
      );

      const h = await model.fit(xs, ys, {
        batchSize: Math.min(128, batchSize),
        epochs: 1,
        validationSplit: 0.1,
        shuffle: true,
        callbacks: {
          onBatchEnd: async () => {
            batchesDone++;
            bar.style.width = Math.min(50 + (batchesDone / totalBatches) * 50, 100) + '%';
            await tf.nextFrame();
          },
        },
      });
      tf.dispose([xs, ys]);

      const acc  = h.history.acc?.[0]     ?? h.history.accuracy?.[0]     ?? 0;
      const loss = h.history.loss?.[0]    ?? 0;
      const vAcc = h.history.val_acc?.[0] ?? h.history.val_accuracy?.[0] ?? 0;

      history.acc.push(acc);
      history.loss.push(loss);
      history.val_acc.push(vAcc);
      drawChart('trainChartCanvas', history);

      document.getElementById('trainMetricAcc').textContent    = (acc  * 100).toFixed(1) + '%';
      document.getElementById('trainMetricLoss').textContent   = loss.toFixed(4);
      document.getElementById('trainMetricValAcc').textContent = (vAcc * 100).toFixed(1) + '%';
      status.textContent = `Training… epoch ${epoch + 1} / ${EPOCHS}`;
    }

    status.className = 'status done';
    status.textContent = `Done — ${(history.val_acc.at(-1) * 100).toFixed(1)}% val accuracy`;
    showDrawSection();
  } catch (err) {
    console.error(err);
    status.className = 'status error';
    status.textContent = 'Error: ' + err.message;
    trainBtn.disabled = false;
  }
}

// ─── Drawing Canvas ───────────────────────────────────────────────────────────
function showDrawSection() {
  const sec = document.getElementById('drawSection');
  sec.classList.remove('hidden');
  sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

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
    const cx = ((e.touches ? e.touches[0].clientX : e.clientX) - r.left) * (canvas.width  / r.width);
    const cy = ((e.touches ? e.touches[0].clientY : e.clientY) - r.top)  * (canvas.height / r.height);
    return [cx, cy];
  }

  function start(e) {
    e.preventDefault(); drawing = true;
    [lastX, lastY] = getPos(e);
    ctx.beginPath(); ctx.arc(lastX, lastY, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill();
    predict();
  }
  function draw(e) {
    e.preventDefault(); if (!drawing) return;
    const [x, y] = getPos(e);
    ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
    [lastX, lastY] = [x, y]; predict();
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
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearPredBars();
    document.getElementById('featureMapsSection').classList.add('hidden');
  });
}

// ─── Prediction ───────────────────────────────────────────────────────────────
function clearPredBars() {
  const c = document.getElementById('predBars');
  c.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    c.insertAdjacentHTML('beforeend', `
      <div class="pred-row">
        <span class="pred-digit">${i}</span>
        <div class="pred-bar-wrap"><div class="pred-bar-fill" id="bar${i}" style="width:0%"></div></div>
        <span class="pred-pct" id="pct${i}">0%</span>
      </div>`);
  }
}

let predTimeout = null;
function predict() {
  if (!model) return;
  clearTimeout(predTimeout);
  predTimeout = setTimeout(() => {
    tf.tidy(() => {
      const input = tf.browser.fromPixels(document.getElementById('drawCanvas'), 1)
        .toFloat().div(255).resizeBilinear([28, 28]).reshape([1, 28, 28, 1]);
      const probs = Array.from(model.predict(input).dataSync());
      const top   = probs.indexOf(Math.max(...probs));
      probs.forEach((p, i) => {
        document.getElementById('bar' + i).style.width    = (p * 100).toFixed(1) + '%';
        document.getElementById('bar' + i).className      = 'pred-bar-fill' + (i === top ? ' top' : '');
        document.getElementById('pct' + i).textContent    = (p * 100).toFixed(0) + '%';
      });
      showFeatureMaps(input);
    });
  }, 30);
}

async function showFeatureMaps(input) {
  const section   = document.getElementById('featureMapsSection');
  const container = document.getElementById('featureMaps');
  const sub   = tf.model({ inputs: model.inputs, outputs: model.layers[0].output });
  const fmaps = sub.predict(input);
  const [, fH, fW, nF] = fmaps.shape;
  const data = await fmaps.array();
  tf.dispose(fmaps);
  container.innerHTML = '';
  for (let f = 0; f < nF; f++) {
    const c = document.createElement('canvas');
    c.width = fW; c.height = fH;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(fW, fH);
    let mn = Infinity, mx = -Infinity;
    for (let y = 0; y < fH; y++) for (let x = 0; x < fW; x++) {
      const v = data[0][y][x][f]; if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const rng = mx - mn || 1;
    for (let y = 0; y < fH; y++) for (let x = 0; x < fW; x++) {
      const idx = (y * fW + x) * 4;
      const v   = Math.round(((data[0][y][x][f] - mn) / rng) * 255);
      img.data[idx] = v; img.data[idx+1] = Math.round(v * 0.4);
      img.data[idx+2] = 255; img.data[idx+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    container.appendChild(c);
  }
  section.classList.remove('hidden');
}

// ─── Mode toggle ──────────────────────────────────────────────────────────────
function initModeToggle() {
  if (MOBILE) {
    document.getElementById('mobileWarning').classList.remove('hidden');
  }

  document.getElementById('modePreBtn').addEventListener('click', () => {
    document.getElementById('modePreBtn').classList.add('active');
    document.getElementById('modeTrainBtn').classList.remove('active');
    document.getElementById('prePanel').classList.remove('hidden');
    document.getElementById('trainPanel').classList.add('hidden');
  });

  document.getElementById('modeTrainBtn').addEventListener('click', () => {
    document.getElementById('modeTrainBtn').classList.add('active');
    document.getElementById('modePreBtn').classList.remove('active');
    document.getElementById('trainPanel').classList.remove('hidden');
    document.getElementById('prePanel').classList.add('hidden');
  });

  document.getElementById('trainBtn').addEventListener('click', runTraining);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  clearPredBars();
  initCanvas();
  initModeToggle();
  runPretrained(); // start loading pre-trained model automatically
});
