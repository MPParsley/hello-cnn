'use strict';

let model = null;

// ─── Chart ────────────────────────────────────────────────────────────────────
function drawChart(history) {
  const canvas = document.getElementById('chartCanvas');
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
    ctx.fillStyle = '#9e9eb8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((1 - i / 5).toFixed(1), PAD.left - 6, y + 4);
  }

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

  plotLine(history.acc,     '#6c63ff', 'Train Acc',  PAD.top + 14);
  plotLine(history.val_acc, '#43e97b', 'Val Acc',    PAD.top + 28);
  plotLine(history.loss,    '#ff6584', 'Train Loss', PAD.top + 42);

  ctx.fillStyle = '#9e9eb8';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) {
    ctx.fillText(i + 1, PAD.left + (i / (n - 1)) * w, H - PAD.bottom + 16);
  }
  ctx.fillText('Epoch', PAD.left + w / 2, H - 4);
}

// ─── Model Loading ────────────────────────────────────────────────────────────
async function loadModel() {
  const status = document.getElementById('status');
  const bar    = document.getElementById('progressBar');

  try {
    status.className = 'status training';
    status.textContent = 'Loading model weights…';
    bar.style.width = '30%';

    model = await tf.loadLayersModel('./model/model.json');
    bar.style.width = '70%';

    const res = await fetch('./model/history.json');
    const history = await res.json();
    bar.style.width = '100%';

    const finalAcc = history.val_acc.at(-1);
    status.className = 'status done';
    status.textContent = `Ready — ${(finalAcc * 100).toFixed(1)}% accuracy on MNIST test set`;

    document.getElementById('metricsSection').classList.remove('hidden');
    document.getElementById('metricAcc').textContent    = (history.acc.at(-1)  * 100).toFixed(1) + '%';
    document.getElementById('metricLoss').textContent   = history.loss.at(-1).toFixed(4);
    document.getElementById('metricValAcc').textContent = (finalAcc * 100).toFixed(1) + '%';

    document.getElementById('lossChart').classList.remove('hidden');
    drawChart(history);

    document.getElementById('drawSection').classList.remove('hidden');
  } catch (err) {
    console.error(err);
    status.className = 'status error';
    status.textContent = 'Error loading model: ' + err.message;
  }
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

// ─── Prediction ───────────────────────────────────────────────────────────────
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
    document.getElementById('bar' + i).style.width = (p * 100).toFixed(1) + '%';
    document.getElementById('bar' + i).className = 'pred-bar-fill' + (i === top ? ' top' : '');
    document.getElementById('pct' + i).textContent = (p * 100).toFixed(0) + '%';
  });
}

let predTimeout = null;

function predict() {
  if (!model) return;
  clearTimeout(predTimeout);
  predTimeout = setTimeout(() => {
    tf.tidy(() => {
      const canvas = document.getElementById('drawCanvas');
      const input  = tf.browser.fromPixels(canvas, 1).toFloat()
        .div(255)
        .resizeBilinear([28, 28])
        .reshape([1, 28, 28, 1]);
      const probs = Array.from(model.predict(input).dataSync());
      renderPredBars(probs);
      showFeatureMaps(input);
    });
  }, 30);
}

// ─── Feature Maps ─────────────────────────────────────────────────────────────
async function showFeatureMaps(input) {
  const section   = document.getElementById('featureMapsSection');
  const container = document.getElementById('featureMaps');
  const subModel  = tf.model({ inputs: model.inputs, outputs: model.layers[0].output });
  const fmaps     = subModel.predict(input);
  const [, fH, fW, nFilters] = fmaps.shape;
  const data = await fmaps.array();
  tf.dispose(fmaps);

  container.innerHTML = '';
  for (let f = 0; f < nFilters; f++) {
    const canvas = document.createElement('canvas');
    canvas.width = fW; canvas.height = fH;
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
      img.data[idx] = v; img.data[idx+1] = Math.round(v * 0.4);
      img.data[idx+2] = 255; img.data[idx+3] = 255;
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
  loadModel();
});
