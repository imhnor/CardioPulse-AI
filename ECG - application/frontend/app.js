'use strict';

// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

const CANONICAL_LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'];

// ── State ────────────────────────────────────────────────────────────────
let heaFile = null;
let datFile = null;
let isProcessing = false;

// ── DOM Helper ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// Inputs
const heaInput = $('heaInput');
const datInput = $('datInput');

// UI Elements
const wfdbStatus = $('wfdbStatus');
const wfdbStatusTxt = $('wfdbStatusText');

const uploadCard = $('uploadCard');
const processingCard = $('processingCard');
const resultCard = $('resultCard');
const errorCard = $('errorCard');
const placeholderCard = $('placeholderCard');

const detectBtn = $('detectBtn');

// File labels (FIXED IDs)
const heaFileName = $('heaFileName');
const datFileName = $('datFileName');

const formatSelect = $('formatSelect');

// ── Helpers ───────────────────────────────────────────────────────────────
function show(el) { el.classList.remove('card--hidden'); }
function hide(el) { el.classList.add('card--hidden'); }

function showError(msg) {
  hide(processingCard);
  hide(resultCard);
  show(uploadCard);

  $('errorMessage').textContent = msg;
  show(errorCard);
}

function updateStatus() {
  const fmt = formatSelect.value;
  wfdbStatus.style.display = 'flex';

  if (fmt === 'WFDB') {
    if (!heaFile || !datFile) {
      wfdbStatusTxt.textContent = 'Waiting for both .hea and .dat files';
      detectBtn.disabled = true;
      return;
    }
    const h = heaFile.name.replace(/\.hea$/i,'').toLowerCase();
    const d = datFile.name.replace(/\.dat$/i,'').toLowerCase();
    if (h !== d) {
      wfdbStatusTxt.textContent = 'File names must match';
      detectBtn.disabled = true;
      return;
    }
  } else {
    if (!heaFile) {
      wfdbStatusTxt.textContent = `Waiting for file`;
      detectBtn.disabled = true;
      return;
    }
  }

  wfdbStatusTxt.textContent = 'Ready — click Detect';
  detectBtn.disabled = false;
}

// ── File Handlers ───────────────────────────────────────────────────────
formatSelect.addEventListener('change', () => {
  const fmt = formatSelect.value;
  if (fmt === 'WFDB') {
    $('datInputContainer').style.display = 'block';
    $('heaInputLabel').textContent = 'Select .hea file';
    heaInput.accept = '.hea';
  } else {
    $('datInputContainer').style.display = 'none';
    const exts = { 'CSV': '.csv', 'MAT': '.mat', 'SCP': '.scp', 'XML': '.xml' };
    $('heaInputLabel').textContent = `Select ${exts[fmt]} file`;
    heaInput.accept = exts[fmt];
    datFile = null;
    $('datFileName').textContent = 'No .dat file selected';
  }
  updateStatus();
});

heaInput.addEventListener('change', (e) => {
  heaFile = e.target.files?.[0] || null;
  heaFileName.textContent = heaFile ? heaFile.name : 'No file selected';
  updateStatus();
});

datInput.addEventListener('change', (e) => {
  datFile = e.target.files?.[0] || null;
  datFileName.textContent = datFile ? datFile.name : 'No .dat file selected';
  updateStatus();
});

// ── Upload + Predict ─────────────────────────────────────────────────────
async function startPipeline() {
  if (isProcessing) return;

  const fmt = formatSelect.value;
  if (fmt === 'WFDB' && (!heaFile || !datFile)) {
    showError("Please select both .hea and .dat files");
    return;
  }
  if (fmt !== 'WFDB' && !heaFile) {
    showError("Please select a file to upload");
    return;
  }

  isProcessing = true;

  hide(errorCard);
  hide(resultCard);
  hide(uploadCard);
  show(processingCard);

  try {
    // ── Upload ─────────────────────────────────────────────
    const formData = new FormData();
    formData.append('files', heaFile, heaFile.name);
    if (fmt === 'WFDB') {
      formData.append('files', datFile, datFile.name);
    }

    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });

    const uploadData = await res.json();

    if (!res.ok) {
      throw new Error(uploadData.detail || 'Upload failed');
    }

    // ── Predict ───────────────────────────────────────────
    const predForm = new FormData();
    predForm.append('session_id', uploadData.session_id);

    const predRes = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: predForm
    });

    const predData = await predRes.json();

    if (!predRes.ok) {
      throw new Error(predData.detail || 'Prediction failed');
    }

    hide(processingCard);
    renderResult(predData);

  } catch (err) {
    showError(err.message);
  }

  isProcessing = false;
}

let currentSignal = null;
let currentWaveView = 'all';

// ── Result ───────────────────────────────────────────────────────────────
function renderResult(data) {
  show(resultCard);


  $('diagnosisName').textContent = data.top_diagnosis || "Unknown";
  $('diagnosisConfidence').textContent =
    ((data.top_confidence || 0) * 100).toFixed(1) + "%";

  const banner = $('diagnosisBanner');
  banner.className = data.is_normal ? 'normal' : 'abnormal';

  if (data.labels_info) {
    const probaList = $('probaList');
    probaList.innerHTML = '';
    data.labels_info.forEach(info => {
      const row = document.createElement('div');
      row.className = 'proba-row';
      row.innerHTML = `
        <span class="proba-name" style="color: ${info.color}">${info.name}</span>
        <div class="proba-bar-bg"><div class="proba-bar-fg" style="width: ${info.percent}%; background: ${info.color}"></div></div>
        <span class="proba-val">${info.percent}%</span>
      `;
      probaList.appendChild(row);
    });
  }

  if (data.preprocessing) {
    const prepTags = $('prepTags');
    prepTags.innerHTML = '';
    const addTag = (text) => {
      const t = document.createElement('span');
      t.className = 'prep-tag';
      t.textContent = text;
      prepTags.appendChild(t);
    };
    if (data.preprocessing.resampled) addTag(`Resampled to ${data.preprocessing.target_fs}Hz`);
    if (data.preprocessing.padded) addTag('Padded to 10s');
    if (data.preprocessing.trimmed) addTag('Trimmed to 10s');
  }

  if (data.signal) {
    currentSignal = data.signal;
    renderWaveform();
  }
}

function setWaveView(view) {
  currentWaveView = view;
  ['All', 'Limb', 'Chest'].forEach(v => {
    const btn = $('btnWave' + v);
    if (btn) btn.classList.remove('active');
  });
  const activeBtn = $('btnWave' + view.charAt(0).toUpperCase() + view.slice(1));
  if (activeBtn) activeBtn.classList.add('active');
  if (currentSignal) renderWaveform();
}
window.setWaveView = setWaveView;

// ── Waveform ─────────────────────────────────────────────────────────────
// ── Waveform ─────────────────────────────────────────────────────────────
function renderWaveform() {
  if (!currentSignal) return;
  const fs = 100;
  const n = currentSignal[0].length;

  // Clinical 12-lead layout with Red Grid
  const traces = [];
  
  if (currentWaveView === 'all') {
    // 3x4 layout + 1 rhythm strip
    const rowOffsets = [15, 10, 5];
    const gridMap = [
      [0, 3, 6, 9],  // I, aVR, V1, V4
      [1, 4, 7, 10], // II, aVL, V2, V5
      [2, 5, 8, 11]  // III, aVF, V3, V6
    ];

    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 4; c++) {
        let leadIdx = gridMap[r][c];
        let startIdx = parseInt(c * 2.5 * fs);
        let endIdx = parseInt((c + 1) * 2.5 * fs);
        
        let x = Array.from({length: endIdx - startIdx}, (_, i) => (startIdx + i) / fs);
        let y = currentSignal[leadIdx].slice(startIdx, endIdx).map(v => v + rowOffsets[r]);
        
        traces.push({ x: x, y: y, mode: 'lines', line: { color: 'black', width: 1.5 }, hoverinfo: 'none' });
        // Lead Label
        traces.push({
          x: [c * 2.5 + 0.1], y: [rowOffsets[r] + 1.2],
          mode: 'text', text: [CANONICAL_LEADS[leadIdx]],
          textposition: 'top right', textfont: { size: 14, color: 'black' }, hoverinfo: 'none'
        });
      }
    }
    // Rhythm Strip (Lead II)
    let xRhythm = Array.from({length: n}, (_, i) => i / fs);
    let yRhythm = currentSignal[1].map(v => v + 0);
    traces.push({ x: xRhythm, y: yRhythm, mode: 'lines', line: { color: 'black', width: 1.5 }, hoverinfo: 'none' });
    traces.push({
      x: [0.1], y: [1.2], mode: 'text', text: ['II'],
      textposition: 'top right', textfont: { size: 14, color: 'black' }, hoverinfo: 'none'
    });

    const gridColor = '#ffcccc';
    const majorGridColor = '#ff9999';

    const layout = {
      title: '12-Lead Clinical ECG Layout (10s)',
      height: 800,
      showlegend: false,
      plot_bgcolor: '#fff',
      paper_bgcolor: '#fff',
      margin: { t: 50, b: 50, l: 30, r: 30 },
      xaxis: {
        range: [0, 10], dtick: 0.2, minor: { dtick: 0.04, gridcolor: gridColor },
        gridcolor: majorGridColor, zeroline: false, showticklabels: true,
        title: 'Time (s)'
      },
      yaxis: {
        range: [-2, 18], dtick: 0.5, minor: { dtick: 0.1, gridcolor: gridColor },
        gridcolor: majorGridColor, zeroline: false, showticklabels: false
      }
    };

    Plotly.react('waveformPlot', traces, layout, { responsive: true, staticPlot: true });

  } else {
    // Limb or Chest (Fallback simple layout)
    let leadsToShow = currentWaveView === 'limb' ? [0,1,2,3,4,5] : [6,7,8,9,10,11];
    const layout = {
      height: leadsToShow.length * 150,
      showlegend: false,
      grid: { rows: leadsToShow.length, columns: 1, pattern: 'independent' },
    };
    let t = Array.from({ length: n }, (_, i) => i / fs);
    
    leadsToShow.forEach((leadIdx, i) => {
      traces.push({
        x: t, y: currentSignal[leadIdx], mode: 'lines', line: { color: '#1f77b4', width: 1.5 },
        xaxis: 'x', yaxis: 'y' + (i === 0 ? '' : (i + 1))
      });
      layout['yaxis' + (i === 0 ? '' : (i + 1))] = { title: CANONICAL_LEADS[leadIdx], zeroline: false };
      layout['xaxis' + (i === 0 ? '' : (i + 1))] = { zeroline: false };
    });
    Plotly.react('waveformPlot', traces, layout, { responsive: true });
  }
}

function downloadECG() {
  if (!currentSignal) return;
  Plotly.downloadImage('waveformPlot', {
    format: 'png',
    width: 1200,
    height: currentWaveView === 'all' ? 1600 : 800,
    filename: 'ECGPredict_Report'
  });
}
window.downloadECG = downloadECG;

// ── Events ───────────────────────────────────────────────────────────────
detectBtn.addEventListener('click', startPipeline);

$('resultResetBtn').addEventListener('click', () => {
  hide(resultCard);
  show(uploadCard);
  heaFile = null;
  datFile = null;
  heaInput.value = '';
  datInput.value = '';
  heaFileName.textContent = 'No file selected';
  datFileName.textContent = 'No .dat file selected';
  updateStatus();
});

$('errorResetBtn').addEventListener('click', () => {
  hide(errorCard);
  show(uploadCard);
});

// ── Init ────────────────────────────────────────────────────────────────
show(placeholderCard);