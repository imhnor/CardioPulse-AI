/**
 * app.js — ECGPredict Frontend Logic
 *
 * Handles:
 *  - Drag & drop + file picker upload
 *  - Two-step API flow: /api/upload → /api/predict
 *  - Step-by-step processing status updates
 *  - Result rendering (confidence bars, diagnosis banner, preprocessing info)
 *  - Error display with descriptive messages
 *  - Server health polling
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// Label colours must match backend LABEL_INFO
const LABEL_COLORS = {
  NORM: '#22c55e',
  MI:   '#ef4444',
  STTC: '#f97316',
  CD:   '#a855f7',
  HYP:  '#3b82f6',
};

const CANONICAL_LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'];

// ── State ─────────────────────────────────────────────────────────────────────
let uploadedSessionId  = null;
let uploadedFileInfo   = null;
let isProcessing       = false;

// ── DOM References ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone        = $('dropZone');
const fileInput       = $('fileInput');
const browseBtn       = $('browseBtn');
const uploadCard      = $('uploadCard');
const infoCard        = $('infoCard');
const processingCard  = $('processingCard');
const placeholderCard = $('placeholderCard');
const resultCard      = $('resultCard');
const errorCard       = $('errorCard');
const predictBtn      = $('predictBtn');
const resetBtn        = $('resetBtn');
const resultResetBtn  = $('resultResetBtn');
const errorResetBtn   = $('errorResetBtn');
const serverDot       = $('serverDot');
const serverStatus    = $('serverStatus');

// ── Helpers ───────────────────────────────────────────────────────────────────

function show(el) { el.classList.remove('card--hidden'); }
function hide(el) { el.classList.add('card--hidden'); }

function setStep(stepId, state, desc = '') {
  const step    = $(stepId);
  const iconEl  = step.querySelector('.step-icon');
  const descEl  = $(`${stepId}-desc`);

  // Remove all state classes
  iconEl.className = 'step-icon';

  const icons = {
    waiting: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>`,
    running: `<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-6.219-8.56"/></svg>`,
    done:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
    error:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
  };

  iconEl.innerHTML = icons[state] || icons.waiting;
  iconEl.classList.add(`step-icon--${state}`);

  if (desc) {
    descEl.textContent = desc;
    descEl.className   = 'step-desc ' + (state === 'error' ? 'error' : state === 'done' ? 'done' : '');
  }
}

function resetAllSteps() {
  ['step-upload','step-detect','step-extract','step-preprocess','step-infer'].forEach(id => {
    setStep(id, 'waiting', '—');
  });
}

function formatHz(hz) {
  return `${Math.round(hz)} Hz`;
}

function formatDuration(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(0);
  return `${m}m ${rem}s`;
}

// ── Server Health ─────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (data.status === 'ok') {
      serverDot.className    = 'badge-dot online';
      serverStatus.textContent = data.model_loaded ? 'Model Ready' : 'Server Online (no model)';
    } else {
      throw new Error('bad status');
    }
  } catch {
    serverDot.className      = 'badge-dot offline';
    serverStatus.textContent = 'Server Offline';
  }
}

// Poll every 10 seconds
checkHealth();
setInterval(checkHealth, 10_000);

// ── Upload Flow ───────────────────────────────────────────────────────────────

function handleFiles(files) {
  if (isProcessing) return;
  if (!files || files.length === 0) return;

  // Validate extensions
  const allowed = ['.hea', '.dat', '.dcm', '.xml', '.scp'];
  const invalid = [...files].filter(f => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    return !allowed.includes(ext);
  });

  if (invalid.length > 0) {
    showError(`Unsupported file type: "${invalid[0].name}". Accepted formats: WFDB (.hea+.dat), DICOM (.dcm), XML (.xml), SCP-ECG (.scp)`);
    return;
  }

  uploadFiles(files);
}

async function uploadFiles(files) {
  isProcessing = true;
  resetAllSteps();
  hideAllResults();

  hide(uploadCard);
  show(processingCard);
  $('processingMessage').textContent = 'Uploading ECG file...';

  setStep('step-upload', 'running', 'Sending to server...');

  const formData = new FormData();
  [...files].forEach(f => formData.append('files', f));

  let uploadData;
  try {
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body:   formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed (HTTP ${res.status})`);
    }

    uploadData = await res.json();
  } catch (e) {
    setStep('step-upload', 'error', e.message);
    showError(e.message);
    isProcessing = false;
    return;
  }

  setStep('step-upload', 'done', `${[...files].map(f => f.name).join(' + ')}`);

  // Format detection result
  setStep('step-detect', 'done', uploadData.format_display);

  // Lead extraction result
  const nLeads = uploadData.n_leads;
  setStep('step-extract', 'done', `${nLeads} leads · ${formatHz(uploadData.fs)} · ${formatDuration(uploadData.duration_s)}`);

  // Store session
  uploadedSessionId = uploadData.session_id;
  uploadedFileInfo  = uploadData;

  // Show ECG info card and wait for predict button
  populateInfoCard(uploadData);
  hide(processingCard);
  show(infoCard);
  $('processingMessage').textContent = 'Ready for analysis';
  isProcessing = false;
}

function populateInfoCard(data) {
  $('infoFormatDisplay').textContent = data.format_display;
  $('infoFormat').textContent        = data.format_display;
  $('infoLeads').textContent         = `${data.n_leads} / 12`;
  $('infoFs').textContent            = formatHz(data.fs);
  $('infoDuration').textContent      = formatDuration(data.duration_s);
  $('infoSamples').textContent       = data.n_samples.toLocaleString();

  // Patient info
  const pi = data.patient_info || {};
  const patParts = [];
  if (pi.PatientName && pi.PatientName !== 'None') patParts.push(pi.PatientName);
  if (pi.PatientAge)  patParts.push(`Age ${pi.PatientAge}`);
  if (pi.PatientSex)  patParts.push(pi.PatientSex);
  $('infoPatient').textContent = patParts.length ? patParts.join(' · ') : 'Not available';

  // Lead pills — show canonical 12, grey out missing
  const foundSet    = new Set(data.lead_names.map(l => l.toUpperCase()));
  const pillsEl     = $('leadsPills');
  pillsEl.innerHTML = '';
  CANONICAL_LEADS.forEach(lead => {
    const pill    = document.createElement('span');
    pill.className = 'lead-pill';
    pill.textContent = lead;
    // Try to find this lead in found set
    const found = [...foundSet].some(l => l === lead.toUpperCase());
    if (!found) pill.classList.add('missing');
    pillsEl.appendChild(pill);
  });
}

// ── Predict Flow ──────────────────────────────────────────────────────────────

async function runPrediction() {
  if (!uploadedSessionId || isProcessing) return;
  isProcessing = true;

  hide(infoCard);
  resetAllSteps();
  // Replay completed steps from upload
  setStep('step-upload',  'done', 'File uploaded');
  setStep('step-detect',  'done', uploadedFileInfo.format_display);
  setStep('step-extract', 'done', `${uploadedFileInfo.n_leads} leads · ${formatHz(uploadedFileInfo.fs)}`);
  setStep('step-preprocess', 'running', 'Resampling & normalising...');
  $('processingMessage').textContent = 'Running AI analysis...';
  show(processingCard);

  await sleep(300);
  setStep('step-preprocess', 'done', `100 Hz · z-score norm · (1, 12, 1000)`);
  setStep('step-infer', 'running', 'ResNet1D forward pass...');

  const formData = new FormData();
  formData.append('session_id', uploadedSessionId);

  let predData;
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body:   formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Prediction failed (HTTP ${res.status})`);
    }

    predData = await res.json();
  } catch (e) {
    setStep('step-infer', 'error', e.message);
    showError(e.message);
    isProcessing = false;
    uploadedSessionId = null;
    return;
  }

  setStep('step-infer', 'done', `Done — ${predData.top_diagnosis}`);
  $('processingMessage').textContent = 'Analysis complete!';

  await sleep(400);
  hide(processingCard);
  renderResult(predData);
  isProcessing = false;
  uploadedSessionId = null;
}

function renderResult(data) {
  // ── Diagnosis Banner ──────────────────────────────────────────────────────
  const banner = $('diagnosisBanner');
  const icon   = $('diagnosisIcon');
  const status = $('diagnosisStatus');
  const name   = $('diagnosisName');
  const conf   = $('diagnosisConfidence');

  if (data.is_normal) {
    banner.className = 'diagnosis-banner normal';
    icon.className   = 'diagnosis-icon normal';
    icon.textContent = '✅';
    status.className = 'diagnosis-status normal';
    status.textContent = '✓ Normal';
  } else {
    banner.className = 'diagnosis-banner abnormal';
    icon.className   = 'diagnosis-icon abnormal';
    icon.textContent = '⚠️';
    status.className = 'diagnosis-status abnormal';
    status.textContent = '⚠ Abnormal — Pathology Detected';
  }

  name.textContent = data.top_diagnosis;
  conf.textContent = `Confidence: ${(data.top_confidence * 100).toFixed(1)}%`;

  // ── Probability Bars ──────────────────────────────────────────────────────
  const probaList = $('probaList');
  probaList.innerHTML = '';

  (data.labels_info || []).forEach(label => {
    const item = document.createElement('div');
    item.className = 'proba-item fade-in';

    const pct    = label.percent;
    const active = label.predicted;
    const color  = LABEL_COLORS[label.code] || '#888';

    item.innerHTML = `
      <span class="proba-code">${label.code}</span>
      <div class="proba-bar-wrap">
        <div class="proba-bar" style="width:0%;background:${color}" data-target="${pct}"></div>
      </div>
      <span class="proba-pct">${pct}%</span>
      <span class="proba-badge ${active ? 'active' : 'inactive'}">${active ? 'HIGH' : '—'}</span>
    `;
    probaList.appendChild(item);
  });

  // Animate bars after DOM paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probaList.querySelectorAll('.proba-bar').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    });
  });

  // ── Preprocessing Tags ────────────────────────────────────────────────────
  const prepTags = $('prepTags');
  prepTags.innerHTML = '';
  const prep = data.preprocessing || {};

  const tags = [
    { label: `Input: ${prep.original_fs} Hz`,  highlight: false },
    { label: `→ 100 Hz${prep.resampled ? ' (resampled)' : ''}`, highlight: prep.resampled },
    { label: '1000 samples',                   highlight: false },
    { label: 'z-score normalised',             highlight: false },
    { label: '(1, 12, 1000) tensor',           highlight: false },
  ];
  if (prep.padded)  tags.push({ label: 'Zero-padded',  highlight: true });
  if (prep.trimmed) tags.push({ label: 'Trimmed',      highlight: true });

  tags.forEach(t => {
    const tag       = document.createElement('span');
    tag.className   = 'prep-tag' + (t.highlight ? ' highlight' : '');
    tag.textContent = t.label;
    prepTags.appendChild(tag);
  });

  show(resultCard);
}

// ── Error Display ─────────────────────────────────────────────────────────────

function showError(message) {
  hideAllResults();
  hide(processingCard);
  hide(infoCard);
  show(uploadCard);
  $('errorMessage').textContent = message;
  show(errorCard);
}

function hideAllResults() {
  hide(resultCard);
  hide(errorCard);
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetToUpload() {
  uploadedSessionId = null;
  uploadedFileInfo  = null;
  isProcessing      = false;
  fileInput.value   = '';
  hide(infoCard);
  hide(processingCard);
  hide(resultCard);
  hide(errorCard);
  show(uploadCard);
  show(placeholderCard);
}

// ── Event Listeners ───────────────────────────────────────────────────────────

// Browse button opens file picker
browseBtn.addEventListener('click', () => fileInput.click());

// Click on drop zone also opens file picker
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

// File input change
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

// Drag events
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});
dropZone.addEventListener('dragleave', e => {
  if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('dragging');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  handleFiles(e.dataTransfer.files);
});

// Global drag-over prevention
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop',     e => e.preventDefault());

// Predict button
predictBtn.addEventListener('click', runPrediction);

// Reset buttons
resetBtn.addEventListener('click',       resetToUpload);
resultResetBtn.addEventListener('click', resetToUpload);
errorResetBtn.addEventListener('click',  resetToUpload);

// ── Utility ───────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init ──────────────────────────────────────────────────────────────────────
// Show placeholder on load
show(placeholderCard);
