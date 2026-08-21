const dropzone = document.getElementById('dropzone');
const dropzoneText = document.getElementById('dropzone-text');
const fileInput = document.getElementById('file-input');
const submitBtn = document.getElementById('submit-btn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const overallBadge = document.getElementById('overall-badge');
const stageList = document.getElementById('stage-list');
const pipelineNote = document.getElementById('pipeline-note');
const demoToggleWrap = document.getElementById('demo-toggle-wrap');
const demoToggle = document.getElementById('demo-toggle');

let selectedFile = null;
let backendReady = false;

// ---- Pipeline status ----
async function loadStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();

    const stages = data.stages || {};
    for (const li of stageList.querySelectorAll('li')) {
      const key = li.dataset.stage;
      const ready = !!stages[key];
      li.classList.toggle('ready', ready);
      li.classList.toggle('pending', !ready);
    }

    backendReady = !!data.ready;
    if (backendReady) {
      overallBadge.textContent = 'Model ready';
      overallBadge.className = 'badge ready';
      pipelineNote.textContent = 'All checkpoints loaded — predictions below are from the live model.';
      demoToggleWrap.style.display = 'none';
      demoToggle.checked = false;
    } else {
      overallBadge.textContent = 'Training in progress';
      overallBadge.className = 'badge pending';
      pipelineNote.textContent = data.message ||
        'The fusion classifier is still being trained (pending GPU access). ' +
        'Use demo mode below to preview the interface with a simulated result — ' +
        'no code changes will be needed once the real checkpoints are dropped in.';
      demoToggleWrap.style.display = 'flex';
      demoToggle.checked = true;
    }
  } catch (err) {
    overallBadge.textContent = 'Status unavailable';
    overallBadge.className = 'badge error';
    pipelineNote.textContent = 'Could not reach the backend status endpoint: ' + err;
    demoToggleWrap.style.display = 'flex';
    demoToggle.checked = true;
  }
}
loadStatus();

// ---- File selection ----
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  dropzoneText.textContent = `Selected: ${file.name}`;
  submitBtn.disabled = false;
  resultEl.style.display = 'none';
  statusEl.textContent = '';
}

// ---- Submit ----
submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  submitBtn.disabled = true;
  resultEl.style.display = 'none';

  const useDemo = demoToggle.checked || !backendReady;
  statusEl.textContent = useDemo
    ? 'Running demo pipeline (simulated result)...'
    : 'Analyzing... this can take a little while on CPU.';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const endpoint = useDemo ? '/predict/demo' : '/predict';
    const response = await fetch(endpoint, { method: 'POST', body: formData });
    const data = await response.json();

    if (!response.ok) {
      statusEl.textContent = `Error: ${data.detail || 'unknown error'}`;
      submitBtn.disabled = false;
      return;
    }

    statusEl.textContent = '';
    resultEl.className = data.prediction;
    resultEl.style.display = 'block';
    const pFake = (data.fake_probability * 100).toFixed(1);
    const pReal = (data.real_probability * 100).toFixed(1);
    resultEl.innerHTML =
      '<div class="label">' +
      (data.prediction === 'fake' ? 'Likely AI-generated / manipulated' : 'Likely authentic') +
      '</div>' +
      '<div class="prob">P(fake) = ' + pFake + '% &middot; P(real) = ' + pReal + '%</div>' +
      (data.demo ? '<div class="demo-tag">Simulated result — demo mode</div>' : '');
  } catch (err) {
    statusEl.textContent = `Request failed: ${err}`;
  } finally {
    submitBtn.disabled = false;
  }
});
