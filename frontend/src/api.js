// Thin wrappers around the FastAPI backend contract. Every page imports
// from here rather than calling fetch() directly, so if the backend's
// routes ever change shape, this is the one file that needs updating.

export async function getStatus() {
  const res = await fetch('/status');
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

export async function getMetrics() {
  const res = await fetch('/metrics');
  if (!res.ok) throw new Error(`metrics ${res.status}`);
  return res.json();
}

async function postVideo(endpoint, file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(endpoint, { method: 'POST', body: formData });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || `request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

export function predict(file) {
  return postVideo('/predict', file);
}

export function predictDemo(file) {
  return postVideo('/predict/demo', file);
}
