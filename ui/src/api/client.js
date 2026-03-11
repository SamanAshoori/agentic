const API_BASE = "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function getStatus() {
  return request("/status");
}

export async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Upload failed");
  }
  return res.json();
}

export async function getColumns() {
  return request("/columns");
}

export async function analyzeTarget(target) {
  return request("/analyze/target", {
    method: "POST",
    body: JSON.stringify({ target }),
  });
}

export async function runStats() {
  return request("/run/stats", { method: "POST" });
}

export async function runModel(payload) {
  return request("/run/model", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runEvaluate() {
  return request("/run/evaluate", { method: "POST" });
}

export async function confirmStage(stage, payload) {
  return request(`/confirm/${stage}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getOutput(stage) {
  return request(`/output/${stage}`);
}

export async function resetSession() {
  return request("/reset", { method: "POST" });
}

export async function runDescriptives(target) {
  return request("/run/descriptives", {
    method: "POST",
    body: JSON.stringify({ target }),
  });
}

export async function runLogistic(payload) {
  return request("/run/logistic", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function downloadReport() {
  const a = document.createElement("a");
  a.href = `${API_BASE}/download/report`;
  a.download = "ml_pipeline_report.html";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
