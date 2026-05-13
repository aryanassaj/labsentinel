/**
 * Early Cancer Signal Detector — Frontend
 * Handles form submission, API calls, and result rendering.
 */

const API_BASE = window.location.origin;

// ── Utility ──────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

// ── Model info (fetched on load) ──────────────────────────────────────────────

let _modelInfo = null;

async function fetchModelInfo() {
  try {
    const resp = await fetch(`${API_BASE}/model/info`);
    if (resp.ok) {
      _modelInfo = await resp.json();
    }
  } catch {
    // silent — not critical
  }
}

// ── Reference ranges for real-time input validation ──────────────────────────

const REF_RANGES = {
  wbc:        { lo: 4.0,  hi: 11.0 },
  hgb_male:   { lo: 13.5, hi: 17.5 },
  hgb_female: { lo: 12.0, hi: 15.5 },
  hct_male:   { lo: 41.0, hi: 53.0 },
  hct_female: { lo: 36.0, hi: 46.0 },
  plt:        { lo: 150,  hi: 400  },
  mcv:        { lo: 80,   hi: 100  },
  rdw:        { lo: 11.5, hi: 14.5 },
  neut_pct:   { lo: 40,   hi: 75   },
  lymph_pct:  { lo: 20,   hi: 45   },
  albumin:    { lo: 3.5,  hi: 5.0  },
  alt:        { lo: 7,    hi: 56   },
  ast:        { lo: 10,   hi: 40   },
  crp:        { lo: 0,    hi: 3.0  },
  bilirubin:  { lo: 0.1,  hi: 1.2  },
  calcium:    { lo: 8.5,  hi: 10.5 },
  ldh:        { lo: 140,  hi: 280  },
};

function getRange(name) {
  const sex = $("sex")?.value || "male";
  if (name === "hgb") return sex === "male" ? REF_RANGES.hgb_male : REF_RANGES.hgb_female;
  if (name === "hct") return sex === "male" ? REF_RANGES.hct_male : REF_RANGES.hct_female;
  return REF_RANGES[name];
}

function validateInput(input) {
  const name = input.name;
  const val = parseFloat(input.value);
  if (!input.value || isNaN(val)) {
    input.classList.remove("abnormal", "borderline");
    return;
  }
  const range = getRange(name);
  if (!range) return;
  if (val < range.lo || val > range.hi) {
    const deviation = Math.max(
      (range.lo - val) / range.lo,
      (val - range.hi) / range.hi
    );
    if (deviation > 0.2) {
      input.classList.add("abnormal");
      input.classList.remove("borderline");
    } else {
      input.classList.add("borderline");
      input.classList.remove("abnormal");
    }
  } else {
    input.classList.remove("abnormal", "borderline");
  }
}

document.querySelectorAll("input[type=number]").forEach(inp => {
  inp.addEventListener("input", () => validateInput(inp));
});
$("sex").addEventListener("change", () => {
  ["hgb", "hct"].forEach(name => {
    const inp = document.querySelector(`input[name="${name}"]`);
    if (inp) validateInput(inp);
  });
});


// ── Form Collection ───────────────────────────────────────────────────────────

function collectFormData() {
  const numericFields = [
    "age", "wbc", "rbc", "hgb", "hct", "plt", "mcv", "mch", "mchc", "rdw",
    "neut_pct", "lymph_pct", "mono_pct", "eos_pct", "baso_pct",
    "glucose", "bun", "creatinine", "sodium", "potassium", "chloride",
    "co2", "calcium", "total_protein", "albumin", "bilirubin",
    "alt", "ast", "alkphos", "uric_acid", "ldh", "phosphorus",
    "ferritin", "iron", "tibc", "crp",
  ];
  const data = { sex: $("sex").value };

  // Include optional patient_id
  const pidEl = $("patient_id");
  if (pidEl && pidEl.value.trim() !== "") {
    data.patient_id = pidEl.value.trim();
  }

  numericFields.forEach(field => {
    const inp = document.getElementById(field);
    if (inp && inp.value !== "") {
      data[field] = parseFloat(inp.value);
    }
  });
  return data;
}

function validateForm(data) {
  if (!data.age || data.age < 18 || data.age > 120) {
    return "Please enter a valid age (18–120).";
  }
  if (!data.sex) {
    return "Please select biological sex.";
  }
  return null;
}


// ── API Call ──────────────────────────────────────────────────────────────────

async function analyzeLabs(data) {
  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `Server error: ${resp.status}`);
  }
  return resp.json();
}


// ── Gauge Rendering ───────────────────────────────────────────────────────────

function drawGauge(canvas, score) {
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height - 4;
  const r = 50;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.lineWidth = 10;
  ctx.strokeStyle = "#e9ecef";
  ctx.stroke();

  // Score arc
  const angle = Math.PI + (Math.PI * Math.min(score / 100, 1));
  const color = score < 10 ? "#16a34a" : score < 20 ? "#d97706" : score < 40 ? "#ea580c" : "#dc2626";
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.strokeStyle = color;
  ctx.stroke();

  // Needle
  const needleAngle = Math.PI + (Math.PI * Math.min(score / 100, 1));
  const nx = cx + r * 0.7 * Math.cos(needleAngle);
  const ny = cy + r * 0.7 * Math.sin(needleAngle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}


// ── Result Rendering ──────────────────────────────────────────────────────────

function renderSubtypes(subtypeRisks) {
  const grid = $("subtypeGrid");
  grid.innerHTML = "";
  const entries = Object.entries(subtypeRisks);
  if (entries.length === 0) {
    hide($("subtypesSection"));
    return;
  }
  show($("subtypesSection"));

  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, pct]) => {
      const color = pct < 10 ? "#16a34a" : pct < 20 ? "#d97706" : "#dc2626";
      const card = document.createElement("div");
      card.className = "subtype-card";
      card.innerHTML = `
        <div class="subtype-name">${name}</div>
        <div class="subtype-pct" style="color:${color}">${pct.toFixed(1)}%</div>
        <div class="subtype-bar">
          <div class="subtype-fill" style="width:${Math.min(pct * 2, 100)}%;background:${color}"></div>
        </div>
      `;
      grid.appendChild(card);
    });
}

function renderDrivers(drivers, containerId, cardClass) {
  const container = $(containerId);
  container.innerHTML = "";

  if (drivers.length === 0) {
    container.innerHTML = '<p style="font-size:12px;color:#6c757d">None identified.</p>';
    return;
  }

  const maxShap = Math.max(...drivers.map(d => Math.abs(d.shap_value)));

  drivers.forEach(driver => {
    const pct = maxShap > 0 ? Math.abs(driver.shap_value) / maxShap * 100 : 0;
    const card = document.createElement("div");
    card.className = `driver-card ${cardClass}`;
    card.innerHTML = `
      <div class="driver-header">
        <span class="driver-name">${escapeHtml(driver.display_name)}</span>
        <span class="driver-value">${formatValue(driver.name, driver.value)}</span>
      </div>
      <div class="shap-bar-wrap">
        <div class="shap-bar">
          <div class="shap-fill" style="width:${pct.toFixed(1)}%"></div>
        </div>
        <span class="shap-label">impact: ${driver.shap_value > 0 ? "+" : ""}${driver.shap_value.toFixed(3)}</span>
      </div>
      <div class="driver-interpretation">${escapeHtml(driver.interpretation)}</div>
    `;
    container.appendChild(card);
  });
}

function formatValue(name, value) {
  const units = {
    wbc: " ×10⁹/L", rbc: " ×10¹²/L", hgb: " g/dL", hct: "%",
    plt: " ×10⁹/L", mcv: " fL", mch: " pg", rdw: "%",
    neut_pct: "%", lymph_pct: "%", mono_pct: "%", eos_pct: "%",
    albumin: " g/dL", alt: " U/L", ast: " U/L", alkphos: " U/L",
    bilirubin: " mg/dL", glucose: " mg/dL", creatinine: " mg/dL",
    calcium: " mg/dL", ldh: " U/L", crp: " mg/L", ferritin: " ng/mL",
    nlr: "", plr: "", sii: "", egfr: " mL/min", age: " yrs",
  };
  const unit = units[name] || "";
  const decimals = Math.abs(value) < 10 ? 2 : Math.abs(value) < 100 ? 1 : 0;
  return value.toFixed(decimals) + unit;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderResults(report, formData) {
  const pct = report.risk_percent;
  const tier = report.risk_tier;

  // Patient ID line
  const pidLine = $("patientIdLine");
  if (formData.patient_id) {
    pidLine.textContent = `Report for Patient: ${escapeHtml(String(formData.patient_id))}`;
    show(pidLine);
  } else {
    hide(pidLine);
  }

  // Risk card
  const card = $("riskCard");
  card.className = `risk-card tier-${tier}`;
  $("riskScore").textContent = pct.toFixed(1) + "%";
  $("riskTier").textContent = tier.charAt(0).toUpperCase() + tier.slice(1) + " Risk";
  $("clinicalSummary").textContent = report.clinical_summary;

  // Operating sensitivity / specificity
  const opEl = $("operatingPoint");
  if (
    report.operating_sensitivity !== undefined &&
    report.operating_sensitivity !== null &&
    report.operating_specificity !== undefined &&
    report.operating_specificity !== null
  ) {
    const sens = Math.round(report.operating_sensitivity * 100);
    const spec = Math.round(report.operating_specificity * 100);
    opEl.textContent = `Model operating point: ${sens}% sensitivity · ${spec}% specificity`;
    show(opEl);
  } else {
    hide(opEl);
  }

  // Gauge
  drawGauge($("riskGauge"), pct);

  // Subtypes
  renderSubtypes(report.subtype_risks || {});

  // Drivers
  renderDrivers(report.top_drivers, "driversList", "risk-driver");
  renderDrivers(report.protective_factors, "protectiveList", "protective-driver");

  // Missing labs
  if (report.missing_labs && report.missing_labs.length > 0) {
    show($("missingSection"));
    $("missingText").textContent =
      `Consider adding: ${report.missing_labs.join(", ")}. ` +
      `These labs improve model accuracy.`;
  } else {
    hide($("missingSection"));
  }

  // Show PDF download button
  show($("downloadPdfBtn"));

  // Model version footer
  const footerEl = $("modelFooter");
  if (_modelInfo) {
    const ver = _modelInfo.version || "1.0.0";
    const participants = _modelInfo.approx_participants || "53,448";
    footerEl.textContent = `Model v${ver} · Trained on NHANES 1999–2018 · ${participants} participants`;
    show(footerEl);
  }
}


// ── PDF Download ──────────────────────────────────────────────────────────────

let _lastFormData = null;

$("downloadPdfBtn").addEventListener("click", async () => {
  if (!_lastFormData) return;

  const btn = $("downloadPdfBtn");
  const origText = btn.textContent;
  btn.textContent = "Generating PDF…";
  btn.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/predict/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_lastFormData),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `Server error: ${resp.status}`);
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const today = new Date().toISOString().slice(0, 10);
    const a = document.createElement("a");
    a.href = url;
    a.download = `labsentinel_report_${today}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

  } catch (err) {
    $("errorMessage").textContent = `PDF generation failed: ${err.message}`;
    show($("errorState"));
  } finally {
    btn.textContent = origText;
    btn.disabled = false;
  }
});


// ── PDF Upload & Auto-fill ────────────────────────────────────────────────────

function setPdfState(state) {
  // state: "idle" | "loading" | "success" | "error"
  const zone = $("pdfUploadZone");
  const inner = $("pdfUploadInner");
  const loading = $("pdfLoading");
  const success = $("pdfSuccess");
  const error = $("pdfError");

  hide(loading); hide(success); hide(error);
  zone.classList.remove("pdf-zone-success", "pdf-zone-error");

  if (state === "idle") {
    show(inner);
  } else if (state === "loading") {
    hide(inner);
    show(loading);
  } else if (state === "success") {
    hide(inner);
    show(success);
    zone.classList.add("pdf-zone-success");
  } else if (state === "error") {
    show(inner);
    show(error);
    zone.classList.add("pdf-zone-error");
  }
}

function autofillForm(extracted) {
  const stringFields = { sex: true, patient_id: true };
  let filled = 0;

  for (const [key, val] of Object.entries(extracted)) {
    const el = document.getElementById(key);
    if (!el) continue;

    if (key === "sex") {
      // Select element
      const opt = [...el.options].find(o => o.value === val.toLowerCase());
      if (opt) { el.value = opt.value; filled++; }
    } else if (key === "patient_id") {
      el.value = String(val);
      filled++;
    } else if (typeof val === "number" && !isNaN(val)) {
      el.value = val;
      validateInput(el);
      filled++;
    }
  }
  return filled;
}

async function handlePdfUpload(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    $("pdfErrorMsg").textContent = "Please select a PDF file.";
    setPdfState("error");
    return;
  }

  setPdfState("loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(`${API_BASE}/parse-labs-pdf`, {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || `Server error: ${resp.status}`);
    }

    const count = autofillForm(data.extracted);
    $("pdfSuccessMsg").textContent = `Filled ${count} field${count !== 1 ? "s" : ""} from "${file.name}"`;
    setPdfState("success");

  } catch (err) {
    $("pdfErrorMsg").textContent = err.message;
    setPdfState("error");
  }
}

// Wire up upload button and file input
$("pdfUploadBtn").addEventListener("click", () => $("pdfFileInput").click());
$("pdfFileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handlePdfUpload(file);
  e.target.value = ""; // reset so same file can be re-selected
});

// Clear buttons
$("pdfClearBtn").addEventListener("click", () => setPdfState("idle"));
$("pdfErrorClearBtn").addEventListener("click", () => {
  $("pdfErrorMsg").textContent = "";
  setPdfState("idle");
});

// Drag and drop
const zone = $("pdfUploadZone");
zone.addEventListener("dragover", (e) => {
  e.preventDefault();
  zone.classList.add("pdf-zone-drag");
});
zone.addEventListener("dragleave", () => zone.classList.remove("pdf-zone-drag"));
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("pdf-zone-drag");
  const file = e.dataTransfer.files[0];
  if (file) handlePdfUpload(file);
});


// ── Form Submission ────────────────────────────────────────────────────────────

$("labForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = collectFormData();
  const validationError = validateForm(data);

  if (validationError) {
    $("errorMessage").textContent = validationError;
    show($("errorState"));
    return;
  }

  // Store for PDF download
  _lastFormData = data;

  // Hide previous states
  hide($("resultsPlaceholder"));
  hide($("resultsContent"));
  hide($("errorState"));
  show($("loadingState"));

  const submitBtn = $("submitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing…";

  try {
    const report = await analyzeLabs(data);

    hide($("loadingState"));
    show($("resultsContent"));
    renderResults(report, data);

  } catch (err) {
    hide($("loadingState"));
    $("errorMessage").textContent = `Analysis failed: ${err.message}`;
    show($("errorState"));
    show($("resultsPlaceholder"));
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      Analyze Lab Panel`;
  }
});


// ── Load model status + info on startup ───────────────────────────────────────

window.addEventListener("load", async () => {
  // Fetch model info for footer
  await fetchModelInfo();

  try {
    const resp = await fetch(`${API_BASE}/health`);
    const data = await resp.json();
    if (!data.model_loaded) {
      $("errorMessage").textContent =
        "Model not loaded — run `python -m src.models.train` to train the model first.";
      show($("errorState"));
    }
  } catch {
    // API not running yet — silent
  }
});
