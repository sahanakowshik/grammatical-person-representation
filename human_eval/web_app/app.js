const GSHEET_ENDPOINT = "GSHEET_ENDPOINT";
const GSHEET_SECRET = "GSHEET_SECRET";
const CLIENT_ID_KEY = "study_id";



function getClientId() {
  let v = localStorage.getItem(CLIENT_ID_KEY);
  if (!v) {
    v = "c_" + Math.random().toString(16).slice(2) + "_" + Date.now().toString(16);
    localStorage.setItem(CLIENT_ID_KEY, v);
  }
  return v;
}

async function uploadDatasetResultsToSheet(dataset, ratingsObj) {
    const outRows = [];
  
    for (const [rowIndexStr, val] of Object.entries(ratingsObj)) {
      const rowIndex = Number(rowIndexStr);
      const row = rows[rowIndex]; // uses the currently-loaded CSV rows for this model
  
      outRows.push({
        row_index: rowIndex,
        prompt: row ? getPrompt(row) : "",   // <-- add prompt
        label_a: val?.a?.label || "",
        ts_a: val?.a?.ts || "",
        label_b: val?.b?.label || "",
        ts_b: val?.b?.ts || ""
      });
    }
  
    const payload = {
      secret: GSHEET_SECRET,
      client_id: getClientId(),
      dataset_key: dataset.key,
      dataset_label: dataset.label,
      rows: outRows
    };
  
    const res = await fetch(GSHEET_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(payload),
    });
  
    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch { json = { ok: false, error: text }; }
    if (!json.ok) throw new Error(json.error || "upload failed");
    return json;
  }
  
  

const DATASETS_URL = "./datasets.json";

// Stores ratings per dataset:
// ratings key: human_eval_v7::ratings::<datasetKey>
const STORAGE_PREFIX = "human_eval";
const KEY_DONE = `${STORAGE_PREFIX}::done`; // JSON: { [datasetKey]: true/false }

let datasets = [];
let datasetKey = null;
let datasetLabel = null;
let datasetCsv = null;

let rows = [];
let idx = 0;

// ratings[rowIndex] = { a: {label, ts}, b: {label, ts} }
let ratings = {};

const startView = document.getElementById("startView");
const evalView = document.getElementById("evalView");

const elStartStatus = document.getElementById("startStatus");
const elStartHint = document.getElementById("startHint");
const elDatasetSelect = document.getElementById("datasetSelect");
const elStartBtn = document.getElementById("startBtn");

const elFinishEvalBtn = document.getElementById("finishEvalBtn");
// const elDownloadAllBtn = document.getElementById("downloadAllBtn");
const elClearAllBtn = document.getElementById("clearAllBtn");

const elStatus = document.getElementById("status");
const elProgress = document.getElementById("progress");
const elCurrentRow = document.getElementById("currentRow");
const elModelPill = document.getElementById("modelPill");

const elPromptBox = document.getElementById("promptBox");
const elABox = document.getElementById("aBox");
const elBBox = document.getElementById("bBox");
const elAMeta = document.getElementById("aMeta");
const elBMeta = document.getElementById("bMeta");

const elPrevBtn = document.getElementById("prevBtn");
const elNextBtn = document.getElementById("nextBtn");
const elNextUnratedBtn = document.getElementById("nextUnratedBtn");
const elFinishModelBtn = document.getElementById("finishModelBtn");

const elResetBtn = document.getElementById("resetBtn");
const elHint = document.getElementById("hint");

function showView(which) {
  startView.classList.toggle("active", which === "start");
  evalView.classList.toggle("active", which === "eval");
}

function nowUnix() {
  return Math.floor(Date.now() / 1000);
}

function setStartStatus(msg) { elStartStatus.textContent = msg; }
function setStatus(msg) { elStatus.textContent = msg; }

function doneMapLoad() {
  try {
    const raw = localStorage.getItem(KEY_DONE);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function doneMapSave(m) {
  localStorage.setItem(KEY_DONE, JSON.stringify(m));
}

function markDatasetDone(key, val) {
  const m = doneMapLoad();
  m[key] = !!val;
  doneMapSave(m);
}

function isDatasetDone(key) {
  const m = doneMapLoad();
  return !!m[key];
}

function ratingsKeyFor(key) {
  return `${STORAGE_PREFIX}::ratings::${key}`;
}

function loadRatingsFor(key) {
  try {
    const raw = localStorage.getItem(ratingsKeyFor(key));
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveRatingsFor(key, obj) {
  localStorage.setItem(ratingsKeyFor(key), JSON.stringify(obj));
}

function clearRatingsFor(key) {
  localStorage.removeItem(ratingsKeyFor(key));
}

function clearAllProgress() {
  // remove all dataset ratings
  for (const d of datasets) clearRatingsFor(d.key);
  localStorage.removeItem(KEY_DONE);
}

function getRowRating(i) { return ratings[String(i)] || {}; }

function setRowRating(i, side, label) {
  const k = String(i);
  if (!ratings[k]) ratings[k] = {};
  ratings[k][side] = { label, ts: nowUnix() };
  saveRatingsFor(datasetKey, ratings);
}

function isRowComplete(i) {
  const r = getRowRating(i);
  return !!(r.a?.label && r.b?.label);
}

function completedCount() {
  let c = 0;
  for (let i = 0; i < rows.length; i++) if (isRowComplete(i)) c++;
  return c;
}

function allComplete() {
  return rows.length > 0 && completedCount() === rows.length;
}

function updateButtonHighlights(side, chosenLabel) {
  const container = side === "a" ? document.getElementById("aButtons") : document.getElementById("bButtons");
  [...container.querySelectorAll("button")].forEach(b => {
    b.classList.toggle("chosen", chosenLabel && b.dataset.label === chosenLabel);
  });
}

function hasABColumns(fields) {
  const f = new Set(fields || []);
  const hasAB = f.has("response_a") && f.has("response_b");
  const hasLegacy = f.has("positive_response") && f.has("negative_response");
  return { hasAB, hasLegacy };
}

function getPrompt(row) { return String(row.prompt ?? ""); }
function getA(row, mode) { return mode.hasAB ? String(row.response_a ?? "") : String(row.positive_response ?? ""); }
function getB(row, mode) { return mode.hasAB ? String(row.response_b ?? "") : String(row.negative_response ?? ""); }

function getASource(row) { return row.response_a_source ?? ""; }
function getBSource(row) { return row.response_b_source ?? ""; }

function updateUI(mode) {
  if (!rows.length) return;

  const row = rows[idx];

  elModelPill.textContent = `model: ${datasetLabel}`;
  elCurrentRow.textContent = `row: ${idx + 1}/${rows.length}`;

  elPromptBox.textContent = getPrompt(row) || "(empty)";
  elABox.textContent = getA(row, mode) || "(empty)";
  elBBox.textContent = getB(row, mode) || "(empty)";

  const done = completedCount();
  elProgress.textContent = `${done}/${rows.length} rows complete`;

  const r = getRowRating(idx);
  const aLabel = r.a?.label || "";
  const bLabel = r.b?.label || "";

  if (aLabel) { elAMeta.textContent = `Selected: ${aLabel}`; updateButtonHighlights("a", aLabel); }
  else { elAMeta.textContent = "Not rated yet"; updateButtonHighlights("a", null); }

  if (bLabel) { elBMeta.textContent = `Selected: ${bLabel}`; updateButtonHighlights("b", bLabel); }
  else { elBMeta.textContent = "Not rated yet"; updateButtonHighlights("b", null); }

  const completeRow = isRowComplete(idx);
  const isLastRow = idx === rows.length - 1;

  elPrevBtn.disabled = idx === 0;
  elNextUnratedBtn.disabled = false;
  elNextBtn.disabled = !completeRow || isLastRow;

  const finished = allComplete();
  elFinishModelBtn.style.display = (finished && isLastRow) ? "inline-block" : "none";

  if (finished) {
    setStatus("All rows complete");
    elHint.textContent = "Done. Click Finish model to return to model selection.";
  } else if (!aLabel && !bLabel) {
    setStatus("Unrated");
    elHint.textContent = "Rate both Response A and Response B to enable Next.";
  } else if (completeRow) {
    setStatus("Complete");
    elHint.textContent = "Row complete. You can still change selections.";
  } else {
    setStatus("Partially rated");
    elHint.textContent = "Rate both Response A and Response B to enable Next.";
  }

  elResetBtn.disabled = false;
}

function gotoIndex(newIdx, mode) {
  idx = Math.max(0, Math.min(rows.length - 1, newIdx));
  updateUI(mode);
}

function gotoNextUnrated(mode) {
  for (let j = idx; j < rows.length; j++) {
    if (!isRowComplete(j)) { idx = j; updateUI(mode); return; }
  }
  for (let j = 0; j < idx; j++) {
    if (!isRowComplete(j)) { idx = j; updateUI(mode); return; }
  }
  setStatus("All rows complete");
  updateUI(mode);
}

function onRate(side, label, mode) {
  setRowRating(idx, side, label);
  updateUI(mode);
}

function csvEscape(s) {
  const v = String(s ?? "");
  return `"${v.replaceAll('"', '""')}"`;
}

function buildCombinedResultsCSV() {
  // Build one CSV containing all datasets + all rows + labels
  // Includes sources if present in response_a/response_b files.
  const lines = [];
  let includeSources = false;

  // First pass: detect if any dataset has source columns
  for (const d of datasets) {
    const r = loadRatingsFor(d.key);
    if (Object.keys(r).length > 0) {
      // We'll decide sources later based on actual row objects during download; keep it simple:
      // include source columns always; empty if not present.
      includeSources = true;
      break;
    }
  }

  const header = [
    "dataset_key","dataset_label","csv_path",
    "row_index",
    "prompt",
    "response_a","response_b",
    "label_a","ts_a",
    "label_b","ts_b"
  ];
  if (includeSources) header.push("response_a_source","response_b_source");
  lines.push(header.join(","));

  return { lines, includeSources };
}

// async function downloadAllResults() {
//   // We re-load each CSV to include prompt/response text in the export.
//   const { lines, includeSources } = buildCombinedResultsCSV();

//   for (const d of datasets) {
//     const res = await fetch(d.csv, { cache: "no-store" });
//     if (!res.ok) throw new Error(`Failed to fetch ${d.csv} (${res.status})`);
//     const text = await res.text();

//     const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
//     if (parsed.errors?.length) {
//       console.error(parsed.errors);
//       throw new Error(`CSV parse error in ${d.csv}`);
//     }

//     const fields = parsed.meta.fields || [];
//     const mode = hasABColumns(fields);

//     if (!fields.includes("prompt")) throw new Error(`Missing prompt column in ${d.csv}`);
//     if (!(mode.hasAB || mode.hasLegacy)) {
//       throw new Error(`CSV must include response_a/response_b or positive_response/negative_response in ${d.csv}`);
//     }

//     const dataRows = parsed.data;
//     const rmap = loadRatingsFor(d.key);

//     for (let i = 0; i < dataRows.length; i++) {
//       const row = dataRows[i];
//       const r = (rmap[String(i)] || {});
//       const a = r.a || {};
//       const b = r.b || {};

//       const line = [
//         csvEscape(d.key),
//         csvEscape(d.label),
//         csvEscape(d.csv),
//         csvEscape(i),
//         csvEscape(getPrompt(row)),
//         csvEscape(getA(row, mode)),
//         csvEscape(getB(row, mode)),
//         csvEscape(a.label ?? ""),
//         csvEscape(a.ts ?? ""),
//         csvEscape(b.label ?? ""),
//         csvEscape(b.ts ?? "")
//       ];

//       if (includeSources) {
//         line.push(csvEscape(getASource(row)), csvEscape(getBSource(row)));
//       }

//       lines.push(line.join(","));
//     }
//   }

//   const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
//   const url = URL.createObjectURL(blob);
//   const a = document.createElement("a");
//   a.href = url;
//   a.download = "all_models_ratings.csv";
//   document.body.appendChild(a);
//   a.click();
//   a.remove();
//   URL.revokeObjectURL(url);
// }

function refreshStartScreen() {
    const doneMap = doneMapLoad();
  
    [...elDatasetSelect.options].forEach(opt => {
      const key = opt.value;
      const isDone = !!doneMap[key];
  
      if (!opt.dataset.baseLabel) opt.dataset.baseLabel = opt.textContent;
      opt.textContent = isDone ? `${opt.dataset.baseLabel} (completed)` : opt.dataset.baseLabel;
  
      // Optional: don't disable options so user can re-run if you want
      opt.disabled = false;
    });
  
    const doneCount = datasets.filter(d => doneMap[d.key]).length;
    setStartStatus(`Models complete: ${doneCount}/${datasets.length}`);
  
    const allDone = doneCount === datasets.length && datasets.length > 0;
    elFinishEvalBtn.style.display = allDone ? "inline-block" : "none";
    elFinishEvalBtn.disabled = !allDone;
  
    // If currently-selected model is done, jump to first unfinished
    const selectedKey = elDatasetSelect.value;
    const selectedDone = selectedKey ? !!doneMap[selectedKey] : false;
  
    if (selectedDone) {
      const firstUnfinished = [...elDatasetSelect.options].find(o => !doneMap[o.value]);
      if (firstUnfinished) {
        elDatasetSelect.value = firstUnfinished.value;
      }
    }
  
    const nowSelectedKey = elDatasetSelect.value;
    const nowSelectedDone = nowSelectedKey ? !!doneMap[nowSelectedKey] : false;
  
    elStartBtn.disabled = nowSelectedDone;
    elStartBtn.textContent = nowSelectedDone ? "Model completed" : "Start evaluation";
  
    elStartHint.textContent = allDone
      ? "All models are complete. Click Finish evaluation."
      : "Pick a model and click Start evaluation. You can do models in any order.";
}
  

async function loadDatasetsList() {
    setStartStatus("Loading models…");
  
    const res = await fetch(DATASETS_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to fetch ${DATASETS_URL} (${res.status})`);
  
    datasets = await res.json();
    if (!Array.isArray(datasets) || datasets.length === 0) {
      throw new Error("datasets.json is empty or invalid.");
    }
  
    elDatasetSelect.innerHTML = "";
    for (const d of datasets) {
      const opt = document.createElement("option");
      opt.value = d.key;
      opt.textContent = d.label;
      elDatasetSelect.appendChild(opt);
    }
  
    // elDatasetSelect.addEventListener("change", refreshStartScreen);
    elFinishEvalBtn.addEventListener("click", () => {
        elStartHint.textContent = "Evaluation complete. You may now close this page.";
      });
  
    // ✅ DEFINE doneMap HERE
    const doneMap = doneMapLoad();
  
    // Select first unfinished model
    const firstAvailable = [...elDatasetSelect.options].find(o => !doneMap[o.value]);
    if (firstAvailable) {
      elDatasetSelect.value = firstAvailable.value;
    }
  
    elStartBtn.disabled = false;
    elClearAllBtn.style.display = "inline-block";
  
    refreshStartScreen();
  }
  

async function startEvaluation() {
  const chosenKey = elDatasetSelect.value;
  const d = datasets.find(x => x.key === chosenKey);
  if (!d) throw new Error(`Unknown dataset key: ${chosenKey}`);

  datasetKey = d.key;
  datasetLabel = d.label;
  datasetCsv = d.csv;

  showView("eval");
  setStatus(`Loading ${datasetLabel}…`);
  elModelPill.textContent = `model: ${datasetLabel}`;

  const res = await fetch(datasetCsv, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${datasetCsv} (${res.status})`);
  const text = await res.text();

  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
  if (parsed.errors?.length) {
    console.error(parsed.errors);
    throw new Error("CSV parse error. Check console for details.");
  }

  const fields = parsed.meta.fields || [];
  const mode = hasABColumns(fields);

  if (!fields.includes("prompt")) throw new Error("Missing required column: prompt");
  if (!(mode.hasAB || mode.hasLegacy)) {
    throw new Error("CSV must include either (response_a,response_b) or (positive_response,negative_response).");
  }

  rows = parsed.data;
  if (!rows.length) throw new Error("CSV is empty.");

  ratings = loadRatingsFor(datasetKey);
  idx = 0;

  elPrevBtn.onclick = () => gotoIndex(idx - 1, mode);
  elNextBtn.onclick = () => gotoIndex(idx + 1, mode);
  elNextUnratedBtn.onclick = () => gotoNextUnrated(mode);

  document.querySelectorAll("button[data-side][data-label]").forEach(btn => {
    btn.onclick = () => onRate(btn.dataset.side, btn.dataset.label, mode);
  });

  elResetBtn.onclick = () => {
    clearRatingsFor(datasetKey);
    markDatasetDone(datasetKey, false);
    ratings = {};
    idx = 0;
    setStatus("Progress reset");
    updateUI(mode);
  };

  elFinishModelBtn.onclick = async () => {
    elFinishModelBtn.disabled = true;
    try {
      // (Optional safety) only allow finishing if everything is rated
      if (!allComplete()) {
        alert("Please complete all rows before finishing this model.");
        elFinishModelBtn.disabled = false;
        return;
      }
  
      await uploadDatasetResultsToSheet(
        { key: datasetKey, label: datasetLabel },
        loadRatingsFor(datasetKey)
      );
  
      markDatasetDone(datasetKey, true);
  
      // reset eval state + return to model selection page
      rows = [];
      idx = 0;
      ratings = {};
      datasetKey = null;
      datasetLabel = null;
      datasetCsv = null;
  
      showView("start");
      refreshStartScreen();
    } catch (e) {
        elFinishModelBtn.disabled = false;
        alert("Upload failed: " + e.message + "\nPlease try again.");
    }
  };

  elFinishModelBtn.disabled = false;
  gotoNextUnrated(mode);
  setStatus("Ready");
  updateUI(mode);
}

elStartBtn.addEventListener("click", () => {
  startEvaluation().catch(err => {
    console.error(err);
    showView("start");
    setStartStatus("Error");
    elStartHint.textContent = String(err.message || err);
  });
});

elFinishEvalBtn.addEventListener("click", () => {
    elStartHint.textContent = "Evaluation complete. You may now close this page.";
});

// elFinishEvalBtn.addEventListener("click", () => {
//   // Enable download button only after finishing evaluation
//   elDownloadAllBtn.style.display = "inline-block";
//   elDownloadAllBtn.disabled = false;
//   elStartHint.textContent = "You can now download the combined results CSV.";
// });

// elDownloadAllBtn.addEventListener("click", () => {
//   downloadAllResults().catch(err => {
//     console.error(err);
//     setStartStatus("Error");
//     elStartHint.textContent = String(err.message || err);
//   });
// });

elClearAllBtn.addEventListener("click", () => {
  clearAllProgress();
  setStartStatus("Progress cleared");
//   elDownloadAllBtn.style.display = "none";
//   elDownloadAllBtn.disabled = true;
  refreshStartScreen();
});

showView("start");
loadDatasetsList().catch(err => {
  console.error(err);
  setStartStatus("Error");
  elStartHint.textContent = String(err.message || err);
});
