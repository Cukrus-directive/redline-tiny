// redline-tiny web UI: talks to the local server (redline/server.py). No frameworks, works offline.
"use strict";
const $ = (id) => document.getElementById(id);
const state = { mode: "edit", image: null, imageName: "", crop: "center", steps: 8, busy: false, ready: false,
                current: null, history: [], limits: { edit_tokens: 64 }, tokenTimer: 0, tokensOk: true };

// ---------- server status ----------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function setStatus(cls, text) {
  $("status").className = "status " + cls;
  $("status-text").textContent = text;
}

async function poll() {
  try {
    const s = await api("/api/status");
    if (s.limits) state.limits = s.limits;
    state.ready = s.state === "ready" || s.state === "busy";
    if (s.state === "loading") setStatus("loading", s.message || "Loading models…");
    else if (s.state === "error") setStatus("error", s.message || "Error");
    else if (s.state === "busy") setStatus("busy", "Working on the NPU");
    else setStatus("ready", `NPU ready${s.load_s ? ` (loaded in ${Math.round(s.load_s)} s)` : ""}`);
    if (state.busy && s.job) showProgress(s.job);
  } catch {
    state.ready = false;
    setStatus("error", "Server not running: start it with “Start redline-tiny”");
  }
  updateGo();
  setTimeout(poll, state.busy ? 700 : 2500);
}

// ---------- mode, inputs ----------
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.mode === mode;
    t.classList.toggle("active", on); t.setAttribute("aria-selected", on);
  });
  const edit = mode === "edit";
  $("drop").hidden = !edit;
  $("crop-row").hidden = !edit || !state.image;
  $("prompt-label").textContent = edit ? "What should change?" : "Describe the image";
  $("prompt").placeholder = edit ? "e.g. make the car bright red" : "e.g. a kookaburra on a snow gum branch, misty mountains behind";
  $("again").hidden = false;
  checkTokens();
  updateGo();
}

function loadFile(f) {
  if (!f || !f.type.startsWith("image/")) return showError("That isn't an image file.");
  const rd = new FileReader();
  rd.onload = () => {
    state.image = rd.result; state.imageName = f.name;
    $("drop-preview").src = rd.result; $("drop-preview").hidden = false; $("drop-empty").hidden = true;
    $("crop-row").hidden = false;
    updateGo();
  };
  rd.readAsDataURL(f);
}

function segment(id, key, cast) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    state[key] = cast(b.dataset.v);
  });
}

function checkTokens() {
  clearTimeout(state.tokenTimer);
  const p = $("prompt").value.trim();
  const c = $("counter");
  if (!p) { c.innerHTML = "&nbsp;"; c.classList.remove("over"); state.tokensOk = true; updateGo(); return; }
  state.tokenTimer = setTimeout(async () => {
    try {
      const r = await api("/api/tokens", { kind: state.mode, prompt: p });
      state.tokensOk = r.ok;
      c.textContent = r.max ? `${r.tokens} / ${r.max} tokens` + (r.ok ? "" : " — too long for an edit, please shorten it")
                            : `${r.tokens} tokens`;
      c.classList.toggle("over", !r.ok);
    } catch { c.innerHTML = "&nbsp;"; state.tokensOk = true; }
    updateGo();
  }, 300);
}

function updateGo() {
  const edit = state.mode === "edit";
  const go = $("go");
  go.textContent = state.busy ? "Working…" : edit ? "Edit photo" : "Create image";
  go.disabled = state.busy || !state.ready || !$("prompt").value.trim() || (edit && !state.image) || !state.tokensOk;
}

function showError(msg) { $("err").textContent = msg; $("err").hidden = !msg; }

// ---------- progress ----------
let tStart = 0;
function showProgress(job) {
  $("progress").hidden = false;
  $("stage-name").textContent = job.stage + (job.step ? ` ${job.step}/${job.steps}` : "");
  $("elapsed").textContent = `${Math.round(job.elapsed ?? (Date.now() - tStart) / 1000)} s`;
  $("fill").style.width = `${Math.min(99, Math.round((job.fraction || 0) * 100))}%`;
}

async function run() {
  showError("");
  const edit = state.mode === "edit";
  const seedTxt = $("seed").value.trim();
  const body = { kind: edit ? "edit" : "generate", prompt: $("prompt").value.trim(), steps: state.steps,
                 seed: seedTxt === "" ? null : Number(seedTxt) };
  if (edit) { body.image = state.image.split(",")[1]; body.crop = state.crop; body.name = state.imageName; }
  state.busy = true; tStart = Date.now(); updateGo();
  $("placeholder").hidden = $("compare").hidden; $("stage").classList.add("busy");
  showProgress({ stage: edit ? "Preparing the photo" : "Reading the prompt", fraction: 0.01 });
  setTimeout(poll, 300);
  try {
    const r = await api("/api/run", body);
    show(r);
    await loadHistory();
  } catch (e) {
    showError(e.message);
  } finally {
    state.busy = false; $("progress").hidden = true; $("stage").classList.remove("busy"); updateGo();
  }
}

// ---------- results ----------
function show(r) {
  state.current = r;
  $("placeholder").hidden = true; $("compare").hidden = false;
  $("img-after").src = r.url;
  const hasBefore = !!r.before;
  $("compare").classList.toggle("single", !hasBefore);
  if (hasBefore) { $("img-before").src = r.before; setSplit(0.5); }
  $("result-bar").hidden = false;
  const t = r.timings || {};
  $("meta").textContent = `${r.kind === "edit" ? "Edit" : "New image"} · seed ${r.seed} · ${t.total ? t.total.toFixed(1) + " s" : ""}` +
                          (r.prompt ? ` · “${r.prompt.length > 80 ? r.prompt.slice(0, 77) + "…" : r.prompt}”` : "");
  $("download").href = r.url; $("download").setAttribute("download", r.id + ".png");
  document.querySelectorAll(".strip button").forEach((b) => b.classList.toggle("sel", b.dataset.id === r.id));
}

function setSplit(f) {
  f = Math.max(0, Math.min(1, f));
  $("before-wrap").style.width = `${f * 100}%`;
  $("handle").style.left = `${f * 100}%`;
}

function dragCompare() {
  const box = $("compare");
  let down = false;
  const move = (e) => { if (!down) return; const r = box.getBoundingClientRect(); setSplit((e.clientX - r.left) / r.width); };
  box.addEventListener("pointerdown", (e) => { down = true; box.setPointerCapture(e.pointerId); move(e); });
  box.addEventListener("pointermove", move);
  box.addEventListener("pointerup", () => { down = false; });
}

async function loadHistory() {
  try { state.history = (await api("/api/history")).items || []; } catch { return; }
  const strip = $("strip"); strip.innerHTML = "";
  for (const h of state.history) {
    const b = document.createElement("button");
    b.dataset.id = h.id; b.title = h.prompt;
    b.innerHTML = `<img loading="lazy" alt="" src="${h.thumb || h.url}">`;
    b.onclick = () => show(h);
    if (state.current && state.current.id === h.id) b.classList.add("sel");
    strip.appendChild(b);
  }
  $("history-count").textContent = state.history.length ? `${state.history.length} saved in the outputs folder` : "nothing yet";
  const want = decodeURIComponent(location.hash.slice(1));
  if (want && !state.current) { const h = state.history.find((x) => x.id === want); if (h) show(h); }
}

async function editResult() {
  if (!state.current) return;
  const blob = await (await fetch(state.current.url)).blob();
  setMode("edit");
  loadFile(new File([blob], state.current.id + ".png", { type: "image/png" }));
  $("prompt").value = ""; $("prompt").focus(); checkTokens();
}

// ---------- wiring ----------
function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
  const drop = $("drop"), file = $("file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => loadFile(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) { setMode("edit"); loadFile(f); }
  });
  segment("crop", "crop", String); segment("steps", "steps", Number);
  $("prompt").addEventListener("input", () => { checkTokens(); updateGo(); });
  $("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !$("go").disabled) run(); });
  $("go").addEventListener("click", run);
  $("seed-reuse").addEventListener("click", () => { if (state.current) $("seed").value = state.current.seed; });
  $("again").addEventListener("click", editResult);
  $("open-folder").addEventListener("click", () => api("/api/open-folder", { id: state.current?.id }).catch((e) => showError(e.message)));
  dragCompare();
  setMode("edit");
  poll();
  loadHistory();
}
init();
