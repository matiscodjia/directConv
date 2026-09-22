DC.header("/lab");
const { $, esc, num, bytes, rate, millis } = DC;
const S = { cfg: null, res: 10, device: "stm32", cam: 60, lenIdx: 3, strategy: null, pick: null, preview: new Map(), run: null, play: null, busy: false };

const SCENARIOS = [
  { id: "fits", label: "1 · It fits", res: "96×54", device: "stm32", pick: "stack" },
  { id: "wall", label: "2 · The wall", res: "1280×720", device: "stm32", pick: null },
  { id: "shrink", label: "3 · Shrink it", res: "1280×720", device: "stm32", pick: "reduced" },
  { id: "stream", label: "4 · Stream it", res: "1280×720", device: "stm32", pick: "stream" },
  { id: "mac", label: "5 · On a computer", res: "1280×720", device: "host", pick: "heap" },
];
const CARDS = {
  stack: ["Full frame", "on the stack"],
  heap: ["Full frame", "on the heap · needs an allocator"],
  reduced: ["Smaller frame", "on the stack · the biggest that fits"],
  stream: ["Streaming", "on the stack · one row at a time"],
};
const ORDER = ["stack", "heap", "reduced", "stream"];
const STEPS = ["decode", "numpy", "rust", "memory"];
const cur = () => S.cfg.ladder[S.res];
const dev = () => S.cfg.devices[S.device];
const sizeIdx = (name) => S.cfg.ladder.findIndex((p) => p.name === name);
const save = () => DC.store.set("lab", { res: S.res, device: S.device, cam: S.cam, lenIdx: S.lenIdx });

function lengthSteps() {
  const max = cur().max_frames, steps = S.cfg.length_steps.filter((n) => n <= max);
  if (steps[steps.length - 1] !== max && max > steps[steps.length - 1]) steps.push(max);
  return steps;
}
const frames = () => lengthSteps()[Math.min(S.lenIdx, lengthSteps().length - 1)];

// ---------------------------------------------------------------- chart (built once, then animated)
const CH = { W: 640, H: 300, m: { l: 52, r: 12, t: 12, b: 50 }, lo: 3, hi: 8 };
const px = (i) => CH.m.l + (i * (CH.W - CH.m.l - CH.m.r)) / (S.cfg.ladder.length - 1);
const py = (b) => CH.m.t + ((CH.hi - Math.log10(Math.max(b, 10 ** CH.lo))) * (CH.H - CH.m.t - CH.m.b)) / (CH.hi - CH.lo);
const tf = (id, x, y = 0) => { $(id).style.transform = `translate(${x}px, ${y}px)`; };

function buildChart(animate = true) {
  const box = $("chart").getBoundingClientRect();
  CH.W = Math.max(320, Math.round(box.width)); CH.H = Math.max(170, Math.round(box.height));
  $("chart").setAttribute("viewBox", `0 0 ${CH.W} ${CH.H}`);
  const { W, H, m, lo, hi } = CH, L = S.cfg.ladder, pw = W - m.l - m.r, ph = H - m.t - m.b;
  let g = `<defs><clipPath id="plot"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/></clipPath></defs>`;
  for (let e = lo; e <= hi; e++) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${py(10 ** e)}" y2="${py(10 ** e)}"/><text x="${m.l - 6}" y="${py(10 ** e) + 3.5}" text-anchor="end">${e < 6 ? num(10 ** (e - 3)) + " kB" : num(10 ** (e - 6)) + " MB"}</text>`;
  g += `<g clip-path="url(#plot)"><rect class="wall tr" id="c-wall" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/></g>` +
    `<line class="wallline tr" id="c-wallline" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/><text class="tr" id="c-walltext" x="${m.l + 6}" y="${m.t + 11}" style="fill:var(--bad)">a full frame no longer fits</text>`;
  for (const [cls, key] of [["direct", "direct_bytes"], ["stream", "stream_bytes"]]) g += `<polyline class="curve ${cls}" ${animate ? "" : 'style="animation:none;stroke-dashoffset:0"'} pathLength="1" points="${L.map((p, i) => `${px(i)},${py(p[key])}`).join(" ")}"/>`;
  g += `<g class="tr" id="c-budget"><line x1="${m.l}" x2="${W - m.r}" y1="0" y2="0" stroke="var(--text)" stroke-width="1.3" stroke-dasharray="6 4"/><text id="c-budget-t" x="${m.l + 6}" y="-5" text-anchor="start" style="fill:var(--text)"></text></g>` +
    `<g class="tr" id="c-ver"><line x1="${m.l}" x2="${W - m.r}" y1="0" y2="0" stroke="var(--ok)" stroke-width="1.1" stroke-dasharray="2 3"/><text id="c-ver-t" x="${px(3) + 8}" y="12" text-anchor="start" style="fill:var(--ok)"></text></g>` +
    `<line class="tr" id="c-mark" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" opacity="0.55"/>` +
    `<circle class="dot tr" id="c-dd" r="5.5" cx="0" cy="0" fill="var(--accent)"/><circle class="dot tr" id="c-ds" r="5.5" cx="0" cy="0" fill="var(--stream)"/>`;
  g += L.map((p, i) => `<text class="xl" data-i="${i}" transform="translate(${px(i)},${H - m.b + 12}) rotate(-38)" text-anchor="end">${esc(p.name)}</text>`).join("");
  g += `<text x="${W - m.r - 4}" y="${H - m.b - 26}" text-anchor="end" style="fill:var(--accent)">● full frame (stack or heap)</text><text x="${W - m.r - 4}" y="${H - m.b - 12}" text-anchor="end" style="fill:var(--stream)">● streaming, one row at a time</text>`;
  $("chart").innerHTML = g;
}

function updateChart() {
  const L = S.cfg.ladder, d = dev(), { m, W } = CH, p = cur();
  const wall = L.findIndex((q) => q.direct_bytes > d.budget);
  const wx = wall > 0 ? (px(wall - 1) + px(wall)) / 2 : wall === 0 ? m.l : W + 10;
  tf("c-wall", wx - m.l); tf("c-wallline", wx - m.l);
  const left = wx > W * 0.6;
  $("c-walltext").setAttribute("text-anchor", left ? "end" : "start"); tf("c-walltext", left ? wx - 6 - (m.l + 6) : wx - m.l);
  $("c-walltext").style.opacity = wall > 0 ? 1 : 0;
  tf("c-budget", 0, py(d.budget));
  $("c-budget-t").textContent = `${bytes(d.budget)} ${d.on_chip ? "predicted limit" : "budget"}`;
  tf("c-ver", 0, d.verified ? py(d.verified) : py(1e3)); $("c-ver").style.opacity = d.verified ? 1 : 0;
  $("c-ver-t").textContent = d.verified ? `${bytes(d.verified)} ran on the chip` : "";
  tf("c-mark", px(S.res)); tf("c-dd", px(S.res), py(p.direct_bytes)); tf("c-ds", px(S.res), py(p.stream_bytes));
  document.querySelectorAll("#chart .xl").forEach((t) => t.classList.toggle("on", +t.dataset.i === S.res));

  const over = p.direct_bytes / d.budget, term = d.on_chip ? "predicted limit" : "budget";
  const unconfirmed = d.verified && over <= 1 && p.direct_bytes > d.verified ? ` This is beyond the ${bytes(d.verified)} run on the chip: a prediction.` : "";
  $("verdict").innerHTML = over <= 1
    ? `At <b>${esc(p.name)}</b> a full frame needs <b>${bytes(p.direct_bytes)}</b> (${num(100 * over)} % of the ${term}): <b style="color:var(--ok)">it fits.</b>${unconfirmed}`
    : `At <b>${esc(p.name)}</b> a full frame needs <b>${bytes(p.direct_bytes)}</b>, <b style="color:var(--bad)">${num(over, over < 10 ? 1 : 0)}× the ${term}.</b> Streaming needs only ${bytes(p.stream_bytes)}.`;
}

// ---------------------------------------------------------------- strategy cards (three meters each)
function buildCards() {
  $("cards").innerHTML = ORDER.map((id) => `<button type="button" class="card2" data-id="${id}" aria-pressed="false">
    <div class="head"><h3>${CARDS[id][0]}</h3><span class="pill" data-b></span></div><div class="where">${CARDS[id][1]}</div>
    <div class="meters">${["Resolution", "Memory", "Speed"].map((k, i) => `<div class="meter"><span class="k">${k}</span><span class="bar"><i data-m="${i}"></i></span><span class="v" data-v="${i}"></span></div>`).join("")}</div>
    <p class="take" data-t></p></button>`).join("");
  document.querySelectorAll(".card2").forEach((el) => el.addEventListener("click", () => { if (el.disabled || S.strategy === el.dataset.id) return; S.strategy = el.dataset.id; updateCards(); maybePlay(); }));
}

function currentCards() {
  const name = cur().name, measured = S.run && S.run.resolution.name === name;
  const src = measured ? S.run.cards : S.preview.get(name);
  return { cards: src ? src[S.device] : null, measured };
}

function speedOf(c, measured) {
  const d = dev();
  if (!c.fits || (d.on_chip && c.id === "heap")) return null;
  const ms = d.on_chip ? c.chip_ms : measured ? c.host_ms : null;
  return ms ? { ms, fps: 1000 / ms } : { pending: true };
}

function view(id, c, measured) {
  const d = dev(), heapHS = measured && S.run.heap_vs_stack[S.device];
  if (c.unneeded) return { badge: ["not needed", ""], off: true, take: "The full frame already fits: nothing to give up." };
  const ratio = c.bytes / c.budget, kept = c.pixels_kept ?? 1, sp = speedOf(c, measured);
  const badge = c.fits ? [c.predicted ? "predicted fit" : "fits", c.predicted ? "warn" : "ok"] : c.bytes <= c.budget ? ["unavailable", "bad"] : ["does not fit", "bad"];
  const pct = (v) => num(100 * v, v < 0.1 ? 1 : 0) + " %";
  const cls = (v, hi, mid) => (v >= hi ? "" : v >= mid ? "mid" : "low");
  const meters = [
    c.size ? { w: Math.max(2, 100 * kept), c: cls(kept, 0.9, 0.3), v: pct(kept) } : { w: 0, c: "none", v: "—" },
    { w: 100 * Math.min(1, ratio), c: ratio <= 0.75 ? "" : ratio <= 1 ? "mid" : "low", v: ratio > 1 ? num(ratio, ratio < 10 ? 1 : 0) + "×" : pct(ratio), t: `${num(c.bytes)} B` },
    sp === null ? { w: 0, c: "none", v: "—" } : sp.pending ? { w: 0, c: "none", v: "run" } : { w: 100 * Math.min(1, sp.fps / S.cam), c: cls(sp.fps / S.cam, 1, 0.5), v: rate(sp.ms) },
  ];
  let take = "";
  if (id === "stack") take = c.fits ? "<b>Fast and simple</b>, but only for frames this small." : `Needs <b>${num(ratio, ratio < 10 ? 1 : 0)}×</b> ${d.on_chip ? "what the chip can hold" : "the stack limit"}.`;
  else if (id === "heap") take = d.on_chip ? "<b>Does not exist on the chip</b>: the bare-metal build has no allocator." : `Works on a computer. ${heapHS ? `It took ${num(heapHS.ratio, 2)}× the stack's time: <b>no cost in time</b>.` : "<b>Costs memory, not time.</b>"}`;
  else if (id === "reduced") take = c.size ? `<b>Fast and small</b>, but keeps only ${pct(kept)} of the pixels (${esc(c.size.name)}).` : "No frame of the ladder fits.";
  else take = `<b>The whole picture in ${bytes(c.bytes)}.</b>` + (sp && sp.fps ? (sp.fps >= S.cam ? " It keeps up with the camera." : ` It pays in speed: ${num(100 * (1 - sp.fps / S.cam))} % of frames dropped.`) : "");
  return { badge, meters, take, off: !c.playable };
}

function updateCards() {
  const { cards, measured } = currentCards();
  ORDER.forEach((id) => {
    const el = document.querySelector(`.card2[data-id="${id}"]`);
    if (!cards) return;
    const c = cards[id], v = view(id, c, measured);
    const b = el.querySelector("[data-b]"); b.textContent = v.badge[0]; b.className = "pill " + v.badge[1];
    el.disabled = !c.playable; el.classList.toggle("off", !!v.off);
    el.setAttribute("aria-pressed", String(S.strategy === id));
    (v.meters || [{ w: 0, c: "none", v: "" }, { w: 0, c: "none", v: "" }, { w: 0, c: "none", v: "" }]).forEach((m, i) => {
      const bar = el.querySelector(`[data-m="${i}"]`); bar.style.width = m.w + "%"; bar.className = m.c;
      const val = el.querySelector(`[data-v="${i}"]`); val.textContent = m.v; val.title = m.t || "";
    });
    el.querySelector("[data-t]").innerHTML = v.take;
  });
}

function pickDefault() {
  const { cards } = currentCards();
  if (!cards) return;
  if (S.pick && cards[S.pick] && cards[S.pick].playable) { S.strategy = S.pick; S.pick = null; return; }
  if (S.strategy && cards[S.strategy] && cards[S.strategy].playable) return;
  S.strategy = ["stream", "stack", "heap", "reduced"].find((id) => cards[id].playable) || null;
}

// ---------------------------------------------------------------- result
let rvfc = null, playTimer = null, playSeq = 0;
function playInSync() {
  const clips = ["v-original", "v-edges", "v-blur"].map($);
  Promise.all(clips.map((v) => new Promise((ok) => { if (v.readyState >= 3) ok(); else v.addEventListener("canplay", ok, { once: true }); })))
    .then(() => clips.forEach((v) => { v.currentTime = 0; v.play().catch(() => {}); }));
  const master = clips[0];
  master.ontimeupdate = () => clips.slice(1).forEach((v) => { if (Math.abs(v.currentTime - master.currentTime) > 0.15) v.currentTime = master.currentTime; });
}

function measureDisplay(video) {
  if (rvfc) rvfc.video.cancelVideoFrameCallback(rvfc.id);
  if (!("requestVideoFrameCallback" in video)) return;
  let t0 = null, f0 = 0;
  const tick = (now, meta) => {
    if (t0 === null) { t0 = now; f0 = meta.presentedFrames; }
    else if (now - t0 >= 1000) {
      const el = $("m-disp"); if (el) el.textContent = num(((meta.presentedFrames - f0) * 1000) / (now - t0)) + " fps";
      t0 = now; f0 = meta.presentedFrames;
    }
    rvfc.id = video.requestVideoFrameCallback(tick);
  };
  rvfc = { video, id: video.requestVideoFrameCallback(tick) };
}

function drawTimeline(p) {
  const cv = $("tl"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = 26;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement), ok = css.getPropertyValue("--ok").trim(), bad = css.getPropertyValue("--bad").trim();
  const bins = Math.max(20, Math.floor(w / 3)), cnt = new Float32Array(bins);
  p.processed_indices.forEach((i) => { cnt[Math.min(bins - 1, Math.floor((i * bins) / p.frames))]++; });
  const per = p.frames / bins, bw = w / bins;
  for (let b = 0; b < bins; b++) {
    const f = Math.min(1, cnt[b] / per);
    g.fillStyle = f > 0.98 ? ok : f < 0.02 ? bad : `color-mix(in srgb, ${ok} ${Math.round(f * 100)}%, ${bad})`;
    g.fillRect(b * bw, 0, Math.ceil(bw) - 0.5, h);
  }
}

function rafPlayhead() {
  const v = $("v-original"), ph = $("playhead"), tl = $("tl");
  if (v && v.duration && !document.hidden && !$("filled").hidden) ph.style.left = (v.currentTime / v.duration) * tl.clientWidth + "px";
  requestAnimationFrame(rafPlayhead);
}

function metric(k, to, fmt, sub, id) {
  return `<div class="metric"><div class="k">${k}</div><div class="v" ${id ? `id="${id}"` : ""} data-to="${to}" data-fmt="${fmt}">—</div><div class="s">${sub}</div></div>`;
}
const FMT = { n0: (v) => num(v), fps: (v) => num(v, v < 10 ? 2 : 0) + " fps", ms: (v) => millis(v), b: (v) => bytes(v), s: (v) => num(v, 1) + " s", pct: (v) => num(v) + " %" };

function renderDist() {
  const r = S.run, R = r.resolution, rows = [["numpy", r.numpy, "var(--numpy)"]];
  for (const [id, label] of [["heap", "frugal_ml heap"], ["stream", "frugal_ml stream"], ["stack", "frugal_ml stack"]]) {
    const m = r.measured[`${id}@${R.h}x${R.w}`]; if (m) rows.push([label, m, "var(--accent)"]);
  }
  const lo = Math.log10(Math.min(...rows.map((x) => x[1].p5_ms)) * 0.8), hi = Math.log10(Math.max(...rows.map((x) => x[1].p95_ms)) * 1.25);
  const X = (ms) => 118 + ((Math.log10(ms) - lo) / (hi - lo)) * 250;
  $("dist").setAttribute("viewBox", `0 0 420 ${22 + rows.length * 30}`);
  $("dist").innerHTML = rows.map(([label, m, col], i) => {
    const y = 16 + i * 30;
    return `<text x="0" y="${y + 4}">${label}</text><line x1="${X(m.p5_ms)}" x2="${X(m.p95_ms)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="5" stroke-linecap="round" opacity="0.4"/>` +
      `<circle cx="${X(m.ms_per_frame)}" cy="${y}" r="5" fill="${col}"/><text x="${X(m.p95_ms) + 8}" y="${y + 4}">${millis(m.ms_per_frame)}</text>`;
  }).join("");
}

function renderResult() {
  const p = S.play, r = S.run, stale = r && r.resolution.name !== cur().name;
  $("stale").hidden = !stale;
  if (stale) $("stale").textContent = `Showing the last run, at ${r.resolution.name}. Press Run to measure ${cur().name}.`;
  const has = !!(p && r);
  $("empty").hidden = has; $("filled").hidden = !has; $("r-title").hidden = !has; $("zoom-hint").hidden = !has;
  if (!has) return;
  $("r-title").textContent = `${CARDS[p.strategy][0]}, ${CARDS[p.strategy][1].split(" · ")[0]} · ${p.device === "stm32" ? "STM32" : "this computer"}`;
  for (const k of ["original", "edges", "blur"]) { const v = $("v-" + k); v.src = p.videos[k]; v.classList.toggle("px", p.size.w <= 256); }
  playInSync(); measureDisplay($("v-edges"));

  const chip = S.cfg.devices[p.device].on_chip, c = r.cards[p.device][p.strategy];
  const m = r.measured[`${p.strategy === "reduced" ? "stack" : p.strategy}@${p.size.h}x${p.size.w}`];
  $("metrics").innerHTML =
    metric("Camera", p.camera_fps, "fps", `${num(p.frames)} frames · ${num(p.duration_s, 1)} s`) +
    metric("Kept", p.processed, "n0", `of ${num(p.frames)} · ${num(p.dropped_pct)} % dropped`) +
    metric("Output rate", p.effective_fps, "fps", chip ? (p.dropped ? "limited by the chip" : "keeps up") : "limited by the camera") +
    metric("Capacity", 1000 / p.ms_per_frame, "fps", chip ? "chip, projected" : "this computer, measured") +
    metric("Per frame", p.ms_per_frame, "ms", chip ? "chip, projected" : m ? `measured · 5–95 %: ${millis(m.p5_ms)}–${millis(m.p95_ms)}` : "measured") +
    metric("Shown by browser", 0, "n0", "frames really displayed", "m-disp");
  document.querySelectorAll("#metrics .v").forEach((el) => { if (el.id === "m-disp") { el.textContent = "…"; return; } DC.countUp(el, +el.dataset.to, FMT[el.dataset.fmt]); });

  $("tl-len").textContent = `${num(p.duration_s, 1)} s at ${num(p.camera_fps)} fps`;
  requestAnimationFrame(() => drawTimeline(p));
  renderDist();
  const speed = p.camera_fps / r.clip_fps;
  $("hint").textContent = [
    p.dropped ? `${num(p.dropped)} of ${num(p.frames)} frames arrived while the device was busy. Edges and blur freeze on the last frame it finished; the input keeps running.` : "Every frame was handled in time.",
    p.clamped ? `Camera rate lowered to ${p.camera_fps} fps: the most H.264 carries at this size.` : "",
    Math.abs(speed - 1) > 0.15 ? `The clip is ${num(r.clip_fps)} fps, so motion looks ${num(speed, 1)}× ${speed > 1 ? "faster" : "slower"}.` : "",
    r.looped ? `The ${r.source_frames}-frame clip plays forward and back to fill ${num(r.frames)} frames.` : "",
  ].filter(Boolean).join(" ");
}

function maybePlay() {
  clearTimeout(playTimer);
  playTimer = setTimeout(async () => {
    if (!S.run || S.run.resolution.name !== cur().name || !S.strategy || S.busy) return;
    const my = ++playSeq;
    $("runmsg").className = "runmsg"; $("runmsg").textContent = "Encoding the video…";
    try {
      const r = await DC.job("/api/play", { run_id: S.run.run_id, strategy: S.strategy, device: S.device, camera_fps: S.cam }, (j) => { if (my === playSeq) $("runmsg").textContent = `${j.stage}${j.pct != null ? ` · ${num(100 * j.pct)} %` : ""}`; });
      if (my !== playSeq) return;
      S.play = r; renderResult(); $("runmsg").textContent = "";
    } catch (e) { if (my === playSeq) { $("runmsg").className = "runmsg err"; $("runmsg").textContent = e.message; } }
  }, 250);
}

// ---------------------------------------------------------------- run
function setStepper(step, pct) {
  const idx = STEPS.indexOf(step);
  [...$("stepper").children].forEach((el, i) => { el.className = idx < 0 ? "" : i < idx ? "done" : i === idx ? "on" : ""; if (i === idx) el.style.setProperty("--pct", Math.round(100 * (pct ?? 0.4)) + "%"); });
}

async function doRun() {
  if (S.busy) return;
  S.busy = true; $("run").disabled = true; setStepper("decode", 0.05);
  $("runmsg").className = "runmsg"; $("runmsg").textContent = "Starting…";
  try {
    const preset = $("preset").value;
    const r = await DC.job("/api/run", { resolution: cur().name, frames: frames(), ...(preset === "__url__" ? { url: $("url").value } : { preset }) }, (j) => {
      setStepper(j.step, j.pct); $("runmsg").textContent = `${j.stage}${j.pct != null ? ` · ${num(100 * j.pct)} %` : ""}`;
    });
    S.run = r; S.play = null; S.preview.set(r.resolution.name, r.cards);
    setStepper("done"); [...$("stepper").children].forEach((el) => (el.className = "done"));
    const b = Object.values(r.measured)[0]?.batches ?? 0;
    $("runmsg").textContent = `Measured on ${num(r.frames)} frames, ${b} batches per strategy. Every output equals numpy's.`;
    pickDefault(); updateCards(); renderResult(); S.busy = false; maybePlay();
  } catch (e) { $("runmsg").className = "runmsg err"; $("runmsg").textContent = e.message; setStepper("none"); }
  finally { S.busy = false; $("run").disabled = false; }
}

// ---------------------------------------------------------------- state -> UI
async function refresh() {
  const p = cur();
  $("res-big").textContent = p.name;
  $("res-sub").textContent = `${num((p.h * p.w) / 1e6, 2)} MP`;
  $("cam").max = p.max_camera_fps; if (S.cam > p.max_camera_fps) S.cam = p.max_camera_fps;
  $("cam").value = S.cam; $("cam-big").textContent = S.cam + " fps";
  const steps = lengthSteps(); S.lenIdx = Math.min(S.lenIdx, steps.length - 1);
  $("len").max = steps.length - 1; $("len").value = S.lenIdx;
  $("len-big").textContent = `${num(steps[S.lenIdx])} frames`;
  $("len-sub").textContent = `≈ ${num(steps[S.lenIdx] / S.cam, 1)} s at ${S.cam} fps`;
  $("device-note").textContent = dev().on_chip ? "Times on the chip are projected from cycle counts measured on the board." : "Times are measured live on this computer.";
  document.querySelectorAll("#device button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.d === S.device)));
  document.querySelectorAll("#cam-presets .chip").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.fps === S.cam)));
  ["res", "cam", "len"].forEach((id) => DC.paintRange($(id)));
  updateChart(); updateCards(); renderResult(); save();
  if (!S.preview.has(p.name)) { S.preview.set(p.name, await (await fetch("/api/preview?resolution=" + encodeURIComponent(p.name))).json()); pickDefault(); updateCards(); }
}

async function applyScenario(id) {
  const sc = SCENARIOS.find((s) => s.id === id);
  if (!sc || S.busy) return;
  S.device = sc.device; S.res = sizeIdx(sc.res); S.pick = sc.pick; S.strategy = null; S.play = null;
  await refresh(); await doRun();
}

async function init() {
  S.cfg = await DC.config();
  const saved = DC.store.get("lab", {});
  S.res = saved.res ?? 10; S.device = saved.device ?? "stm32"; S.cam = saved.cam ?? 60; S.lenIdx = saved.lenIdx ?? 3;
  for (const name of S.cfg.presets) $("preset").add(new Option(name, name));
  $("preset").add(new Option("Paste a URL…", "__url__"));
  $("scen").innerHTML = SCENARIOS.map((s) => `<button type="button" class="chip" data-s="${s.id}">${s.label}</button>`).join("");
  $("cam-presets").innerHTML = [30, 60, 120, 240].map((f) => `<button type="button" class="chip" data-fps="${f}">${f}</button>`).join("");
  buildChart(); buildCards();
  let rz; new ResizeObserver(() => { clearTimeout(rz); rz = setTimeout(() => { const b = $("chart").getBoundingClientRect(); if (Math.abs(b.width - CH.W) > 2 || Math.abs(b.height - CH.H) > 2) { buildChart(false); updateChart(); } }, 120); }).observe($("chart"));

  $("res").addEventListener("input", () => { S.res = +$("res").value; refresh(); });
  $("cam").addEventListener("input", () => { S.cam = +$("cam").value; refresh(); });
  $("cam").addEventListener("change", maybePlay);
  $("len").addEventListener("input", () => { S.lenIdx = +$("len").value; refresh(); });
  $("preset").addEventListener("change", () => { $("url").hidden = $("preset").value !== "__url__"; });
  $("run").addEventListener("click", doRun);
  document.querySelectorAll("#device button").forEach((b) => b.addEventListener("click", () => { S.device = b.dataset.d; S.strategy = null; S.play = null; refresh().then(() => { pickDefault(); updateCards(); maybePlay(); }); }));
  document.querySelectorAll("#cam-presets .chip").forEach((b) => b.addEventListener("click", () => { S.cam = Math.min(+b.dataset.fps, cur().max_camera_fps); refresh(); maybePlay(); }));
  document.querySelectorAll("#scen .chip").forEach((b) => b.addEventListener("click", () => applyScenario(b.dataset.s)));
  $("videos").addEventListener("click", (e) => {
    const fig = e.target.closest("figure"); if (!fig) return;
    const on = fig.hasAttribute("data-on");
    $("videos").querySelectorAll("figure").forEach((f) => f.removeAttribute("data-on"));
    if (on) $("videos").removeAttribute("data-focus"); else { fig.setAttribute("data-on", ""); $("videos").setAttribute("data-focus", fig.dataset.k); }
    setTimeout(() => S.play && drawTimeline(S.play), 50);
  });
  window.addEventListener("resize", () => S.play && drawTimeline(S.play));

  // coming back from another page: show the last run again
  const last = (await (await fetch("/api/last")).json()).run;
  if (last && !new URLSearchParams(location.search).get("scenario")) {
    S.run = last; S.preview.set(last.resolution.name, last.cards);
    S.res = sizeIdx(last.resolution.name);
    const steps = lengthSteps(); S.lenIdx = Math.max(0, steps.findIndex((n) => n >= last.frames));
  }
  await refresh(); pickDefault(); updateCards();
  if (S.run) maybePlay();
  requestAnimationFrame(rafPlayhead);
  const sc = new URLSearchParams(location.search).get("scenario");
  if (sc) applyScenario(sc);
}
init().catch((e) => { $("runmsg").className = "runmsg err"; $("runmsg").textContent = "Could not reach the demo server: " + e.message; });
