// Shared helpers for every page: formatting, the header, the API client, small animations.
const DC = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = (n, d = 0) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
  const bytes = (b) => (b >= 1e6 ? num(b / 1e6, 1) + " MB" : b >= 1e3 ? num(b / 1e3, 1) + " kB" : num(b) + " B");
  const rate = (ms) => { const v = 1000 / ms; return num(v, v >= 100 ? 0 : v >= 10 ? 1 : 2) + " fps"; };
  const millis = (ms) => num(ms, ms >= 100 ? 0 : ms >= 10 ? 1 : ms >= 1 ? 2 : 3) + " ms";
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const store = {
    get(k, d) { try { const v = localStorage.getItem("dc." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("dc." + k, JSON.stringify(v)); } catch { /* private mode: fine */ } },
  };

  let cfg = null;
  const config = async () => (cfg ??= await (await fetch("/api/config")).json());

  // Starts a job and follows it: onProgress({step, stage, pct}) until it is done.
  async function job(path, body, onProgress = () => {}) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    for (;;) {
      const j = await (await fetch("/api/job/" + data.job)).json();
      if (j.state === "done") return j.result;
      if (j.state === "error") throw new Error(j.error);
      onProgress(j);
      await new Promise((r) => setTimeout(r, 200));
    }
  }

  // Tweens a number in an element.
  function countUp(el, to, format = num, ms = 800) {
    if (!el) return;
    if (reduced || !isFinite(to)) { el.textContent = format(to); return; }
    const from = el._v ?? 0, t0 = performance.now();
    el._v = to;
    cancelAnimationFrame(el._raf);
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - p, 3);
      el.textContent = format(from + (to - from) * e);
      if (p < 1) el._raf = requestAnimationFrame(tick);
    };
    el._raf = requestAnimationFrame(tick);
  }

  function header(active) {
    const links = [["/", "Home"], ["/lab", "Lab"], ["/why", "Why not numpy?"], ["/method", "Method"]];
    const el = $("top");
    el.className = "top";
    el.innerHTML = `<a class="brand" href="/"><span class="mark"><i></i><i></i><i></i></span>directConv</a>
      <nav aria-label="Pages">${links.map(([h, t]) => `<a href="${h}" ${h === active ? 'aria-current="page"' : ""}>${t}</a>`).join("")}</nav>
      <span class="spacer"></span>${active === "/lab" ? "" : '<a class="btn primary" href="/lab">Open the lab →</a>'}`;
  }

  // range inputs: paint the filled part of the track
  function paintRange(input) {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--fill", p + "%");
  }

  return { $, esc, num, bytes, rate, millis, reduced, store, config, job, countUp, header, paintRange };
})();
