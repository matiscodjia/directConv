DC.header("/why");
(async () => {
  const { $, num, millis } = DC;
  const last = (await (await fetch("/api/last")).json()).run;

  // What we show: the author's numbers until the reader has run the lab, then the reader's own.
  let d;
  if (last) {
    const R = last.resolution, heap = last.measured[`heap@${R.h}x${R.w}`], b = last.numpy.breakdown;
    d = { total: b.total_ms, copy: b.copy_ms, gemm: b.gemm_ms, frugal: heap.ms_per_frame, mb: b.copy_mb_per_frame, view: b.view_is_zero_copy,
      note: `Your last run: ${DC.num(last.frames)} frames at ${R.name}, measured on this computer.` };
  } else {
    d = { total: 10.72, copy: 9.23, gemm: 1.22, frugal: 0.55, mb: 33, view: true,
      note: "Example, measured on the author's Mac at 1280×720. Run the lab to replace it with your own numbers." };
  }
  // copy and matrix product are timed on their own, so they can add up to a little more than the whole call
  const denom = Math.max(d.total, d.copy + d.gemm), other = Math.max(0, d.total - d.copy - d.gemm);
  $("src-note").textContent = d.note;
  $("w-view").textContent = d.view ? "no copy (it shares the frame's memory, checked)" : "copy";
  $("w-copy").textContent = `${num(d.mb, 0)} MB, ${millis(d.copy)} (${num((100 * d.copy) / d.total)} % of the time)`;
  $("w-gemm").textContent = millis(d.gemm);
  $("w-frugal").textContent = `${millis(d.frugal)} per frame, ${num(d.total / d.frugal, 1)}× faster than numpy`;

  const seg = (cls, ms, label) => `<div class="seg2 ${cls}" data-w="${(100 * ms) / denom}">${label}</div>`;
  $("lane-numpy").innerHTML = seg("copy", d.copy, `copy ${millis(d.copy)}`) + seg("gemm", d.gemm, millis(d.gemm)) + (other > 0.02 * d.total ? seg("", other, "") : "");
  $("lane-frugal").innerHTML = seg("mac", d.frugal, d.frugal / denom > 0.12 ? millis(d.frugal) : "");
  $("tot-numpy").textContent = `${millis(d.total)} per frame`;
  $("tot-frugal").textContent = `${millis(d.frugal)} per frame`;
  requestAnimationFrame(() => setTimeout(() => document.querySelectorAll(".seg2").forEach((el) => { el.style.width = el.dataset.w + "%"; }), 150));

  const t = await (await fetch("/tradeoffs.json")).json().catch(() => null);
  if (t) {
    $("t-src").textContent = `Measured on ${t.cpu}, ${t.date}, ${t.hw}×${t.hw} frames.`;
    $("tradeoffs").tBodies[0].innerHTML = t.rows.map((r, i) => {
      const win = r.ratio >= 1.05, times = win ? r.ratio : 1 / r.ratio, w = Math.min(1, Math.log10(times) / Math.log10(25)) * 150;
      return `<tr class="${i === 0 ? "demo" : ""}"><td>${r.c}</td><td>${r.k}</td><td>${millis(r.numpy_ms)}</td><td>${millis(r.rust_ms)}</td>` +
        `<td><div class="ratio"><span>${win ? num(r.ratio, 1) + "× faster" : "numpy " + num(1 / r.ratio, 1) + "× faster"}</span><i class="${win ? "" : "lose"}" data-w="${Math.max(4, w)}"></i></div></td></tr>`;
    }).join("");
    requestAnimationFrame(() => setTimeout(() => document.querySelectorAll(".ratio i").forEach((el) => { el.style.width = el.dataset.w + "px"; }), 200));
  }
})();
