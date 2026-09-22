DC.header("/");
(async () => {
  const cfg = await DC.config();
  const L = cfg.ladder, dev = cfg.devices.stm32, $ = DC.$;
  // log scale, 1 kB .. 100 MB, like the lab's chart
  const pos = (b) => Math.max(0, Math.min(1, (Math.log10(Math.max(b, 1e3)) - 3) / 5));
  $("w-budget").style.left = $("w-budget2").style.left = pos(dev.budget) * 100 + "%";

  const hd = L.find((p) => p.name === "1280×720");
  DC.countUp($("n-stack"), 126.4, (v) => DC.num(v, 1) + " kB");
  DC.countUp($("n-frame"), hd.direct_bytes, (v) => DC.bytes(v));
  DC.countUp($("n-stream"), hd.stream_bytes, (v) => DC.bytes(v));

  // the wall, animated: walk the ladder, watch the full frame cross the dashed line
  let i = 0;
  const step = () => {
    const p = L[i], over = p.direct_bytes / dev.budget;
    $("w-size").textContent = p.name;
    const full = $("w-full");
    full.style.width = pos(p.direct_bytes) * 100 + "%";
    full.style.background = over <= 1 ? "var(--ok)" : "var(--bad)";
    $("w-stream").style.width = pos(p.stream_bytes) * 100 + "%";
    $("w-full-b").textContent = `${DC.bytes(p.direct_bytes)} needed`;
    const v = $("w-verdict");
    v.textContent = over <= 1 ? "Fits on the stack." : `${DC.num(over, over < 10 ? 1 : 0)}× too big for the stack. Streaming still fits.`;
    v.style.color = over <= 1 ? "var(--ok)" : "var(--bad)";
    i = (i + 1) % L.length;
  };
  step();
  if (!DC.reduced) setInterval(step, 1100);
  else i = L.findIndex((p) => p.name === "1280×720"), step();
})();
