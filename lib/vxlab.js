// A 3D lab: a voxel picture you configure with sliders and presets. It redraws on every change, turns
// in 90° steps, says what each block is when you hover it, and shows the real numbers beside it.
import { h } from "./core.js";
import { Scene } from "./voxel.js";
import { slider, segmented, stats } from "./widgets.js";

/** A Scene that turns everything a quarter-turn `rot` times about the vertical axis, and keeps hover notes. */
export class LabScene extends Scene {
  constructor(rot = 0) { super(); this.rot = rot; this.tips = new Map(); }
  put(x, y, z, color, faces = {}) {
    let X = x, Y = y;
    for (let r = 0; r < this.rot; r++) [X, Y] = [-Y, X];
    return super.put(X, Y, z, color, faces);
  }
  /** Register a hover note and get the key to pass as faces.hit. */
  tip(html) { const k = String(this.tips.size); this.tips.set(k, html); return k; }
}

/**
 * controls: [{ key, label, min, max, step, value, fmt } | { key, label, options: [{value, label}], value }]
 * presets: [{ label, set: {key: value} }]
 * draw(scene, cfg): fill the scene. read(cfg): { stats: [[label, value, tone?]], say: html }
 */
export function vxLab({ title, controls = [], presets, draw, read, px = 5, height = 320, fixed = 0, hint = "Hover or tap a block to see what it is." }) {
  const cfg = {}, els = {};
  for (const c of controls) cfg[c.key] = c.value;
  let rot = 0, tips = new Map();
  const pic = h("div.vxl-pic", { style: { minHeight: `${height}px` } });
  const tip = h("div.vxl-tip", { "aria-live": "polite" }, hint);
  const r0 = read(cfg);
  const st = stats(r0.stats.map(([label, value, tone]) => ({ label, value, tone })));
  const say = h("div.vxl-say");
  let pre = null;
  const changed = (k, v) => { cfg[k] = v; if (pre) pre.value = -1; update(); };
  const ctrls = controls.map((c) => {
    const el = c.options
      ? segmented({ label: c.label, options: c.options, value: c.value, onChange: (v) => changed(c.key, v) })
      : slider({ label: c.label, min: c.min, max: c.max, step: c.step || 1, value: c.value, fmt: c.fmt, onInput: (v) => changed(c.key, v) });
    els[c.key] = el; return el;
  });
  if (presets) pre = segmented({ label: "Presets", options: presets.map((p, i) => ({ value: i, label: p.label })), value: -1, onChange: (i) => {
    Object.assign(cfg, presets[i].set);
    for (const k in presets[i].set) if (els[k]) els[k].value = presets[i].set[k];
    update();
  } });
  const turn = (d) => { rot = (rot + d + 4) % 4; update(); };
  const rotBtns = h("div.vxl-rot",
    h("button.btn.small", { type: "button", "aria-label": "Turn left", title: "Turn left", onclick: () => turn(3) }, "⟲"),
    h("button.btn.small", { type: "button", "aria-label": "Turn right", title: "Turn right", onclick: () => turn(1) }, "⟳"));
  const show = (e) => { const k = e.target?.dataset?.h; if (k != null && tips.has(k)) tip.innerHTML = tips.get(k); };
  pic.addEventListener("pointerover", show);
  pic.addEventListener("click", show);

  function update() {
    const s = new LabScene(rot);
    draw(s, cfg);
    tips = s.tips;
    const svg = s.svg({ cls: "vx", label: title || "3D model" });
    const [, W, H] = /viewBox="0 0 (\d+) (\d+)"/.exec(svg) || [0, 60, 40];
    const box = Math.max(240, (pic.clientWidth || 460) - 20);
    // fixed: keep one scale so size changes show as size changes; otherwise fit the box
    const k = fixed ? Math.min(fixed, box / W) : Math.max(1, Math.min(px, box / W, (height - 20) / H));
    pic.innerHTML = svg.replace("<svg ", `<svg width="${Math.round(W * k)}" height="${Math.round(H * k)}" `);
    const r = read(cfg);
    r.stats.forEach(([, v, tone], i) => st.set(i, v, tone || ""));
    say.innerHTML = r.say || "";
  }
  const el = h("div.stage.vxl",
    h("div.stage-controls", ctrls, pre),
    h("div.stage-body.vxl-body",
      h("div.vxl-view", h("span.vxl-tag", "3D · turn it"), rotBtns, pic, tip),
      h("div.vxl-side", title ? h("div.vxl-title", title) : null, st.el, say)));
  update();
  // first layout pass: redraw once the box has a real width
  requestAnimationFrame(() => update());
  return el;
}
