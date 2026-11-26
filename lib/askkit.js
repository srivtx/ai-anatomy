// Shared helpers for the Ask Neuro pictures.
import { h, s } from "./core.js";

export const box = (cls, ...kids) => h(`div.av.${cls}`, ...kids);
export const svg = (w, hh, ...kids) => s("svg", { viewBox: `0 0 ${w} ${hh}`, class: "av-svg", role: "img" }, ...kids);
/** Paint the part of a slider's track left of the thumb (CSS reads --p). */
export const fill = (inp) => inp.style.setProperty("--p", `${(100 * (inp.value - inp.min)) / (inp.max - inp.min || 1)}%`);
export const rangeIn = (label, min, max, step, value, onInput) => {
  const out = h("output"), inp = h("input", { type: "range", min, max, step, value, "aria-label": label });
  const upd = () => { fill(inp); out.textContent = (+inp.value).toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0); onInput(+inp.value); };
  inp.addEventListener("input", upd);
  fill(inp);
  return { el: h("label.av-sl", h("span", label), inp, out), set(v) { inp.value = v; upd(); }, init: upd };
};

