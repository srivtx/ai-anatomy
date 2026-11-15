// A notebook viewer: open a Colab project in place, read its explanation and code cell by cell,
// see what each cell does (sets up, trains, plots, checks) and what results to expect.
import { h } from "./core.js";
import { colabUrl } from "./catalog.js";

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** A small Markdown renderer for notebook text: headings, lists, bold, italics, code, links and maths. */
function md(src) {
  const inline = (t) => esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\$\$([^$]+)\$\$|\$([^$\n]+)\$/g, (_, a, b) => `<code class="math">${a || b}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const out = []; let list = null, para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } if (list) { out.push(`<${list.t}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.t}>`); list = null; } };
  const lines = src.split("\n");
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[li + 1] || "")) {   // a table
      flush();
      const row = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = row(line); li += 2; const rows = [];
      while (li < lines.length && /^\s*\|/.test(lines[li])) rows.push(row(lines[li++]));
      li--;
      out.push(`<div class="nbv-tw"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const hd = line.match(/^(#{1,4})\s+(.*)/), ul = line.match(/^\s*[-*]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (hd) { flush(); const n = Math.min(4, hd[1].length + 1); out.push(`<h${n}>${inline(hd[2])}</h${n}>`); }
    else if (ul || ol) { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } const t = ul ? "ul" : "ol"; if (!list || list.t !== t) { flush(); list = { t, items: [] }; } list.items.push((ul || ol)[1]); }
    else if (!line.trim()) flush();
    else if (list) list.items[list.items.length - 1] += " " + line.trim();
    else para.push(line.trim());
  }
  flush();
  return out.join("");
}

/** Python syntax colouring in one pass. */
const KW = "False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield";
const BI = "print|range|len|enumerate|zip|list|dict|set|tuple|int|float|str|sum|min|max|abs|sorted|isinstance|super|open|map|round|any|all";
const TOK = new RegExp(`(#[^\\n]*)|([rbfu]{0,2}"""[\\s\\S]*?"""|[rbfu]{0,2}'''[\\s\\S]*?'''|[rbfu]{0,2}"(?:\\\\.|[^"\\\\\\n])*"|[rbfu]{0,2}'(?:\\\\.|[^'\\\\\\n])*')|(@[\\w.]+)|\\b(${KW})\\b|\\b(${BI})(?=\\()|\\b(\\d+\\.?\\d*(?:e[-+]?\\d+)?)\\b|\\b(def|class)\\b`, "g");
function py(code) {
  let out = "", last = 0;
  code.replace(TOK, (m, com, str, dec, kw, bi, num, _d, i) => {
    out += esc(code.slice(last, i));
    const c = com ? "c" : str ? "s" : dec ? "d" : kw ? "k" : bi ? "b" : "n";
    out += `<span class="t-${c}">${esc(m)}</span>`; last = i + m.length; return m;
  });
  return out + esc(code.slice(last));
}

/** What a code cell does, from what it calls. */
function kinds(code, first) {
  const k = [];
  if (first || /^\s*(import|from)\s/m.test(code) && code.split("\n").length < 14) k.push(["setup", "Sets up"]);
  if (/^\s*(class|def)\s/m.test(code)) k.push(["model", "Defines"]);
  if (/\.backward\(|\.step\(\)|optim|-=\s*lr\b|\bfor\s+(_|step|epoch|it|t|i)\s+in\s+range\(\s*(steps|epochs|n_steps|iters|STEPS)/.test(code)) k.push(["train", "Trains"]);
  if (/#\s*Exercise\b/i.test(code)) k.push(["ex", "Exercise"]);
  if (/plt\.|\.plot\(|imshow|scatter\(/.test(code)) k.push(["plot", "Plots"]);
  if (/\bassert\b/.test(code)) k.push(["check", "Checks"]);
  if (!k.length && /print\(/.test(code)) k.push(["print", "Measures"]);
  return k;
}

let dlg;
/** Open the viewer for project p (from catalog PROJECTS). */
export async function openNotebook(p, { results = [] } = {}) {
  dlg?.remove();
  const body = h("div.nbv-body", h("div.nbv-loading", "Loading the notebook…"));
  const nav = h("nav.nbv-nav", { "aria-label": "Notebook sections" });
  dlg = h("dialog.nbv", { "aria-label": `Notebook preview: ${p.title}` },
    h("header.nbv-h",
      h("div.nbv-ht", h("div.eyebrow", `Project ${p.n ?? parseInt(p.file, 10)} · ${p.hw === "CPU" ? "runs on CPU" : `Colab GPU (${p.hw})`} · ~${p.mins} min`), h("h2", p.title)),
      h("div.nbv-act",
        h("a.btn.primary", { href: colabUrl(p.file), target: "_blank", rel: "noopener" }, "Open in Colab"),
        h("button.nbv-x", { type: "button", "aria-label": "Close", onclick: (e) => e.target.closest("dialog").close() }, "×"))),
    h("div.nbv-main", nav, body));
  const me = dlg;
  document.body.append(me);
  me.addEventListener("close", () => { me.remove(); if (dlg === me) document.documentElement.classList.remove("nbv-open"); });
  me.addEventListener("click", (e) => { if (e.target === me) me.close(); });
  document.documentElement.classList.add("nbv-open");
  dlg.showModal();
  dlg.querySelector(".nbv-x").focus({ preventScroll: true });

  let nb;
  try { const r = await fetch(`/notebooks/${p.file}.ipynb`); if (!r.ok) throw new Error(r.status); nb = await r.json(); }
  catch { body.replaceChildren(h("p.nbv-loading", "Couldn't load the notebook here. ", h("a", { href: colabUrl(p.file), target: "_blank", rel: "noopener" }, "Open it in Colab →"))); return; }

  const cells = nb.cells.filter((c) => !/colab-badge/.test([].concat(c.source).join("")));
  const items = [], secs = [];
  let codeN = 0, sec = null;
  const counts = { ex: 0, setup: 0, model: 0, train: 0, plot: 0, check: 0, print: 0 };
  for (const c of cells) {
    const src = [].concat(c.source).join("");
    if (!src.trim()) continue;
    if (c.cell_type === "markdown") {
      const hd = src.match(/^#{1,3}\s+(.*)$/m);
      const el = h("div.nbv-md", { html: md(src) });
      if (hd) { sec = { title: hd[1].replace(/[*`]/g, ""), el, kinds: new Set() }; el.id = `nbv-s${secs.length}`; secs.push(sec); }
      items.push(el);
    } else {
      codeN++;
      const k = kinds(src, codeN === 1);
      k.forEach(([id]) => { counts[id]++; sec?.kinds.add(id); });
      const lines = src.replace(/\n$/, "").split("\n").length;
      items.push(h("div.nbv-cell",
        h("div.nbv-cell-h", h("span.nbv-in", `In [${codeN}]`), ...k.map(([id, w]) => h(`span.nbv-k.k-${id}`, w)), h("span.nbv-lines", `${lines} line${lines === 1 ? "" : "s"}`)),
        h("pre.nbv-code", h("code", { html: py(src.replace(/\n$/, "")) }))));
    }
  }
  // what you should see
  const expect = [...(p.done || []), ...results];
  if (expect.length || p.stretch?.length) {
    const end = h("section.nbv-end", { id: "nbv-end" },
      expect.length ? [h("h3", "What you should see"), h("ul.nbv-check", expect.map((t) => h("li", t)))] : null,
      p.stretch?.length ? [h("h3", "Then stretch it"), h("ul.nbv-stretch", p.stretch.map((t) => h("li", t)))] : null);
    items.push(end);
  }
  const sum = Object.entries(counts).filter(([, n]) => n).map(([k, n]) => h(`span.nbv-k.k-${k}`, `${n} ${{ ex: "exercise", setup: "setup", model: "define", train: "train", plot: "plot", check: "check", print: "measure" }[k]}`));
  nav.replaceChildren(
    h("div.nbv-nl", `${codeN} code cells`), h("div.nbv-sum", sum),
    h("div.nbv-nl", "Sections"),
    h("ol", secs.map((s, i) => h("li", h("a", { href: `#nbv-s${i}`, onclick: (e) => { e.preventDefault(); s.el.scrollIntoView({ behavior: "smooth", block: "start" }); } },
      h("span", s.title), h("span.nbv-dots", [...s.kinds].map((k) => h(`i.k-${k}`))))))),
    expect.length ? h("a.nbv-jump", { href: "#nbv-end", onclick: (e) => { e.preventDefault(); body.querySelector("#nbv-end").scrollIntoView({ behavior: "smooth" }); } }, "What you should see ↓") : null);
  body.replaceChildren(...items);
}
