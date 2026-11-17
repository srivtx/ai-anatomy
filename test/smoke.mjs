// Loads the map and every live chapter in a real browser, pokes each control
// and fails on any console error or horizontal overflow.
// Usage: node test/smoke.mjs [chapterId] [--shots dir]
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { ALL } from "../lib/catalog.js";

const { chromium } = createRequire(import.meta.url)("playwright"); // preinstalled globally
const args = process.argv.slice(2);
const shots = args.includes("--shots") ? args[args.indexOf("--shots") + 1] : null;
const only = args.find((a) => !a.startsWith("--") && a !== shots);
const PORT = 4199;
const server = spawn("node", ["serve.mjs"], { env: { ...process.env, PORT: String(PORT) }, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 600));
const base = `http://localhost:${PORT}`;

const browser = await chromium.launch();
let failed = 0;

async function visit(path, name, poke) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("response", (r) => r.status() >= 400 && errors.push(`${r.status()} ${r.url()}`));
  page.on("console", (m) => m.type() === "error" && !m.text().includes("fonts.g") && errors.push(m.text()));
  await page.goto(base + path, { waitUntil: "load" });
  await page.waitForTimeout(300);
  if (poke) await poke(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(150);
  const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  const ok = !errors.length && !overflow && !mobileOverflow;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${errors.length ? "  errors: " + errors.join(" | ") : ""}${overflow ? "  desktop overflow" : ""}${mobileOverflow ? "  mobile overflow" : ""}`);
  await page.close();
}

async function pokeAll(page) {
  // controls may re-render their own panel, so a stale handle is not a failure
  for (const t of await page.$$(".toggle")) await t.click({ timeout: 1000 }).catch(() => {});
  for (const b of await page.$$(".stage .seg button")) await b.click().catch(() => {});
  for (const r of await page.$$('input[type="range"]'))
    await r.evaluate((el) => { el.value = el.max; el.dispatchEvent(new Event("input", { bubbles: true })); });
  for (const p of await page.$$(".predict")) { const o = await p.$(".opt"); if (o) await o.click({ timeout: 1000 }).catch(() => {}); }
  for (const st of await page.$$(".stepper")) {
    const next = await st.$(".step-nav .primary");
    for (let i = 0; i < 12 && next && !(await next.isDisabled()); i++) await next.click();
  }
  for (const b of await page.$$(".stage .btn")) await b.click().catch(() => {});
  await page.waitForTimeout(1200);
}

if (!only || only === "studio") {
  // the 3D studio: every level of every model, plus clicking, double-clicking, keys and the toolbar
  const studio = async (p) => {
    if (!(await p.$(".st-canvas"))) throw new Error("no studio canvas");
    const box = await (await p.$(".st-canvas")).boundingBox(), cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await p.mouse.move(cx, cy); await p.mouse.click(cx, cy); await p.mouse.dblclick(cx, cy); await p.waitForTimeout(900);
    await p.focus(".st-canvas"); for (const k of ["ArrowLeft", "ArrowUp", "+", "-", "f", "Escape"]) await p.keyboard.press(k);
    for (const b of await p.$$(".st-tools .btn")) await b.click();
    // edit whatever is selected, then every tab and every design slider
    for (const b of await p.$$(".st-edit .seg button")) await b.click().catch(() => {});
    for (const t of await p.$$(".st-tabs button")) {
      await t.click(); await p.waitForTimeout(150);
      for (const r of await p.$$('.st-pane:not([hidden]) input[type="range"]')) await r.evaluate((el) => { el.value = el.max; el.dispatchEvent(new Event("input", { bubbles: true })); });
    }
    await p.waitForTimeout(700);
  };
  for (const m of ["tiny", "gpt2", "llama8"]) await visit(`/studio?m=${m}`, `studio ${m}`, studio);
  for (const at of ["embed", "pos", "b2", "b2/attn", "b2/attn/h3", "b2/mlp", "b2/ln2", "lnf", "head"]) await visit(`/studio?m=gpt2&at=${at}`, `studio ${at}`, studio);
  await visit("/studio?m=llama8&at=b9/mlp", "studio llama mlp", studio);
  await visit("/studio?m=mistral7", "studio mistral", studio);
  await visit("/studio?m=mixtral", "studio mixtral", studio);
  await visit("/studio?m=mixtral&at=b3/mlp", "studio mixtral experts", studio);
  // learn: every level, its buttons, the + / - block tools
  const { LEVELS, CHALLENGES } = await import("../lib/studio/learn.js");
  const learn = async (p) => {
    await studio(p);
    for (const b of await p.$$(".st-blocks .btn")) await b.click();
    for (const b of await p.$$(".st-lvnav .btn")) { await b.click().catch(() => {}); await p.waitForTimeout(150); }
  };
  for (const l of LEVELS) await visit(`/studio?mode=learn&lv=${l.id}`, `studio level ${l.id}`, learn);
  // challenges: the list, then each one with its hint open
  await visit("/studio?mode=ch", "studio challenges", async (p) => { await p.click(".st-ch"); await p.waitForTimeout(300); });
  for (const c of CHALLENGES) await visit(`/studio?mode=ch&c=${c.id}`, `studio challenge ${c.id}`, async (p) => { await p.click(".st-hint2 summary"); await studio(p); });
  // speed levers, recipes, and the whole Colab loop with a pasted result
  await visit("/studio?m=gpt2", "studio levers and recipes", async (p) => {
    await p.click('.st-tabs button:has-text("Speed")');
    for (let i = 0; i < 4; i++) { const b = (await p.$$(".sp-lever .btn"))[i]; if (b) { await b.click(); await p.waitForTimeout(150); } }
    await p.click('.st-tabs button:has-text("Design")');
    for (let i = 0; i < 8; i++) { const b = (await p.$$(".st-recipes .btn"))[i]; if (b) { await b.click(); await p.waitForTimeout(120); } }
  });
  // a token's trip: the player steps through every stop, an edit stops it cleanly, the key help opens
  await visit("/studio?m=gpt2", "studio token flow", async (p) => {
    await p.click(".st-play"); await p.waitForTimeout(900);
    if (await p.$eval(".st-flowcap", (el) => el.hidden)) throw new Error("flow caption missing");
    await p.click(".st-fb.main"); const k0 = await p.textContent(".st-flowk");   // pause, then step on and back
    await p.click(".st-fb >> nth=2"); await p.waitForTimeout(100);
    if ((await p.textContent(".st-flowk")) === k0) throw new Error("step forward did nothing");
    await p.click(".st-fb >> nth=0"); await p.waitForTimeout(100);
    if ((await p.textContent(".st-flowk")) !== k0) throw new Error("step back did nothing");
    await p.click(".st-flowsp .st-fb"); await p.click(".st-fb.main"); await p.waitForTimeout(400);
    await p.click(".st-blocks button >> nth=0"); await p.waitForTimeout(200);
    if (!(await p.$eval(".st-flowcap", (el) => el.hidden))) throw new Error("flow did not stop on edit");
    if (!(await p.$(".st-dlt"))) throw new Error("no HUD delta after an edit");
    await p.click(".st-keybtn"); if (await p.$eval(".st-keys", (el) => el.hidden)) throw new Error("key help did not open");
    await p.goto(p.url().replace(/\?.*/, "?m=gpt2&at=b2")); await p.waitForTimeout(500);
    await p.click(".st-play"); await p.waitForTimeout(1200); await p.click(".st-fb.x");
    if (!(await p.$eval(".st-flowcap", (el) => el.hidden))) throw new Error("close did not close");
  });
  // time machine, change log, pin and compare, the sticker book, a saved picture, number keys
  await visit("/studio?m=gpt2", "studio time machine and stickers", async (p) => {
    await p.click(".st-tl li >> nth=1 >> button"); await p.waitForTimeout(500);
    if (!(await p.$(".st-jump .st-chg li"))) throw new Error("timeline jump shows no changes");
    await p.click(".st-pinbtn"); await p.click(".st-blocks button >> nth=0"); await p.waitForTimeout(200);
    if (!(await p.$(".st-cmpt tbody tr"))) throw new Error("no compare table");
    if (!(await p.$(".st-log .st-chg li"))) throw new Error("no change log");
    await p.click(".st-cmp button >> nth=0"); await p.waitForTimeout(200);
    await p.click(".st-tlh .btn"); await p.waitForTimeout(1500); await p.click(".st-tlh .btn");
    const [dl] = await Promise.all([p.waitForEvent("download"), p.click('.st-tools .btn[title^="Save"]')]);
    if (!(await dl.suggestedFilename()).endsWith(".png")) throw new Error("picture is not a png");
    await p.click(".st-bookbtn"); if (!(await p.$(".stk-grid li[data-got='1']"))) throw new Error("no sticker earned");
    await p.keyboard.press("Escape"); await p.keyboard.press("2");
    if ((await p.$eval(".st-tabs [aria-selected='true']", (e) => e.textContent)).indexOf("Predict") < 0) throw new Error("key 2 did not open Predict");
  });
  for (const m of ["gpt3", "llama2"]) await visit(`/studio?m=${m}`, `studio ${m}`);
  await visit("/studio?m=gpt2&at=b1/attn/h1", "studio head scan", async (p) => {
    await p.click(".st-play"); await p.keyboard.press("Space"); await p.keyboard.press("Period"); await p.waitForTimeout(200);
    if ((await p.$$(".st-scan span")).length !== 5) throw new Error("scan bars missing");
  });
  await visit("/studio?m=gpt2&at=head", "studio pick the next word", async (p) => {
    await p.$eval(".st-smp input[type=range]", (e) => { e.value = 0.1; e.dispatchEvent(new Event("input")); });
    await p.click(".st-smp-b button >> nth=1"); await p.waitForTimeout(200);
    const w = await p.$$eval(".st-smp-hist span", (a) => a.map((e) => e.textContent));
    if (w.length !== 10 || w.some((x) => x !== "mat")) throw new Error(`at temperature 0.1 expected 10 x mat, got ${w}`);
  });
  for (const lv of ["long", "experts"]) await visit(`/studio?mode=learn&lv=${lv}`, `studio level ${lv}`);
  await visit("/studio?mode=learn&lv=mlp", "studio guess and mix-up", async (p) => {
    await p.click(".st-guess-o button >> nth=1"); if (!(await p.$(".st-guess-why"))) throw new Error("guess did not reveal");
    await p.click(".st-myth summary");
  });
  const run = (f) => ({ label: f, finals: [f === "a" ? 1.62 : 2.1, f === "a" ? 1.64 : 2.2], curve: [[0, 4.2], [500, 2.4], [1500, f === "a" ? 1.63 : 2.15]], diverged: false, tps: 1000, params: 1e6, grads0: [0.3, 0.1, 0.02, 0.004, 1e-3, 2e-4], acts: [1, 2, 4, 8, 16, 32], sample: "To be, or not" });
  const line = "STUDIO_RESULT:" + Buffer.from(JSON.stringify({ v: 1, id: "x", size: "S", steps: 1500, seeds: 2, device: "test", quick: true, runs: { baseline: run("a"), design: run("b") } })).toString("base64url");
  await visit("/studio?m=gpt2&x=eyJyZXNpZHVhbCI6ZmFsc2V9", "studio test tab", async (p) => {
    await p.click('.st-tabs button:has-text("Test")');
    for (let i = 2; i >= 0; i--) { const b = (await p.$$(".st-pane:not([hidden]) .seg button"))[i]; if (b) { await b.click(); await p.waitForTimeout(100); } }
    await p.fill(".lab-ta", line); await p.waitForTimeout(300);
    if (!(await p.$(".lab-res"))) throw new Error("result card missing");
    for (let i = 0; i < 3; i++) { const b = (await p.$$(".lab-res .row .btn"))[i]; if (b) { await b.click(); await p.waitForTimeout(250); } }
  });
}

if (!only || only === "quiz") {
  if (!only) { await visit("/", "home"); await visit("/roadmap", "roadmap"); }
  const playOne = async (p) => { const o = await p.$(".qz-opt, .qp-chip, .qp-bit, .qp-cell, .qp-kc, .qp-gate button"); if (o) await o.click(); else await p.$eval(".qp-range", (el) => { el.value = el.max; el.dispatchEvent(new Event("input", { bubbles: true })); }).catch(() => {}); await p.waitForTimeout(200); };
  await visit("/quiz", "quiz", async (p) => { await p.click(".qz-word"); await p.click(".qz-go"); await playOne(p); });
  for (const k of ["estimate", "order", "build"]) await visit(`/quiz?kind=${k}`, `quiz ${k}`, async (p) => { await p.click(".qz-go"); for (let i = 0; i < 4; i++) { await playOne(p); const b = await p.$(".qz-nav .primary:not([disabled])"); if (!b) break; await b.click(); } });
  await visit("/settings", "settings", async (p) => { for (const b of await p.$$(".set-seg button")) await b.click().catch(() => {}); });
}
for (const c of ALL.filter((c) => c.live && (!only || c.id === only))) await visit(`/learn?c=${c.id}`, c.id, pokeAll);
await browser.close();
server.kill();
console.log(failed ? `${failed} page(s) failed` : "all pages ok");
process.exit(failed ? 1 : 0);
