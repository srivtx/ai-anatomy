import { h, rng, loop } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { LoopNet, maze } from "../lib/looped.js";

const TRAIN_N = 7;

export default function (root, { id }) {
  let net, steps, avg, r, testN = 21, m, T = 60, curve = null;
  const pl = plane({ width: 420, height: 420, xr: [0, 1], yr: [0, 1], label: "Maze and the network's flood fill" });
  const st = stats([{ label: "Training steps (7×7 mazes)" }, { label: "Training loss" }, { label: "Iterations this maze needs" }, { label: "Cells right after T iterations" }]);
  const solveC = xyChart({ width: 460, height: 280, xLabel: "test-time iterations T", yLabel: "mazes fully solved", xDomain: [0, 150], yDomain: [0, 1] });
  const testSet = {};
  function init() {
    net = new LoopNet(1); steps = 0; avg = 0; r = rng(1); curve = null;
    newMaze(); drawCurve();
  }
  function newMaze() { const r2 = rng(Math.floor(Math.random() * 1e6)); m = maze(testN, r2, 0.66, true); drawMaze(); }
  function drawMaze() {
    const R = net.run(m, T).states[T], n = m.n;
    pl.paint(() => {
      pl.clear();
      const ctx = pl.ctx, cw = pl.W / n, chh = pl.H / n, C = pl.colors;
      for (let c = 0; c < n * n; c++) {
        const x = (c % n) * cw, y = ((c / n) | 0) * chh;
        if (!m.open[c]) { ctx.fillStyle = C.ink; ctx.globalAlpha = 0.85; ctx.fillRect(x, y, cw, chh); ctx.globalAlpha = 1; continue; }
        ctx.fillStyle = C.surface; ctx.fillRect(x, y, cw, chh);
        ctx.fillStyle = C.accent; ctx.globalAlpha = R[c]; ctx.fillRect(x, y, cw, chh); ctx.globalAlpha = 1;
        // cells the network gets wrong: reachable but dark, or lit but unreachable
        if ((R[c] > 0.5) !== !!m.reach[c] && steps) { ctx.strokeStyle = C.bad; ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, cw - 4, chh - 4); }
      }
      pl.dot(cw / 2, chh / 2, Math.min(cw, chh) * 0.3, C.good);
      const ex = n * n - 1; pl.dot(((ex % n) + 0.5) * cw, (((ex / n) | 0) + 0.5) * chh, Math.min(cw, chh) * 0.3, m.reach[ex] ? C.good : C.bad);
    });
    let ok = 0; for (let c = 0; c < n * n; c++) ok += (R[c] > 0.5) === !!m.reach[c];
    st.set(0, String(steps)); st.set(1, steps ? avg.toFixed(4) : "–", avg < 0.01 && steps ? "good" : "");
    st.set(2, String(m.depth + 1)); st.set(3, `${((ok / (n * n)) * 100).toFixed(1)}%`, ok === n * n ? "good" : T < m.depth + 1 ? "warn" : "bad");
  }
  function drawCurve() {
    if (!curve) { solveC.update([{ name: "Train first", color: "var(--ink-3)", points: [[0, 0]] }], { xDomain: [0, 150], yDomain: [0, 1] }); return; }
    solveC.update(curve, { xDomain: [0, 150], yDomain: [0, 1], marks: [{ x: T, label: "T" }] });
  }
  function computeCurve() {
    const sizes = [[7, "var(--c3)"], [13, "var(--c2)"], [21, "var(--accent)"], [31, "var(--c4)"]];
    curve = sizes.map(([n, color]) => {
      const set = (testSet[n] ??= Array.from({ length: n > 21 ? 8 : 16 }, (_, i) => maze(n, rng(500 + i * 7 + n), 0.66, true)));
      const solvedAt = new Array(151).fill(0);
      for (const mz of set) {
        const { states } = net.run(mz, 150);
        for (let t = 0; t <= 150; t += 5) solvedAt[t] += mz.reach.every((y, c) => (states[t][c] > 0.5) === !!y) / set.length;
      }
      return { name: `${n}×${n}`, color, points: solvedAt.map((v, t) => [t, v]).filter(([t]) => t % 5 === 0) };
    });
    drawCurve();
  }
  const lp = loop(() => {
    for (let k = 0; k < 4; k++) { const batch = Array.from({ length: 8 }, () => maze(TRAIN_N, r)); const l = net.train(batch, 6 + Math.floor(r() * 8), 0.03, "steps"); avg = steps ? 0.95 * avg + 0.05 * l : l; steps++; }
    drawMaze();
    if (steps >= 400) { lp.stop(); computeCurve(); drawMaze(); }
  }, pl.el);
  let anim = null;
  function play() { T = 0; tSlider.value = 0; clearInterval(anim); anim = setInterval(() => { T = Math.min(150, T + 1); tSlider.value = T; drawMaze(); drawCurve(); if (T >= Math.min(150, m.depth + 8)) clearInterval(anim); }, 60); }
  const tSlider = slider({ label: "Test-time iterations T", min: 0, max: 150, value: T, onInput: (v) => { T = v; drawMaze(); drawCurve(); } });

  root.append(
    section(null, prose(`<p>A normal network has a fixed depth: 32 layers means 32 rounds of computation, for an easy question and a hard one alike. A <b>looped</b> (or recurrent-depth) model reuses the <b>same</b> block of layers again and again, so you can choose how many times to run it at test time. More loops, more computation, with no new parameters. Unlike chain of thought, the extra thinking happens in the hidden state rather than in written tokens. It's an old idea (Universal Transformers, 2018; Adaptive Computation Time, 2016) that came back strongly in 2025.</p>`)),
    section("Train small, then think longer on bigger problems",
      prose(`<p>One tiny rule (a network with 7 inputs and 10 hidden units, about 100 parameters) is applied to every cell of a maze, all cells at once, over and over. Each cell sees whether it's open, whether it's the start, and its own and its four neighbours' current values. It's trained only on <b>${TRAIN_N}×${TRAIN_N}</b> mazes with up to 13 iterations. Then we test it on mazes up to 31×31, which need far more iterations than it ever ran in training. Violet shows which cells it believes are reachable from the green start; red outlines are mistakes.</p>`),
      h("div.stage",
        h("div.stage-controls",
          button("Train", () => lp.start(), "primary"), button("Pause", () => lp.stop()), button("Reset", () => { lp.stop(); init(); }),
          tSlider,
          segmented({ label: "Test maze size", options: [9, 15, 21, 31].map((v) => ({ value: v, label: `${v}×${v}` })), value: testN, onChange: (v) => { testN = v; newMaze(); } }),
          button("New maze", newMaze), button("▶ Watch it think", play)),
        h("div.stage-body", st.el, h("div.lab", h("div.canvas-box", pl.el), h("div", solveC.el, h("p.canvas-cap", "After training: the fraction of test mazes solved perfectly, by size and number of iterations. Bigger mazes need more iterations, and get them for free."))))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Train (about 400 steps), pick a 31×31 maze and press <b>Watch it think</b>. The reachable region floods outward one cell per iteration, through corridors it has never seen at this scale.</li><li>Set T below “iterations this maze needs”: the far side is still dark. The network isn't wrong, it simply hasn't finished thinking. Give it more iterations and it completes the job.</li><li>Leave T at 150 on a small maze: the answer stays put. A well-trained looped model reaches a stable fixed point, so extra iterations don't hurt. Models without this property can “overthink” and drift (Bansal et al., 2022, fixed it by re-feeding the input every iteration, as we do here).</li></ul>"),
      callout("note", "Honest footnote", "<p>Here every iteration is supervised: after t loops, cells within t steps of the start should be lit (a form of <b>deep supervision</b>). Trained only on the final answer, this tiny model doesn't learn in the time a web page allows. Research models use similar aids: progressive losses, deep supervision (as in HRM and TRM), and random iteration counts during training so the model can't rely on a particular depth.</p>"),
      predict({
        chapter: id,
        question: "Why can a looped model solve a maze larger than any it was trained on, while a fixed 12-layer network can't?",
        options: ["It has more parameters", "Its computation can grow with the problem: more loops at test time, while a fixed-depth network can pass information only ~12 cells away", "It memorised larger mazes", "Larger mazes are easier"],
        answer: 1,
        explain: "<p>Information in a maze has to travel along corridors, one step per round of local computation. A fixed-depth network has a fixed reach. A looped network can iterate until the answer has propagated. That's <b>length generalisation</b> through depth, and it's the same reason chain of thought helps: both add serial computation steps.</p>",
      }),
    ),
    section("Looping in language models",
      table(["Approach", "What loops", "Notes"], [
        ["Universal Transformer (Dehghani et al., 2018)", "One shared transformer block, applied many times", "With adaptive halting per token"],
        ["ALBERT (2019)", "All layers share weights", "Fewer parameters, same compute"],
        ["Looped transformers (Giannou et al., 2023; Yang et al., 2023)", "A few shared layers", "Can emulate programs; better at learning in-context algorithms"],
        ["Coconut (Hao et al., 2024)", "The last hidden state is fed back as the next input embedding", "“Continuous thought”: reasoning without words, can keep several options open"],
        ["Recurrent depth (Geiping et al., 2025)", "A middle block repeated a random number of times in training", "A 3.5B model that improves with more loops at test time, like a reasoning model that thinks silently"],
        ["HRM and TRM (2025)", "Small networks recursing at two speeds / one tiny network recursing on its own answer", "TRM: about 7M parameters, strong on Sudoku, mazes and ARC-AGI puzzles"],
        ["Mixture-of-depths, early exit", "Skip layers for easy tokens", "The reverse: spend less compute where it isn't needed"],
      ]),
      prose("<p>The trade-off with chain of thought: latent loops are cheaper (no tokens to decode, no KV cache growth) and can carry richer information than a word, but they are invisible. You can read a chain of thought; you can't read a hidden state without interpretability tools. That matters for oversight.</p>"),
    ),
    section("In code",
      code({
        "Recurrent depth": `class LoopedLM(nn.Module):
    def __init__(self, prelude, core, coda):
        super().__init__()
        self.prelude, self.core, self.coda = prelude, core, coda   # core: a few transformer blocks
    def forward(self, tokens, n_loops):
        e = self.prelude(tokens)
        s = torch.randn_like(e)                     # latent state
        for _ in range(n_loops):                    # same weights every time
            s = self.core(s + e)                    # re-inject the input each loop ("recall")
        return self.coda(s)

# training: sample n_loops randomly (e.g. Poisson around 8) so the model works at any depth;
# at test time, turn n_loops up on hard inputs`,
      }, "Project 14 trains a looped transformer on addition and shows it extrapolates to longer numbers with more loops."),
    ),
  );
  init();
}
