import { h, rng } from "../lib/core.js";
import { slider, stats, predict, code, callout, section, prose, table, button, bars } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { MLP, dataset, bceLogit, sigmoid, softmax } from "../lib/nn.js";

const LOGITS = [["cat", 9.0], ["dog", 6.5], ["fox", 5.2], ["car", -1.0], ["teapot", -2.5]];

/** Adam on binary cross-entropy (soft targets allowed); bs = batch size, Infinity for full batch. */
function train(net, pts, epochs, lr = 0.01, soft = null, bs = Infinity) {
  for (let e = 0; e < epochs; e++) {
    for (let s0 = 0; s0 < pts.length; s0 += bs) {
      const end = Math.min(pts.length, s0 + bs);
      for (let i = s0; i < end; i++) { const z = net.forward(pts[i].x)[0]; net.backward([sigmoid(z) - (soft ? soft[i] : pts[i].y)]); }
      net.step(lr, end - s0);
    }
  }
  return net;
}
const accuracy = (f, pts) => pts.filter((p) => (f(p.x) > 0.5) === !!p.y).length / pts.length;

export default function (root, { id }) {
  // 1. temperature and dark knowledge
  let T = 1;
  const tEl = h("div");
  function drawT() {
    const p = softmax(LOGITS.map((x) => x[1]), T);
    tEl.replaceChildren(bars(LOGITS.map((x, i) => ({ label: x[0], value: p[i] * 100, color: i === 0 ? "var(--accent)" : "var(--c4)" })), { max: 100, fmt: (v) => `${v.toFixed(v < 1 ? 2 : 1)}%` }));
  }

  // 2. teacher -> student, trained live
  const train0 = dataset("spiral", 120, { noise: 0.05, seed: 2 }), few = train0.filter((_, i) => i % 8 < 2), test = dataset("spiral", 600, { noise: 0.05, seed: 77 });
  const transfer = Array.from({ length: 800 }, (_, i) => { const r = rng(i + 5); return { x: [r() * 2.4 - 1.2, r() * 2.4 - 1.2], y: 0 }; });
  let teacher, sHard, sSoft, studentSize = 12, Tk = 1;
  const planes = [0, 1, 2].map(() => plane({ width: 260, height: 260, label: "Decision regions" }));
  const st = stats([{ label: "Teacher (2→32→32→1, 120 labels)" }, { label: "Student: 30 labels only" }, { label: "Student: taught by the teacher" }]);
  const status = h("p.canvas-cap", "Press Train to start. The teacher takes a second or two.");
  function drawModels() {
    const fs = [teacher, sHard, sSoft];
    planes.forEach((pl, k) => { pl.clear(); if (fs[k]) pl.field((x, y) => sigmoid(fs[k].forward([x, y])[0]), { res: 46, alpha: 0.55 }); pl.scatter(k === 1 ? few : train0, { r: k === 1 ? 4 : 2.5 }); });
    fs.forEach((f, k) => f && st.set(k, `${(accuracy((x) => sigmoid(f.forward(x)[0]), test) * 100).toFixed(1)}% test`, k === 2 && sSoft && sHard && accuracy((x) => sigmoid(sSoft.forward(x)[0]), test) > accuracy((x) => sigmoid(sHard.forward(x)[0]), test) ? "good" : ""));
  }
  function run() {
    status.textContent = "Training the teacher…";
    setTimeout(() => {
      teacher = train(new MLP([2, 32, 32, 1], { act: "tanh", seed: 1 }), train0, 1500, 0.01);
      status.textContent = "Training two identical small students…";
      setTimeout(() => {
        sHard = train(new MLP([2, studentSize, studentSize, 1], { act: "tanh", seed: 9 }), few, 1500, 0.01);
        const soft = transfer.map((p) => sigmoid(teacher.forward(p.x)[0] / Tk)); // the teacher labels many unlabelled points, softly
        sSoft = train(new MLP([2, studentSize, studentSize, 1], { act: "tanh", seed: 9 }), transfer, 200, 0.01, soft, 32);
        status.textContent = `Both students have ${sHard.params} parameters; the teacher has ${teacher.params}.`;
        drawModels();
      }, 20);
      drawModels();
    }, 20);
  }

  // 3. pruning
  const pruneChart = xyChart({ height: 220, xLabel: "fraction of weights removed (smallest first)", yLabel: "test accuracy", xDomain: [0, 1], yDomain: [0.4, 1] });
  function drawPrune() {
    if (!teacher) return;
    const all = teacher.W.flatMap((W) => [...W].map(Math.abs)).sort((a, b) => a - b), pts = [];
    for (let f = 0; f <= 0.98; f += 0.04) {
      const th = all[Math.floor(f * (all.length - 1))], saved = teacher.W.map((W) => Float64Array.from(W));
      teacher.W.forEach((W) => W.forEach((v, i) => { if (Math.abs(v) < th) W[i] = 0; }));
      pts.push([f, accuracy((x) => sigmoid(teacher.forward(x)[0]), test)]);
      teacher.W = saved;
    }
    pruneChart.update([{ name: "Magnitude pruning, no retraining", color: "var(--c2)", points: pts, dots: true, r: 3 }], { xDomain: [0, 1], yDomain: [0.4, 1] });
  }

  // 4. merging
  const mst = stats([{ label: "Model A" }, { label: "Model B" }, { label: "Average of A and B" }]);
  const mergeNote = h("p.canvas-cap");
  function merge(sameInit) {
    const half1 = train0.filter((_, i) => i % 4 < 2), half2 = train0.filter((_, i) => i % 4 >= 2); // both classes in each half
    const base = train(new MLP([2, 24, 24, 1], { act: "tanh", seed: 3 }), train0, 150, 0.01); // shared, partly trained "pretraining"
    const clone = (n) => { const c = new MLP([2, 24, 24, 1], { act: "tanh", seed: 3 }); c.W = n.W.map((w) => Float64Array.from(w)); c.b = n.b.map((b) => Float64Array.from(b)); return c; };
    const A = train(sameInit ? clone(base) : new MLP([2, 24, 24, 1], { act: "tanh", seed: 11 }), half1, 600, 0.01);
    const B = train(sameInit ? clone(base) : new MLP([2, 24, 24, 1], { act: "tanh", seed: 12 }), half2, 600, 0.01);
    const M = clone(A); M.W = A.W.map((w, l) => w.map((v, i) => (v + B.W[l][i]) / 2)); M.b = A.b.map((b, l) => b.map((v, i) => (v + B.b[l][i]) / 2));
    [A, B, M].forEach((n, k) => { const a = accuracy((x) => sigmoid(n.forward(x)[0]), test); mst.set(k, `${(a * 100).toFixed(1)}%`, k === 2 ? (a > 0.8 ? "good" : "bad") : ""); });
    mergeNote.textContent = sameInit ? "Both fine-tuned from the same pretrained weights: they stay in the same basin, so their average still works." : "Trained from different random inits: hidden units are in different orders, so averaging mixes unrelated neurons.";
  }

  root.append(
    section(null, prose(`<p>The best model is rarely the one you can afford to serve. <b>Distillation</b> trains a small <i>student</i> to imitate a big <i>teacher</i>. <b>Pruning</b> deletes weights that don't matter. <b>Merging</b> combines separately fine-tuned models by averaging their weights. Together with quantisation (systems level), these are how frontier capability reaches phones and cheap API tiers.</p>`)),
    section("Dark knowledge: the teacher's wrong answers are informative",
      prose("<p>A teacher looking at a cat photo puts most probability on “cat”, but noticeably more on “dog” and “fox” than on “teapot”. Those small numbers say <i>what a cat is like</i>. Hinton, Vinyals and Dean (2015) called this <b>dark knowledge</b>. Raising the softmax temperature T makes it visible, and training the student on these soft targets passes it on.</p>"),
      h("div.stage",
        h("div.stage-controls", slider({ label: "Temperature T", min: 0.5, max: 10, step: 0.1, value: T, fmt: (v) => v.toFixed(1), onInput: (v) => { T = v; drawT(); } })),
        h("div.stage-body", tEl))),
    section("Distil a network, live",
      prose("<p>A big teacher learns the spiral from 120 labelled points. You want a small, cheap model, but you only have 30 labels of your own, plus plenty of <b>unlabelled</b> inputs. Two identical students train: one on your 30 labels, one on the teacher's soft predictions for 800 unlabelled points. That's the everyday LLM situation: you have prompts, and a strong model that can answer them.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Student width (two hidden layers)", min: 4, max: 16, value: studentSize, onInput: (v) => (studentSize = v) }),
          slider({ label: "Distillation temperature", min: 1, max: 5, step: 0.5, value: Tk, fmt: (v) => v.toFixed(1), onInput: (v) => (Tk = v) }),
          button("Train teacher and students", run, "primary")),
        h("div.stage-body", st.el,
          h("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px" } },
            ["Teacher", "Student: 30 labels", "Student: distilled"].map((t, k) => h("div", h("div.eyebrow", t), h("div.canvas-box", planes[k].el)))), status)),
      callout("tip", "Why it works", "<p>Thirty labels barely constrain the spiral. The teacher's soft outputs constrain the whole plane, including the regions between the arms of the spiral, and say how confident to be near the boundary. This is why distilling from a strong model's outputs (often its generated text) is the cheapest way to build a good small LLM.</p>"),
    ),
    section("Pruning: how many weights does it need?",
      prose("<p>Zero out the smallest weights of the trained teacher, without retraining, and measure test accuracy. (Train the teacher above first.)</p>"),
      h("div.stage", h("div.stage-controls", button("Prune the teacher", drawPrune, "primary")), h("div.stage-body", pruneChart.el)),
      prose("<p>Networks tolerate surprising amounts of pruning, especially with a little retraining afterwards. The <b>lottery ticket hypothesis</b> (Frankle and Carbin, 2019) found sparse subnetworks that, trained from their original initialisation, match the full network. In practice, structured pruning (removing whole heads, layers or channels) is what speeds up real hardware, as in NVIDIA's Minitron models.</p>"),
    ),
    section("Merging: average two models",
      h("div.stage",
        h("div.stage-controls", button("Fine-tune A and B from the same model, then average", () => merge(true), "primary"), button("Train A and B from different random starts, then average", () => merge(false))),
        h("div.stage-body", mst.el, mergeNote)),
      predict({
        chapter: id,
        question: "Why does averaging two separately trained networks usually fail, but averaging two fine-tunes of the same base model often works?",
        options: ["Fine-tuned models are smaller", "Hidden units can be permuted freely; independent runs order them differently, while fine-tunes from one base stay aligned in the same low-loss basin", "Averaging always fails", "Fine-tuning uses a lower learning rate, so nothing changes"],
        answer: 1,
        explain: "<p>Swap two hidden neurons (and their weights) and the network computes exactly the same function, so independent networks are like the same circuit with shuffled wiring. Fine-tunes from a shared base stay close in weight space, where loss is roughly convex, so averages (“model soups”, task arithmetic, TIES merging) keep, and often combine, their skills.</p>",
      }),
      table(["Technique", "Smaller by", "Typical cost"], [
        ["Distillation", "Choose any student size", "A training run on teacher outputs"],
        ["Unstructured pruning", "50–90% of weights", "Little accuracy loss, but needs sparse kernels to be faster"],
        ["Structured pruning + distillation", "Remove heads, layers, width", "Real speedups; e.g. 15B → 8B → 4B families"],
        ["Quantisation", "2–8× fewer bytes per weight", "See the precision and inference chapters"],
        ["Merging", "Not smaller: many skills in one model", "No training at all"],
      ]),
    ),
    section("In code",
      code({
        "Distillation loss": `import torch.nn.functional as F

def distill_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    soft = F.kl_div(F.log_softmax(student_logits / T, -1),
                    F.softmax(teacher_logits / T, -1), reduction="batchmean") * T * T
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard        # T*T keeps gradient scale comparable`,
        "Model soup": `def average(state_dicts):
    return {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
merged.load_state_dict(average([ft_a.state_dict(), ft_b.state_dict()]))`,
      }, "Project 27 distils, prunes and quantises a CNN and tables the trade-offs."),
    ),
  );
  drawT(); planes.forEach((p) => p.paint(drawModels));
}
