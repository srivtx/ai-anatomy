import { h, loop } from "../lib/core.js";
import { stats, predict, code, callout, section, prose, table, button, segmented } from "../lib/widgets.js";
import { plane, xyChart } from "../lib/plot.js";
import { MLP } from "../lib/nn.js";
import { arms, trainEpoch, acc, align, lerpNet, meanLoss, argmax, stitch } from "../lib/minds.js";

const EPOCHS = 150;
const BETS = [
  ["Two frontier LLMs from different labs can be stitched at a middle layer with one linear map, losing under 5 points on a typical benchmark", 70],
  ["Fine-tunes of different open base models will be mergeable after permutation-and-rotation alignment, once the bases are above ~30B parameters", 45],
  ["Representational alignment between a model's vision and language encoders will turn out to predict its multimodal benchmark score better than parameter count does", 60],
  ["Interpretability features found by an SAE on one model will have near-copies (cosine > 0.8 after alignment) in most other models of similar capability", 65],
];

export default function (root, { id }) {
  const data = arms(300), test = arms(300, 9), probe = arms(300, 11).map((p) => p.x);
  let width = 64, A, B, Bp, ep = 0, done = false;
  const colors = ["var(--c1)", "var(--c2)", "var(--c3)"];
  const mk = (label) => plane({ width: 190, height: 190, xr: [-1.2, 1.2], yr: [-1.2, 1.2], label });
  const P = { A: mk("Network A"), B: mk("Network B"), mid: mk("Halfway, naive"), midP: mk("Halfway, after aligning") };
  const pc = xyChart({ width: 520, height: 230, xLabel: "t: weights (1 − t)·A + t·B", yLabel: "test loss", xDomain: [0, 1], fmtX: (v) => v.toFixed(1) });
  const st = stats([{ label: "Epoch" }, { label: "Accuracy A / B" }, { label: "Halfway, naive" }, { label: "Halfway, aligned" }, { label: "Stitched A→B" }]);
  const off = document.createElement("canvas");
  function paintNet(pl, m) {
    pl.paint(() => {
      pl.clear();
      const res = 48; off.width = off.height = res;
      const o = off.getContext("2d"), img = o.createImageData(res, res);
      const cs = [pl.colors.c1, pl.colors.c2, pl.colors.c3].map((c) => [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16)));
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
        const [x, y] = pl.inv(((i + 0.5) / res) * 190, ((j + 0.5) / res) * 190), c = cs[argmax(m.forward([x, y]))], k = (j * res + i) * 4;
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
      o.putImageData(img, 0, 0);
      pl.ctx.globalAlpha = 0.4; pl.ctx.imageSmoothingEnabled = false; pl.ctx.drawImage(off, 0, 0, 190, 190); pl.ctx.globalAlpha = 1;
      pl.scatter(data.filter((_, k) => k % 2 === 0), { r: 2.5, color: (p) => [pl.colors.c1, pl.colors.c2, pl.colors.c3][p.y] });
    });
  }
  function reset() {
    lp.stop(); ep = 0; done = false;
    A = new MLP([2, width, width, 3], { act: "relu", seed: 1 }); B = new MLP([2, width, width, 3], { act: "relu", seed: 2 }); Bp = null;
    for (const k of ["mid", "midP"]) P[k].paint(() => P[k].clear());
    pc.update([{ points: [[0, 0]], color: "transparent" }], { yDomain: [0, 3] });
    for (let i = 2; i < 5; i++) st.set(i, "–", "");
    draw(); lp.start();
  }
  function draw() {
    st.set(0, `${ep} / ${EPOCHS}`); st.set(1, `${Math.round(acc(A, test) * 100)}% / ${Math.round(acc(B, test) * 100)}%`);
    paintNet(P.A, A); paintNet(P.B, B);
  }
  function finish() {
    done = true; Bp = align(A, B, probe);
    const ts = Array.from({ length: 11 }, (_, i) => i / 10);
    const naive = ts.map((t) => [t, meanLoss(lerpNet(A, B, t), test)]), al = ts.map((t) => [t, meanLoss(lerpNet(A, Bp, t), test)]);
    pc.update([{ name: "naive straight line", points: naive, color: "var(--c2)", dots: true }, { name: "after re-ordering B's units", points: al, color: "var(--accent)", dots: true }], { yDomain: [0, Math.max(1.5, ...naive.map((p) => p[1])) * 1.05] });
    const mN = lerpNet(A, B, 0.5), mP = lerpNet(A, Bp, 0.5);
    const aN = acc(mN, test), aP = acc(mP, test);
    st.set(2, `${Math.round(aN * 100)}%`, aN < 0.6 ? "bad" : ""); st.set(3, `${Math.round(aP * 100)}%`, aP > 0.8 ? "good" : "");
    st.set(4, `${Math.round(stitch(A, B, probe, test, 1) * 100)}%`);
    paintNet(P.mid, mN); paintNet(P.midP, mP);
  }
  const lp = loop(() => {
    const k = width >= 128 ? 1 : width >= 64 ? 2 : 4;
    for (let i = 0; i < k && ep < EPOCHS; i++) { trainEpoch(A, data, 0.01); trainEpoch(B, data, 0.01); ep++; }
    draw();
    if (ep >= EPOCHS) { lp.stop(); finish(); }
  }, pc.el);

  root.append(
    section(null, prose(`<p>Train the same network twice from different random seeds and you get two sets of weights that look nothing alike. Are they two different solutions, or the same one in disguise? A line of work since 2019 says: increasingly, <b>the same one</b>. Networks that do the same job converge on the same internal solution, up to symmetries like re-ordering their neurons, and the bigger and more capable they get, the more alike they become. This is one of the most surprising and least taught facts about deep learning, and it's what makes model merging, model stitching and transferable interpretability possible.</p>`)),
    section("Two networks, one basin",
      prose(`<p>Below, networks A and B (same shape, different seeds) learn the same three-class task. When they finish, we walk in a straight line through weight space from A to B and measure the loss. Then we re-order B's hidden neurons to match A's, choosing the order by solving an assignment problem on their activations (Hungarian algorithm), and walk again. Re-ordering neurons doesn't change what B computes at all.</p>`),
      predict({ chapter: id, question: "A and B both reach ~90% accuracy. What's the accuracy of the network halfway between them, (A + B)/2, before any re-ordering?",
        options: ["About 90%: both are good, so their average is good", "Much worse, near chance: the average of two good networks is a bad network", "Better than either: averaging is like an ensemble"], answer: 1,
        explain: "Neuron 7 of A and neuron 7 of B do unrelated jobs, so averaging them blends unrelated features into mush. That's the loss barrier. Once B's neurons are put in A's order, the average is nearly as good as either network, more so the wider they are." }),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Hidden width", options: [{ value: 32, label: "32" }, { value: 64, label: "64" }, { value: 128, label: "128 (slower)" }], value: width, onChange: (v) => { width = v; reset(); } }),
          button("Retrain", reset)),
        h("div.stage-body", st.el,
          h("div.lab",
            h("div.agrid4", ...["A", "B", "mid", "midP"].map((k) => h("div.canvas-box", P[k].el))),
            pc.el))),
      callout("note", "What I measured", `<p>Halfway accuracy, naive → aligned: width 64: 44% → 77%; width 128: 40% → 86%, and the loss barrier almost disappears. The wider the network, the better re-ordering alone removes the barrier, as Ainsworth et al. (2022, “Git Re-Basin”) found for ResNets. Entezari et al. (2021) conjectured that, modulo permutations, <b>all</b> SGD solutions of a wide enough network lie in one connected basin. A network with n hidden units has n! copies of every solution; two trainings just land in different copies.</p>`),
    ),
    section("Same mind, different bodies",
      prose(`<p><b>Model stitching</b> asks a sharper question: can network A's first layers feed network B's last layers through just a linear map? If yes, they compute interchangeable representations. In the toy above, stitching A's second layer into B's head scores near the networks' own accuracy (the last stat). At init, it's at chance (~30%).</p>`),
      table(["Finding", "Who", "What it says"], [
        ["Similar representations across seeds", "Kornblith et al., 2019 (CKA)", "Wide networks trained from different seeds have nearly identical representations, measured by centred kernel alignment; differences grow in later layers and in narrow networks"],
        ["Stitching works across seeds and methods", "Bansal, Nakkiran & Barak, 2021", "Even networks trained with different objectives (supervised vs self-supervised) stitch with small loss; better models stitch better"],
        ["The Platonic Representation Hypothesis", "Huh, Cheung, Wang & Isola, 2024", "As vision and language models get more capable, the similarity structure of their representations converges, even across modalities: better LLMs align more with better vision models"],
        ["Embeddings translate without pairs", "Jha et al., 2025 (vec2vec)", "Text embeddings from different models can be mapped onto each other with no paired data at all, recovering much of the original meaning"],
        ["Fine-tunes of one base merge freely", "Wortsman 2022 (soups); Ilharco 2023 (task arithmetic)", "Models fine-tuned from the same pretrained weights stay in one basin, so their weights can be averaged or added (“+ maths skill − toxicity”) without any alignment"],
      ]),
      callout("warn", "Where my toy fails, and why it works at scale", `<p>I also measured linear CKA between A and B here. It's ~0.8 <i>at initialisation</i> and stays around 0.7–0.8: in a 2-D toy every network, trained or not, is mostly a smooth function of the same two input coordinates, so CKA can't see learning. Agreement maps tell a similar non-story. Convergence needs <b>pressure</b>: the argument of Huh et al. (and Cao &amp; Yamins' “contravariance principle”) is that each extra task or datum rules out solutions, and a model good at everything has few representations left to choose from. A 3-class spiral constrains almost nothing. A model trained on the internet constrains a lot. So the effect should be strongest exactly where we can't run it here: big models, broad data. That's the pattern of the published evidence too.</p>`),
    ),
    section("My bets",
      prose("<p>Untested predictions, with my confidence. Each is testable with open models and a few GPU-days.</p>"),
      h("div.finding-grid", BETS.map(([t, c]) => h("div.finding", h("p", { style: { margin: 0 } }, t), h("div", { style: { height: "6px", borderRadius: "3px", marginTop: "10px", background: `linear-gradient(90deg, var(--c5) ${c}%, var(--sunken) ${c}%)` } }), h("small.hint", `my confidence: ${c}%`)))),
    ),
    section("In code",
      code({
        "Weight matching (PyTorch)": `from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def align_mlp(A, B, X):
    """Permute B's hidden units to match A's by activation correlation. A, B: lists of nn.Linear."""
    B = [copy.deepcopy(l) for l in B]; ha, hb = X, X
    for i in range(len(A) - 1):
        ha, hb = torch.relu(A[i](ha)), torch.relu(B[i](hb))
        za = (ha - ha.mean(0)) / (ha.std(0) + 1e-6); zb = (hb - hb.mean(0)) / (hb.std(0) + 1e-6)
        _, perm = linear_sum_assignment(-(za.T @ zb).numpy())       # maximise correlation
        perm = torch.as_tensor(perm)
        B[i].weight.copy_(B[i].weight[perm]); B[i].bias.copy_(B[i].bias[perm])
        B[i + 1].weight.copy_(B[i + 1].weight[:, perm]); hb = hb[:, perm]
    return B

@torch.no_grad()
def lerp(A, B, t):
    out = copy.deepcopy(A)
    for lo, la, lb in zip(out, A, B):
        lo.weight.copy_((1 - t) * la.weight + t * lb.weight); lo.bias.copy_((1 - t) * la.bias + t * lb.bias)
    return out`,
        "Task arithmetic": `base = AutoModelForCausalLM.from_pretrained("base")
math_ft = AutoModelForCausalLM.from_pretrained("base-finetuned-math")
code_ft = AutoModelForCausalLM.from_pretrained("base-finetuned-code")
sd = base.state_dict()
for k in sd:
    tau_math = math_ft.state_dict()[k] - sd[k]          # a "task vector"
    tau_code = code_ft.state_dict()[k] - sd[k]
    sd[k] = sd[k] + 0.7 * tau_math + 0.7 * tau_code     # one model with both skills
base.load_state_dict(sd)
# works because both fine-tunes stayed in the base model's basin: no alignment needed`,
        "Linear CKA": `def cka(X, Y):
    """X: (n, p), Y: (n, q) activations on the same n inputs. 1 = same up to rotation and scale."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())`,
        "Stitching two LLMs": `# hidden states of two different models on the same 10k sentences, at a middle layer
HA = hidden(model_a, texts, layer=12)        # (tokens, d_a)   – align tokens by character spans
HB = hidden(model_b, texts, layer=14)        # (tokens, d_b)
W = torch.linalg.lstsq(HA, HB).solution      # one linear map
# run model_a up to layer 12, map with W, continue in model_b from layer 14, measure perplexity`,
      }, "Capstone 41 (sparse autoencoders on a real model) can be extended to compare features across two models."),
    ),
  );
  reset();
}
