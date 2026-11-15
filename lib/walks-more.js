// More step-through walkthroughs: tokenizers, state-space scans, RL objectives, positions, patches,
// sparse features, clustering and splitting a layer across GPUs. Same widget as walks.js.
import { tensorWalk, op, M, rng, fmt } from "./tensorviz.js";

const r2 = (A, d = 2) => M.round(A, d);
const dims = (t, d, extra = {}) => ({ data: t, dims: d, ...extra });
const col = (v) => v.map((x) => [x]);
const softplus = (x) => Math.log1p(Math.exp(x));
const keysOf = (R, C, l) => Array.from({ length: R * C }, (_, k) => (l == null ? `${Math.floor(k / C)},${k % C}` : `${l},${Math.floor(k / C)},${k % C}`));
const rowK = (i, C) => Array.from({ length: C }, (_, j) => `${i},${j}`);
const colK = (j, R) => Array.from({ length: R }, (_, i) => `${i},${j}`);
const std = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };

export const MORE = {
  // -------------------------------------------------------------------------------------------
  tokenization() {
    const freq = { hug: 10, pug: 5, pun: 12, bun: 4, hugs: 5 }, words = Object.keys(freq), W = 4;
    const PAL = { b: "#cfc8de", g: "#a9cbef", h: "#cdbff4", n: "#a7e0cb", p: "#f6c3d7", s: "#f6dca5", u: "#ddd8e8", ug: "#4f8ef7", un: "#1fa58f", hug: "#7b61ff" };
    const colors = (v) => (v == null ? "#f6f4fa" : PAL[v] || "#9aa3b5");
    const grid = (S) => S.map((t) => [...t, ...Array(W - t.length).fill(null)]);
    const total = (S) => S.reduce((a, t, i) => a + t.length * freq[words[i]], 0);
    const rowNames = words.map((w) => `“${w}” ×${freq[w]}`);
    const T = { F: dims(col(words.map((w) => freq[w])), "word, count", { label: "counts", kind: "prob", bars: 4, rowNames }) };
    let sp = words.map((w) => [...w]);
    const rounds = [];
    for (let r = 0; r < 3; r++) {
      const counts = new Map(), where = new Map();
      sp.forEach((t, i) => t.slice(0, -1).forEach((a, j) => { const k = `${a} ${t[j + 1]}`; counts.set(k, (counts.get(k) || 0) + freq[words[i]]); if (!where.has(k)) where.set(k, []); where.get(k).push([i, j]); }));
      const pairs = [...counts.keys()], best = pairs.reduce((a, b) => (counts.get(b) > counts.get(a) ? b : a)), [A, B] = best.split(" ");
      const landed = [];
      const next = sp.map((t, i) => { const o = []; for (let j = 0; j < t.length; j++) { if (t[j] === A && t[j + 1] === B) { landed.push(`${i},${o.length}`); o.push(A + B); j++; } else o.push(t[j]); } return o; });
      rounds.push({ before: sp, counts, where, pairs, best, A, B, landed, after: next });
      sp = next;
    }
    const tok = (S, label) => dims(grid(S), "word, token", { label, colors, rowNames, chips: true, empty: "(padding: this word has fewer tokens)" });
    rounds.forEach((rd, r) => {
      T[`W${r}`] = tok(rd.before, r ? `after ${r} merge${r > 1 ? "s" : ""}` : "characters");
      T[`C${r}`] = dims([rd.pairs.map((p) => rd.counts.get(p))], "pair", { label: "pair counts", kind: "prob", bars: 4, scale: 20, names: rd.pairs.map((p) => `“${p.replace(" ", "” + “")}”`) });
    });
    T.W3 = tok(sp, "after 3 merges");
    const enc = (w) => { let t = [...w]; for (const rd of rounds) { const o = []; for (let j = 0; j < t.length; j++) { if (t[j] === rd.A && t[j + 1] === rd.B) { o.push(rd.A + rd.B); j++; } else o.push(t[j]); } t = o; } return t; };
    T.E0 = dims([[..."bugs"]], "token", { label: "“bugs”, raw", colors, chips: true });
    T.E1 = dims([[...enc("bugs"), ...Array(4 - enc("bugs").length).fill(null)]], "token", { label: "“bugs”, encoded", colors, chips: true, empty: "(nothing: the word is shorter now)" });
    const steps = [{ title: "Start from single characters", lines: [1, 2], show: ["W0", "F"], text: `<p>A tiny corpus: five words and how often each appears. Every word starts split into characters, so the vocabulary is just the 7 letters used and the corpus is <b>${total(rounds[0].before)}</b> tokens long. Each colour is one token.</p>` }];
    rounds.forEach((rd, r) => {
      steps.push({
        title: `Round ${r + 1}: count every neighbouring pair`, lines: [4, 5, 6, 7], show: [`W${r}`, "→", `C${r}`], speed: 1300,
        op: op.custom(rd.pairs.length, (p) => {
          const k = rd.pairs[p], occ = rd.where.get(k);
          return { hl: { [`W${r}`]: occ.flatMap(([i, j]) => [`${i},${j}`, `${i},${j + 1}`]), [`C${r}`]: [`0,${p}`] }, read: `<b>“${k.replace(" ", "” + “")}”</b> sits side by side in ${occ.map(([i]) => `${words[i]} (×${freq[words[i]]})`).join(" + ")} = <b>${rd.counts.get(k)}</b>` };
        }, (name, key) => (name === `C${r}` ? +key.split(",")[1] : null)),
        text: r === 0 ? "<p>Walk along every word and count each pair of neighbouring tokens, weighted by how often the word appears. Hover a bar to see where that pair occurs.</p>" : `<p>The same count again, but over the new tokens. “${rounds[r - 1].A + rounds[r - 1].B}” is one token now, so it pairs with its neighbours as a single unit.</p>`,
      });
      steps.push({
        title: `Merge the winner: “${rd.A}” + “${rd.B}” → “${rd.A + rd.B}”`, lines: [8, 9, 10], show: [`W${r}`, "→", `W${r + 1}`],
        op: op.custom(1, () => ({ hl: { [`W${r}`]: rd.where.get(rd.best).flatMap(([i, j]) => [`${i},${j}`, `${i},${j + 1}`]), [`W${r + 1}`]: rd.landed }, read: `most frequent pair: <b>“${rd.A}” + “${rd.B}”</b> (${rd.counts.get(rd.best)} times). New token id <b>${7 + r}</b>; the corpus shrinks from ${total(rd.before)} to <b>${total(rd.after)}</b> tokens.` })),
        text: r === 0 ? "<p>Every occurrence of the winning pair is glued into one new token, and the rows get shorter. That's the whole algorithm: count, merge, repeat. Real tokenizers do this about 100,000 times over terabytes of text.</p>" : r === 1 ? "<p>“un” wins over “h” + “ug” by one count (16 against 15). Ties and near-ties like this are why two tokenizers trained on slightly different data split words differently.</p>" : "<p>Merges stack: “hug” is built from “h” and the earlier “ug”. Frequent words end up as a single token; rare ones stay in pieces.</p>",
      });
    });
    steps.push({ title: "Encode a word it has never seen", lines: [12], show: ["E0", "→", "E1"], op: op.custom(1, () => ({ hl: { E0: ["0,1", "0,2"], E1: ["0,1"] }, read: `merges are replayed in the order they were learned: “u”+“g” fires, “u”+“n” and “h”+“ug” find nothing. <b>bugs → b · ug · s</b>, 3 tokens.` })), text: "<p>“bugs” was not in the corpus, but the learned merges still apply. A new word never fails: in the worst case it falls back to single characters (or bytes), which is why a BPE model can read any string.</p>" });
    return tensorWalk({
      title: "Byte-pair encoding, merge by merge",
      lede: "A tokenizer is trained, not written: it learns which characters to glue together by counting. Each cube is one token, coloured by which token it is.",
      code: `words = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
splits = {w: list(w) for w in words}      # start from characters
for step in range(3):
    pairs = Counter()
    for w, toks in splits.items():
        for a, b in zip(toks, toks[1:]):
            pairs[a, b] += words[w]       # count, weighted by frequency
    best = max(pairs, key=pairs.get)      # the most frequent pair
    splits = {w: merge(t, best) for w, t in splits.items()}
    merges.append(best)                   # remembered, in order

encode("bugs")                            # replay the merges -> b, ug, s`,
      tensors: T,
      steps,
      key: [[PAL.u, "letter"], [PAL.ug, "ug"], [PAL.un, "un"], [PAL.hug, "hug"], ["#f6f4fa", "padding"]],
    });
  },

  // -------------------------------------------------------------------------------------------
  "ssm-mamba"() {
    const toks = ["the", "cat", "um", "uh", "sat", "."];
    const X = [[0.2, 0.3], [0.9, 0.9], [-0.4, 0.02], [0.3, 0.02], [-0.7, 0.8], [0.1, 0.4]];
    const A = [-0.2, -0.8, -2, -4], Cv = [0.4, 0.3, 0.2, 0.1];
    const dt = X.map(([, s]) => +softplus(8 * s - 4).toFixed(2));
    const Ab = r2(dt.map((d) => A.map((a) => Math.exp(d * a))));
    const H = []; let h = [0, 0, 0, 0];
    for (let t = 0; t < 6; t++) { h = h.map((v, n) => +(Ab[t][n] * v + (1 - Ab[t][n]) * X[t][0]).toFixed(2)); H.push(h); }
    const y = H.map((r) => +r.reduce((s, v, n) => s + v * Cv[n], 0).toFixed(2));
    const T = {
      X: dims(X, "time, (value, salience)", { label: "x", rowNames: toks, names: ["value", "salience"] }),
      dt: dims(col(dt), "time", { label: "Δ (write)", kind: "prob", bars: 3, rowNames: toks }),
      Ab: dims(Ab, "time, state", { label: "Ā (keep)", kind: "prob", scale: 1, rowNames: toks, names: A.map((a) => `A = ${a}`) }),
      H: dims(H, "time, state", { label: "h", rowNames: toks }),
      y: dims(col(y), "time", { label: "y", rowNames: toks }),
    };
    return tensorWalk({
      title: "A selective scan, token by token",
      lede: "Mamba's core is a running state that each token can choose to overwrite or leave alone. Watch the filler words “um” and “uh” barely touch it.",
      code: `dt = softplus(8 * x[:, 1] - 4)           # write strength, chosen per token
Abar = np.exp(dt[:, None] * A)           # keep-gate per state dim, in (0, 1)
h = np.zeros(4)
for t in range(6):                       # left to right, one token at a time
    h = Abar[t] * h + (1 - Abar[t]) * x[t, 0]   # keep some, write some
    H.append(h)
y = H @ C                                # read the state out`,
      tensors: T,
      steps: [
        { title: "Each token: a value, and how much it matters", lines: [], show: ["X"], text: "<p>Two numbers per token to keep it small: what it says (value) and how important it is (salience). In real Mamba both come from learned projections of a 1,000-plus-wide vector.</p>" },
        { title: "Δ: the token decides how hard to write", lines: [1], show: ["X", "→", "dt"], op: op.rows(["X", "dt"], 6, (t) => `<b>Δ[${toks[t]}]</b> = softplus(8 × ${fmt(X[t][1])} − 4) = <b>${fmt(dt[t])}</b>`, { X: 2, dt: 1 }), text: "<p>This is the “selective” part. In older state-space models Δ was a fixed number; here every token computes its own. “cat” and “sat” get a big Δ, the fillers a tiny one.</p>" },
        { title: "Ā: how much of the old state survives", lines: [2], show: ["dt", "→", "Ab"], op: op.rows(["dt", "Ab"], 6, (t) => `<b>Ā[${toks[t]}]</b> = exp(${fmt(dt[t])} × [${A.join(", ")}]) = [${Ab[t].map((v) => fmt(v)).join(", ")}]`, { dt: 1, Ab: 4 }), text: "<p>Four state dimensions with different decay rates A: the first remembers for a long time, the last forgets fast. A small Δ makes Ā close to 1 in every column, which means “keep everything, write nothing”.</p>" },
        { title: "The scan: keep some, write some", lines: [4, 5, 6], show: ["Ab", "⊙ h + (1−Ā) ×", "X", "→", "H"], speed: 1500, op: op.custom(6, (t) => ({ hl: { Ab: rowK(t, 4), X: [`${t},0`], H: [...(t ? rowK(t - 1, 4) : []), ...rowK(t, 4)] }, read: `<b>h[${toks[t]}]</b> = Ā ⊙ ${t ? `h[${toks[t - 1]}]` : "0"} + (1 − Ā) × ${fmt(X[t][0])}<br>= [${H[t].map((v) => fmt(v)).join(", ")}]` }), (name, key) => (name === "H" ? +key.split(",")[0] : null)), text: "<p>Each row of h is a blend of the row before it and the new value. On “um” and “uh” (Δ ≈ 0.02) the state barely moves: the model has chosen to ignore them. On “sat” it's mostly rewritten: the fast dimensions jump straight to about −0.7, while the slowest one only falls back to zero. Bigger |A|, shorter memory.</p>" },
        { title: "Read the state out", lines: [7], show: ["H", "· C →", "y"], op: op.rows(["H", "y"], 6, (t) => `<b>y[${toks[t]}]</b> = ${H[t].map((v, n) => `${fmt(Cv[n])}×${fmt(v)}`).join(" + ")} = <b>${fmt(y[t])}</b>`, { H: 4, y: 1 }), text: "<p>One output per token, a weighted read of the state. Notice the scan only ever needed the previous row: at inference Mamba stores <b>4 numbers per channel</b>, however long the text gets, where attention keeps every past key and value.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "grpo-dpo"() {
    const ans = ["answer 1", "answer 2", "answer 3", "answer 4", "answer 5", "answer 6"], r = [0.9, 0.1, 0.4, 1.0, 0.0, 0.6];
    const mean = r.reduce((a, b) => a + b, 0) / r.length, sd = std(r);
    const dev = r.map((v) => +(v - mean).toFixed(2)), adv = r.map((v) => +((v - mean) / sd).toFixed(2));
    const len = [5, 3, 4, 5, 2, 4], At = adv.map((a, i) => Array.from({ length: 5 }, (_, j) => (j < len[i] ? a : null)));
    const LP = [[-12.1, -12.9, -15.0, -14.2], [-8.4, -8.6, -9.9, -9.1], [-20.3, -19.8, -22.0, -21.0], [-5.2, -6.0, -7.5, -6.8]], beta = 0.5;
    const LR = LP.map(([a, b, c, d]) => [+(a - b).toFixed(2), +(c - d).toFixed(2)]);
    const mg = LR.map(([w, l]) => +(beta * (w - l)).toFixed(2)), loss = mg.map((m) => +Math.log1p(Math.exp(-m)).toFixed(2));
    const pr = ["pair 1", "pair 2", "pair 3", "pair 4"];
    const T = {
      r: dims(col(r), "answer", { label: "reward", kind: "prob", bars: 4, scale: 1, rowNames: ans }),
      dev: dims(col(dev), "answer", { label: "r − mean", rowNames: ans }),
      A: dims(col(adv), "answer", { label: "advantage", rowNames: ans }),
      At: dims(At, "answer, token", { label: "per token", rowNames: ans, empty: "(the answer had ended)" }),
      LP: dims(LP, "pair, log-prob", { label: "log-probs", center: -12, rowNames: pr, names: ["policy · chosen", "reference · chosen", "policy · rejected", "reference · rejected"] }),
      LR: dims(LR, "pair, (chosen, rejected)", { label: "log-ratio", rowNames: pr, names: ["chosen", "rejected"] }),
      mg: dims(col(mg), "pair", { label: "margin", rowNames: pr }),
      L: dims(col(loss), "pair", { label: "loss", kind: "prob", bars: 3, rowNames: pr }),
    };
    return tensorWalk({
      title: "GRPO and DPO, number by number",
      lede: "Two ways to learn from preferences without a value network. GRPO compares answers within a group; DPO compares a chosen and a rejected answer against a frozen copy of the model.",
      code: `r = reward(prompt, answers)            # 6 answers to the same prompt
adv = (r - r.mean()) / r.std()         # the group is the baseline
loss = -(adv[:, None] * logp).sum()    # every token shares its answer's advantage

# DPO: no reward model, only (chosen, rejected) pairs
lr_w = logp_pi(y_w) - logp_ref(y_w)    # how much likelier than before
lr_l = logp_pi(y_l) - logp_ref(y_l)
loss = -log_sigmoid(beta * (lr_w - lr_l))`,
      tensors: T,
      steps: [
        { title: "GRPO: six answers, six scores", lines: [1], show: ["r"], text: "<p>Sample six answers to one prompt and score each one (a unit test, a checker, a reward model). Rewards on their own say little: is 0.6 good? It depends on what else the model could have said.</p>" },
        { title: "Subtract the group's mean", lines: [2], show: ["r", "−", `${fmt(mean)}`, "→", "dev"], op: op.cells(["r", "dev"], 6, 1, (i) => `<b>${ans[i]}</b>: ${fmt(r[i])} − ${fmt(mean)} = <b>${fmt(dev[i])}</b>`), text: "<p>The other answers are the baseline. Better than the group: orange, push it up. Worse: blue, push it down. No separate value network is needed to estimate “what's normal here”.</p>" },
        { title: "Divide by the spread", lines: [2], show: ["dev", `÷ ${fmt(sd)}`, "→", "A"], op: op.cells(["dev", "A"], 6, 1, (i) => `<b>advantage[${i}]</b> = ${fmt(dev[i])} / ${fmt(sd)} = <b>${fmt(adv[i])}</b>`), text: "<p>Dividing by the standard deviation puts every prompt on the same scale, so an easy prompt where all answers score 0.9 doesn't drown out a hard one. If all six get the same reward, the advantages are all zero and the prompt teaches nothing.</p>" },
        { title: "Every token of an answer shares its advantage", lines: [3], show: ["A", "→", "At"], op: op.rows(["A", "At"], 6, (i) => `${ans[i]}: ${len[i]} tokens, each weighted by <b>${fmt(adv[i])}</b>`, { A: 1, At: 5 }), text: "<p>The log-probability of every token in a good answer is pushed up, and in a bad answer pushed down, by the same amount. Credit isn't assigned token by token: the whole answer is praised or blamed.</p>" },
        { title: "DPO: log-probs under the policy and a frozen reference", lines: [5, 6, 7], show: ["LP", "→", "LR"], op: op.rows(["LP", "LR"], 4, (i) => `${pr[i]}: chosen ${fmt(LP[i][0])} − (${fmt(LP[i][1])}) = <b>${fmt(LR[i][0])}</b>, rejected ${fmt(LP[i][2])} − (${fmt(LP[i][3])}) = <b>${fmt(LR[i][1])}</b>`, { LP: 4, LR: 2 }), text: "<p>For each pair, how much more (orange) or less (blue) likely the current model makes each answer than the frozen starting model did. Comparing against the reference is what keeps the model from drifting far from where it started.</p>" },
        { title: "Margin, then loss", lines: [8], show: ["LR", "→", "mg", "→", "L"], op: op.rows(["LR", "mg", "L"], 4, (i) => `margin = ${beta} × (${fmt(LR[i][0])} − (${fmt(LR[i][1])})) = ${fmt(mg[i])}; loss = −log σ(${fmt(mg[i])}) = <b>${fmt(loss[i])}</b>`, { LR: 2, mg: 1, L: 1 }), text: "<p>The loss is small when the chosen answer has gained more than the rejected one. Pair 3 is the interesting one: the model made <i>both</i> answers less likely, but the rejected one fell further, so the margin is still positive. DPO only cares about the gap, which is one reason it can quietly lower the probability of good answers too.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "rlhf-ppo"() {
    const toks = ["The", "answer", "is", "42", ",", "because", "it", "is"];
    const lo = [-1.2, -0.8, -0.3, -2.4, -0.9, -1.6, -0.7, -0.5], dl = [0.05, -0.1, 0.02, 0.62, 0.15, -0.35, 0.3, 0.1];
    const ln = lo.map((v, i) => +(v + dl[i]).toFixed(2)), A = [0.6, 0.6, 0.5, 0.9, -0.3, -0.5, -0.4, 0.2];
    const rt = ln.map((v, i) => +Math.exp(v - lo[i]).toFixed(2)), rc = rt.map((v) => Math.min(1.2, Math.max(0.8, v)));
    const s1 = rt.map((v, i) => +(v * A[i]).toFixed(2)), s2 = rc.map((v, i) => +(v * A[i]).toFixed(2)), ob = s1.map((v, i) => Math.min(v, s2[i]));
    const c = (v) => col(v);
    const T = {
      lo: dims(c(lo), "token", { label: "log π old", rowNames: toks }), ln: dims(c(ln), "token", { label: "log π new", rowNames: toks }),
      rt: dims(c(rt), "token", { label: "ratio", center: 1, rowNames: toks }), A: dims(c(A), "token", { label: "advantage", rowNames: toks }),
      rc: dims(c(rc), "token", { label: "clipped ratio", center: 1, scale: Math.max(...rt.map((v) => Math.abs(v - 1))), rowNames: toks }),
      s1: dims(c(s1), "token", { label: "ratio × A", rowNames: toks }), s2: dims(c(s2), "token", { label: "clipped × A", rowNames: toks }), ob: dims(c(ob), "token", { label: "objective", rowNames: toks }),
    };
    return tensorWalk({
      title: "The PPO objective, token by token",
      lede: "PPO lets the policy improve, but only a little per step. The trick is one line: clip the probability ratio, and take the more pessimistic of the two.",
      code: `ratio = torch.exp(logp_new - logp_old)    # how much likelier now
surr1 = ratio * adv
surr2 = ratio.clamp(0.8, 1.2) * adv       # same, but the ratio is capped
obj = torch.min(surr1, surr2)             # the more pessimistic one
loss = -obj.mean()`,
      tensors: T,
      steps: [
        { title: "How likely each token was, before and after an update", lines: [1], show: ["lo", "ln"], text: "<p>The same eight tokens, scored by the policy that sampled them (old) and by the policy after a few gradient steps (new). The numbers are log-probabilities, so they're all negative; what matters is how they changed.</p>" },
        { title: "The ratio", lines: [1], show: ["ln", "−", "lo", "→", "rt"], op: op.cells(["lo", "ln", "rt"], 8, 1, (i) => `<b>ratio[“${toks[i]}”]</b> = exp(${fmt(ln[i])} − (${fmt(lo[i])})) = <b>${fmt(rt[i])}</b>`), text: "<p>1 means unchanged (the pale cubes). “42” is now 1.86 times likelier than before: the update moved it a lot. The ratio is coloured around 1, not around 0.</p>" },
        { title: "Weight by the advantage", lines: [2], show: ["rt", "×", "A", "=", "s1"], op: op.cells(["rt", "A", "s1"], 8, 1, (i) => `${fmt(rt[i])} × ${fmt(A[i])} = <b>${fmt(s1[i])}</b>`), text: "<p>Advantage says whether the token was better (orange) or worse (blue) than expected. Maximising ratio × advantage alone would push the probability of “42” up without limit.</p>" },
        { title: "Clip the ratio to [0.8, 1.2]", lines: [3], show: ["rt", "→", "rc", "×", "A", "=", "s2"], op: op.cells(["rt", "rc", "A", "s2"], 8, 1, (i) => `clip(${fmt(rt[i])}, 0.8, 1.2) = ${fmt(rc[i])}; × ${fmt(A[i])} = <b>${fmt(s2[i])}</b>${rt[i] !== rc[i] ? " <span class=\"tw-dim\">(clipped)</span>" : ""}`), text: "<p>Ratios that moved more than 20% either way are capped: “42” and “it” went up too far, “because” went down too far.</p>" },
        { title: "Take the smaller", lines: [4, 5], show: ["s1", "min", "s2", "→", "ob"], op: op.cells(["s1", "s2", "ob"], 8, 1, (i) => `min(${fmt(s1[i])}, ${fmt(s2[i])}) = <b>${fmt(ob[i])}</b>${s1[i] !== s2[i] ? (ob[i] === s2[i] ? " · the clipped side won: <b>no gradient</b> from this token" : " · the unclipped side won: the gradient still flows") : ""}`), text: "<p>Where a good token's ratio already passed 1.2, the clipped value is smaller, so it's chosen and its gradient is zero: no reward for pushing further. Where a bad token (“it”, advantage −0.4) went <i>up</i>, the unclipped value is more negative and is kept, so the model is still pulled back. “because” is a bad token that has already been pushed below 0.8, so it too is left alone. Pessimism in both directions keeps each step small.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "gradient-descent"() {
    const xs = [-1, -0.6, -0.2, 0.2, 0.6, 1], noise = [0.1, -0.15, 0.05, -0.05, 0.12, -0.08];
    const X = xs.map((x) => [x, 1]), y = col(xs.map((x, i) => +(2 * x - 1 + noise[i]).toFixed(2))), n = 6, lr = 0.5;
    const hist = [[0, 0]], losses = [];
    let w = col([0, 0]), first;
    for (let it = 0; it < 4; it++) {
      const p = r2(M.mm(X, w)), e = r2(M.map(p, (v, i) => v - y[i][0])), g = r2(M.mm(M.T(X), e).map((r) => [(r[0] * 2) / n])), w1 = r2(M.map(w, (v, i) => v - lr * g[i][0]));
      losses.push(+(e.reduce((s, [v]) => s + v * v, 0) / n).toFixed(2));
      if (!it) first = { p, e, g, w, w1 };
      w = w1; hist.push([w[0][0], w[1][0]]);
    }
    const H = hist.slice(0, 5), L = [...losses, +(M.mm(X, w).reduce((s, [v], i) => s + (v - y[i][0]) ** 2, 0) / n).toFixed(2)];
    const T = {
      X: dims(X, "example, (x, 1)", { label: "X", names: ["x", "bias input"] }), y: dims(y, "example", { label: "y" }), w: dims(first.w, "param", { label: "w", names: ["value"], rowNames: ["slope", "intercept"], scale: 2 }),
      p: dims(first.p, "example", { label: "prediction" }), e: dims(first.e, "example", { label: "error" }), XT: dims(M.T(X), "(x, 1), example", { label: "Xᵀ" }),
      g: dims(first.g, "param", { label: "gradient", rowNames: ["slope", "intercept"] }), w1: dims(first.w1, "param", { label: "w new", rowNames: ["slope", "intercept"], scale: 2 }),
      H: dims(H, "step, (slope, intercept)", { label: "w over time", rowNames: H.map((_, i) => `step ${i}`), names: ["slope", "intercept"], scale: 2 }),
      L: dims(col(L), "step", { label: "loss", kind: "prob", bars: 4, rowNames: L.map((_, i) => `step ${i}`) }),
    };
    return tensorWalk({
      title: "One step of gradient descent, then four",
      lede: "Fit a line y = slope · x + intercept to six points. The data was made with slope 2 and intercept −1; watch the weights walk there.",
      code: `w = np.zeros(2)                          # slope, intercept
for step in range(4):
    pred = X @ w                         # (6, 2) @ (2,)
    err = pred - y
    grad = 2 * X.T @ err / len(y)        # d(mean squared error) / dw
    w = w - 0.5 * grad                   # a step downhill`,
      tensors: T,
      steps: [
        { title: "The data, and weights that know nothing", lines: [1], show: ["X", "y", "w"], text: "<p>X has a column of x values and a column of 1s, so the second weight acts as the intercept. Both weights start at zero.</p>" },
        { title: "Predict", lines: [3], show: ["X", "@", "w", "=", "p"], op: op.matmul(T, "X", "w", "p"), speed: 700, text: "<p>With zero weights every prediction is zero: pale cubes.</p>" },
        { title: "How wrong", lines: [4], show: ["p", "−", "y", "→", "e"], op: op.cells(["p", "y", "e"], 6, 1, (i) => `${fmt(first.p[i][0])} − (${fmt(y[i][0])}) = <b>${fmt(first.e[i][0])}</b>`), speed: 700, text: "<p>The error for each point: orange where the prediction is too high, blue where it's too low.</p>" },
        { title: "The gradient: which way is uphill", lines: [5], show: ["XT", "@", "e", "→", "g"], op: op.matmul(T, "XT", "e", "g", { scale: 2 / 6, scaleLabel: "× 2/6" }), speed: 1500, text: "<p>Each weight's gradient pairs its input column with the errors. The slope's gradient is negative (raising the slope would reduce the loss) and the intercept's is positive (lower it).</p>" },
        { title: "Step the other way", lines: [6], show: ["w", "−", "0.5 ×", "g", "→", "w1"], op: op.cells(["w", "g", "w1"], 2, 1, (i) => `${fmt(first.w[i][0])} − 0.5 × ${fmt(first.g[i][0])} = <b>${fmt(first.w1[i][0])}</b>`), text: "<p>Minus the gradient, times the learning rate. That's one step of training, for this model and for a trillion-parameter one alike.</p>" },
        { title: "Repeat", lines: [2, 3, 4, 5, 6], show: ["H", "L"], op: op.rows(["H", "L"], 5, (i) => `step ${i}: slope ${fmt(H[i][0])}, intercept ${fmt(H[i][1])}, loss <b>${fmt(L[i])}</b>`, { H: 2, L: 1 }), speed: 1300, text: "<p>Each step uses the new weights to compute a new gradient. The loss falls fast at first and then slows, because the gradient shrinks as the fit gets better: steps get smaller on their own.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  perceptron() {
    const X = [[2, 1, 1], [1, 2, 1], [2.5, 2, 1], [-1, -1, 1], [-2, 0.5, 1], [0.5, -1.5, 1]], y = [1, 1, 1, -1, -1, -1];
    const w0 = [[-0.5], [0.6], [0.2]], s0 = r2(M.mm(X, w0)), m0 = s0.map((r, i) => [+(r[0] * y[i]).toFixed(2)]);
    const bad = m0.findIndex(([v]) => v <= 0), w1 = w0.map((r, k) => [+(r[0] + y[bad] * X[bad][k]).toFixed(2)]);
    const s1 = r2(M.mm(X, w1)), m1 = s1.map((r, i) => [+(r[0] * y[i]).toFixed(2)]);
    const nm = X.map((_, i) => `point ${i} (${y[i] > 0 ? "+" : "−"})`), wn = ["w₁", "w₂", "bias"];
    const T = {
      X: dims(X, "point, (x₁, x₂, 1)", { label: "X", rowNames: nm, names: ["x₁", "x₂", "bias input"] }), y: dims(col(y), "point", { label: "label", rowNames: nm }),
      w0: dims(w0, "weight", { label: "w", rowNames: wn, scale: 2 }), s0: dims(s0, "point", { label: "score", rowNames: nm }), m0: dims(m0, "point", { label: "score × label", rowNames: nm }),
      w1: dims(w1, "weight", { label: "w after", rowNames: wn, scale: 2 }), s1: dims(s1, "point", { label: "score", rowNames: nm }), m1: dims(m1, "point", { label: "score × label", rowNames: nm }),
    };
    const wrong = m0.map(([v], i) => (v <= 0 ? i : -1)).filter((i) => i >= 0);
    return tensorWalk({
      title: "The perceptron rule, step by step",
      lede: "Six points, two classes, three weights. The learning rule is one line: when you get a point wrong, add it to the weights (or subtract it).",
      code: `scores = X @ w                # one number per point
wrong = (scores * y) <= 0     # sign disagrees with the label
i = np.flatnonzero(wrong)[0]  # take the first mistake
w = w + y[i] * X[i]           # nudge w toward the right answer
scores = X @ w                # try again`,
      tensors: T,
      steps: [
        { title: "Points with a bias column, and their labels", lines: [], show: ["X", "y"], text: "<p>Each point has two coordinates plus a constant 1, so the third weight can shift the boundary away from the origin. Labels are +1 (orange) or −1 (blue).</p>" },
        { title: "Score every point", lines: [1], show: ["X", "@", "w0", "=", "s0"], op: op.matmul(T, "X", "w0", "s0"), speed: 900, text: "<p>A dot product per point. The sign is the prediction.</p>" },
        { title: "Which ones are wrong", lines: [2], show: ["s0", "×", "y", "→", "m0"], op: op.cells(["s0", "y", "m0"], 6, 1, (i) => `${fmt(s0[i][0])} × ${y[i]} = <b>${fmt(m0[i][0])}</b> → ${m0[i][0] > 0 ? "right" : "<b>wrong</b>"}`), speed: 800, text: `<p>Multiply the score by the label: positive means the sign agrees. Blue here is a mistake. Points ${wrong.join(", ")} are on the wrong side.</p>` },
        { title: `Fix point ${bad}: add it to w`, lines: [3, 4], show: ["w0", `+ (${y[bad] > 0 ? "+1" : "−1"}) ×`, "X", "→", "w1"], op: op.custom(3, (k) => ({ hl: { w0: [`${k},0`], X: [`${bad},${k}`], w1: [`${k},0`] }, read: `<b>${wn[k]}</b> = ${fmt(w0[k][0])} + (${y[bad]}) × ${fmt(X[bad][k])} = <b>${fmt(w1[k][0])}</b>` }), (name, key) => (name === "w1" ? +key.split(",")[0] : null)), text: "<p>Adding a positive point to w turns w toward it, so its score goes up next time. That's the entire learning rule, and it's guaranteed to finish if a separating line exists.</p>" },
        { title: "Score again", lines: [5], show: ["X", "@", "w1", "=", "s1", "→", "m1"], op: op.matmul(T, "X", "w1", "s1"), speed: 900, text: `<p>${m1.every(([v]) => v > 0) ? "One update fixed all three mistakes: every score × label is now orange. Training stops when a whole pass makes no mistakes." : "Fewer mistakes now. Repeat until a whole pass makes none."}</p>` },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  interpretability() {
    // 8 feature directions in 6 dimensions (more features than dimensions: superposition)
    const signs = [1, -1, 1, 1, -1, -1], base = [];
    for (let m = 0; m < 32 && base.length < 8; m++) { const v = [1, ...[0, 1, 2, 3, 4].map((b) => ((m >> b) & 1 ? -1 : 1))]; if (base.every((u) => Math.abs(u.reduce((s, x, i) => s + x * v[i], 0)) <= 2)) base.push(v); }
    const D = base.map((v) => v.map((x, i) => +((x * signs[i]) / Math.sqrt(6)).toFixed(2)));
    const feat = ["“code”", "“French”", "“dates”", "“anger”", "“DNA”", "“legal”", "“math”", "“food”"];
    const sup = [[0, 6], [1, 7], [2, 4], [3, 5]];
    const codes = sup.map(([a, b], i) => { const c = Array(8).fill(0); c[a] = +(1.0 + 0.1 * i).toFixed(2); c[b] = 0.8; return c; });
    const X = r2(M.mm(codes, D)), E = M.T(D), pre = r2(M.mm(X, E)), b = -0.75, F = r2(M.map(pre, (v) => Math.max(0, v + b))), Xh = r2(M.mm(F, D)), err = r2(M.map(X, (v, i, j) => v - Xh[i][j]));
    const exN = sup.map(([a, c]) => `${feat[a]} + ${feat[c]}`);
    const T = {
      X: dims(X, "example, dim", { label: "activations", rowNames: exN }), E: dims(E, "dim, feature", { label: "W_enc", names: feat }), pre: dims(pre, "example, feature", { label: "pre-act", rowNames: exN, names: feat }),
      F: dims(F, "example, feature", { label: "features", rowNames: exN, names: feat, scale: 1 }), D: dims(D, "feature, dim", { label: "W_dec", rowNames: feat }),
      Xh: dims(Xh, "example, dim", { label: "rebuilt", rowNames: exN, scale: Math.max(...X.flat().map(Math.abs)) }), err: dims(err, "example, dim", { label: "error", rowNames: exN, scale: Math.max(...X.flat().map(Math.abs)) }),
    };
    return tensorWalk({
      title: "A sparse autoencoder pulls features apart",
      lede: "Eight concepts squeezed into six numbers, so every activation is a blur of several. An SAE maps back up to one slot per concept, keeps only the confident ones, and rebuilds the original.",
      code: `x = model.activations(text)       # (4, 6): dense, every dim busy
pre = x @ W_enc                    # (6, 8): one score per feature
f = relu(pre + b)                  # b < 0: only strong scores survive
x_hat = f @ W_dec                  # (8, 6): rebuild from the few
loss = ((x - x_hat) ** 2).sum() + lam * f.abs().sum()`,
      tensors: T,
      steps: [
        { title: "Activations: dense and unreadable", lines: [1], show: ["X"], text: "<p>Four inputs, six dimensions each. Every dimension is busy in every row; no single cube means “French” or “math”. That's superposition: more concepts than dimensions, overlapping.</p>" },
        { title: "Score every feature", lines: [2], show: ["X", "@", "E", "=", "pre"], op: op.matmul(T, "X", "E", "pre"), speed: 450, text: "<p>One column of W_enc per candidate feature. The true features score high, but the other six don't score zero: overlapping directions leak into each other.</p>" },
        { title: "Threshold: keep only confident features", lines: [3], show: ["pre", `+ (${b}) → ReLU →`, "F"], op: op.rows(["pre", "F"], 4, (i) => `${exN[i]}: ${F[i].filter((v) => v > 0).length} of 8 features fire: ${F[i].map((v, j) => (v > 0 ? `${feat[j]} ${fmt(v)}` : null)).filter(Boolean).join(", ")}`, { pre: 8, F: 8 }), speed: 1600, text: "<p>A negative bias, then ReLU. Leaks below the threshold become exact zeros (pale), and each row is left with the two features that are really there. Sparse and readable: you can put a name on each column.</p>" },
        { title: "Rebuild from the few", lines: [4], show: ["F", "@", "D", "=", "Xh"], op: op.matmul(T, "F", "D", "Xh"), speed: 450, text: "<p>Multiply back out through W_dec. Each output is a weighted sum of just two dictionary rows.</p>" },
        { title: "What's lost", lines: [5], show: ["X", "−", "Xh", "=", "err"], op: op.rows(["X", "Xh", "err"], 4, (i) => `${exN[i]}: error size ${fmt(Math.hypot(...err[i]))} against an activation of size ${fmt(Math.hypot(...X[i]))}`, { X: 6, Xh: 6, err: 6 }), speed: 1400, text: "<p>The right features came back, but the rebuild is too small: the bias took 0.75 off every feature it kept (“code” is 1.0 in the real mix but reads 0.53). That's the well-known shrinkage problem, and it's why newer SAEs (TopK, JumpReLU) use the threshold only to <i>choose</i> features and keep their full values. Real SAEs have millions of features and still leave some “dark matter” unexplained.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "long-context"() {
    const P = 6, th = [1, 0.3], q = [1, 0.2, 0.8, -0.4];
    const Q = Array.from({ length: P }, () => [...q]);
    const Ang = Array.from({ length: P }, (_, m) => th.map((t) => +(m * t).toFixed(2)));
    const rot = (v, m) => { const o = []; for (let p = 0; p < 2; p++) { const a = m * th[p], c = Math.cos(a), s = Math.sin(a), x = v[2 * p], y = v[2 * p + 1]; o.push(x * c - y * s, x * s + y * c); } return o; };
    const Qr = r2(Q.map((v, m) => rot(v, m))), QT = M.T(Qr), S = r2(M.mm(Qr, QT));
    const pos = Array.from({ length: P }, (_, m) => `position ${m}`);
    const T = {
      Q: dims(Q, "position, dim", { label: "q", rowNames: pos }), Ang: dims(Ang, "position, pair", { label: "angle", kind: "prob", rowNames: pos, names: ["pair 1 (fast)", "pair 2 (slow)"] }),
      Qr: dims(Qr, "position, dim", { label: "rotated q", rowNames: pos }), QT: dims(QT, "dim, position", { label: "rotated kᵀ", names: pos }),
      S: dims(S, "query pos, key pos", { label: "scores", rowNames: pos, names: pos }),
    };
    return tensorWalk({
      title: "RoPE: position as a rotation",
      lede: "Rotary embeddings don't add a position vector; they turn each pair of dimensions by an angle that grows with position. The payoff shows up in the scores.",
      code: `theta = [1.0, 0.3]                   # radians per position, one per pair
for m in range(6):                   # the same q placed at every position
    a = m * theta
    q_rot[m] = rotate_pairs(q, a)    # (x, y) -> (x cos a - y sin a, x sin a + y cos a)
scores = q_rot @ k_rot.T             # here k = q, so we see the pattern alone`,
      tensors: T,
      steps: [
        { title: "The same vector at every position", lines: [2], show: ["Q"], text: "<p>Put an identical query at positions 0 to 5. Without position information, attention couldn't tell these apart: every row is the same.</p>" },
        { title: "An angle per position, per pair", lines: [1, 3], show: ["Ang"], op: op.rows(["Ang"], P, (m) => `position ${m}: pair 1 turns ${fmt(Ang[m][0])} rad, pair 2 turns ${fmt(Ang[m][1])} rad`, { Ang: 2 }), text: "<p>Dimensions are grouped in pairs, and each pair spins at its own speed. Fast pairs tell neighbours apart; slow pairs track position over long ranges. Real models use 64 pairs with speeds spread over orders of magnitude.</p>" },
        { title: "Rotate", lines: [4], show: ["Q", "rotate by", "Ang", "→", "Qr"], op: op.rows(["Q", "Ang", "Qr"], P, (m) => `position ${m}: (${fmt(q[0])}, ${fmt(q[1])}) turned by ${fmt(Ang[m][0])} → (${fmt(Qr[m][0])}, ${fmt(Qr[m][1])}); (${fmt(q[2])}, ${fmt(q[3])}) turned by ${fmt(Ang[m][1])} → (${fmt(Qr[m][2])}, ${fmt(Qr[m][3])})`, { Q: 4, Ang: 2, Qr: 4 }), speed: 1400, text: "<p>Rotation changes direction but not length, so the vector keeps its meaning and gains a position. Row 0 isn't rotated at all.</p>" },
        { title: "Scores depend only on distance", lines: [5], show: ["Qr", "@", "QT", "=", "S"], speed: 1500,
          op: op.custom(P, (d) => { const ks = []; for (let m = 0; m < P; m++) for (let n = 0; n < P; n++) if (Math.abs(m - n) === d) ks.push(`${m},${n}`); return { hl: { S: ks }, read: `every pair of positions ${d} apart scores <b>${fmt(S[Math.min(d, P - 1)][0])}</b>${d ? "" : " (a vector with itself)"}` }; }, (name, key) => (name === "S" ? Math.abs(key.split(",").map(Number).reduce((a, b) => a - b)) : null)),
          text: "<p>Rotating both q and k by their own positions means their dot product depends only on the <b>difference</b> m − n: every diagonal of the score table holds one value. The model learns “two tokens back” once, and it works anywhere in the sequence. Past the training length, though, the angles are combinations it has never seen, which is why long-context models stretch θ (position interpolation, YaRN).</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  vision() {
    // an 8x8 picture: a plus sign in the top-left quarter, a diagonal in the bottom-right
    const img = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => +(((i < 4 && j < 4 && (i === 1 || j === 1)) || (i >= 4 && j >= 4 && i === j) ? 0.9 : 0.08) + ((i * 3 + j) % 4) * 0.01).toFixed(2)));
    const patches = [0, 1, 2, 3].map((p) => { const r0 = Math.floor(p / 2) * 4, c0 = (p % 2) * 4, o = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) o.push(img[r0 + i][c0 + j]); return o; });
    const R = rng(23), We = M.rand(16, 4, R, 0.5), E = r2(M.mm(patches, We)), Pos = M.rand(4, 4, R, 0.3), Z = r2(M.add(E, Pos));
    const pn = ["patch 0 (top left)", "patch 1 (top right)", "patch 2 (bottom left)", "patch 3 (bottom right)"];
    const T = {
      img: dims(img, "row, column", { label: "image", kind: "prob", scale: 1 }), P: dims(patches, "patch, pixel", { label: "patches", kind: "prob", scale: 1, rowNames: pn, px: 3 }),
      We: dims(We, "pixel, dim", { label: "W_embed", px: 3 }), E: dims(E, "patch, dim", { label: "embedded", rowNames: pn }), Pos: dims(Pos, "patch, dim", { label: "position", rowNames: pn }), Z: dims(Z, "token, dim", { label: "tokens", rowNames: pn }),
    };
    return tensorWalk({
      title: "Turning an image into tokens",
      lede: "A vision transformer never sees pixels one by one. It cuts the image into squares, flattens each square into a row, and from then on treats them exactly like words.",
      code: `img = load()                           # (8, 8), one channel to keep it small
patches = img.reshape(2, 4, 2, 4)      # cut into a 2 x 2 grid of 4 x 4 squares
patches = patches.transpose(0, 2, 1, 3).reshape(4, 16)  # one row per square
tokens = patches @ W_embed             # (4, 16) @ (16, 4)
tokens = tokens + pos                  # where each square came from`,
      tensors: T,
      steps: [
        { title: "The image", lines: [1], show: ["img"], text: "<p>64 pixels, darker purple = brighter pixel. A plus sign top left, a diagonal bottom right.</p>" },
        { title: "Cut into patches, flatten each one", lines: [2, 3], show: ["img", "→", "P"], speed: 1500, op: op.custom(4, (p) => { const r0 = Math.floor(p / 2) * 4, c0 = (p % 2) * 4, ks = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) ks.push(`${r0 + i},${c0 + j}`); return { hl: { img: ks, P: rowK(p, 16) }, read: `<b>${pn[p]}</b>: rows ${r0}–${r0 + 3}, columns ${c0}–${c0 + 3} of the image become row ${p}, read left to right, top to bottom` }; }, (name, key) => (name === "P" ? +key.split(",")[0] : null)), text: "<p>Each 4×4 square becomes a row of 16 numbers. The plus sign's shape is still there, unrolled. Real ViTs cut a 224×224 photo into 196 squares of 16×16 pixels × 3 colours = 768 numbers each.</p>" },
        { title: "Project each patch to a token", lines: [4], show: ["P", "@", "We", "=", "E"], op: op.matmul(T, "P", "We", "E"), speed: 700, text: "<p>One learned matrix turns 16 pixels into a 4-number embedding, the same for every patch: this is a convolution with stride equal to its size, written as a matmul.</p>" },
        { title: "Add where each patch came from", lines: [5], show: ["E", "+", "Pos", "=", "Z"], op: op.cells(["E", "Pos", "Z"], 4, 4, (i, j) => `${fmt(E[i][j])} + ${fmt(Pos[i][j])} = <b>${fmt(Z[i][j])}</b>`), speed: 500, text: "<p>Flattening threw away the 2D layout, so a learned position vector is added per patch. Now it's four tokens, and the rest of the model is the same transformer that reads text. That interchangeability is why one architecture took over images, audio and video.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  autoencoders() {
    const R = rng(31), a = 1 / Math.sqrt(3);
    const D = [[a, a, a, 0, 0, 0], [0, 0, 0, a, -a, a]].map((r) => r.map((v) => +v.toFixed(2)));
    const zt = [[1.2, 0.3], [-0.8, 0.9], [0.4, -1.1], [-1.0, -0.5]], noise = M.rand(4, 6, R, 0.06);
    const X = r2([...M.add(M.mm(zt, D), noise), [0.9, -0.9, 0.9, -0.9, 0.9, -0.9]]);
    const En = M.T(D), Z = r2(M.mm(X, En)), Xh = r2(M.mm(Z, D)), err = r2(M.map(X, (v, i, j) => v - Xh[i][j]));
    const en = err.map((r) => +Math.hypot(...r).toFixed(2)), nm = ["example 0", "example 1", "example 2", "example 3", "an odd one"];
    const sc = Math.max(...X.flat().map(Math.abs));
    const T = {
      X: dims(X, "example, feature", { label: "x", rowNames: nm }), En: dims(En, "feature, code", { label: "encoder" }), Z: dims(Z, "example, code", { label: "code z", rowNames: nm }),
      D: dims(D, "code, feature", { label: "decoder" }), Xh: dims(Xh, "example, feature", { label: "x rebuilt", rowNames: nm, scale: sc }),
      err: dims(err, "example, feature", { label: "error", rowNames: nm, scale: sc }), en: dims(col(en), "example", { label: "error size", kind: "prob", bars: 4, rowNames: nm }),
    };
    return tensorWalk({
      title: "Squeeze to two numbers, then rebuild",
      lede: "An autoencoder learns to pass data through a narrow middle. Whatever survives the squeeze is what the data is really made of.",
      code: `z = x @ W_enc           # (5, 6) @ (6, 2): six numbers -> two
x_hat = z @ W_dec       # (5, 2) @ (2, 6): two numbers -> six
err = x - x_hat
score = np.linalg.norm(err, axis=1)   # big = this input doesn't fit`,
      tensors: T,
      steps: [
        { title: "Six features per example", lines: [], show: ["X"], text: "<p>The first four rows look like six independent numbers, but they're secretly built from two (plus a little noise): features 0–2 move together, and so do 3–5 (with 4 going the opposite way). The last row breaks that pattern.</p>" },
        { title: "Encode: down to two numbers", lines: [1], show: ["X", "@", "En", "=", "Z"], op: op.matmul(T, "X", "En", "Z"), speed: 700, text: "<p>The encoder's two columns are the two directions the data actually varies along (here they're already learned). Every example becomes a point in 2D.</p>" },
        { title: "Decode: back up to six", lines: [2], show: ["Z", "@", "D", "=", "Xh"], op: op.matmul(T, "Z", "D", "Xh"), speed: 450, text: "<p>Only the two codes are passed on, so the decoder can only rebuild what fits the pattern it learned.</p>" },
        { title: "What didn't fit", lines: [3, 4], show: ["X", "−", "Xh", "=", "err", "→", "en"], op: op.rows(["X", "Xh", "err", "en"], 5, (i) => `${nm[i]}: error size <b>${fmt(en[i])}</b>${i === 4 ? " · it doesn't follow the pattern, so the squeeze can't hold it" : " · just the noise"}`, { X: 6, Xh: 6, err: 6, en: 1 }), speed: 1300, text: "<p>The normal rows come back almost exactly; what's lost is the noise. The odd row can't be described by the two learned directions and comes back badly. That's how autoencoders get used for anomaly detection, and a VAE adds one rule on top: make the codes look like samples from a simple bell curve, so you can sample new ones.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  "training-tricks"() {
    const R = rng(41), d = 6, X = M.rand(4, d, R, 2);   // rng.n has spread 0.5, so 2 gives spread 1
    const Wb = [0, 1, 2].map(() => M.rand(d, d, R, 2)), Wg = Wb.map((W) => M.map(W, (v) => v / Math.sqrt(d)));
    const run = (Ws) => { const out = []; let h = X; for (const W of Ws) { h = r2(M.mm(h, W)); out.push(h); } return out; };
    const B = run(Wb), G = run(Wg), sd = (A) => std(A.flat());
    const ln = r2(B[2].map((r) => { const m = r.reduce((a, b) => a + b, 0) / d, s = std(r) + 1e-5; return r.map((v) => (v - m) / s); }));
    const mask = M.rand(4, d, rng(43), 1).map((r) => r.map((v) => (v > 0 ? 1 : 0))), drop = r2(G[2].map((r, i) => r.map((v, j) => (v * mask[i][j]) / 0.5)));
    const fixed = (data, label) => dims(data, "example, dim", { label, scale: 3 });
    const T = { X: fixed(X, "x"), b1: fixed(B[0], "layer 1"), b2: fixed(B[1], "layer 2"), b3: fixed(B[2], "layer 3"), g1: fixed(G[0], "layer 1"), g2: fixed(G[1], "layer 2"), g3: fixed(G[2], "layer 3"), ln: fixed(ln, "LayerNorm(layer 3)"), m: dims(mask, "example, dim", { label: "keep mask", kind: "prob", scale: 1 }), dr: fixed(drop, "after dropout") };
    const layers = (pre, A) => op.custom(4, (l) => ({ hl: { [l ? `${pre}${l}` : "X"]: keysOf(4, d) }, read: `${l ? `layer ${l}` : "input"}: spread (standard deviation) = <b>${fmt(sd(l ? A[l - 1] : X))}</b>` }), (name) => (name === "X" ? 0 : name.startsWith(pre) ? +name.slice(1) : null));
    return tensorWalk({
      title: "Why initialisation and normalisation matter",
      lede: "Three plain linear layers, no training. Every tensor here is drawn on the same fixed colour scale, so values that blow up turn fully saturated and values that vanish turn pale.",
      code: `h = x
for W in Ws:                       # 3 layers, 6 wide
    h = h @ W                      # W ~ N(0, 1): too big
# fix 1: scale the init so the spread is kept
Ws = [W / np.sqrt(6) for W in Ws]  # "Xavier / He" style
# fix 2: renormalise each row, whatever came in
h = (h - h.mean(-1, keepdims=True)) / h.std(-1, keepdims=True)
# and while training: randomly drop half, scale the rest up
h = h * mask / 0.5`,
      tensors: T,
      steps: [
        { title: "Naive init: the signal explodes", lines: [1, 2, 3], show: ["X", "→", "b1", "→", "b2", "→", "b3"], op: layers("b", B), speed: 1300, text: `<p>Each weight has spread 1 and each output sums 6 products, so the spread grows by about √6 ≈ 2.4 per layer: ${fmt(sd(X))} → ${fmt(sd(B[0]))} → ${fmt(sd(B[1]))} → <b>${fmt(sd(B[2]))}</b>. Thirty layers of that overflows; with the opposite mistake it vanishes to zero. Gradients do the same thing on the way back.</p>` },
        { title: "Scaled init: the spread holds", lines: [4, 5], show: ["X", "→", "g1", "→", "g2", "→", "g3"], op: layers("g", G), speed: 1300, text: `<p>Divide each weight by √(fan-in) and each layer keeps the spread roughly where it was: ${fmt(sd(X))} → ${fmt(sd(G[0]))} → ${fmt(sd(G[1]))} → ${fmt(sd(G[2]))}. (He init uses √(2 / fan-in) to make up for ReLU zeroing half.)</p>` },
        { title: "Or renormalise: LayerNorm", lines: [6, 7], show: ["b3", "→", "ln"], op: op.rows(["b3", "ln"], 4, (i) => `row ${i}: mean ${fmt(B[2][i].reduce((a, b) => a + b, 0) / d)}, spread ${fmt(std(B[2][i]))} → mean 0.00, spread <b>1.00</b>`, { b3: d, ln: d }), text: "<p>LayerNorm doesn't care how badly scaled its input is: each row is shifted to mean 0 and scaled to spread 1. Put one in every block and depth stops being fragile; that's a large part of why transformers can be stacked 100 layers deep.</p>" },
        { title: "Dropout: random holes, scaled up", lines: [8, 9], show: ["g3", "×", "m", "÷ 0.5 →", "dr"], op: op.cells(["g3", "m", "dr"], 4, d, (i, j) => `${fmt(G[2][i][j])} × ${mask[i][j]} / 0.5 = <b>${fmt(drop[i][j])}</b>${mask[i][j] ? "" : " · dropped"}`), speed: 450, text: "<p>During training, about half the values are zeroed at random and the survivors doubled, so the expected value is unchanged. No unit can rely on any single other one being there, which makes the network spread what it knows. At test time dropout is switched off.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  distributed() {
    const R = rng(51), X = M.rand(4, 4, R, 1), W = M.rand(4, 6, R, 0.7), Y = r2(M.mm(X, W));
    const Ws = [0, 1].map((g) => W.map((r) => r.slice(g * 3, g * 3 + 3))), Ys = [0, 1].map((g) => Y.map((r) => r.slice(g * 3, g * 3 + 3)));
    const Gs = [0, 1].map(() => M.rand(4, 6, R, 0.5)), Gm = r2(Gs[0].map((r, i) => r.map((v, j) => (v + Gs[1][i][j]) / 2)));
    const T = {
      X: dims(X, "batch, in", { label: "X" }), W: dims(W, "in, out", { label: "W" }), Y: dims(Y, "batch, out", { label: "Y" }),
      Ws: dims(Ws, "gpu, in, out/2", { label: "W, split", gap: 2, stair: 3 }), Ys: dims(Ys, "gpu, batch, out/2", { label: "Y, split", gap: 2, stair: 3 }),
      Gs: dims(Gs, "gpu, in, out", { label: "gradients", gap: 2, stair: 3 }), Gm: dims(Gm, "in, out", { label: "averaged" }),
    };
    return tensorWalk({
      title: "Splitting one layer across two GPUs",
      lede: "When a matrix is too big for one GPU, cut it. Stacked slabs here are the same tensor living on different GPUs: bottom is GPU 0, top is GPU 1.",
      code: `Y = X @ W                          # (4, 4) @ (4, 6): too big for one GPU
W0, W1 = W[:, :3], W[:, 3:]        # tensor parallel: split the columns
Y0 = X @ W0  # on GPU 0            # both GPUs get the whole X
Y1 = X @ W1  # on GPU 1
Y = all_gather([Y0, Y1])           # stitch the halves back together
# data parallel: same weights, different batches
g = all_reduce([g0, g1]) / 2       # every GPU ends up with the average`,
      tensors: T,
      steps: [
        { title: "The whole layer on one device", lines: [1], show: ["X", "@", "W", "=", "Y"], op: op.matmul(T, "X", "W", "Y"), speed: 300, text: "<p>Four inputs through a layer with six outputs. Pretend W is 50 GB.</p>" },
        { title: "Cut W by columns", lines: [2], show: ["W", "→", "Ws"], speed: 1500, op: op.custom(2, (g) => ({ hl: { W: Array.from({ length: 12 }, (_, k) => `${Math.floor(k / 3)},${g * 3 + (k % 3)}`), Ws: keysOf(4, 3, g) }, read: `columns ${g * 3}–${g * 3 + 2} of W go to <b>GPU ${g}</b>` }), (name, key) => (name === "Ws" ? +key.split(",")[0] : null)), text: "<p>Each GPU stores half the weights. Splitting by output columns means no GPU needs the other's half to do its work.</p>" },
        { title: "Each GPU computes its half of the output", lines: [3, 4], show: ["X", "@", "Ws", "=", "Ys"], speed: 450, op: op.custom(24, (k) => { const g = Math.floor(k / 12), i = Math.floor((k % 12) / 3), j = k % 3; return { hl: { X: rowK(i, 4), Ws: [0, 1, 2, 3].map((q) => `${g},${q},${j}`), Ys: [`${g},${i},${j}`] }, read: `<b>GPU ${g}</b>: Y[${i},${g * 3 + j}] = row ${i} of X · column ${j} of its shard = ${X[i].map((x, q) => `${fmt(x)}×${fmt(Ws[g][q][j])}`).join(" + ")} = <b>${fmt(Ys[g][i][j])}</b>` }; }, (name, key) => { if (name !== "Ys") return null; const [g, i, j] = key.split(",").map(Number); return g * 12 + i * 3 + j; }), text: "<p>Both GPUs work at the same time, on the same X, with no communication. Each produces a slab of the result.</p>" },
        { title: "Gather the halves", lines: [5], show: ["Ys", "→", "Y"], speed: 1500, op: op.custom(2, (g) => ({ hl: { Ys: keysOf(4, 3, g), Y: Array.from({ length: 12 }, (_, k) => `${Math.floor(k / 3)},${g * 3 + (k % 3)}`) }, read: `GPU ${g}'s slab becomes columns ${g * 3}–${g * 3 + 2} of Y: exactly the numbers the single-GPU version made` }), (name, key) => (name === "Ys" ? +key.split(",")[0] : null)), text: "<p>One all-gather and every GPU has the full output, identical to doing it on one device. That communication step is the cost, which is why tensor parallelism stays inside one machine, where GPUs are linked by fast NVLink.</p>" },
        { title: "Data parallel: average the gradients", lines: [6, 7], show: ["Gs", "→ mean →", "Gm"], speed: 450, op: op.custom(24, (k) => { const i = Math.floor(k / 6), j = k % 6; return { hl: { Gs: [`0,${i},${j}`, `1,${i},${j}`], Gm: [`${i},${j}`] }, read: `(${fmt(Gs[0][i][j])} + ${fmt(Gs[1][i][j])}) / 2 = <b>${fmt(Gm[i][j])}</b>` }; }, (name, key) => (name === "Gm" ? key.split(",").map(Number).reduce((i, j) => i * 6 + j) : name === "Gs" ? +key.split(",")[1] * 6 + +key.split(",")[2] : null)), text: "<p>The other way to split: every GPU holds all the weights but sees a different batch, so their gradients differ. An all-reduce averages them, cube by cube, and every GPU takes the same step. Big training runs use both at once, plus pipeline stages, plus sharded optimiser state.</p>" },
      ],
    });
  },

  // -------------------------------------------------------------------------------------------
  unsupervised() {
    const P = [[-1.6, 1.2], [-1.2, 1.6], [-1.4, 0.9], [1.3, 1.1], [1.6, 1.5], [0.2, -1.4], [-0.2, -1.2], [0.4, -1.7]];
    const C0 = [[-0.5, 0.2], [0.6, 0.4], [0.3, -0.3]];
    const Dm = r2(P.map((p) => C0.map((c) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)));
    const asg = Dm.map((r) => r.indexOf(Math.min(...r)));
    const C1 = r2(C0.map((c, j) => { const m = P.filter((_, i) => asg[i] === j); return m.length ? [0, 1].map((k) => m.reduce((s, p) => s + p[k], 0) / m.length) : c; }));
    const pn = P.map((_, i) => `point ${i}`), cn = ["centre A", "centre B", "centre C"];
    const T = {
      P: dims(P, "point, (x, y)", { label: "points", rowNames: pn, names: ["x", "y"] }), C0: dims(C0, "centre, (x, y)", { label: "centres", rowNames: cn, names: ["x", "y"] }),
      D: dims(Dm, "point, centre", { label: "distance²", kind: "prob", rowNames: pn, names: cn }), a: dims(col(asg), "point", { label: "nearest", kind: "ids", rowNames: pn }),
      C1: dims(C1, "centre, (x, y)", { label: "new centres", rowNames: cn, names: ["x", "y"] }),
    };
    return tensorWalk({
      title: "One round of k-means, in numbers",
      lede: "No labels at all: just eight points and three guesses for where the groups are. Two moves, repeated: assign each point to its nearest centre, then move each centre to the middle of its points.",
      code: `C = [[-0.5, 0.2], [0.6, 0.4], [0.3, -0.3]]   # three rough guesses
D = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (8, 3) distances
a = D.argmin(1)                              # nearest centre for each point
C = np.array([P[a == j].mean(0) for j in range(3)])   # move to the middle`,
      tensors: T,
      steps: [
        { title: "Points and starting centres", lines: [1], show: ["P", "C0"], text: "<p>Eight points in 2D (x, y). There are three clumps: top left, top right, bottom. The starting centres are all huddled near the middle.</p>" },
        { title: "Distance from every point to every centre", lines: [2], show: ["P", "↔", "C0", "→", "D"], speed: 500, op: op.custom(24, (k) => { const i = Math.floor(k / 3), j = k % 3; return { hl: { P: rowK(i, 2), C0: rowK(j, 2), D: [`${i},${j}`] }, read: `point ${i} to ${cn[j]}: (${fmt(P[i][0])} − ${fmt(C0[j][0])})² + (${fmt(P[i][1])} − ${fmt(C0[j][1])})² = <b>${fmt(Dm[i][j])}</b>` }; }, (name, key) => (name === "D" ? key.split(",").map(Number).reduce((i, j) => i * 3 + j) : null)), text: "<p>An 8 × 3 table: darker means further. This is the expensive part: points × centres × dimensions.</p>" },
        { title: "Assign: the smallest in each row wins", lines: [3], show: ["D", "→", "a"], op: op.rows(["D", "a"], 8, (i) => `point ${i}: nearest is <b>${cn[asg[i]]}</b> (${fmt(Dm[i][asg[i]])})`, { D: 3, a: 1 }), speed: 900, text: "<p>Each point joins its nearest centre; colours show which group. Even from these huddled guesses, every clump lands with its own centre in one go. Worse starts can take several rounds.</p>" },
        { title: "Update: move each centre to its points' average", lines: [4], show: ["P", "→", "C1"], speed: 1600, op: op.custom(3, (j) => ({ hl: { P: P.flatMap((_, i) => (asg[i] === j ? rowK(i, 2) : [])), C1: rowK(j, 2) }, read: `<b>${cn[j]}</b>: the mean of points ${P.map((_, i) => i).filter((i) => asg[i] === j).join(", ")} = (${fmt(C1[j][0])}, ${fmt(C1[j][1])})` }), (name, key) => (name === "C1" ? +key.split(",")[0] : null)), text: "<p>The centres jump out toward their clumps. Repeat both steps and the assignments stop changing within a few rounds. Each round can only lower the total distance, so it always stops, though maybe at a poor answer: that's why k-means is usually run from several random starts.</p>" },
      ],
    });
  },
};
