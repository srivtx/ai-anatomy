import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button, stepper } from "../lib/widgets.js";

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const winner = (b) => { for (const [a, c, d] of LINES) if (b[a] && b[a] === b[c] && b[a] === b[d]) return b[a]; return b.every(Boolean) ? "draw" : null; };
const moves = (b) => b.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
const other = (p) => (p === "X" ? "O" : "X");
// a crude "policy network": prefer centre, then corners, then edges (what AlphaZero would learn as a prior)
const PRIOR = [3, 1, 3, 1, 4, 1, 3, 1, 3];

/** Monte Carlo tree search. mode "uct": random rollouts; "puct": priors + rollouts, AlphaZero-style selection. */
function mcts(board, player, sims, mode, r) {
  const root = { b: board, p: player, N: 0, W: 0, kids: null };
  const expand = (n) => { const ms = moves(n.b), z = ms.reduce((a, m) => a + PRIOR[m], 0); n.kids = ms.map((m) => { const b = [...n.b]; b[m] = n.p; return { m, b, p: other(n.p), N: 0, W: 0, kids: null, P: mode === "puct" ? PRIOR[m] / z : 1 / ms.length }; }); };
  const rollout = (b, p) => { b = [...b]; for (;;) { const w = winner(b); if (w) return w; const ms = moves(b); b[ms[r.int(ms.length)]] = p; p = other(p); } };
  for (let s = 0; s < sims; s++) {
    const path = [root];
    let n = root;
    while (n.kids && n.kids.length && !winner(n.b)) {
      // pick the child maximising value + exploration bonus (from the perspective of the player to move at n)
      let best = null, bv = -Infinity;
      for (const k of n.kids) {
        const q = k.N ? k.W / k.N : 0;
        const u = mode === "puct" ? 1.5 * k.P * Math.sqrt(n.N + 1) / (1 + k.N) : k.N ? 1.4 * Math.sqrt(Math.log(n.N + 1) / k.N) : Infinity;
        if (q + u > bv) { bv = q + u; best = k; }
      }
      n = best; path.push(n);
    }
    if (!winner(n.b) && !n.kids) expand(n);
    const w = winner(n.b) || rollout(n.b, n.p);
    for (const x of path) { x.N++; const mover = other(x.p); x.W += w === "draw" ? 0.5 : w === mover ? 1 : 0; } // W from the view of whoever moved into x
  }
  return root.kids || [];
}

export default function (root, { id }) {
  let board = Array(9).fill(null), turn = "X", sims = 200, mode = "puct", analysis = [], r = rng(3), score = { you: 0, ai: 0, draw: 0 };
  const grid = h("div", { style: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "6px", width: "min(100%, 330px)" } });
  const st = stats([{ label: "Status" }, { label: "Your score" }, { label: "AI score" }, { label: "Draws" }]);
  const note = h("p.canvas-cap");
  function draw() {
    const maxN = Math.max(1, ...analysis.map((k) => k.N));
    grid.replaceChildren(...board.map((v, i) => {
      const k = analysis.find((x) => x.m === i);
      return h("button", { type: "button", "aria-label": `Square ${i + 1}${v ? ", " + v : ""}`, onclick: () => play(i), disabled: !!v || !!winner(board) || turn !== "X" || null,
        style: { aspectRatio: "1", borderRadius: "12px", border: "1px solid var(--line)", font: "700 38px var(--display)", cursor: v ? "default" : "pointer", position: "relative", color: v === "X" ? "var(--c1)" : "var(--c2)", background: k && !v ? `color-mix(in srgb, var(--c4) ${Math.round((k.N / maxN) * 55)}%, var(--surface))` : "var(--surface)" } },
        v || (k ? h("span", { style: { font: "500 11.5px/1.3 var(--mono)", color: "var(--ink-2)" } }, `${k.N} visits`, h("br"), `win ${(k.W / Math.max(1, k.N) * 100).toFixed(0)}%`) : ""));
    }));
    const w = winner(board);
    st.set(0, w ? (w === "draw" ? "Draw" : w === "X" ? "You win" : "AI wins") : turn === "X" ? "Your move (X)" : "AI thinking…", w === "X" ? "good" : w === "O" ? "bad" : "");
    st.set(1, String(score.you)); st.set(2, String(score.ai)); st.set(3, String(score.draw));
  }
  function finish() { const w = winner(board); if (w) { if (w === "draw") score.draw++; else if (w === "X") score.you++; else score.ai++; } }
  function play(i) {
    if (board[i] || winner(board) || turn !== "X") return;
    board[i] = "X"; turn = "O"; analysis = []; draw();
    if (winner(board)) return finish(), draw();
    setTimeout(() => {
      analysis = mcts(board, "O", sims, mode, r);
      const best = analysis.reduce((a, k) => (k.N > a.N ? k : a), analysis[0]);
      note.textContent = `The AI ran ${sims} simulations and played the most-visited square. Shading and numbers show its search for that move.`;
      board[best.m] = "O"; turn = "X"; finish(); draw();
    }, 60);
  }
  function analyse() { if (winner(board)) return; analysis = mcts(board, turn, sims, mode, r); note.textContent = `Analysis for ${turn} to move: ${sims} simulations.`; draw(); }
  const newGame = (aiFirst) => { board = Array(9).fill(null); analysis = []; turn = "X"; note.textContent = ""; if (aiFirst) { turn = "O"; draw(); setTimeout(() => { analysis = mcts(board, "O", sims, mode, r); const b = analysis.reduce((a, k) => (k.N > a.N ? k : a), analysis[0]); board[b.m] = "O"; turn = "X"; draw(); }, 60); } draw(); };

  root.append(
    section(null, prose(`<p>In 2016 DeepMind's AlphaGo beat Lee Sedol at Go, a game with more positions than atoms in the universe. A year later <b>AlphaZero</b> learned Go, chess and shogi from nothing but the rules, by playing against itself, and beat the best programs at all three. Its recipe has two halves: a neural network with intuition (which moves look promising, who's winning) and <b>Monte Carlo tree search</b>, which uses that intuition to look ahead.</p>`)),
    section("Play against a search",
      prose("<p>You're X. The AI runs Monte Carlo tree search before each move: it repeatedly walks down the tree of possible games, picking moves that are promising <i>or</i> under-explored, plays a random game to the end, and records who won. After the search, the square visited most is its move. Lower the simulations and beat it.</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Simulations per move", min: 5, max: 3000, step: 5, value: sims, onInput: (v) => (sims = v) }),
          segmented({ label: "Search", options: [{ value: "uct", label: "Plain MCTS (UCT)" }, { value: "puct", label: "With a prior (AlphaZero's PUCT)" }], value: mode, onChange: (v) => (mode = v) }),
          button("New game", () => newGame(false), "primary"), button("AI moves first", () => newGame(true)), button("Analyse this position", analyse)),
        h("div.stage-body", st.el, grid, note)),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>At 5 simulations the AI is easy to beat. At 1,000+ it never loses: tic-tac-toe is a draw with perfect play.</li><li>Press Analyse on the empty board with plain MCTS, then with the prior. The prior focuses the search on the centre and corners, so the same number of simulations goes deeper on the moves that matter. AlphaZero's network provides that prior, plus a value estimate that replaces random rollouts entirely.</li></ul>"),
      predict({
        chapter: id,
        question: "AlphaZero trains its network to predict which moves the search ended up choosing. Why does that make it stronger over time?",
        options: ["It memorises openings", "Search with the current network plays better than the network alone, so imitating search results improves the network, which makes the next search better still", "It copies human games", "It increases the number of simulations"],
        answer: 1,
        explain: "<p>That loop is <b>policy improvement by search</b>: search acts as an expert, the network distils the expert, and a better network makes a better expert. The value head learns from the self-play game outcomes. The same idea shows up in LLMs as “search, then train on the best answers” (expert iteration, STaR, and RL from verified samples).</p>",
      }),
    ),
    section("The self-play loop",
      stepper([
        { title: "1. Play against yourself", html: "<p>The current network plays thousands of games against itself, running MCTS at every move (800 simulations per move in the original). No human games at all.</p>" },
        { title: "2. Record what search preferred", html: "<p>At each position, store the visit counts from the search (a better policy than the raw network) and, once the game ends, who won.</p>" },
        { title: "3. Train the network", html: "<p>Policy head: cross-entropy towards the visit-count distribution. Value head: squared error towards the final result. One network, two heads, trained together.</p>" },
        { title: "4. Repeat", html: "<p>The improved network guides better searches, which produce better training targets. AlphaZero reached superhuman chess in about 4 hours of this on Google's TPUs.</p>" },
      ]),
      table(["System", "Year", "What it added"], [
        ["TD-Gammon", "1992", "Self-play + a neural value function reaches expert backgammon"],
        ["AlphaGo", "2016", "Policy and value networks trained on human games, then RL, plus MCTS"],
        ["AlphaZero", "2017", "No human data: pure self-play, one algorithm for three games"],
        ["MuZero", "2019", "Learns its own model of the rules and plans inside it"],
        ["AlphaProof, AlphaGeometry", "2024", "Search plus learning for olympiad maths, with formal verification"],
        ["LLM reasoning", "2024–", "Sampling many reasoning paths and training on the verified ones is search + learning again"],
      ]),
    ),
    section("In code",
      code({
        Python: `import math

def puct_select(node, c_puct=1.5):
    total = math.sqrt(node.N + 1)
    return max(node.children, key=lambda ch:
        (ch.W / ch.N if ch.N else 0)                       # exploitation: average value
        + c_puct * ch.prior * total / (1 + ch.N))          # exploration, guided by the policy net

def simulate(root, net):
    node, path = root, [root]
    while node.children:                                   # 1. select
        node = puct_select(node); path.append(node)
    priors, value = net(node.state)                        # 2. expand + evaluate (no rollout)
    node.expand(priors)
    for n in reversed(path):                               # 3. back up, flipping sign each ply
        n.N += 1; n.W += value; value = -value`,
      }, "Project 25 trains a small policy-value network for tic-tac-toe purely from self-play."),
    ),
  );
  draw();
}
