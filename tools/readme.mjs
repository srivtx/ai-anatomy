// Regenerate README.md from the catalog: node tools/readme.mjs
import { writeFileSync } from "node:fs";
import { LEVELS, PHASES, PROJECTS } from "../lib/catalog.js";
import { WALKED } from "../lib/walklist.js";
import { MODES, chaptersOf } from "../lib/paths.js";
import { THREADS } from "../lib/ask/index.js";
const modes = MODES.map((m) => `| **${m.name}** | ${m.who} | ~${m.hours} h | ${m.stations ? chaptersOf(m, LEVELS).length + " chapters" : "all"}${m.ask ? ` + ${m.stations.flatMap((s) => s.ask || []).length} conversations` : ""} |`).join("\n");
const chapters = LEVELS.flatMap((l) => l.chapters).filter((c) => c.live).length;
const lv = LEVELS.map((l, i) => `**${i}. ${l.name}**: ${l.chapters.map((c) => c.title).join(" · ")}`).join("\n\n");
const ph = PHASES.map((p) => `| ${p.name} | ${p.projects.length} | ${p.projects.map((x) => x.title).join(", ")} |`).join("\n");
writeFileSync("README.md", `# AI Anatomy

How machines learn, from one neuron to the frontier. ${chapters} interactive chapters that train real models in your browser, and ${PROJECTS.length} Colab projects where you build each idea yourself.

![The home page](docs/img/home.png)

Every chapter runs a live simulation in plain JavaScript (no build step, no server-side compute): you predict what will happen, change a setting, and watch a network learn. Each chapter also has field notes on what works in practice, what the big runs found, what is still unknown, and deeper cuts with code. ${WALKED.size} chapters also step through a few lines of real code with real numbers, drawing every tensor as a stack of pixel cubes you can hover for the arithmetic.

| | |
|---|---|
| ![Sharpness locks to 2/η](docs/img/edge-of-stability.png) | ![An attention layer that turns into gradient descent](docs/img/attention-is-gd.png) |
| **The hidden laws of training.** Sharpness climbs until it hits 2/η, then stays there. | **Every way machines learn now.** Train an attention layer on regression tasks and watch it become one step of gradient descent. |
| ![Permutation alignment](docs/img/converging-minds.png) | ![Puzzles vision models fail](docs/img/vlm-puzzles.png) |
| **Different networks, same mind.** Two seeds look unrelated until their neurons are re-ordered. | **How machines see (and don't).** Puzzles a child solves that vision-language models still miss. |
| ![pass@k before and after RL](docs/img/rl-narrows.png) | ![Thirty generations of models trained on models](docs/img/self-consumption.png) |
| **What RL really does to a model.** pass@1 rises while pass@k falls: RL sharpens more than it teaches. | **Models eating their own outputs.** Each generation forgets more of the tail of the distribution. |

## Start where you are

Nobody learns ${chapters} chapters at once. [\`/paths\`](paths.html) offers seven learning modes, each a route through the same course for one goal, drawn as a metro line, with the Colab projects that fit it. Pick one and the rest of the site follows it: chapters say whether they're on your path, *Next* follows your route, the quiz can draw only from it, and the roadmap marks its projects.

| Mode | For | Time | Stops |
|---|---|---|---|
${modes}

[\`/ask\`](ask.html) is **Ask Neuro**, a Socratic path for complete beginners: ${THREADS.length} short conversations in very easy English, from “what does it mean to learn?” to agents and why models make things up. Neuro asks, you answer, and every idea comes with a small picture to play with: label spam, paint pixels, fit a line, roll a ball downhill, wire a neuron into an AND gate, fail at XOR with one line, watch blame flow backwards, tokenise your own sentence, explore a word map, race a CPU against a GPU, judge two answers, step an agent through its loop. Wrong answers get a hint and a second try; every conversation ends with a recap and the new words.

## The Architecture Studio

[\`/studio\`](studio.html) is a real transformer in 3D, drawn at its true proportions, that you open level by level (model › block › attention › one head) and change part by part. It has three modes:

- **Learn**: eleven levels that build a transformer one part at a time, from a bare lookup table to the modern Llama recipe, the KV cache and mixture of experts, with Neuro explaining why each part exists and checks that tick as you go. Each level asks you to guess first (answered with numbers from our own lab runs) and clears up a common mix-up. Clear a level to get three numbers to keep, then review them as flashcards.
- **Challenges**: real design problems (shrink the KV cache 4×, fit 100 users on one GPU, read 128k tokens cheaply, the best model for $500) checked live against the predictor.
- **Build**: start from Tiny GPT, GPT-2, GPT-3, Llama 2, Llama 3 8B, Mistral 7B or Mixtral 8×7B (exact parameter counts), ride the time machine from 2019 to 2024 and see what changed at each step, apply recipes, edit anything, pin a design to compare against (it stands next to yours in 3D), see where its parameters live, which devices it fits on at 16, 8 or 4 bits, how one H100's 80 GB fills up with chats, how its loss falls as it reads more, and save a picture of it.

Press ▶ **Watch a token** (or Space) to follow one token through the model stop by stop, with pause, step and speed controls. Inside an attention head it scans word by word; on the readout you can turn the temperature and pick the next word yourself. Earn stickers as you go.

Five panels: **Inspect** (what a part is and why it's there), **Predict** (what will happen before training, each line tagged exact / physics / law / rule of thumb / test it), **Speed** (what makes it faster on an H100 and what doesn't), **Test** (one click to a Colab run of your design against its baseline, then paste the result back to see if the prediction held, and colour the 3D blocks by the measured gradients) and **Design**.

## The curriculum

${lv}

## Colab projects

| Phase | Projects | |
|---|---|---|
${ph}

Every notebook runs end to end on a CPU in a quick mode (\`QUICK=1\`), with offline fallbacks when Hugging Face or an API key isn't available. The expected results quoted on the site were measured from these notebooks; where a run needs a GPU we didn't have, the result is labelled as a prediction.

## Honesty conventions

- Results are either measured (by these notebooks or simulations), quoted from published work with a citation, or labelled as a prediction, intuition or bet with a confidence.
- When a toy fails to show a published effect, the chapter says so and explains why it should hold at scale.

## Run it locally

\`\`\`bash
node serve.mjs            # http://localhost:4173
node test/smoke.mjs       # loads every page in Chromium and fails on any error
\`\`\`

Notebooks are written as \`notebooks/src/*.py\` with \`# %%\` cells and built with \`python tools/build_notebooks.py\`; \`tools/test_notebooks.sh\` runs them in quick mode.

## Deploy

It's a static site. On Vercel: import the repository, choose the "Other" framework preset, leave the build command empty, deploy. \`vercel.json\` already maps \`/learn?c=…\` and the other clean URLs.
`);
console.log("README.md:", chapters, "chapters,", PROJECTS.length, "projects");
