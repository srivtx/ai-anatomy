// Plain-word definitions for the hardware level. The first use of each term in a chapter gets a dotted
// underline; hover, focus or tap it for a one- or two-sentence definition. Later uses stay plain.
// Each entry: [spellings matched (longest first wins), definition]. Lowercase spellings match any case;
// spellings with capitals (SM, HBM, PTX) match exactly.

export const GLOSS = [
  [["transistors", "transistor"], "A tiny electrical switch with no moving parts. A voltage on its gate lets current through or blocks it. A modern chip has tens of billions."],
  [["MOSFET"], "The kind of transistor in every chip (metal-oxide-semiconductor field-effect transistor). A voltage on its gate pulls in electrons and opens a channel for current."],
  [["semiconductors", "semiconductor"], "A material, like silicon, that conducts electricity only a little, and whose conductivity we can turn up or down with impurities or a voltage."],
  [["doping"], "Adding a few atoms of another element to silicon on purpose. Phosphorus gives spare electrons (n-type); boron leaves gaps called holes (p-type)."],
  [["band gap"], "The energy an electron needs to break free and carry current. Silicon's is 1.1 eV: big enough to switch it off, small enough to switch it on."],
  [["p-n junction"], "Where p-type and n-type silicon meet. Current crosses it easily one way and hardly at all the other way."],
  [["capacitance"], "How much charge something holds per volt. Every wire and every transistor gate has some, and filling it with charge costs energy."],
  [["capacitors", "capacitor"], "Two conductors separated by an insulator. It stores charge, like a tiny bucket for electrons."],
  [["leakage"], "Current that trickles through a transistor that is supposed to be off. It wastes power even when nothing is switching."],
  [["tunnelling", "tunneling"], "A quantum effect: an electron can pass straight through a barrier that is thin enough, which a ball never could."],
  [["eV"], "Electron volt: a tiny unit of energy, what one electron gains moving across one volt."],
  [["logic gates", "logic gate"], "A tiny circuit that computes a yes/no answer from its inputs, such as AND, OR or NOT. A few transistors each."],
  [["NAND"], "A logic gate whose output is 0 only when both inputs are 1. Every other logic function can be built from NANDs alone."],
  [["flip-flops", "flip-flop"], "A small circuit (about 20 transistors) that holds one bit and only takes a new value on a clock tick."],
  [["registers", "register"], "The fastest, smallest storage in a processor: a few bytes held right beside the arithmetic units."],
  [["clock speed", "clock"], "A signal that ticks billions of times a second. Every part of the chip moves one step forward on each tick."],
  [["GHz"], "Gigahertz: a billion clock ticks per second."],
  [["pipelining", "pipeline"], "Splitting work into stages like an assembly line, so the next item can start before the last one is finished."],
  [["branch prediction", "branch predictor"], "Guessing which way an if-statement will go before the answer is known, so the processor keeps busy instead of waiting."],
  [["out-of-order"], "Running instructions in a different order from the program, as soon as their inputs are ready, while making the results look in order."],
  [["speculation", "speculative"], "Doing work before knowing whether it's needed, and throwing it away if the guess was wrong."],
  [["instructions", "instruction"], "One basic command for a processor, such as “add these two numbers” or “load this value from memory”."],
  [["machine code"], "Instructions written as numbers (bits), which the processor reads and carries out directly."],
  [["assembly"], "Machine instructions spelled in words, one per line, like ADD or MOV, so people can read and write them."],
  [["assembler"], "A simple translator that turns each line of assembly into the bytes of one machine instruction, mostly by looking it up in a table."],
  [["stack machine"], "A machine that keeps its working numbers on a pile: push values on top, and each operation takes the top two and pushes back the answer."],
  [["hexadecimal", "hex"], "Writing numbers in base 16, with digits 0–9 then A–F. Each hex digit is exactly 4 bits, so a byte is two hex digits."],
  [["bytes", "byte"], "Eight bits: enough for a number from 0 to 255, or one letter of plain text."],
  [["compilers", "compiler"], "A program that translates code people write into instructions a machine can run."],
  [["interpreter"], "A program that reads your code and carries it out step by step, instead of translating all of it to machine code first."],
  [["abstraction"], "Hiding details behind something simpler, so you can use it without knowing how it works inside."],
  [["caches", "cache"], "A small, fast memory close to the processor that keeps copies of recently used data, so it doesn't have to be fetched from far away again."],
  [["SRAM"], "Static RAM: memory that keeps each bit in 6 transistors. Fast but bulky, so it's used for caches and on-chip memory."],
  [["DRAM"], "Dynamic RAM: memory that keeps each bit as charge in a tiny capacitor. Dense and cheap, but slower, and it leaks, so it must be rewritten every few milliseconds."],
  [["HBM"], "High Bandwidth Memory: stacks of DRAM chips placed right beside the GPU and joined to it by thousands of wires."],
  [["bandwidth"], "How much data can move per second. Think of the width of a pipe."],
  [["latency"], "How long one request takes, from asking to getting the answer. Think of the length of a pipe."],
  [["throughput"], "How much work gets done per second in total, however long each single piece takes."],
  [["cores", "core"], "One independent processor inside a chip, running its own stream of instructions."],
  [["threads", "thread"], "One stream of instructions being run. On a GPU, each thread usually handles one small piece of the data."],
  [["warps", "warp"], "A group of 32 GPU threads that run the same instruction at the same moment, each on its own data."],
  [["SMs", "SM"], "Streaming Multiprocessor: one of the GPU's building blocks, with its own cores, registers and shared memory. An H100 has 132 that work."],
  [["tensor cores", "tensor core"], "A unit inside the GPU that multiplies small blocks of numbers in one step: far more multiply-adds per instruction than ordinary cores."],
  [["shared memory"], "Fast memory inside each SM, which the threads of one block use to pass data to each other."],
  [["kernels", "kernel"], "A function that runs on the GPU, started across thousands or millions of threads at once."],
  [["PCIe"], "The link between the CPU's side of the computer and the GPU card. Much slower than the GPU's own memory."],
  [["PTX"], "NVIDIA's portable, assembly-like code. The driver turns it into the real machine code (SASS) for your exact GPU."],
  [["SASS"], "The real machine code that one generation of NVIDIA GPU runs."],
  [["CUDA"], "NVIDIA's way to program GPUs: write the code for one thread, in C++, and launch it on millions."],
  [["Triton"], "A Python-like language for writing GPU kernels one block of data at a time; it compiles to fast GPU code."],
  [["fusion"], "Merging several GPU steps into one kernel, so data stays on the chip instead of going back to memory in between."],
  [["SIMT"], "Single Instruction, Multiple Threads: one instruction drives many threads at once, each on its own data."],
  [["multiply-adds", "multiply-add", "multiply-accumulate", "MACs", "MAC", "FMA"], "Multiply two numbers and add the result to a running total. The one operation AI does trillions of times."],
  [["FLOPs", "FLOP/s", "FLOP"], "Floating-point operation: one add or one multiply on decimal-point numbers. FLOP/s counts how many happen each second."],
  [["floating-point", "floating point"], "A way of storing numbers like scientific notation: some bits hold the digits, some hold the power of two."],
  [["bf16"], "A 16-bit number format with the range of 32-bit floats but fewer digits. The usual format for training."],
  [["fp8"], "An 8-bit number format: few digits, but very cheap to multiply. Used on H100 and later GPUs."],
  [["matrices", "matrix"], "A grid of numbers. Multiplying matrices is how a neural-network layer mixes its inputs."],
  [["vectors", "vector"], "A list of numbers, such as the 64 pixels of a small drawing."],
  [["operands", "operand"], "A number an instruction works on, such as the two inputs of an add."],
  [["wafers", "wafer"], "A thin disc of pure silicon, 300 mm across, on which hundreds of chips are made at once."],
  [["nanometres", "nanometre", "nm"], "A billionth of a metre. A silicon atom is about 0.2 nm across."],
  [["shaders", "shader"], "A small program the GPU runs once for every vertex or every pixel. Before CUDA, researchers disguised maths as shaders."],
  [["rasterisation", "rasterization"], "Working out which pixels a shape, such as a triangle, covers."],
  [["vertices", "vertex"], "A corner point of a triangle in a 3D model."],
  [["Dennard scaling"], "The rule (1974) that as transistors shrink, their power falls just as fast, so chips stay equally cool. It broke down around 2005."],
  [["Moore's law"], "The observation (1965, updated 1975) that the number of transistors on a chip doubles about every two years."],
  [["Landauer"], "Landauer's principle: erasing one bit must give off at least kT·ln 2 of heat, about 3 zeptojoules at room temperature."],
  [["Amdahl"], "Amdahl's law: if part of a job can't be split up, that part limits the total speed-up, however many processors you add."],
  [["superposition"], "In a linear system, effects simply add up without disturbing each other. Two beams of light pass through each other unchanged."],
  [["JIT", "just-in-time"], "Just-in-time compilation: turning code into machine code while the program runs, once it's clear which parts are busy."],
  [["torch.compile"], "PyTorch's compiler: it captures your model as a graph and writes fused GPU kernels for it."],
  [["voltage", "volts"], "The push on electric charge: the energy each unit of charge gains or loses. 1 volt is 1 joule per coulomb. A chip's core runs on under 1 volt."],
  [["amps", "amperes"], "The unit of current: how much charge flows past a point each second. 1 amp is about 6 × 10¹⁸ electrons a second."],
  [["resistance"], "How hard a material makes it for current to flow, in ohms. Electrons bump into vibrating atoms and impurities and lose energy as heat."],
  [["lithography"], "Printing patterns on a chip with light: shine light through a mask onto a light-sensitive coating, then etch where the light fell."],
  [["EUV"], "Extreme ultraviolet: light of 13.5 nm wavelength, used since 2019 to print the finest layers of chips. Everything absorbs it, so the machines use mirrors in a vacuum."],
  [["wafers", "wafer"], "A thin, polished disc sliced from one big crystal of silicon, usually 300 mm across. Hundreds of chips are made on it at once and then cut apart."],
  [["chiplets", "chiplet"], "A small die that is combined with others in one package to act as one big chip. Smaller dies are less likely to contain a defect."],
  [["reticle"], "The mask holding one layer's pattern. The largest area a scanner can print in one shot (~26 × 33 mm) is called the reticle limit."],
];

const ESC = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const KEY = new Map();                     // spelling (lowercased unless it has capitals) → entry index
GLOSS.forEach(([sp], i) => sp.forEach((s) => KEY.set(/[A-Z]/.test(s) ? s : s.toLowerCase(), i)));
const ALL = [...KEY.keys()].sort((a, b) => b.length - a.length);
const RX = new RegExp(`(?<![\\w-])(${ALL.map((s) => (/[A-Z]/.test(s) ? ESC(s) : ESC(s).replace(/[a-z]/g, (ch) => `[${ch}${ch.toUpperCase()}]`))).join("|")})(?![\\w-])`, "g");
const lookup = (m) => (KEY.has(m) ? KEY.get(m) : KEY.get(m.toLowerCase()));
const SKIP = "a, code, pre, button, label, h1, h2, h3, h4, summary, svg, canvas, table, .gterm, .predict, .xt-tabs, .stats, .stage-controls, .eyebrow, .crumbs, .vx-row, .tw, .walk";

let pop;
function popover() {
  if (pop) return pop;
  pop = document.createElement("div"); pop.className = "gl-pop"; pop.setAttribute("role", "tooltip"); pop.id = "gl-pop"; pop.hidden = true;
  document.body.append(pop);
  let cur = null;
  const show = (el) => {
    cur = el; const [title, def] = [el.dataset.t, GLOSS[+el.dataset.k][1]];
    pop.replaceChildren(Object.assign(document.createElement("b"), { textContent: title }), document.createTextNode(" " + def));
    pop.hidden = false; el.setAttribute("aria-describedby", "gl-pop");
    const r = el.getBoundingClientRect(), w = Math.min(320, window.innerWidth - 24);
    pop.style.width = `${w}px`;
    const left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2));
    const below = r.bottom + 8 + pop.offsetHeight < window.innerHeight;
    pop.style.left = `${left + window.scrollX}px`;
    pop.style.top = `${(below ? r.bottom + 8 : r.top - 8 - pop.offsetHeight) + window.scrollY}px`;
  };
  const hide = () => { if (cur) cur.removeAttribute("aria-describedby"); cur = null; pop.hidden = true; };
  document.addEventListener("mouseover", (e) => { const el = e.target.closest?.(".gterm"); if (el) show(el); else if (cur && !e.target.closest?.(".gl-pop")) hide(); });
  document.addEventListener("focusin", (e) => { const el = e.target.closest?.(".gterm"); if (el) show(el); });
  document.addEventListener("focusout", (e) => { if (e.target.closest?.(".gterm")) hide(); });
  document.addEventListener("click", (e) => { const el = e.target.closest?.(".gterm"); if (el) { if (cur === el && !pop.hidden) hide(); else show(el); } else if (!e.target.closest?.(".gl-pop")) hide(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
  window.addEventListener("scroll", hide, { passive: true });
  return pop;
}

/** Underline the first use of each glossary term inside root, now and whenever new text appears (steppers, why-chains). */
export function glossify(root) {
  popover();
  const used = () => new Set([...root.querySelectorAll(".gterm")].map((e) => e.dataset.k));
  function run() {
    const seen = used();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (!n.nodeValue.trim() || !n.parentElement || n.parentElement.closest(SKIP) || !n.parentElement.closest("p, li") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const text = node.nodeValue; RX.lastIndex = 0; let m, last = 0; const frag = []; let hit = false;
      while ((m = RX.exec(text))) {
        const k = lookup(m[1]); if (k === undefined || seen.has(String(k))) continue;
        if (/^assembly$/i.test(m[1]) && /^ line/.test(text.slice(m.index + m[1].length))) continue;   // an assembly line is not assembly language
        seen.add(String(k)); hit = true;
        frag.push(document.createTextNode(text.slice(last, m.index)));
        const sp = document.createElement("span"); sp.className = "gterm"; sp.tabIndex = 0; sp.dataset.k = String(k); sp.dataset.t = m[1];
        sp.textContent = m[1]; frag.push(sp); last = m.index + m[1].length;
      }
      if (hit) { frag.push(document.createTextNode(text.slice(last))); node.replaceWith(...frag); }
    }
  }
  run();
  let t = 0;
  new MutationObserver((recs) => {
    if (recs.every((r) => [...r.addedNodes].every((n) => n.nodeType === 1 && n.classList?.contains("gterm")))) return;
    clearTimeout(t); t = setTimeout(run, 60);
  }).observe(root, { childList: true, subtree: true });
}
