import { h, rng } from "../lib/core.js";
import { slider, segmented, stats, predict, code, callout, section, prose, table, button, stepper } from "../lib/widgets.js";
import { xyChart, plane } from "../lib/plot.js";
import { gauss } from "../lib/nn.js";

const SR = 8000, DUR = 1.2;
// a perceptual-ish dark-to-bright ramp for spectrogram magnitudes (works on both themes because it paints its own background)
const RAMP = [[12, 8, 38], [70, 30, 120], [160, 50, 130], [230, 100, 80], [252, 200, 90], [252, 250, 200]];
const ramp = (t) => { t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1); const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i; return RAMP[i].map((v, k) => Math.round(v + (RAMP[i + 1][k] - v) * f)); };

function synth(kind, f0, noise, seed = 1) {
  const r = rng(seed), n = Math.round(SR * DUR), x = new Float32Array(n);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    let v = 0;
    if (kind === "tones") v = 0.5 * Math.sin(2 * Math.PI * f0 * t) + 0.35 * Math.sin(2 * Math.PI * f0 * 2.5 * t);
    else if (kind === "chirp") { const f = f0 * 0.5 + (3600 - f0 * 0.5) * (t / DUR); phase += (2 * Math.PI * f) / SR; v = 0.6 * Math.sin(phase); }
    else if (kind === "vowel") {
      // harmonics of the pitch, shaped by two formants that glide from an "ah" to an "ee"
      const g = t / DUR, F1 = 750 - 470 * g, F2 = 1100 + 1200 * g;
      for (let k = 1; k * f0 < SR / 2; k++) { const fk = k * f0, a = Math.exp(-(((fk - F1) / 230) ** 2)) + 0.7 * Math.exp(-(((fk - F2) / 300) ** 2)) + 0.02; v += a * Math.sin(2 * Math.PI * fk * t); }
      v *= 0.25;
    } else { const period = Math.round(SR / (f0 / 20)); v = i % period < 8 ? 0.9 * (1 - (i % period) / 8) : 0; }
    x[i] = v + noise * gauss(r) * 0.3;
  }
  return x;
}

function stft(x, N) {
  const hop = N / 4, frames = Math.floor((x.length - N) / hop) + 1, bins = N / 2;
  const win = Float32Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
  const cos = new Float32Array(N * bins), sin = new Float32Array(N * bins);
  for (let k = 0; k < bins; k++) for (let i = 0; i < N; i++) { cos[k * N + i] = Math.cos((2 * Math.PI * k * i) / N); sin[k * N + i] = Math.sin((2 * Math.PI * k * i) / N); }
  const out = [];
  for (let f = 0; f < frames; f++) {
    const col = new Float32Array(bins), o = f * hop;
    for (let k = 0; k < bins; k++) { let re = 0, im = 0; for (let i = 0; i < N; i++) { const v = x[o + i] * win[i]; re += v * cos[k * N + i]; im -= v * sin[k * N + i]; } col[k] = 20 * Math.log10(Math.hypot(re, im) + 1e-6); }
    out.push(col);
  }
  return out;
}

const mel = (f) => 2595 * Math.log10(1 + f / 700), imel = (m) => 700 * (10 ** (m / 2595) - 1);

export default function (root, { id }) {
  // ---------- 1. waveform and spectrogram ----------
  let kind = "vowel", f0 = 140, noise = 0.02, N = 256, useMel = false, nMel = 40, x;
  const wave = xyChart({ height: 150, xLabel: "time (ms)", yLabel: "amplitude", xDomain: [0, 40], yDomain: [-1.2, 1.2] });
  const spec = h("canvas", { width: 600, height: 220, style: { width: "100%", height: "260px", display: "block", borderRadius: "10px", border: "1px solid var(--line)", imageRendering: "pixelated" } });
  const sst = stats([{ label: "Window" }, { label: "Time resolution" }, { label: "Frequency resolution" }, { label: "Frames per second" }]);
  function drawSpec() {
    x = synth(kind, f0, noise);
    const zoom = Array.from({ length: 321 }, (_, i) => [i * 0.125, x[Math.round(i * 0.125 * SR / 1000) + 2000]]);
    wave.update([{ name: "Waveform (a 40 ms slice)", color: "var(--accent)", points: zoom }], { xDomain: [0, 40], yDomain: [-1.2, 1.2] });
    let S = stft(x, N);
    if (useMel) {
      // triangular filters equally spaced on the mel scale; energies summed then logged
      const bins = N / 2, edges = Array.from({ length: nMel + 2 }, (_, i) => imel((mel(SR / 2) * i) / (nMel + 1)));
      S = S.map((col) => Float32Array.from({ length: nMel }, (_, m) => {
        let e = 0;
        for (let k = 0; k < bins; k++) { const f = (k * SR) / N, w = Math.max(0, Math.min((f - edges[m]) / (edges[m + 1] - edges[m]), (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]))); e += w * 10 ** (col[k] / 10); }
        return 10 * Math.log10(e + 1e-9);
      }));
    }
    const F = S.length, B = S[0].length, ctx = spec.getContext("2d");
    spec.width = F; spec.height = B;
    const img = ctx.createImageData(F, B);
    let mx = -Infinity; for (const c of S) for (const v of c) mx = Math.max(mx, v);
    for (let f = 0; f < F; f++) for (let b = 0; b < B; b++) { const c = ramp((S[f][b] - (mx - 55)) / 55), k = ((B - 1 - b) * F + f) * 4; img.data.set([...c, 255], k); }
    ctx.putImageData(img, 0, 0);
    sst.set(0, `${N} samples`); sst.set(1, `${((N / SR) * 1000).toFixed(0)} ms`); sst.set(2, useMel ? `${nMel} mel bands` : `${(SR / N).toFixed(1)} Hz`); sst.set(3, String(Math.round(SR / (N / 4))));
  }
  let actx;
  function play() {
    actx ??= new (window.AudioContext || window.webkitAudioContext)();
    const buf = actx.createBuffer(1, x.length, SR); buf.copyToChannel(x, 0);
    const src = actx.createBufferSource(), g = actx.createGain(); g.gain.value = 0.5;
    src.buffer = buf; src.connect(g).connect(actx.destination); src.start();
  }

  // ---------- 2. aliasing ----------
  let fa = 1000, fs = 8000;
  const alias = xyChart({ width: 460, height: 240, xLabel: "time (ms)", yLabel: "amplitude", xDomain: [0, 5], yDomain: [-1.3, 1.3] });
  const ast = stats([{ label: "Nyquist limit fs / 2" }, { label: "What the samples say" }]);
  function drawAlias() {
    const ny = fs / 2, k = Math.round(fa / fs), fal = Math.abs(fa - k * fs);
    const tru = Array.from({ length: 1001 }, (_, i) => { const t = i * 0.005; return [t, Math.sin(2 * Math.PI * fa * t / 1000)]; });
    const smp = []; for (let t = 0; t <= 5; t += 1000 / fs) smp.push([t, Math.sin(2 * Math.PI * fa * t / 1000)]);
    // the lowest-frequency sine through the same samples (sign flips for folded frequencies)
    const sign = fa - k * fs >= 0 ? 1 : -1;
    const rec = Array.from({ length: 1001 }, (_, i) => { const t = i * 0.005; return [t, sign * Math.sin(2 * Math.PI * fal * t / 1000)]; });
    alias.update([
      { name: `True signal, ${fa} Hz`, color: "var(--ink-3)", points: tru },
      { name: `Simplest sine through the samples, ${Math.round(fal)} Hz`, color: "var(--c2)", points: rec, dash: true },
      { name: "Samples", color: "var(--accent)", points: smp, dots: true, line: false, r: 3.5 },
    ], { xDomain: [0, 5], yDomain: [-1.3, 1.3] });
    ast.set(0, `${ny} Hz`); ast.set(1, `${Math.round(fal)} Hz`, fa > ny ? "bad" : "good");
  }

  // ---------- 3. mel scale ----------
  const melC = xyChart({ width: 460, height: 240, xLabel: "frequency (Hz)", yLabel: "mel", xDomain: [0, 8000] });

  // ---------- 4. residual vector quantisation ----------
  const R = rng(3);
  const pts = Array.from({ length: 500 }, (_, i) => { const a = R() * Math.PI * 2, m = i % 3; return m === 0 ? [Math.cos(a) * 0.8 + gauss(R) * 0.06, Math.sin(a) * 0.8 + gauss(R) * 0.06] : m === 1 ? [0.2 + gauss(R) * 0.15, -0.1 + gauss(R) * 0.25] : [-0.35 + gauss(R) * 0.08, 0.3 + gauss(R) * 0.08]; });
  let K = 4, L = 2;
  const rv = plane({ width: 420, height: 420, xr: [-1.15, 1.15], yr: [-1.15, 1.15], label: "Residual vector quantisation" });
  const rst = stats([{ label: "Bits per vector" }, { label: "Codebook entries stored" }, { label: "Reconstruction error" }, { label: "One flat codebook would need" }]);
  function kmeans(data, k) {
    const r = rng(11); let C = Array.from({ length: k }, () => data[Math.floor(r() * data.length)].slice());
    for (let it = 0; it < 25; it++) {
      const sum = C.map(() => [0, 0, 0]);
      for (const p of data) { let bi = 0, bd = Infinity; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; bi = j; } }); sum[bi][0] += p[0]; sum[bi][1] += p[1]; sum[bi][2]++; }
      C = C.map((c, j) => (sum[j][2] ? [sum[j][0] / sum[j][2], sum[j][1] / sum[j][2]] : data[Math.floor(r() * data.length)].slice()));
    }
    return C;
  }
  function drawRvq() {
    let resid = pts.map((p) => p.slice()); const recon = pts.map(() => [0, 0]);
    for (let l = 0; l < L; l++) {
      const C = kmeans(resid, K);
      resid.forEach((p, i) => { let bi = 0, bd = Infinity; C.forEach((c, j) => { const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2; if (d < bd) { bd = d; bi = j; } }); recon[i][0] += C[bi][0]; recon[i][1] += C[bi][1]; resid[i] = [p[0] - C[bi][0], p[1] - C[bi][1]]; });
    }
    const err = resid.reduce((a, p) => a + p[0] ** 2 + p[1] ** 2, 0) / pts.length;
    rv.paint(() => {
      rv.clear(); rv.axes();
      pts.forEach((p, i) => rv.line(rv.X(p[0]), rv.Y(p[1]), rv.X(recon[i][0]), rv.Y(recon[i][1]), { color: rv.colors.line, lw: 0.75 }));
      rv.scatter(pts.map((p) => ({ x: p })), { color: rv.colors["ink-3"], r: 2, ring: false });
      const uniq = [...new Map(recon.map((p) => [p.map((v) => v.toFixed(4)).join(), p])).values()];
      rv.scatter(uniq.map((p) => ({ x: p })), { color: rv.colors.accent, r: 4.5 });
    });
    const bits = L * Math.log2(K);
    rst.set(0, String(bits)); rst.set(1, String(L * K)); rst.set(2, err.toFixed(4), err < 0.005 ? "good" : ""); rst.set(3, `${(K ** L).toLocaleString()} entries`);
  }

  // ---------- 5. codec bitrate ----------
  let fr = 75, nq = 8, cbBits = 10;
  const bst = stats([{ label: "Bitrate" }, { label: "Tokens per second of audio" }, { label: "Tokens in a 1-minute clip" }]);
  function drawBits() {
    const kbps = (fr * nq * cbBits) / 1000, tps = fr * nq;
    bst.set(0, `${kbps.toFixed(kbps < 10 ? 1 : 0)} kbps`); bst.set(1, tps.toLocaleString(), tps > 300 ? "warn" : "good"); bst.set(2, (tps * 60).toLocaleString());
  }

  root.append(
    section(null, prose(`<p>Sound is a single number changing over time: air pressure, sampled thousands of times a second. Speech recognition, text-to-speech, music generation and voice assistants all start by deciding how to turn that long, fast-wiggling signal into something a network can digest. There are two answers you need to know: the <b>spectrogram</b>, which shows which frequencies are present at each moment, and <b>audio tokens</b> from a neural codec, which turn sound into a sequence a language model can predict.</p>`)),
    section("From waveform to spectrogram",
      prose("<p>A short-time Fourier transform slides a window along the signal and measures how much of each frequency is in each window. Time runs left to right, frequency bottom to top, brightness is energy (in decibels). Everything here is computed in your browser; press Play to hear it.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Sound", options: [{ value: "vowel", label: "Vowel “ah→ee”" }, { value: "tones", label: "Two tones" }, { value: "chirp", label: "Rising chirp" }, { value: "clicks", label: "Clicks" }], value: kind, onChange: (v) => { kind = v; drawSpec(); } }),
          slider({ label: "Pitch", min: 80, max: 400, step: 5, value: f0, fmt: (v) => `${v} Hz`, onInput: (v) => { f0 = v; drawSpec(); } }),
          slider({ label: "Noise", min: 0, max: 1, step: 0.05, value: noise, fmt: (v) => v.toFixed(2), onInput: (v) => { noise = v; drawSpec(); } }),
          segmented({ label: "Window", options: [64, 128, 256, 512].map((v) => ({ value: v, label: `${v}` })), value: N, onChange: (v) => { N = v; drawSpec(); } }),
          segmented({ label: "Frequency axis", options: [{ value: false, label: "Linear (Hz)" }, { value: true, label: "Mel bands" }], value: false, onChange: (v) => { useMel = v; drawSpec(); } }),
          button("▶ Play", () => play(), "primary")),
        h("div.stage-body", sst.el, wave.el, spec, h("p.canvas-cap", "Spectrogram, 0–4 kHz, 1.2 seconds. Dark = quiet, bright = loud."))),
      callout("tip", "Things to try", "<ul style='margin:0;padding-left:18px'><li>Pick the vowel. The evenly spaced horizontal lines are harmonics of the pitch; the two bright bands that slide are the <b>formants</b>, resonances of the mouth that make “ah” different from “ee”. Change the pitch: the harmonics move, the formants don't. That's why you recognise a vowel whoever says it.</li><li>Switch the window between 64 and 512 on the clicks and on the tones. Short windows locate clicks sharply in time but smear frequencies; long windows separate frequencies but blur time. You can't have both: it's the uncertainty principle for signals.</li><li>Turn on mel bands. Low frequencies get many narrow bands and high frequencies a few wide ones, roughly like the ear. Whisper's input is exactly this: an 80-band log-mel spectrogram.</li></ul>"),
    ),
    section("Sampling and aliasing",
      prose("<p>You can only capture frequencies below half the sampling rate (the <b>Nyquist limit</b>). Anything higher doesn't disappear: it masquerades as a lower frequency. That's why CDs sample at 44.1 kHz (hearing tops out near 20 kHz), why speech models often use 16 kHz (speech lives mostly below 8 kHz), and why audio is low-pass filtered before downsampling.</p>"),
      h("div.lab",
        h("div.stage",
          h("div.stage-controls",
            slider({ label: "Signal frequency", min: 200, max: 7800, step: 100, value: fa, fmt: (v) => `${v} Hz`, onInput: (v) => { fa = v; drawAlias(); } }),
            slider({ label: "Sampling rate", min: 2000, max: 16000, step: 500, value: fs, fmt: (v) => `${v} Hz`, onInput: (v) => { fs = v; drawAlias(); } })),
          h("div.stage-body", ast.el, alias.el)),
        h("div", melC.el, h("p.canvas-cap", "The mel scale: equal steps in mel sound like equal steps in pitch. Doubling from 100 to 200 Hz is a big jump; 4,000 to 4,100 Hz is barely audible."))),
      predict({
        chapter: id,
        question: "A 7,000 Hz tone is sampled at 8,000 Hz. What frequency do the samples look like?",
        options: ["7,000 Hz", "4,000 Hz", "1,000 Hz", "15,000 Hz"],
        answer: 2,
        explain: "<p>It folds around the sampling rate: |7,000 − 8,000| = 1,000 Hz. Set the sliders to check. The samples of a 7 kHz sine and a (phase-flipped) 1 kHz sine are identical, so no algorithm can tell them apart afterwards.</p>",
      }),
    ),
    section("Audio as tokens: residual vector quantisation",
      prose("<p>To let a language model <i>generate</i> audio, you need discrete tokens. Neural codecs (SoundStream, EnCodec, Mimi) encode each short frame of audio into a vector, then <b>quantise</b> it: replace it with the nearest entry in a codebook and send only the entry's number. One codebook is coarse. <b>Residual</b> VQ quantises what's left over with a second codebook, then a third, and so on. Each level refines the last. Here each point is a 2D “frame”; grey lines connect it to its reconstruction.</p>"),
      h("div.stage",
        h("div.stage-controls",
          segmented({ label: "Codebook size", options: [2, 4, 8, 16].map((v) => ({ value: v, label: String(v) })), value: K, onChange: (v) => { K = v; drawRvq(); } }),
          slider({ label: "Residual levels", min: 1, max: 6, value: L, onInput: (v) => { L = v; drawRvq(); } })),
        h("div.stage-body", rst.el, h("div.canvas-box", rv.el))),
      callout("note", "Why residual", "<p>Six levels of 16 entries give 24 bits per frame from just 96 stored vectors. A single flat codebook with the same precision would need 16<sup>6</sup> ≈ 16.8 million entries. Bonus: the first level alone is a decent coarse version, so a model can predict level 1 first and fill in finer levels later. (Many speech LMs, like Moshi, do exactly this.)</p>"),
      h("div.stage",
        h("div.stage-controls",
          slider({ label: "Frames per second", min: 12.5, max: 75, step: 12.5, value: fr, fmt: (v) => `${v} Hz`, onInput: (v) => { fr = v; drawBits(); } }),
          slider({ label: "Codebooks (levels)", min: 1, max: 32, value: nq, onInput: (v) => { nq = v; drawBits(); } }),
          slider({ label: "Bits per codebook", min: 8, max: 14, value: cbBits, fmt: (v) => `${v} (${2 ** v} entries)`, onInput: (v) => { cbBits = v; drawBits(); } })),
        h("div.stage-body", bst.el, h("p.canvas-cap", "Defaults: EnCodec at 24 kHz, 6 kbps. Try Mimi's settings: 12.5 Hz, 8 codebooks of 2,048 (11 bits) ≈ 1.1 kbps. For comparison, the words in speech are only 3–4 text tokens per second."))),
    ),
    section("The models",
      stepper([
        { title: "Whisper: listen", html: "<p>Audio is resampled to 16 kHz and cut into 30-second chunks. An 80-band log-mel spectrogram (25 ms windows every 10 ms, so 3,000 frames) goes through two convolutions that halve the length to 1,500, then a transformer <b>encoder</b>. A transformer <b>decoder</b> writes the transcript token by token, attending to the encoder. Trained on 680,000 hours of weakly labelled web audio (Radford et al., 2022), which is why it's robust to accents and noise.</p>" },
        { title: "Text to speech", html: "<p>Modern TTS is usually a language model over codec tokens: given text (and a few seconds of a voice prompt), predict the audio tokens, then the codec's decoder turns them back into a waveform. Some systems use diffusion or flow matching on spectrograms or latents instead.</p>" },
        { title: "Speech in, speech out", html: "<p>The cascade (speech recognition → text LLM → TTS) is simple but slow and loses tone of voice. Native speech models put audio tokens directly in the LLM's sequence, so they can hear emotion and respond in a fraction of a second. Moshi (Kyutai, 2024) models the user's and its own audio streams in parallel, so it can listen while it talks.</p>" },
      ]),
      table(["Representation", "Used by", "Good for"], [
        ["Raw waveform", "WaveNet, vocoders", "Highest fidelity; very long sequences (16,000 per second)"],
        ["Log-mel spectrogram", "Whisper, most recognisers and many TTS acoustic models", "Recognition; compact and perceptual"],
        ["Codec tokens (RVQ)", "AudioLM, VALL-E, MusicGen, Moshi", "Generation with language models"],
        ["Self-supervised features", "wav2vec 2.0, HuBERT", "Speech representations learned without transcripts"],
      ]),
    ),
    section("In code",
      code({
        "Log-mel": `import torchaudio
wav, sr = torchaudio.load("clip.wav")
wav = torchaudio.functional.resample(wav, sr, 16000)
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)(wav)
logmel = (mel + 1e-10).log10()          # (1, 80, frames): Whisper-style input`,
        "Residual VQ": `def rvq_encode(x, codebooks):            # x: (frames, dim); codebooks: list of (K, dim)
    codes, resid = [], x
    for C in codebooks:
        idx = torch.cdist(resid, C).argmin(-1)   # nearest entry per frame
        codes.append(idx)
        resid = resid - C[idx]                    # quantise what's left at the next level
    return torch.stack(codes, -1)                 # (frames, n_levels) integer tokens`,
      }),
    ),
  );
  drawSpec(); drawAlias(); drawRvq(); drawBits();
  // mel curve plus the triangular filters for 12 bands (alternate colours so neighbours are distinguishable)
  const nB = 12, edges = Array.from({ length: nB + 2 }, (_, i) => imel((mel(8000) * i) / (nB + 1)));
  const tri = (par) => edges.slice(0, nB).flatMap((e, m) => (m % 2 === par ? [[edges[m], 0], [edges[m + 1], mel(8000)], [edges[m + 2], 0]] : []));
  melC.update([
    { name: "mel(f) = 2595 · log10(1 + f / 700)", color: "var(--accent)", points: Array.from({ length: 161 }, (_, i) => [i * 50, mel(i * 50)]) },
    { name: "12 mel filters (scaled)", color: "var(--c2)", points: tri(0), dash: true },
    { name: "", color: "var(--c3)", points: tri(1), dash: true },
  ], { xDomain: [0, 8000] });
}
