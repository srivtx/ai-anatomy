// The Studio's 3D engine: our voxel stickers, drawn live with WebGL2.
// Every block is a box with a light top, a lit screen-left side and a shaded screen-right side, like the
// stickers. The scene renders small (big pixels), then a second pass draws the black outline around every
// object and the accent outline around the selected one. You can orbit, pan, zoom and click to pick.
// No dependencies.

const hex = (c) => { const n = parseInt(c.slice(1), 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; };
/** Lighten a hex colour toward white (the sticker top face uses 0.18). */
export const lift = (c, f = 0.18) => { const [r, g, b] = hex(c); const m = (v) => Math.round((v + (1 - v) * f) * 255).toString(16).padStart(2, "0"); return `#${m(r)}${m(g)}${m(b)}`; };
/** Darken a hex colour (f < 1). */
export const dim = (c, f) => { const [r, g, b] = hex(c); const m = (v) => Math.round(v * f * 255).toString(16).padStart(2, "0"); return `#${m(r)}${m(g)}${m(b)}`; };

const reduceMotion = () => document.documentElement.dataset.motion === "reduce" || (matchMedia("(prefers-reduced-motion: reduce)").matches && document.documentElement.dataset.motion !== "full");

// ---------------------------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------------------------
const STRIDE = 13; // pos3 size3 col3 top3 id1

/** Collects boxes and the objects they belong to. Object 0 is "nothing". */
export class Build {
  constructor() { this.f = []; this.meta = [null]; this.bb = [null]; }
  /** Register an object (something you can hover, click and maybe open). Returns its id. */
  obj(meta) { this.meta.push(meta); this.bb.push(null); return this.meta.length - 1; }
  /** One solid box. */
  box(x, y, z, w, d, h, col, id = 0, top) {
    const c = hex(col), t = hex(top || lift(col));
    this.f.push(x, y, z, w, d, h, ...c, ...t, id);
    if (id) { const b = this.bb[id]; if (!b) this.bb[id] = [x, y, z, x + w, y + d, z + h]; else { b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y); b[2] = Math.min(b[2], z); b[3] = Math.max(b[3], x + w); b[4] = Math.max(b[4], y + d); b[5] = Math.max(b[5], z + h); } }
    return this;
  }
  /** A slab of unit cells (cell size s, height h) with checkered tops; top(i, j) may return a colour. */
  cells(x, y, z, nx, ny, h, col, id = 0, { s = 1, top, color } = {}) {
    const a = lift(col, 0.3), b = lift(col, 0.16);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const c = color ? color(i, j) || col : col;
      const t = (top && top(i, j)) || (c === col ? ((i + j) % 2 ? a : b) : lift(c, (i + j) % 2 ? 0.3 : 0.16));
      this.box(x + i * s, y + j * s, z, s, s, h, c, id, t);
    }
    return this;
  }
  bounds(id) {
    if (id) return this.bb[id];
    const f = this.f, b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < f.length; i += STRIDE) {
      b[0] = Math.min(b[0], f[i]); b[1] = Math.min(b[1], f[i + 1]); b[2] = Math.min(b[2], f[i + 2]);
      b[3] = Math.max(b[3], f[i] + f[i + 3]); b[4] = Math.max(b[4], f[i + 1] + f[i + 4]); b[5] = Math.max(b[5], f[i + 2] + f[i + 5]);
    }
    return b;
  }
}

// ---------------------------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------------------------
const VS = `#version 300 es
layout(location=0) in vec3 a_c; layout(location=1) in float a_f;
layout(location=2) in vec3 i_p; layout(location=3) in vec3 i_s; layout(location=4) in vec3 i_col; layout(location=5) in vec3 i_top; layout(location=6) in float i_id;
uniform mat4 u_vp; uniform mat3 u_v;
out vec3 v_c; flat out vec4 v_id;
const vec3 N[6] = vec3[6](vec3(0,0,1), vec3(0,0,-1), vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0));
void main() {
  int f = int(a_f + .5);
  gl_Position = u_vp * vec4(i_p + a_c * i_s, 1.);
  vec3 nv = u_v * N[f];
  // like the stickers: light top, the side facing screen-left in the base colour, screen-right shaded
  v_c = f == 0 ? i_top : f == 1 ? i_col * .5 : (nv.x < 0. ? i_col : i_col * .72);
  uint u = uint(i_id + .5);
  v_id = vec4(float(u & 255u), float((u >> 8) & 255u), float((u >> 16) & 255u), 255.) / 255.;
}`;
const FS = `#version 300 es
precision highp float;
in vec3 v_c; flat in vec4 v_id;
layout(location=0) out vec4 o_c; layout(location=1) out vec4 o_id;
void main() { o_c = vec4(v_c, 1.); o_id = v_id; }`;
const VS2 = `#version 300 es
void main() { vec2 p = vec2(gl_VertexID == 1 ? 3. : -1., gl_VertexID == 2 ? 3. : -1.); gl_Position = vec4(p, 0., 1.); }`;
const FS2 = `#version 300 es
precision highp float;
uniform sampler2D u_c, u_i, u_d; uniform ivec2 u_lo; uniform int u_S;
uniform vec4 u_hover, u_sel; uniform vec3 u_line, u_selc;
out vec4 o;
bool same(vec4 a, vec4 b) { return all(lessThan(abs(a - b), vec4(.002))); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) / u_S;
  vec4 id = texelFetch(u_i, p, 0); float d = texelFetch(u_d, p, 0).r;
  bool line = false, sel = false;
  for (int k = 0; k < 8; k++) {
    ivec2 off = k == 0 ? ivec2(1,0) : k == 1 ? ivec2(-1,0) : k == 2 ? ivec2(0,1) : k == 3 ? ivec2(0,-1) : k == 4 ? ivec2(2,0) : k == 5 ? ivec2(-2,0) : k == 6 ? ivec2(0,2) : ivec2(0,-2);
    ivec2 q = clamp(p + off, ivec2(0), u_lo - 1);
    vec4 n = texelFetch(u_i, q, 0);
    if (n.a < .5 || same(n, id)) continue;
    bool isSel = u_sel.a > .5 && same(n, u_sel);
    if (k >= 4 && !isSel) continue;           // the second ring is only for the selection
    if (id.a < .5 || texelFetch(u_d, q, 0).r < d) { line = true; if (isSel) sel = true; }
  }
  if (sel) { o = vec4(u_selc, 1.); return; }
  if (line) { o = vec4(u_line, 1.); return; }
  if (id.a < .5) { o = vec4(0.); return; }
  vec3 c = texelFetch(u_c, p, 0).rgb;
  if (u_hover.a > .5 && same(id, u_hover)) c = mix(c, vec3(1.), .25);
  o = vec4(c, 1.);
}`;

function prog(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [t, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function cube() {
  const F = [
    [0, [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]], [1, [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]],
    [2, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]], [3, [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]],
    [4, [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]], [5, [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]],
  ];
  const out = [];
  for (const [f, q] of F) for (const k of [0, 1, 2, 0, 2, 3]) out.push(...q[k], f);
  return new Float32Array(out);
}

// small vector helpers
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------
export const ISO = { yaw: Math.PI / 4, pitch: Math.PI / 6 }; // 2:1 isometric, the sticker angle

export class Engine {
  /** on: { hover(id, x, y), click(id), open(id), frame() } */
  constructor(canvas, on = {}) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error("no webgl2");
    this.gl = gl; this.cv = canvas; this.on = on;
    this.cam = { t: [0, 0, 0], yaw: ISO.yaw, pitch: ISO.pitch, zoom: 10 };
    this.px = 2; this.hover = 0; this.sel = 0; this.n = 0; this.meta = [null]; this.build = null;
    this.line = [0.078, 0.071, 0.094]; this.selc = [0.36, 0.27, 0.84];
    this.p1 = prog(gl, VS, FS); this.p2 = prog(gl, VS2, FS2);
    this.u1 = { vp: gl.getUniformLocation(this.p1, "u_vp"), v: gl.getUniformLocation(this.p1, "u_v") };
    this.u2 = Object.fromEntries(["u_c", "u_i", "u_d", "u_lo", "u_S", "u_hover", "u_sel", "u_line", "u_selc"].map((k) => [k, gl.getUniformLocation(this.p2, k)]));
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, cube(), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 12);
    this.ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.ib);
    const L = [[2, 3, 0], [3, 3, 3], [4, 3, 6], [5, 3, 9], [6, 1, 12]];
    for (const [loc, n, off] of L) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, STRIDE * 4, off * 4); gl.vertexAttribDivisor(loc, 1); }
    gl.bindVertexArray(null);
    this.vao2 = gl.createVertexArray();
    this.fb = null; this.lo = [0, 0];
    this.ok = (id) => (this.meta[id] && !this.meta[id].decor ? id : 0);
    this.input();
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas);
    this.resize();
  }

  /** Pixel size in CSS pixels: 2 = chunky sticker pixels, 1 = fine. */
  setPixel(px) { this.px = px; this.resize(); }
  setTheme({ line, sel }) { if (line) this.line = hex(line); if (sel) this.selc = hex(sel); this.draw(); }

  setScene(b) {
    const gl = this.gl;
    this.build = b; this.meta = b.meta; this.n = b.f.length / STRIDE;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ib); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(b.f), gl.STATIC_DRAW);
    this.hover = 0; this.sel = 0; this.draw();
  }

  resize() {
    const gl = this.gl, r = this.cv.getBoundingClientRect(), dpr = Math.min(3, window.devicePixelRatio || 1);
    if (!r.width || !r.height) return;
    const S = Math.max(1, Math.round(this.px * dpr));
    const lo = [Math.ceil((r.width * dpr) / S), Math.ceil((r.height * dpr) / S)];
    this.S = S; this.cssW = r.width; this.cssH = r.height;
    this.cv.width = lo[0] * S; this.cv.height = lo[1] * S;
    if (lo[0] !== this.lo[0] || lo[1] !== this.lo[1] || !this.fb) {
      this.lo = lo;
      for (const t of this.tex || []) gl.deleteTexture(t);
      if (this.fb) gl.deleteFramebuffer(this.fb);
      const mk = (ifmt, fmt, type) => { const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, lo[0], lo[1], 0, fmt, type, null); for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST); for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE); return t; };
      this.tex = [mk(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), mk(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), mk(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT)];
      this.fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex[0], 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.tex[1], 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.tex[2], 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    this.renderNow();
  }

  /** The camera's basis: right, up, forward (into the screen), and the half-extents of the view. */
  basis(c = this.cam) {
    const cp = Math.cos(c.pitch), dir = [cp * Math.cos(c.yaw), cp * Math.sin(c.yaw), Math.sin(c.pitch)];
    const f = norm([-dir[0], -dir[1], -dir[2]]), r = norm(cross(f, [0, 0, 1])), u = cross(r, f);
    const hh = c.zoom, hw = c.zoom * (this.cssW / this.cssH || 1);
    return { r, u, f, dir, hh, hw };
  }
  matrices() {
    const c = this.cam, { r, u, f, dir, hh, hw } = this.basis(), D = 1500;
    const e = [c.t[0] + dir[0] * D, c.t[1] + dir[1] * D, c.t[2] + dir[2] * D];
    const V = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, e), -dot(u, e), dot(f, e), 1];
    const n = 1, fa = 3000;
    const P = [1 / hw, 0, 0, 0, 0, 1 / hh, 0, 0, 0, 0, -2 / (fa - n), 0, 0, 0, -(fa + n) / (fa - n), 1];
    const VP = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += P[k * 4 + j] * V[i * 4 + k]; VP[i * 4 + j] = s; }
    return { VP, R: new Float32Array([r[0], u[0], -f[0], r[1], u[1], -f[1], r[2], u[2], -f[2]]) };
  }

  /** Screen position (CSS px) of a world point. */
  project(p) {
    const { r, u, hh, hw } = this.basis(), d = sub(p, this.cam.t);
    return [(dot(r, d) / hw * 0.5 + 0.5) * this.cssW, (0.5 - dot(u, d) / hh * 0.5) * this.cssH];
  }

  draw() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.renderNow(); }); }

  renderNow() {
    const gl = this.gl;
    if (!this.fb || !this.cssW) return;
    if (this.tw) this.stepTween();
    const { VP, R } = this.matrices();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.viewport(0, 0, this.lo[0], this.lo[1]);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]); gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]); gl.clearBufferfi(gl.DEPTH_STENCIL, 0, 1, 0);
    gl.enable(gl.DEPTH_TEST);
    if (this.n) {
      gl.useProgram(this.p1); gl.uniformMatrix4fv(this.u1.vp, false, VP); gl.uniformMatrix3fv(this.u1.v, false, R);
      gl.bindVertexArray(this.vao); gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.n); gl.bindVertexArray(null);
    }
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.cv.width, this.cv.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.p2);
    this.tex.forEach((t, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); });
    const U = this.u2, idv = (id) => [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255, id ? 1 : 0];
    gl.uniform1i(U.u_c, 0); gl.uniform1i(U.u_i, 1); gl.uniform1i(U.u_d, 2);
    gl.uniform2i(U.u_lo, this.lo[0], this.lo[1]); gl.uniform1i(U.u_S, this.S);
    gl.uniform4fv(U.u_hover, idv(this.hover)); gl.uniform4fv(U.u_sel, idv(this.sel));
    gl.uniform3fv(U.u_line, this.line); gl.uniform3fv(U.u_selc, this.selc);
    gl.bindVertexArray(this.vao2); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
    this.on.frame?.();
    if (this.tw) this.draw();
  }

  /** The object id under a CSS-pixel point (0 = nothing). */
  pick(x, y) {
    const gl = this.gl; if (!this.fb) return 0;
    const lx = Math.floor((x / this.cssW) * this.lo[0]), ly = this.lo[1] - 1 - Math.floor((y / this.cssH) * this.lo[1]);
    if (lx < 0 || ly < 0 || lx >= this.lo[0] || ly >= this.lo[1]) return 0;
    const px = new Uint8Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb); gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(lx, ly, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return px[3] ? px[0] | (px[1] << 8) | (px[2] << 16) : 0;
  }

  /** The whole object-id buffer (low-res, bottom row first): which object is visible at every pixel. */
  ids() {
    const gl = this.gl, [w, h] = this.lo, data = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb); gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { w, h, data };
  }

  // ----- camera -----
  /** A camera that frames a box [x0,y0,z0,x1,y1,z1] at the current angle. */
  framing(b, { yaw = this.cam.yaw, pitch = this.cam.pitch, margin = 1.18 } = {}) {
    const t = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
    const c = { t, yaw, pitch, zoom: 1 }, { r, u } = this.basis(c), asp = this.cssW / this.cssH || 1;
    let ex = 0, ey = 0;
    for (const x of [b[0], b[3]]) for (const y of [b[1], b[4]]) for (const z of [b[2], b[5]]) { const d = sub([x, y, z], t); ex = Math.max(ex, Math.abs(dot(r, d))); ey = Math.max(ey, Math.abs(dot(u, d))); }
    c.zoom = Math.max(ey, ex / asp, 2) * margin;
    return c;
  }
  /** Move the camera to `to` over ms (instantly with reduced motion). */
  fly(to, ms = 420) {
    const from = { ...this.cam, t: [...this.cam.t] };
    if (ms <= 0 || reduceMotion()) { this.tw = null; this.cam = { ...from, ...to, t: [...(to.t || from.t)] }; this.draw(); return Promise.resolve(); }
    let dy = (to.yaw ?? from.yaw) - from.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    return new Promise((done) => { this.tw = { from, to: { ...from, ...to, yaw: from.yaw + dy }, t0: performance.now(), ms, done }; this.draw(); });
  }
  stepTween() {
    const w = this.tw, k = Math.min(1, (performance.now() - w.t0) / w.ms), e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    const L = (a, b) => a + (b - a) * e;
    this.cam = { t: w.from.t.map((v, i) => L(v, w.to.t[i])), yaw: L(w.from.yaw, w.to.yaw), pitch: L(w.from.pitch, w.to.pitch), zoom: Math.exp(L(Math.log(w.from.zoom), Math.log(w.to.zoom))) };
    if (k >= 1) { const d = w.done; this.tw = null; d(); }
  }
  orbit(dyaw, dpitch) { this.tw = null; const c = this.cam; c.yaw += dyaw; c.pitch = Math.max(0.08, Math.min(1.45, c.pitch + dpitch)); this.draw(); }
  pan(dx, dy) { this.tw = null; const { r, u, hh } = this.basis(), k = (2 * hh) / this.cssH; this.cam.t = this.cam.t.map((v, i) => v - r[i] * dx * k + u[i] * dy * k); this.draw(); }
  zoomAt(f, x = this.cssW / 2, y = this.cssH / 2) {
    this.tw = null;
    const { r, u, hh, hw } = this.basis(), nx = (x / this.cssW) * 2 - 1, ny = 1 - (y / this.cssH) * 2;
    const z1 = Math.max(0.6, Math.min(400, this.cam.zoom * f)), s = 1 - z1 / this.cam.zoom;
    this.cam.t = this.cam.t.map((v, i) => v + (r[i] * nx * hw + u[i] * ny * hh) * s);
    this.cam.zoom = z1; this.draw();
  }

  // ----- input: drag to turn, right-drag or shift-drag to pan, wheel or pinch to zoom, click to pick -----
  input() {
    const cv = this.cv, pts = new Map();
    let moved = 0, last = null, pinch = 0;
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("pointerdown", (e) => {
      cv.setPointerCapture(e.pointerId); pts.set(e.pointerId, pos(e)); moved = 0; last = pos(e);
      if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); }
    });
    cv.addEventListener("pointermove", (e) => {
      const p = pos(e);
      if (!pts.has(e.pointerId)) {
        const id = this.pick(...p);
        if (this.ok(id) !== this.hover) { this.hover = this.ok(id); this.draw(); }
        this.on.hover?.(this.hover, p[0], p[1]);
        return;
      }
      const prev = pts.get(e.pointerId); pts.set(e.pointerId, p);
      if (pts.size === 2) {
        const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) this.zoomAt(pinch / d, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        pinch = d; this.pan((p[0] - prev[0]) / 2, (p[1] - prev[1]) / 2); moved += 99; return;
      }
      const dx = p[0] - prev[0], dy = p[1] - prev[1]; moved += Math.abs(dx) + Math.abs(dy);
      if (moved < 4) return;
      if (e.buttons & 2 || e.shiftKey) this.pan(dx, dy); else this.orbit(-dx * 0.009, dy * 0.009);
      if (this.hover) { this.hover = 0; this.on.hover?.(0); }
    });
    const end = (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.delete(e.pointerId); if (pts.size < 2) pinch = 0;
      if (moved < 4 && e.type === "pointerup" && e.button === 0) {
        const ok = this.ok(this.pick(...pos(e))), now = performance.now();
        const dbl = ok && last && this.lastClick && now - this.lastClick.t < 380 && this.lastClick.id === ok;
        this.lastClick = { t: now, id: ok };
        if (dbl) this.on.open?.(ok); else this.on.click?.(ok);
      }
    };
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
    cv.addEventListener("pointerleave", () => { if (!pts.size && this.hover) { this.hover = 0; this.draw(); this.on.hover?.(0); } });
    cv.addEventListener("wheel", (e) => { e.preventDefault(); const p = pos(e); this.zoomAt(Math.exp((e.deltaY * (e.deltaMode ? 30 : 1)) * 0.0016), p[0], p[1]); }, { passive: false });
  }
  destroy() { this.ro.disconnect(); }
}
