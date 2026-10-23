// ltmwa browser engine, word transformer that runs on this page
"use strict";

// math utils

function linear(x, W, b, out) {
  for (let o = 0; o < W.length; o++) {
    const row = W[o];
    let s = b ? b[o] : 0;
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(x, w, b, out, eps = 1e-5) {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  let v = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mean;
    v += d * d;
  }
  const inv = 1 / Math.sqrt(v / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  return out;
}

// exact gelu: 0.5 * x * (1 + erf(x / sqrt(2))), erf via Abramowitz-Stegun 7.1.26
function gelu(x) {
  const z = x / Math.SQRT2;
  const az = Math.abs(z);
  const t = 1 / (1 + 0.3275911 * az);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-az * az);
  const erf = z < 0 ? -y : y;
  return 0.5 * x * (1 + erf);
}

function softmaxInPlace(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    a[i] = Math.exp(a[i] - m);
    s += a[i];
  }
  const inv = 1 / s;
  for (let i = 0; i < a.length; i++) a[i] *= inv;
}

// tokenizer (matches python)

function ltmwaTokenize(text) {
  text = text.toLowerCase()
    .replace(/[‘’]/g, "'").replace(/[“”]/g, '"')
    .replace(/[—–]/g, "-").replace(/…/g, "...")
    .replace(/\n/g, " <n> ");
  return text.match(/<n>|[a-z]+(?:'[a-z]+)?|[\u0900-\u097f]+|[0-9]+|[^\s\w]/gu) || [];
}

function ltmwaDetok(toks) {
  let s = toks.join(" ");
  s = s.replace(/ ([.,!?;:)\]}'"])/g, "$1");
  s = s.replace(/([(\\[]) /g, "$1");
  s = s.replace(/ n't/g, "n't").replace(/ 's/g, "'s")
       .replace(/ 're/g, "'re").replace(/ 've/g, "'ve")
       .replace(/ 'll/g, "'ll").replace(/ 'd/g, "'d");
  s = s.replace(/ <n> /g, "\n").replace(/ <n>/g, "\n").replace(/<n> /g, "\n");
  return s.trim();
}

// word transformer

class WordTransformer {
  constructor(meta) {
    const cfg = meta.config;
    this.D = cfg.d_model;
    this.L = cfg.n_layer;
    this.H = cfg.n_head;
    this.CTX = cfg.ctx;
    this.V = cfg.vocab_size;
    this.hd = this.D / this.H;
    this.vocab = meta.vocab;
    this.w2i = new Map();
    for (let i = 0; i < this.vocab.length; i++) this.w2i.set(this.vocab[i], i);
    this.NL = 2; // <n>
    this.UNK = 1;
    this.nParams = cfg.n_params;
    this.tensors = meta.tensors;
    this._bin = null;

    const D = this.D;
    this._x = new Float32Array(D);
    this._a = new Float32Array(D);
    this._b = new Float32Array(D);
    this._qkv = new Float32Array(3 * D);
    this._mlp = new Float32Array(4 * D);
    this._scores = new Float32Array(this.CTX);
    this._logits = new Float32Array(this.V);
    this.kCache = [];
    this.vCache = [];
    for (let l = 0; l < this.L; l++) {
      this.kCache.push(new Float32Array(this.H * this.CTX * this.hd));
      this.vCache.push(new Float32Array(this.H * this.CTX * this.hd));
    }
    // filled by loadWeights()
    this.tokEmb = null;   // [V][D] rows
    this.wpe = null;      // [CTX][D] rows
    this.blocks = null;
    this.lnF = null;
    this.headB = null;
  }

  async loadWeights(binUrl) {
    const buf = await (await fetch(binUrl)).arrayBuffer();
    const dv = new DataView(buf);
    const raw = {};
    for (const t of this.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (t.dtype === "i8") {
        const arr = new Float32Array(n);
        const s = t.scale;
        for (let i = 0; i < n; i++) arr[i] = dv.getInt8(t.offset + i) * s;
        raw[t.name] = { data: arr, shape: t.shape };
      } else {
        raw[t.name] = { data: new Float32Array(buf, t.offset, n).slice(), shape: t.shape };
      }
    }
    const rows = (t) => {
      const [ro, co] = t.shape;
      const out = new Array(ro);
      for (let r = 0; r < ro; r++) out[r] = t.data.subarray(r * co, (r + 1) * co);
      return out;
    };
    this.tokEmb = rows(raw["tok_emb"]);
    this.wpe = rows(raw["wpe"]);
    this.blocks = [];
    for (let l = 0; l < this.L; l++) {
      const g = (n) => raw[`b${l}.${n}`];
      this.blocks.push({
        ln1w: g("ln1_w").data, ln1b: g("ln1_b").data,
        qkvW: rows(g("qkv_w")), qkvB: g("qkv_b").data,
        projW: rows(g("proj_w")), projB: g("proj_b").data,
        ln2w: g("ln2_w").data, ln2b: g("ln2_b").data,
        fcW: rows(g("mlp_fc_w")), fcB: g("mlp_fc_b").data,
        outW: rows(g("mlp_out_w")), outB: g("mlp_out_b").data,
      });
    }
    this.lnF = { w: raw["ln_f_w"].data, b: raw["ln_f_b"].data };
    this.headB = raw["head_b"].data;
  }

  resetCache() {
    for (let l = 0; l < this.L; l++) {
      this.kCache[l].fill(0);
      this.vCache[l].fill(0);
    }
  }

  encode(text) {
    const ids = [];
    for (const t of ltmwaTokenize(text)) {
      const i = this.w2i.get(t);
      ids.push(i === undefined ? this.UNK : i);
    }
    return ids;
  }

  step(tokenId, pos) {
    const { D, H, hd } = this;
    const x = this._x, a = this._a, b = this._b;
    const emb = this.tokEmb[tokenId], pe = this.wpe[pos];
    for (let i = 0; i < D; i++) x[i] = emb[i] + pe[i];

    for (let l = 0; l < this.L; l++) {
      const B = this.blocks[l];
      layerNorm(x, B.ln1w, B.ln1b, a);
      linear(a, B.qkvW, B.qkvB, this._qkv);
      const qkv = this._qkv;
      const kC = this.kCache[l], vC = this.vCache[l];
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < H; h++) {
        const ho = h * hd;
        const kcBase = (h * this.CTX + pos) * hd;
        for (let d = 0; d < hd; d++) {
          kC[kcBase + d] = qkv[D + ho + d];
          vC[kcBase + d] = qkv[2 * D + ho + d];
        }
        const scores = this._scores;
        for (let p = 0; p <= pos; p++) {
          const kb = (h * this.CTX + p) * hd;
          let s = 0;
          for (let d = 0; d < hd; d++) s += qkv[ho + d] * kC[kb + d];
          scores[p] = s * scale;
        }
        const sub = scores.subarray(0, pos + 1);
        softmaxInPlace(sub);
        for (let d = 0; d < hd; d++) {
          let s = 0;
          for (let p = 0; p <= pos; p++) s += sub[p] * vC[(h * this.CTX + p) * hd + d];
          b[ho + d] = s;
        }
      }
      linear(b, B.projW, B.projB, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
      layerNorm(x, B.ln2w, B.ln2b, a);
      linear(a, B.fcW, B.fcB, this._mlp);
      const m = this._mlp;
      for (let i = 0; i < m.length; i++) m[i] = gelu(m[i]);
      linear(m, B.outW, B.outB, a);
      for (let i = 0; i < D; i++) x[i] += a[i];
    }
    return x;
  }

  logitsFor(hidden) {
    layerNorm(hidden, this.lnF.w, this.lnF.b, this._a);
    const logits = this._logits;
    const a = this._a, D = this.D;
    for (let o = 0; o < this.V; o++) {
      const row = this.tokEmb[o];
      let s = this.headB[o];
      for (let i = 0; i < D; i++) s += row[i] * a[i];
      logits[o] = s;
    }
    return logits;
  }

  generate(prompt, maxNew = 60, temp = 0.8, topK = 50) {
    this.resetCache();
    let ids = this.encode(prompt);
    if (ids.length > this.CTX) ids = ids.slice(ids.length - this.CTX);
    let hidden = null, pos = 0;
    for (const t of ids) {
      hidden = this.step(t, pos).slice();
      pos++;
    }
    const out = [];
    let unkRun = 0;
    for (let n = 0; n < maxNew; n++) {
      const logits = this.logitsFor(hidden);
      const scaled = Array.from(logits, (v) => v / temp);
      if (topK && topK < scaled.length) {
        const thr = scaled.slice().sort((p, q) => q - p)[topK - 1];
        for (let i = 0; i < scaled.length; i++)
          if (scaled[i] < thr) scaled[i] = -1e9;
      }
      const probs = Float32Array.from(scaled);
      softmaxInPlace(probs);
      let r = Math.random(), acc = 0, next = 0;
      for (let i = 0; i < probs.length; i++) {
        acc += probs[i];
        if (r <= acc) { next = i; break; }
      }
      if (next === this.NL || next === 0) break;
      if (next === this.UNK) { if (++unkRun > 3) break; }
      else unkRun = 0;
      out.push(this.vocab[next]);
      hidden = this.step(next, pos).slice();
      pos++;
      if (pos >= this.CTX) break;
    }
    return ltmwaDetok(out);
  }
}
