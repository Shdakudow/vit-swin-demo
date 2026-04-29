import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Grid3x3, Eye, Layers, Play, Pause, RotateCcw, ChevronRight,
  Box, Network, GitBranch, Move, BookOpen,
  ArrowRight, Hash, Crosshair, Workflow, Microscope, Sparkles,
  Upload, Loader2, Brain, HelpCircle, Lightbulb, Check, X
} from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/* TeX inline + block — call KaTeX directly (skipping react-katex which has
   spotty React-19 support) and inject the rendered HTML. throwOnError:false
   shows a red error inline rather than blowing up the whole page. */
function renderTeX(src, displayMode) {
  try {
    return katex.renderToString(typeof src === 'string' ? src : String(src), {
      throwOnError: false,
      displayMode,
      strict: 'ignore',
      output: 'html',
    });
  } catch (err) {
    return `<span style="color:#fca5a5;font-family:monospace">${String(err && err.message || err)}</span>`;
  }
}

const TeX = ({ children }) => (
  <span dangerouslySetInnerHTML={{ __html: renderTeX(children, false) }} />
);

const TeXBlock = ({ children }) => (
  <div className="my-1" dangerouslySetInnerHTML={{ __html: renderTeX(children, true) }} />
);

/* renderWithMath — splits a plain string on $...$ delimiters and renders the
   math segments via KaTeX. Lets us write QUESTION text / option labels /
   explanations in plain JS strings while still getting proper LaTeX rendering
   for the inline math fragments. */
function renderWithMath(text) {
  if (text == null) return null;
  if (typeof text !== 'string') return text;
  const parts = text.split(/(\$[^$]+\$)/g);
  return parts.map((part, i) => {
    if (part.length > 2 && part.startsWith('$') && part.endsWith('$')) {
      const math = part.slice(1, -1);
      return (
        <span
          key={i}
          className="inline-block align-middle"
          dangerouslySetInnerHTML={{ __html: renderTeX(math, false) }}
        />
      );
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

/* =========================================================
   Tiny linalg helpers (deterministic, seeded).
   Used to drive *real* attention-style computations so the
   visualizations reflect actual math, not hand-drawn fakes.
   ========================================================= */

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randMatrix(rows, cols, rng) {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) {
    // Box-Muller for vaguely normal weights
    const u1 = Math.max(rng(), 1e-9);
    const u2 = rng();
    m[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.35;
  }
  return { data: m, rows, cols };
}

function matmul(A, B) {
  const out = new Float32Array(A.rows * B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < A.cols; k++) {
      const a = A.data[i * A.cols + k];
      if (a === 0) continue;
      for (let j = 0; j < B.cols; j++) {
        out[i * B.cols + j] += a * B.data[k * B.cols + j];
      }
    }
  }
  return { data: out, rows: A.rows, cols: B.cols };
}

function softmaxRows(M) {
  const out = new Float32Array(M.data.length);
  for (let i = 0; i < M.rows; i++) {
    let m = -Infinity;
    for (let j = 0; j < M.cols; j++) m = Math.max(m, M.data[i * M.cols + j]);
    let s = 0;
    for (let j = 0; j < M.cols; j++) {
      const v = Math.exp(M.data[i * M.cols + j] - m);
      out[i * M.cols + j] = v;
      s += v;
    }
    for (let j = 0; j < M.cols; j++) out[i * M.cols + j] /= s;
  }
  return { data: out, rows: M.rows, cols: M.cols };
}

function transpose(A) {
  const out = new Float32Array(A.data.length);
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++)
      out[j * A.rows + i] = A.data[i * A.cols + j];
  return { data: out, rows: A.cols, cols: A.rows };
}

/* =========================================================
   Procedural test image. We draw something with a clear
   spatial structure so patch boundaries are visually obvious.
   ========================================================= */

function drawTestImage(ctx, size, variant = 0) {
  // Background: warm dusk gradient
  const bg = ctx.createLinearGradient(0, 0, 0, size);
  bg.addColorStop(0, '#1e293b');
  bg.addColorStop(0.55, '#7c3a4a');
  bg.addColorStop(1, '#f59e0b');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  // Soft "sun"
  const sunX = size * 0.7, sunY = size * 0.32, sunR = size * 0.13;
  const sun = ctx.createRadialGradient(sunX, sunY, 0, sunX, sunY, sunR);
  sun.addColorStop(0, '#fde68a');
  sun.addColorStop(0.6, '#fbbf24');
  sun.addColorStop(1, 'rgba(251,191,36,0)');
  ctx.fillStyle = sun;
  ctx.beginPath();
  ctx.arc(sunX, sunY, sunR * 1.7, 0, Math.PI * 2);
  ctx.fill();

  // Mountain silhouettes
  ctx.fillStyle = '#0f172a';
  ctx.beginPath();
  ctx.moveTo(0, size * 0.75);
  ctx.lineTo(size * 0.18, size * 0.55);
  ctx.lineTo(size * 0.32, size * 0.68);
  ctx.lineTo(size * 0.5, size * 0.45);
  ctx.lineTo(size * 0.7, size * 0.62);
  ctx.lineTo(size * 0.85, size * 0.5);
  ctx.lineTo(size, size * 0.7);
  ctx.lineTo(size, size);
  ctx.lineTo(0, size);
  ctx.closePath();
  ctx.fill();

  // Foreground tree-ish shape (left)
  ctx.fillStyle = '#020617';
  ctx.fillRect(size * 0.12, size * 0.7, size * 0.025, size * 0.25);
  ctx.beginPath();
  ctx.arc(size * 0.135, size * 0.7, size * 0.07, 0, Math.PI * 2);
  ctx.fill();

  // Teal accent shape (object of interest)
  ctx.fillStyle = '#14b8a6';
  ctx.beginPath();
  ctx.arc(size * 0.4, size * 0.82, size * 0.05, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#f43f5e';
  ctx.fillRect(size * 0.55, size * 0.78, size * 0.06, size * 0.06);

  // grain
  if (variant === 0) {
    const img = ctx.getImageData(0, 0, size, size);
    const rng = mulberry32(7);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (rng() - 0.5) * 16;
      img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
      img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
      img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n));
    }
    ctx.putImageData(img, 0, 0);
  }
}

/* Pair-demo image: deliberately laid out so the Self-Attention demo
   has obvious "answers". Three pairs of distinct shape+colour combos:
       red square  ┐    yellow tri (N)     ┌ blue circle
                   │                       │
                   │     (background)      │
                   │                       │
       blue circle ┘    yellow tri (S)     └ red square
   Each shape has a colour-matched twin elsewhere — a query patch on
   a red square should attend strongly to the OTHER red square, and
   similarly for the blue circles and yellow triangles. Designed to
   align with patchSize=48 / SIZE=240. */
function drawPairsDemoImage(ctx, size) {
  // Slightly lighter background so coloured shapes pop more.
  const bg = ctx.createLinearGradient(0, 0, 0, size);
  bg.addColorStop(0, '#0c1424');
  bg.addColorStop(1, '#050810');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  // Three pairs of distinct shape+colour combos, scattered around the
  // image so any patch a student clicks can find a "twin" elsewhere.
  // Bigger, more saturated colours than before so the patch grid shows
  // clearly different tints from cell to cell.
  const r = size * 0.13;
  const tri = (cx, cy, rr, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(cx, cy - rr);
    ctx.lineTo(cx + rr, cy + rr * 0.85);
    ctx.lineTo(cx - rr, cy + rr * 0.85);
    ctx.closePath();
    ctx.fill();
  };
  const square = (cx, cy, rr, color) => {
    ctx.fillStyle = color;
    ctx.fillRect(cx - rr, cy - rr, rr * 2, rr * 2);
  };
  const circle = (cx, cy, rr, color) => {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.fill();
  };

  // Pair A — bright red squares (NW + SE)
  square(size * 0.18, size * 0.20, r, '#ff3b3b');
  square(size * 0.82, size * 0.80, r, '#ff3b3b');

  // Pair B — saturated blue circles (NE + SW)
  circle(size * 0.82, size * 0.20, r, '#2f7bff');
  circle(size * 0.18, size * 0.80, r, '#2f7bff');

  // Pair C — yellow triangles (top-centre + bottom-centre) so the centre
  // column of patches has its own twin set.
  tri(size * 0.50, size * 0.18, r * 0.95, '#ffd23a');
  tri(size * 0.50, size * 0.82, r * 0.95, '#ffd23a');

  // tiny dark dot to anchor the very centre — breaks pure symmetry.
  ctx.fillStyle = '#94a3b8';
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, size * 0.02, 0, Math.PI * 2);
  ctx.fill();

  // light grain so each patch has a unique pixel signature even within
  // a single colour region (helps the attention math distinguish them).
  const img = ctx.getImageData(0, 0, size, size);
  const rng = mulberry32(11);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (rng() - 0.5) * 10;
    img.data[i]     = Math.max(0, Math.min(255, img.data[i]     + n));
    img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
    img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n));
  }
  ctx.putImageData(img, 0, 0);
}

/* =========================================================
   Patch features: average RGB of each patch, projected to
   a small embedding dim. We use these to drive plausible
   attention patterns later — patches that look alike attend
   to one another.
   ========================================================= */

function computePatchFeatures(canvas, patchSize, embedDim = 16, seed = 42) {
  const ctx = canvas.getContext('2d');
  const size = canvas.width;
  const grid = size / patchSize;
  const N = grid * grid;
  const raw = new Float32Array(N * 6); // [meanR, meanG, meanB, varR, varG, varB]
  const img = ctx.getImageData(0, 0, size, size).data;

  for (let py = 0; py < grid; py++) {
    for (let px = 0; px < grid; px++) {
      let sR = 0, sG = 0, sB = 0, n = 0;
      const x0 = px * patchSize, y0 = py * patchSize;
      for (let y = y0; y < y0 + patchSize; y++) {
        for (let x = x0; x < x0 + patchSize; x++) {
          const i = (y * size + x) * 4;
          sR += img[i]; sG += img[i + 1]; sB += img[i + 2];
          n++;
        }
      }
      const mR = sR / n / 255, mG = sG / n / 255, mB = sB / n / 255;
      let vR = 0, vG = 0, vB = 0;
      for (let y = y0; y < y0 + patchSize; y++) {
        for (let x = x0; x < x0 + patchSize; x++) {
          const i = (y * size + x) * 4;
          vR += (img[i] / 255 - mR) ** 2;
          vG += (img[i + 1] / 255 - mG) ** 2;
          vB += (img[i + 2] / 255 - mB) ** 2;
        }
      }
      const idx = (py * grid + px) * 6;
      raw[idx] = mR; raw[idx + 1] = mG; raw[idx + 2] = mB;
      raw[idx + 3] = vR / n; raw[idx + 4] = vG / n; raw[idx + 5] = vB / n;
    }
  }

  // Project to embedDim with a deterministic random matrix
  const rng = mulberry32(seed);
  const W = randMatrix(6, embedDim, rng);
  const X = { data: raw, rows: N, cols: 6 };
  return matmul(X, W);
}

/* =========================================================
   Reusable UI atoms
   ========================================================= */

/* Locked palette: amber = ViT / primary highlight, teal = Swin / cool data,
   rose = [CLS] / specialness, slate = neutral. Removed violet. */
const Tag = ({ children, color = 'amber' }) => {
  const palette = {
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    teal: 'bg-teal-500/10 text-teal-300 border-teal-500/30',
    rose: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    slate: 'bg-slate-500/10 text-slate-300 border-slate-500/30',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-mono tracking-wide ${palette[color] || palette.amber}`}>
      {children}
    </span>
  );
};

/* Eq — inline LaTeX equation, rendered by calling KaTeX directly and
   injecting the resulting HTML. Wrapped in an amber-tinted pill so it
   stands out from prose. */
const Eq = ({ children }) => {
  const src = typeof children === 'string'
    ? children
    : Array.isArray(children) ? children.join('') : String(children);
  return (
    <span
      className="inline-block align-middle bg-amber-500/5 border border-amber-500/20 rounded px-1.5 py-0.5 text-amber-200"
      dangerouslySetInnerHTML={{ __html: renderTeX(src, false) }}
    />
  );
};

const Section = ({ icon: Icon, title, kicker, children }) => (
  <section className="mb-10">
    {kicker && (
      <div className="font-mono text-[11px] tracking-[0.2em] text-amber-400/80 uppercase mb-1">
        {kicker}
      </div>
    )}
    <div className="flex items-center gap-2.5 mb-4">
      {Icon && (
        <div className="w-8 h-8 rounded-md bg-amber-500/12 border border-amber-500/30 flex items-center justify-center shrink-0">
          <Icon size={16} className="text-amber-300" />
        </div>
      )}
      <h2 className="text-2xl font-serif text-slate-100 tracking-tight">{title}</h2>
    </div>
    <div className="text-slate-300/90 leading-relaxed">{children}</div>
  </section>
);

const Card = ({ children, className = '' }) => (
  <div className={`bg-slate-900/60 border border-slate-700/60 rounded-lg ${className}`}>
    {children}
  </div>
);

/* Takeaway — small end-of-tab summary box. Three slots:
   "What you should now understand" (mandatory),
   "Common confusion" (optional), and
   "Try in the Live Demo" (optional, action-y). Standardises every
   tab's wrap-up so reading the whole site has a real cadence. */
function Takeaway({ understand = [], confusion = [], tryIt = '' }) {
  return (
    <Card className="p-5 mt-8">
      <div className="grid sm:grid-cols-3 gap-5">
        <div>
          <div className="text-[10px] font-mono tracking-[0.2em] text-amber-400/80 uppercase mb-2">
            What you should now understand
          </div>
          <ul className="space-y-1.5 text-[13px] text-slate-200 leading-snug">
            {understand.map((u, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-amber-400/70 shrink-0">·</span>
                <span>{u}</span>
              </li>
            ))}
          </ul>
        </div>
        {confusion.length > 0 && (
          <div>
            <div className="text-[10px] font-mono tracking-[0.2em] text-rose-400/80 uppercase mb-2">
              Common confusion
            </div>
            <ul className="space-y-1.5 text-[13px] text-slate-300 leading-snug">
              {confusion.map((c, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-rose-400/60 shrink-0">!</span>
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {tryIt && (
          <div>
            <div className="text-[10px] font-mono tracking-[0.2em] text-teal-400/80 uppercase mb-2">
              Try in the Live Demo
            </div>
            <p className="text-[13px] text-slate-300 leading-snug">{tryIt}</p>
          </div>
        )}
      </div>
    </Card>
  );
}

const Slider = ({ label, value, min, max, step = 1, onChange, suffix, options }) => (
  <div>
    <div className="flex items-baseline justify-between mb-1.5">
      <label className="text-[12px] font-mono text-slate-400 uppercase tracking-wider">{label}</label>
      <span className="font-mono text-amber-300 text-sm">{value}{suffix}</span>
    </div>
    {options ? (
      <div className="flex gap-1">
        {options.map(opt => (
          <button
            key={opt}
            onClick={() => onChange(opt)}
            className={`flex-1 px-2 py-1.5 rounded text-xs font-mono border transition-all
              ${value === opt
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
          >
            {opt}
          </button>
        ))}
      </div>
    ) : (
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1 accent-amber-400 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />
    )}
  </div>
);

const Toggle = ({ label, value, onChange }) => (
  <button
    onClick={() => onChange(!value)}
    className={`flex items-center justify-between gap-3 w-full px-3 py-2 rounded border transition-all
      ${value
        ? 'bg-amber-500/10 border-amber-500/40 text-amber-200'
        : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600'}`}
  >
    <span className="text-[12px] font-mono uppercase tracking-wider">{label}</span>
    <div className={`w-8 h-4 rounded-full relative transition-all ${value ? 'bg-amber-500/60' : 'bg-slate-700'}`}>
      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${value ? 'left-4' : 'left-0.5'}`} />
    </div>
  </button>
);

/* =========================================================
   TAB 1 — Overview
   ========================================================= */

function OverviewCostDemo() {
  const [side, setSide] = useState(768);
  const P = 16, M = 7;
  const N = Math.round((side / P) ** 2);
  const vitCost = N * N;
  const swinCost = N * M * M;
  const max = Math.max(vitCost, swinCost);
  const ratio = vitCost / Math.max(1, swinCost);
  const fmt = (v) => v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)} K` : `${v}`;

  return (
    <Card className="p-5">
      <div className="flex items-baseline justify-between gap-2 mb-4 flex-wrap">
        <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/80">
          Try this · why locality matters
        </div>
        <span className="text-[10px] font-mono text-slate-500">push the slider →</span>
      </div>
      <div className="grid md:grid-cols-[200px_1fr] gap-5 items-center">
        <div>
          <Slider
            label="Image side (px)"
            value={side}
            options={[224, 384, 512, 768, 1024, 1536]}
            onChange={setSide}
          />
          <div className="mt-3 text-[12px] font-mono leading-relaxed">
            <div className="text-slate-400">P = 16 px · M = 7</div>
            <div className="text-amber-300 mt-1">N = {N.toLocaleString()} patches</div>
          </div>
        </div>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-[11px] font-mono mb-1">
              <span className="text-amber-300">ViT · O(N²) attention pairs</span>
              <span className="text-amber-200">{fmt(vitCost)}</span>
            </div>
            <div className="h-3 bg-slate-800 rounded overflow-hidden">
              <div className="h-full bg-amber-400 transition-all duration-300" style={{ width: `${(vitCost / max) * 100}%` }}/>
            </div>
          </div>
          <div>
            <div className="flex justify-between text-[11px] font-mono mb-1">
              <span className="text-teal-300">Swin · O(M²·N) attention pairs</span>
              <span className="text-teal-200">{fmt(swinCost)}</span>
            </div>
            <div className="h-3 bg-slate-800 rounded overflow-hidden">
              <div className="h-full bg-teal-400 transition-all duration-300" style={{ width: `${(swinCost / max) * 100}%` }}/>
            </div>
          </div>
          <div className="pt-2 mt-1 border-t border-slate-800 text-[12px] text-slate-400 leading-snug">
            ViT does <span className="text-amber-300 font-medium">{ratio < 10 ? ratio.toFixed(1) : Math.round(ratio).toLocaleString()}×</span> the
            pairwise attention work of Swin at this resolution. Push to 1024+ and the gap is what kills ViT for high-res tasks.
          </div>
        </div>
      </div>
    </Card>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-8">
      <Section icon={BookOpen} kicker="01 — Setting the stage" title="Two ways to look at an image">
        <p className="mb-4 max-w-3xl">
          Convolutional networks were the unchallenged champions of computer vision for almost a decade.
          Then in 2020, <span className="text-amber-300 font-medium">Vision Transformer</span> (ViT) showed
          that a model originally built for language — pure self-attention, no convolutions — could match
          ConvNets at scale. A year later, <span className="text-amber-300 font-medium">Swin Transformer</span> reintroduced
          the inductive bias ConvNets had quietly contributed all along: locality and hierarchy.
        </p>
        <p className="max-w-3xl">
          This demo walks through both architectures one mechanism at a time. Every visualization runs real
          computations on a small synthetic image — patch embedding, attention, window partitioning, shifts,
          and merges — so the math stays close to the picture.
        </p>
      </Section>

      <div className="grid md:grid-cols-2 gap-5">
        <Card className="p-5">
          <div>
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              <Tag color="amber">ViT · 2020</Tag>
              <Tag color="slate">Dosovitskiy et al.</Tag>
              <a
                href="https://arxiv.org/abs/2010.11929"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-amber-300 hover:text-amber-200 underline decoration-dotted underline-offset-2"
              >
                arXiv:2010.11929 ↗
              </a>
            </div>
            <h3 className="font-serif text-xl text-slate-100 mb-3">Vision Transformer</h3>
            <p className="text-slate-300 text-sm mb-4 leading-relaxed">
              An image is just a sequence of patches. Embed each patch linearly, prepend a [CLS] token,
              add position embeddings, and feed the whole thing through a stack of standard Transformer encoders.
              <span className="text-amber-300"> Every patch attends to every other patch in every layer.</span>
            </p>
            <div className="space-y-2 text-[13px] font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Tokens</span><span className="text-slate-200">{`(H·W / P²) + 1`}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Attention</span><span className="text-slate-200">global</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Complexity</span><span className="text-rose-300">O(N²·d)</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Resolution</span><span className="text-slate-200">single, fixed</span>
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-5">
          <div>
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              <Tag color="teal">Swin · 2021</Tag>
              <Tag color="slate">Liu et al.</Tag>
              <a
                href="https://arxiv.org/abs/2103.14030"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[11px] font-mono text-teal-300 hover:text-teal-200 underline decoration-dotted underline-offset-2"
              >
                arXiv:2103.14030 ↗
              </a>
            </div>
            <h3 className="font-serif text-xl text-slate-100 mb-3">Swin Transformer</h3>
            <p className="text-slate-300 text-sm mb-4 leading-relaxed">
              Restrict attention to non-overlapping local windows of size <Eq>{String.raw`M \times M`}</Eq>, then alternate
              with <span className="text-teal-300">shifted</span> windows so information crosses window
              boundaries. Periodically merge patches to build a feature pyramid like a ConvNet.
            </p>
            <div className="space-y-2 text-[13px] font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Tokens</span><span className="text-slate-200">windows of M²</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Attention</span><span className="text-slate-200">local + shifted</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Complexity</span><span className="text-teal-300">O(M²·N·d)</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Resolution</span><span className="text-slate-200">hierarchical</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Card className="p-5">
        <h3 className="font-serif text-lg text-slate-100 mb-4 flex items-center gap-2">
          <Workflow size={18} className="text-amber-300" /> What you'll explore
        </h3>
        <div className="grid md:grid-cols-2 gap-x-8 gap-y-3 text-sm">
          {[
            ['02', 'Patch embedding', 'How an image becomes a sequence'],
            ['03', 'Position + [CLS]', 'Restoring spatial order'],
            ['04', 'Self-attention', 'Q, K, V from scratch'],
            ['05', 'Multi-head', 'Why one head is never enough'],
            ['06', 'ViT pipeline', 'End-to-end forward pass'],
            ['07', 'Window attention', "Swin's locality trick"],
            ['08', 'Shifted windows', 'The key idea, visualized'],
            ['09', 'Hierarchy', 'Patch merging across stages'],
          ].map(([n, title, desc]) => (
            <div key={n} className="flex gap-3 group">
              <span className="font-mono text-[11px] text-amber-400/60 mt-0.5">{n}</span>
              <div>
                <div className="text-slate-100 font-medium">{title}</div>
                <div className="text-slate-400 text-xs">{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Live "why locality matters" cost slider — primes the rest of the
          tabs by making the quadratic-vs-linear gap visceral up front. */}
      <OverviewCostDemo />

      <Takeaway
        understand={[
          'ViT and Swin are two answers to the same question: how do you build an image model out of pure attention?',
          'ViT keeps it simple — patches in, global attention, [CLS] out. Swin adds locality (windows) and a CNN-style feature pyramid.',
        ]}
        confusion={['"Newer" ≠ "better". The two papers optimize different things; choice depends on the task.']}
        tryIt="Open the Live Demo and watch how each model attends to the same cat image — ViT's pattern is global, Swin's is windowed."
      />
    </div>
  );
}

/* =========================================================
   TAB 2 — Patch Embedding
   ========================================================= */

function PatchTab() {
  const [patchSize, setPatchSize] = useState(32);
  const [showGrid, setShowGrid] = useState(true);
  const [showNumbers, setShowNumbers] = useState(true);
  const [hoveredPatch, setHoveredPatch] = useState(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const SIZE = 256;
  const D_DEMO = 12; // shown embedding dim (real ViT-Base is 768; 12 stays readable)

  // Deterministic projection matrix E ∈ ℝ^(P²·3 × D), rebuilt only when patchSize changes.
  const projE = useMemo(() => {
    const inDim = patchSize * patchSize * 3;
    const rng = mulberry32(7919);
    return randMatrix(inDim, D_DEMO, rng);
  }, [patchSize]);

  // Read the hovered patch's pixels off the canvas and project to D dims.
  const hoveredVector = useMemo(() => {
    if (hoveredPatch === null || !canvasRef.current) return null;
    const ctx = canvasRef.current.getContext('2d');
    const grid = SIZE / patchSize;
    const py = Math.floor(hoveredPatch / grid);
    const px = hoveredPatch % grid;
    let data;
    try {
      data = ctx.getImageData(px * patchSize, py * patchSize, patchSize, patchSize).data;
    } catch {
      return null;
    }
    const flat = new Float32Array(patchSize * patchSize * 3);
    for (let i = 0; i < patchSize * patchSize; i++) {
      flat[i * 3]     = data[i * 4]     / 255;
      flat[i * 3 + 1] = data[i * 4 + 1] / 255;
      flat[i * 3 + 2] = data[i * 4 + 2] / 255;
    }
    const out = new Float32Array(D_DEMO);
    for (let d = 0; d < D_DEMO; d++) {
      let s = 0;
      for (let k = 0; k < flat.length; k++) s += flat[k] * projE.data[k * D_DEMO + d];
      out[d] = s;
    }
    return { flat, embedding: out };
  }, [hoveredPatch, patchSize, projE]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = SIZE; canvas.height = SIZE;
    drawTestImage(canvas.getContext('2d'), SIZE);
  }, []);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = SIZE; canvas.height = SIZE;
    ctx.clearRect(0, 0, SIZE, SIZE);
    if (showGrid) {
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
      ctx.lineWidth = 1;
      for (let x = patchSize; x < SIZE; x += patchSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.stroke();
      }
      for (let y = patchSize; y < SIZE; y += patchSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(SIZE, y); ctx.stroke();
      }
    }
    if (showNumbers) {
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      const grid = SIZE / patchSize;
      for (let py = 0; py < grid; py++) {
        for (let px = 0; px < grid; px++) {
          const idx = py * grid + px;
          const x = px * patchSize + 3, y = py * patchSize + 11;
          ctx.fillRect(x - 2, y - 9, 18, 12);
          ctx.fillStyle = '#fbbf24'; ctx.fillText(idx, x, y);
          ctx.fillStyle = 'rgba(0,0,0,0.65)';
        }
      }
    }
    if (hoveredPatch !== null) {
      const grid = SIZE / patchSize;
      const py = Math.floor(hoveredPatch / grid);
      const px = hoveredPatch % grid;
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(px * patchSize, py * patchSize, patchSize, patchSize);
    }
  }, [patchSize, showGrid, showNumbers, hoveredPatch]);

  const grid = SIZE / patchSize;
  const numPatches = grid * grid;

  // Render flattened patch sequence
  const patchSequence = useMemo(() => {
    const arr = [];
    for (let i = 0; i < numPatches; i++) arr.push(i);
    return arr;
  }, [numPatches]);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * SIZE;
    const px = Math.floor(x / patchSize);
    const py = Math.floor(y / patchSize);
    setHoveredPatch(py * grid + px);
  };

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Drag the "Patch size" slider to see how the same image splits into more or fewer tokens.',
        'Hover any patch on the grid — its actual RGB pixels are shown flattening into a vector and projecting to a D-dim embedding.',
        'Toggle "Show grid" / "Show numbers" to focus on the visual flow vs. the math.',
      ]}/>
      <Section icon={Grid3x3} kicker="02 — From image to sequence" title="Patch embedding">
        <p className="max-w-3xl mb-3">
          A Transformer expects a sequence of vectors. ViT produces this sequence the simplest way imaginable:
          slice the image into a grid of <Eq>{String.raw`P \times P`}</Eq> patches, flatten each patch into a vector of
          length <Eq>{String.raw`P^2 \cdot C`}</Eq>, and project it linearly to dimension <Eq>D</Eq>.
        </p>
        <p className="max-w-3xl text-slate-400 text-sm">
          The standard ViT-Base uses <Eq>P=16</Eq> on <Eq>{String.raw`224 \times 224`}</Eq> inputs, giving 196 tokens. Drag the
          patch size below to feel the trade-off: smaller patches = more tokens = quadratically more
          attention compute.
        </p>
      </Section>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Input image (256×256)</div>
          <div
            className="relative w-[256px] h-[256px] rounded-lg overflow-hidden ring-1 ring-slate-700"
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoveredPatch(null)}
          >
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
          </div>
          <div className="mt-4 space-y-3">
            <Slider
              label="Patch size P"
              value={patchSize}
              options={[8, 16, 32, 64]}
              onChange={setPatchSize}
            />
            <div className="flex gap-2">
              <Toggle label="Grid" value={showGrid} onChange={setShowGrid} />
              <Toggle label="Indices" value={showNumbers} onChange={setShowNumbers} />
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <div className="grid grid-cols-3 gap-4 mb-4">
              <Stat label="Patches per side" value={grid} />
              <Stat label="Total tokens N" value={numPatches} accent />
              <Stat label="Patch dim P²·3" value={patchSize * patchSize * 3} />
            </div>
            <div className="font-mono text-[12px] text-slate-400 leading-relaxed bg-slate-950/50 border border-slate-800 rounded p-3">
              <span className="text-amber-300">x</span> ∈ ℝ<sup>H×W×C</sup> →
              reshape → <span className="text-amber-300">x_p</span> ∈ ℝ<sup>N×(P²·C)</sup> →
              <span className="text-amber-300"> z₀</span> = [<span className="text-rose-300">x_class</span>;
              x_p<sup>1</sup>E; x_p<sup>2</sup>E; …; x_p<sup>N</sup>E] + E_pos
            </div>
          </Card>

          <Card className="p-5">
            <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
              <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/80">
                What happens to {hoveredPatch !== null ? <>patch <span className="text-amber-300">#{hoveredPatch}</span></> : 'one patch'}
              </div>
              <span className="text-[10px] font-mono text-slate-500">
                hover any patch on the image — numbers update live
              </span>
            </div>

            <div className="grid md:grid-cols-[110px_auto_1fr] gap-4 items-start">
              {/* Step 1 — raw patch */}
              <div className="text-center">
                <div className="text-[10px] font-mono text-slate-500 mb-1">1 · raw patch</div>
                <div className="w-[100px] h-[100px] rounded ring-1 ring-amber-500/40 overflow-hidden mx-auto">
                  <PatchThumb index={hoveredPatch ?? 0} patchSize={patchSize} grid={grid} />
                </div>
                <div className="text-[10px] font-mono text-slate-500 mt-1">
                  {patchSize}×{patchSize}×3
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  = {(patchSize * patchSize * 3).toLocaleString()} values
                </div>
                {hoveredVector && (
                  <div className="mt-2">
                    <div className="text-[9px] font-mono text-slate-500 mb-1">first 9 R,G,B:</div>
                    <div className="grid grid-cols-3 gap-px text-[9px] font-mono text-slate-300 bg-slate-950 rounded p-1 border border-slate-800">
                      {Array.from(hoveredVector.flat.slice(0, 27)).map((v, i) => (
                        <div key={i} className={`text-center ${i % 3 === 0 ? 'text-rose-300' : i % 3 === 1 ? 'text-emerald-300' : 'text-sky-300'}`}>
                          {v.toFixed(2)}
                        </div>
                      ))}
                    </div>
                    <div className="text-[9px] font-mono text-slate-500 mt-1">
                      …{(hoveredVector.flat.length - 27).toLocaleString()} more
                    </div>
                  </div>
                )}
              </div>

              {/* Step 2 — flatten + ×E */}
              <div className="flex flex-col items-center mt-6">
                <div className="text-[10px] font-mono text-slate-500 mb-1">2 · flatten + ×E</div>
                <div className="text-amber-400/70 font-mono text-2xl">→</div>
                <div className="px-2.5 py-1 rounded bg-amber-500/15 border border-amber-500/40 text-[10px] font-mono text-amber-200 mt-1">
                  × E
                </div>
                <div className="text-[9px] font-mono text-slate-500 mt-1 text-center leading-tight">
                  E ∈ ℝ<sup>{patchSize*patchSize*3}×{D_DEMO}</sup>
                  <br/>(learned)
                </div>
              </div>

              {/* Step 3 — D-dim embedding with real numbers */}
              <div>
                <div className="text-[10px] font-mono text-slate-500 mb-1">
                  3 · patch embedding · D = {D_DEMO} <span className="text-slate-600">(real ViT-Base uses 768)</span>
                </div>
                {hoveredVector ? (
                  <div className="space-y-0.5 bg-slate-950 rounded p-2 border border-slate-800">
                    {Array.from(hoveredVector.embedding).map((v, i) => {
                      const t = Math.min(1, Math.abs(v) / 8);
                      return (
                        <div key={i} className="flex items-center gap-2 text-[10px] font-mono">
                          <span className="text-slate-500 w-5 text-right">{i}</span>
                          <div className="flex-1 h-2 bg-slate-800/80 rounded relative overflow-hidden">
                            <div className="absolute top-0 bottom-0 w-px bg-slate-600" style={{ left: '50%' }}/>
                            <div className="absolute top-0 bottom-0 transition-[width,left] duration-150" style={{
                              left: v >= 0 ? '50%' : `${50 - t * 50}%`,
                              width: `${t * 50}%`,
                              background: v >= 0 ? 'rgba(245,158,11,0.85)' : 'rgba(20,184,166,0.85)',
                            }}/>
                          </div>
                          <span className={`w-12 text-right ${v >= 0 ? 'text-amber-300' : 'text-teal-300'}`}>
                            {v >= 0 ? '+' : ''}{v.toFixed(2)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500 italic text-center py-8 border border-dashed border-slate-700/50 rounded">
                    Hover a patch on the image to compute its embedding →
                  </div>
                )}
              </div>
            </div>

            <p className="text-[12px] text-slate-400 mt-4 leading-relaxed">
              Every patch follows the same path: flatten its {patchSize*patchSize*3} pixel values into a
              vector, multiply by the same learned matrix <Eq>E</Eq>, get a {D_DEMO}-dim embedding (real
              ViT-Base uses D = 768; we show 12 here for readability). Different patches → different
              embeddings; same patch → identical embedding every time.
            </p>
          </Card>
        </div>
      </div>

      <Takeaway
        understand={[
          'Each P×P patch is flattened into a vector and projected to D dims by a single shared learned matrix E.',
          'After patching you have a sequence of N = (H·W)/P² embeddings — exactly the shape a transformer wants.',
        ]}
        confusion={['Patches throw away spatial order. Position embeddings (next tab) put it back.']}
        tryIt="Try patch size 16 vs 64. Smaller patches = more tokens = quadratically more attention compute."
      />
    </div>
  );
}

const Stat = ({ label, value, accent }) => (
  <div>
    <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500">{label}</div>
    <div className={`text-2xl font-serif ${accent ? 'text-amber-300' : 'text-slate-100'}`}>{value}</div>
  </div>
);

function PatchThumb({ index, patchSize, grid, highlighted, onHover }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = patchSize; c.height = patchSize;
    const ctx = c.getContext('2d');
    const tmp = document.createElement('canvas');
    tmp.width = 256; tmp.height = 256;
    drawTestImage(tmp.getContext('2d'), 256);
    const py = Math.floor(index / grid);
    const px = index % grid;
    ctx.drawImage(tmp, px * patchSize, py * patchSize, patchSize, patchSize, 0, 0, patchSize, patchSize);
  }, [index, patchSize, grid]);
  return (
    <div
      onMouseEnter={onHover}
      className={`aspect-square relative rounded overflow-hidden ring-1 transition-all ${highlighted ? 'ring-rose-400 ring-2 z-10 scale-110' : 'ring-slate-700/60'}`}
    >
      <canvas ref={ref} className="w-full h-full" />
    </div>
  );
}

/* =========================================================
   TAB 3 — Position Embedding
   ========================================================= */

const POS_METHODS = [
  { id: 'learned',  name: 'Learned (untrained)',     kind: 'classic' },
  { id: 'learnedT', name: 'Learned (post-training)', kind: 'classic' },
  { id: 'sin1d',    name: '1D Sinusoidal',           kind: 'classic' },
  { id: 'sin2d',    name: '2D Sinusoidal (axial)',   kind: 'classic' },
  { id: 'rope',     name: 'RoPE · 2D Rotary',        kind: 'modern'  },
  { id: 'alibi',    name: 'ALiBi · linear bias',     kind: 'modern'  },
  { id: 'relbias',  name: 'Swin Relative Bias',      kind: 'modern'  },
];

const PaperLink = ({ href, label, color = 'amber' }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className={`inline-flex items-center gap-1 text-[10px] font-mono align-middle ml-1
      ${color === 'teal' ? 'text-teal-300 hover:text-teal-200' : 'text-amber-300 hover:text-amber-200'}
      underline decoration-dotted underline-offset-2`}
  >
    {label} ↗
  </a>
);

const POS_METHOD_INFO = {
  learned: (
    <>
      <strong className="text-amber-200">Learned (untrained).</strong> ViT's actual choice — but at <em>init</em>.
      A free <Eq>{String.raw`E_{\text{pos}}\in\mathbb{R}^{N\times D}`}</Eq> with random Gaussian rows.
      No spatial structure baked in — only the diagonal lights up because each position is similar to itself.
      Spatial relations have to be <em>learned</em> from the data.
      <PaperLink href="https://arxiv.org/abs/2010.11929" label="ViT · arXiv:2010.11929" />
    </>
  ),
  learnedT: (
    <>
      <strong className="text-amber-200">Learned (post-training).</strong> What ViT-Base's position embeddings
      actually look like after ImageNet training (Dosovitskiy et al., 2021, Fig. 7). The model discovers
      a smooth 2D structure on its own — nearby positions develop similar embeddings. We approximate it
      as <Eq>{String.raw`\exp(-\|p_i - p_j\|^2 / 2\sigma^2)`}</Eq>.
      <PaperLink href="https://arxiv.org/abs/2010.11929" label="ViT · arXiv:2010.11929" />
    </>
  ),
  sin1d: (
    <>
      <strong className="text-amber-200">1D Sinusoidal</strong> · the original "Attention Is All You Need" recipe.
      Patches are flattened to row-major index <Eq>i</Eq>; alternating dims use
      <Eq>{String.raw`\sin(i\cdot\omega_k)`}</Eq>, <Eq>{String.raw`\cos(i\cdot\omega_k)`}</Eq>.
      Cheap, no params — but the 2D image structure is collapsed to a single dimension.
      <PaperLink href="https://arxiv.org/abs/1706.03762" label="Vaswani et al. · arXiv:1706.03762" />
    </>
  ),
  sin2d: (
    <>
      <strong className="text-amber-200">2D Sinusoidal (axial)</strong> · the natural vision adaptation.
      Half the dims encode <Eq>{String.raw`p_x`}</Eq>, the other half encode <Eq>{String.raw`p_y`}</Eq> with
      sin/cos at log-spaced frequencies. Used in DETR, Stable-Diffusion's UNet, and many ViT variants.
      Notice the cross-shaped bright row + column around the clicked patch.
      <PaperLink href="https://arxiv.org/abs/2005.12872" label="DETR · arXiv:2005.12872" />
    </>
  ),
  rope: (
    <>
      <strong className="text-teal-200">RoPE · Rotary Position Embeddings</strong> (Su et al., 2021)
      · the default in modern LLMs (LLaMA, Qwen, GPT-NeoX). Instead of <em>adding</em> a vector,
      RoPE <em>rotates</em> Q and K dim-pairs by an angle proportional to position.
      Q·K then depends only on the <em>relative</em> offset:
      <Eq>{String.raw`q_i^\top k_j = f(i-j)`}</Eq>. We extend it to 2D by splitting dims between rows / cols.
      <PaperLink href="https://arxiv.org/abs/2104.09864" label="RoFormer · arXiv:2104.09864" color="teal" />
    </>
  ),
  alibi: (
    <>
      <strong className="text-teal-200">ALiBi · Attention with Linear Biases</strong> (Press et al., 2022)
      · used in MPT, BLOOM. No position embeddings at all — instead, a <em>linear distance penalty</em>
      <Eq>{String.raw`-m \cdot |i - j|`}</Eq> is added directly to attention logits.
      Strong locality bias and trivially extrapolates to longer sequences. We use 2D Manhattan distance for vision.
      <PaperLink href="https://arxiv.org/abs/2108.12409" label="ALiBi · arXiv:2108.12409" color="teal" />
    </>
  ),
  relbias: (
    <>
      <strong className="text-teal-200">Swin Relative Position Bias</strong> (Liu et al., 2021)
      · a learnable scalar <Eq>{String.raw`B_{\Delta x, \Delta y}`}</Eq> per relative offset, added to attention
      logits inside each window. Doesn't add a vector to tokens — just biases attention by relative position.
      We approximate "what gets learned" as a smooth 2D Gaussian falloff.
      <PaperLink href="https://arxiv.org/abs/2103.14030" label="Swin · arXiv:2103.14030" color="teal" />
    </>
  ),
};

/* computePositionSim — produces an N×N spatial-similarity matrix for the
   chosen positional-encoding method. Methods that add a vector to tokens
   (learned, sinusoidal) build a posMatrix and take cosine similarity.
   Methods that only modify attention (RoPE, ALiBi, Swin relative bias)
   compute the implied similarity between positions directly. */
function computePositionSim(method, N) {
  const grid = Math.round(Math.sqrt(N));
  const D = 64;
  const sim = new Float32Array(N * N);

  const posSimFromMatrix = (pos) => {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let dot = 0, na = 0, nb = 0;
        for (let d = 0; d < D; d++) {
          const a = pos[i * D + d], b = pos[j * D + d];
          dot += a * b; na += a * a; nb += b * b;
        }
        sim[i * N + j] = dot / (Math.sqrt(na * nb) + 1e-8);
      }
    }
  };

  if (method === 'learned') {
    const rng = mulberry32(7);
    const pos = new Float32Array(N * D);
    for (let k = 0; k < pos.length; k++) pos[k] = rng() * 2 - 1;
    posSimFromMatrix(pos);
  } else if (method === 'learnedT') {
    for (let i = 0; i < N; i++) {
      const ix = i % grid, iy = (i / grid) | 0;
      for (let j = 0; j < N; j++) {
        const jx = j % grid, jy = (j / grid) | 0;
        const dx = ix - jx, dy = iy - jy;
        sim[i * N + j] = Math.exp(-(dx * dx + dy * dy) / (2 * 0.9 * 0.9));
      }
    }
  } else if (method === 'sin1d') {
    const pos = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < D; d++) {
        const k = Math.floor(d / 2);
        const omega = Math.pow(10000, -2 * k / D);
        pos[i * D + d] = (d % 2 === 0) ? Math.sin(i * omega) : Math.cos(i * omega);
      }
    }
    posSimFromMatrix(pos);
  } else if (method === 'sin2d') {
    const pos = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      const py = (i / grid) | 0, px = i % grid;
      for (let d = 0; d < D; d++) {
        const k = Math.floor(d / 4);
        const omega = Math.pow(10000, -2 * k / D);
        if (d % 4 === 0) pos[i * D + d] = Math.sin(px * omega);
        else if (d % 4 === 1) pos[i * D + d] = Math.cos(px * omega);
        else if (d % 4 === 2) pos[i * D + d] = Math.sin(py * omega);
        else pos[i * D + d] = Math.cos(py * omega);
      }
    }
    posSimFromMatrix(pos);
  } else if (method === 'rope') {
    // 2D RoPE: split dims half for row, half for column. Q·K reduces to a
    // sum of cos((Δp)·ω_k) terms — that's exactly what we plot.
    const halfPairs = (D / 4) | 0; // pairs allocated per axis
    for (let i = 0; i < N; i++) {
      const ix = i % grid, iy = (i / grid) | 0;
      for (let j = 0; j < N; j++) {
        const jx = j % grid, jy = (j / grid) | 0;
        const dx = ix - jx, dy = iy - jy;
        let s = 0;
        for (let k = 0; k < halfPairs; k++) {
          const omega = Math.pow(10000, -2 * k / (D / 2));
          s += Math.cos(dx * omega);
          s += Math.cos(dy * omega);
        }
        sim[i * N + j] = s / (2 * halfPairs);
      }
    }
  } else if (method === 'alibi') {
    // Attention bias = -m · |Δ|. Convert to "similarity" via exp(bias) so
    // it overlays on the same 0..1 colourmap as the others.
    const m = 0.6;
    for (let i = 0; i < N; i++) {
      const ix = i % grid, iy = (i / grid) | 0;
      for (let j = 0; j < N; j++) {
        const jx = j % grid, jy = (j / grid) | 0;
        const d = Math.abs(ix - jx) + Math.abs(iy - jy);
        sim[i * N + j] = Math.exp(-m * d);
      }
    }
  } else if (method === 'relbias') {
    // Swin's bias is a *learned* table indexed by relative (Δx, Δy). We
    // approximate "what gets learned" with a smooth, mildly anisotropic
    // Gaussian — the actual learned bias has been observed to look like
    // this in published heatmaps.
    for (let i = 0; i < N; i++) {
      const ix = i % grid, iy = (i / grid) | 0;
      for (let j = 0; j < N; j++) {
        const jx = j % grid, jy = (j / grid) | 0;
        const dx = ix - jx, dy = iy - jy;
        const r2 = dx * dx + dy * dy;
        sim[i * N + j] = Math.exp(-r2 / (2 * 1.1 * 1.1));
      }
    }
  } else {
    // Fallback: identity
    for (let i = 0; i < N; i++) sim[i * N + i] = 1;
  }

  return sim;
}

function PositionTab() {
  const [posOn, setPosOn] = useState(true);
  const [shuffle, setShuffle] = useState(false);
  const [posMethod, setPosMethod] = useState('sin2d');
  const N = 16;

  const simMatrix = useMemo(() => computePositionSim(posMethod, N), [posMethod, N]);

  const order = useMemo(() => {
    const arr = Array.from({ length: N }, (_, i) => i);
    if (shuffle) {
      const rng = mulberry32(1337);
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    }
    return arr;
  }, [shuffle]);

  return (
    <div className="space-y-10">
      <HowToUseBadge instructions={[
        'Use the "Try it" toggles to see what happens when position embeddings are removed and patches are shuffled.',
        'Pick an encoding from the Classic / Modern row (Sinusoidal, RoPE, ALiBi, Swin Relative Bias, etc.) — each has an arXiv link.',
        'Click any patch in the 4×4 grid; the explorer shows how strongly the chosen encoding treats every other patch as "nearby".',
      ]}/>
      <Section icon={Hash} kicker="03 — Spatial order, restored" title="Position embeddings & [CLS] token">
        <p className="max-w-3xl mb-3 text-slate-300">
          The last tab turned an image into N patch vectors. Two things are still missing before we can
          run a real ViT classifier:
        </p>
        <ol className="list-decimal pl-6 text-slate-300 space-y-1.5 mb-3 max-w-3xl">
          <li>The transformer has <em>no idea where each patch came from</em> — top-left or bottom-right look identical to it.</li>
          <li>The transformer outputs <em>N vectors</em>, one per patch — but classification needs <em>one</em> answer per image.</li>
        </ol>
        <p className="max-w-3xl text-slate-400 text-sm">
          ViT fixes the first with <span className="text-amber-300">position embeddings</span> and the
          second with a special <span className="text-rose-300">[CLS]</span> token. We'll handle them in
          that order.
        </p>
      </Section>

      {/* Part 1 — position embeddings */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <div className="w-7 h-7 rounded-full bg-amber-500/20 border border-amber-500/50 flex items-center justify-center font-mono text-amber-300 text-sm">1</div>
          <h3 className="font-serif text-2xl text-slate-100 tracking-tight">Position embeddings</h3>
        </div>
        <p className="text-slate-300 mb-5 max-w-3xl leading-relaxed">
          Self-attention is <em>permutation-equivariant</em>: if you shuffle the input tokens, the output
          shuffles the same way — but the actual content the model computes is the same. So the raw patch
          sequence carries <strong>no spatial information</strong>. The fix: add a learned vector
          {' '}<Eq>{String.raw`E_{\text{pos}}[i]`}</Eq> to each token to mark "this came from position i".
        </p>

        <div className="grid lg:grid-cols-[minmax(0,260px)_minmax(0,1fr)] gap-5 items-start">
          <Card className="p-4">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
              Try it · 4×4 patch sequence
            </div>
            {/* 4×4 grid matches the actual image arrangement — no wasted
                horizontal space, and visually accurate to N=16. */}
            <div className="grid grid-cols-4 gap-1 mb-3 max-w-[200px] mx-auto">
              {order.map(i => (
                <div
                  key={i}
                  className="aspect-square rounded border flex flex-col items-center justify-center font-mono text-amber-200 transition-all"
                  style={{
                    background: posOn ? 'rgba(245,158,11,0.10)' : 'rgba(100,116,139,0.10)',
                    borderColor: posOn ? 'rgba(245,158,11,0.40)' : 'rgba(100,116,139,0.40)',
                  }}
                >
                  <span className="text-[11px] leading-none">{i}</span>
                  {posOn && <span className="text-[8px] text-teal-300 leading-none mt-0.5">@{i}</span>}
                </div>
              ))}
            </div>
            <div className="space-y-2">
              <Toggle label="Shuffle patches" value={shuffle} onChange={setShuffle} />
              <Toggle label="Add position embeddings" value={posOn} onChange={setPosOn} />
            </div>
            <div className="mt-3 text-[12px] leading-relaxed">
              {!posOn && shuffle && <span className="text-rose-300">No position info + shuffled → the model treats this as the same input as the original. Spatial structure is lost.</span>}
              {!posOn && !shuffle && <span className="text-slate-400">No position info. It happens to look "right" because we didn't shuffle, but the model isn't actually using the order.</span>}
              {posOn && shuffle && <span className="text-amber-200">Shuffled, but each patch carries its position tag (the small <span className="text-teal-300">@i</span>). The model can recover the original spatial structure.</span>}
              {posOn && !shuffle && <span className="text-teal-300">Each token now carries both content (patch) <em>and</em> location (the <span className="text-teal-300">@i</span> tag).</span>}
            </div>
          </Card>

          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/80 mb-2">
              Pick an encoding · click a patch · see its spatial similarity
            </div>

            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500 mb-1.5">Classic</div>
            <div className="flex flex-wrap gap-1.5 mb-2">
              {POS_METHODS.filter(m => m.kind === 'classic').map(m => (
                <button
                  key={m.id}
                  onClick={() => setPosMethod(m.id)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-mono border transition-all
                    ${posMethod === m.id
                      ? 'bg-amber-500/20 border-amber-500/60 text-amber-100'
                      : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-200'}`}
                >
                  {m.name}
                </button>
              ))}
            </div>

            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500 mb-1.5">Modern · used in recent LLMs / vision models</div>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {POS_METHODS.filter(m => m.kind === 'modern').map(m => (
                <button
                  key={m.id}
                  onClick={() => setPosMethod(m.id)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-mono border transition-all
                    ${posMethod === m.id
                      ? 'bg-teal-500/20 border-teal-500/60 text-teal-100'
                      : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-200'}`}
                >
                  <span className="opacity-60 mr-1">★</span>{m.name}
                </button>
              ))}
            </div>

            <PositionSimExplorer data={simMatrix} N={N} key={posMethod} />

            <div className="mt-3 text-[12px] text-slate-300 leading-relaxed border-l-2 border-amber-500/40 pl-3">
              {POS_METHOD_INFO[posMethod]}
            </div>
            <div className="mt-2 text-[11px] text-slate-500 italic leading-relaxed">
              Click any patch in the 4×4 grid. The other cells light up by the chosen method's
              implicit similarity between two positions — bright = "the model treats these as nearby",
              dim = "treated as far apart".
            </div>
          </Card>
        </div>
      </div>

      {/* Part 2 — [CLS] token */}
      <div>
        <div className="flex items-center gap-3 mb-3">
          <div className="w-7 h-7 rounded-full bg-rose-500/20 border border-rose-500/50 flex items-center justify-center font-mono text-rose-300 text-sm">2</div>
          <h3 className="font-serif text-2xl text-slate-100 tracking-tight">The [CLS] token</h3>
        </div>
        <p className="text-slate-300 mb-5 max-w-3xl leading-relaxed">
          The transformer eats N tokens and produces N output vectors. But classification needs one answer
          per image. ViT prepends a single learnable <Eq>{String.raw`[\text{CLS}]`}</Eq> embedding (a learned parameter, not
          tied to any patch) to the sequence. Through self-attention, [CLS] gathers information from every
          patch token in every layer. After the final block, its output vector goes straight into a linear
          classifier — and the patch outputs are ignored for classification.
        </p>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            Sequence flow with [CLS]
          </div>

          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <div className="text-[11px] font-mono text-slate-500 w-12 shrink-0">in:</div>
            <div className="w-10 h-10 rounded bg-rose-500/25 border-2 border-rose-400 flex items-center justify-center font-mono text-[10px] text-rose-200 font-bold">CLS</div>
            <span className="text-slate-600">|</span>
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="w-10 h-10 rounded bg-amber-500/10 border border-amber-500/40 flex items-center justify-center font-mono text-xs text-amber-200">P{i}</div>
            ))}
            <span className="text-slate-500 text-sm">… (N total)</span>
          </div>

          <div className="flex items-center gap-3 my-3 ml-12">
            <div className="text-amber-400/70 font-mono text-lg">↓</div>
            <div className="px-3 py-1.5 rounded bg-slate-800/60 border border-slate-700 text-[11px] font-mono text-slate-300">
              L stacked Transformer blocks
            </div>
            <div className="text-slate-500 text-[11px]">[CLS] gathers info from all patches at every layer</div>
          </div>

          <div className="flex items-center gap-2 mb-3 flex-wrap">
            <div className="text-[11px] font-mono text-slate-500 w-12 shrink-0">out:</div>
            <div className="w-10 h-10 rounded bg-rose-500/40 border-2 border-rose-400 flex items-center justify-center font-mono text-[10px] text-rose-100 font-bold shadow-lg shadow-rose-500/30 ring-2 ring-rose-400/40">CLS</div>
            <span className="text-slate-600">|</span>
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="w-10 h-10 rounded bg-slate-800/40 border border-slate-700 flex items-center justify-center font-mono text-xs text-slate-500" title="Patch outputs are ignored for classification">P{i}</div>
            ))}
            <span className="text-slate-600 text-sm">… (ignored for classification)</span>
          </div>

          <div className="flex items-center gap-3 mt-4 ml-12">
            <div className="text-rose-300 font-mono text-lg">↓</div>
            <div className="px-3 py-1.5 rounded bg-rose-500/15 border border-rose-500/40 text-[12px] text-rose-100">
              Linear classifier
            </div>
            <div className="text-amber-300 font-mono text-lg">→</div>
            <div className="px-3 py-1.5 rounded bg-amber-500/15 border border-amber-500/40 text-[12px] text-amber-100">
              Class probabilities
            </div>
          </div>

          <p className="text-[12px] text-slate-400 italic mt-5 max-w-3xl leading-relaxed">
            Why is [CLS] enough? Because self-attention lets it pull information from every patch.
            By the final layer, [CLS] is a learned summary of the whole image — exactly what the classifier needs.
            For tasks like segmentation, the patch outputs aren't ignored; for classification specifically,
            only [CLS] matters.
          </p>
        </Card>
      </div>

      <Takeaway
        understand={[
          'Self-attention is permutation-invariant — without position embeddings, top-left and bottom-right look the same to the model.',
          '[CLS] is a learned vector prepended to the sequence; its final-layer output is what the classifier reads.',
        ]}
        confusion={['[CLS] is NOT a patch — it has no spatial location, only a learned embedding.']}
        tryIt="Click between top-5 classes in the Live Demo — the heatmap shows where [CLS] gathered evidence for each."
      />
    </div>
  );
}

function PosHeatmap({ data, rows, cols }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = cols; c.height = rows;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(cols, rows);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < data.length; i++) { mn = Math.min(mn, data[i]); mx = Math.max(mx, data[i]); }
    for (let i = 0; i < data.length; i++) {
      const t = (data[i] - mn) / (mx - mn);
      // amber → black colormap
      const r = Math.floor(245 * t);
      const g = Math.floor(158 * t * 0.8);
      const b = Math.floor(11 * t * 0.4);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, rows, cols]);
  return <canvas ref={ref} className="w-full rounded border border-slate-800" style={{ imageRendering: 'pixelated', height: '180px' }} />;
}

function SimHeatmap({ data, N }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = N; c.height = N;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(N, N);
    for (let i = 0; i < data.length; i++) {
      const t = (data[i] + 1) / 2;
      const r = Math.floor(20 + 225 * t);
      const g = Math.floor(184 * t);
      const b = Math.floor(166 * t);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, N]);
  return <canvas ref={ref} className="w-full aspect-square rounded border border-slate-800" style={{ imageRendering: 'pixelated' }} />;
}

/* PositionSimExplorer — interactive replacement for the abstract similarity
   heatmap. Renders a 4×4 patch grid (the actual spatial layout). Click any
   patch and the rest of the grid is colour-coded by cosine similarity to
   the clicked one — bright = similar embedding, dim = different. The
   spatial-structure lesson ("near patches share embeddings") becomes a
   visible pattern instead of a row to read off a heatmap. */
function PositionSimExplorer({ data, N }) {
  const side = Math.round(Math.sqrt(N));
  const [pick, setPick] = useState(side * (side / 2 | 0) + (side / 2 | 0)); // start at centre

  // Min-max normalise the picked row so the colour gradient stretches across
  // the actual range, not the theoretical [-1, 1]. Without this, real similarity
  // values (typically 0.6–1.0 for a 16-patch grid) all look uniformly bright.
  let minSim = Infinity, maxSim = -Infinity;
  if (pick != null) {
    for (let i = 0; i < N; i++) {
      if (i === pick) continue;
      const s = data[pick * N + i];
      if (s < minSim) minSim = s;
      if (s > maxSim) maxSim = s;
    }
  }
  const span = Math.max(1e-9, maxSim - minSim);

  // High-contrast magma-style ramp so cell-to-cell density jumps are obvious.
  // t in [0, 1] from min similarity in the row → max similarity.
  function ramp(t) {
    const stops = [
      [0.00,   8,  10,  35],
      [0.30,  60,  20,  90],
      [0.55, 200,  60,  90],
      [0.80, 250, 150,  50],
      [1.00, 255, 240, 110],
    ];
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const a = stops[i - 1], b = stops[i];
        const u = (t - a[0]) / (b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u];
      }
    }
    return stops[stops.length - 1].slice(1);
  }

  return (
    <div className="grid sm:grid-cols-[auto_1fr] gap-4 items-start">
      <div>
        <div
          className="grid gap-1 p-1 rounded bg-slate-950 border border-slate-800"
          style={{ gridTemplateColumns: `repeat(${side}, minmax(0, 1fr))`, width: 240, height: 240 }}
        >
          {Array.from({ length: N }, (_, i) => {
            const isPicked = pick === i;
            const sim = pick == null ? 0 : data[pick * N + i];
            const t = isPicked ? 1 : Math.pow((sim - minSim) / span, 0.85);
            const [r, g, b] = ramp(t);
            return (
              <button
                key={i}
                onClick={() => setPick(i)}
                className={`flex items-center justify-center rounded font-mono text-[10px] leading-tight transition-all
                  ${isPicked ? 'ring-2 ring-rose-400 z-10 scale-105' : 'ring-1 ring-slate-700/30'}`}
                style={{
                  background: isPicked
                    ? 'rgba(244, 63, 94, 0.65)'
                    : `rgb(${r | 0}, ${g | 0}, ${b | 0})`,
                  color: isPicked ? '#fff' : t > 0.55 ? '#0a0e1a' : '#e2e8f0',
                }}
                title={`patch ${i} · cos similarity ${sim.toFixed(4)}`}
              >
                {isPicked ? '★' : sim.toFixed(2)}
              </button>
            );
          })}
        </div>
        <div className="text-[10px] font-mono text-slate-500 mt-1 text-center">
          rose ★ = picked · bright yellow = most similar in this row · dark = least
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-[12px] text-slate-300 leading-snug">
          You're looking at the actual 4×4 spatial arrangement of patches, not an abstract heatmap.
          Each cell shows the cosine similarity of <em>that patch's position embedding</em> to the
          one you clicked. Colour scales relative to the row's range so differences pop.
        </div>
        <div className="text-[11px] font-mono text-slate-400 space-y-1">
          {pick != null && (() => {
            const sims = [];
            for (let i = 0; i < N; i++) sims.push({ i, s: data[pick * N + i] });
            sims.sort((a, b) => b.s - a.s);
            const top = sims.filter(o => o.i !== pick).slice(0, 3);
            const bot = sims.slice(-2);
            return (
              <>
                <div>
                  <span className="text-amber-300">most similar to #{pick}:</span> {top.map(o => `#${o.i} (${o.s.toFixed(4)})`).join(' · ')}
                </div>
                <div>
                  <span className="text-slate-500">least similar:</span> {bot.map(o => `#${o.i} (${o.s.toFixed(4)})`).join(' · ')}
                </div>
                <div className="text-slate-500">
                  range in this row: {minSim.toFixed(4)} → {maxSim.toFixed(4)}
                </div>
              </>
            );
          })()}
        </div>
        <p className="text-[11px] text-slate-400 italic leading-snug">
          Try the picks: corners are most similar to other corners; centres to other centres. That's
          what gives self-attention its sense of geometry.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   TAB 4 — Self-Attention
   ========================================================= */

function AttentionTab() {
  const [patchSize, setPatchSize] = useState(60);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [seed, setSeed] = useState(7);
  const [step, setStep] = useState(0);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const SIZE = 240; // chosen so 30 / 48 / 60 px patches all divide cleanly
  const grid = SIZE / patchSize;
  const N = grid * grid;
  const D = 16;

  const [computed, setComputed] = useState({ Q: null, K: null, V: null, scores: null, attn: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = SIZE; canvas.height = SIZE;
    drawPairsDemoImage(canvas.getContext('2d'), SIZE);
    const X = computePatchFeatures(canvas, patchSize, D, seed);
    const rng = mulberry32(seed);
    const Wq = randMatrix(D, D, rng);
    const Wk = randMatrix(D, D, rng);
    const Wv = randMatrix(D, D, rng);
    // Boost projection variance so softmax peaks instead of going uniform
    // (random init at σ≈0.35 produces dot products too small to peak after /√d).
    const projBoost = 2.5;
    for (let i = 0; i < Wq.data.length; i++) Wq.data[i] *= projBoost;
    for (let i = 0; i < Wk.data.length; i++) Wk.data[i] *= projBoost;
    const Q = matmul(X, Wq);
    const K = matmul(X, Wk);
    const V = matmul(X, Wv);
    const Kt = transpose(K);
    const scoresMat = matmul(Q, Kt);
    for (let i = 0; i < scoresMat.data.length; i++) scoresMat.data[i] /= Math.sqrt(D);
    const attn = softmaxRows(scoresMat);
    setComputed({ Q, K, V, scores: scoresMat, attn });
  }, [patchSize, seed]);

  const { Q, K, V, scores, attn } = computed;

  // Draw overlay: per-step distinct visualizations with numeric labels.
  useEffect(() => {
    const c = overlayRef.current; if (!c) return;
    c.width = SIZE; c.height = SIZE;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, SIZE, SIZE);

    // Guard against stale data (matrices recomputed asynchronously after a
    // patch-size change) and out-of-range selections.
    const dataValid = (M) => M && M.data && M.data.length === N * N;
    const selValid = selectedPatch !== null && selectedPatch >= 0 && selectedPatch < N;

    const drawNum = (text, x, y) => {
      const fs = Math.max(9, Math.floor(patchSize / 4.2));
      ctx.font = `${fs}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)';
      ctx.strokeText(text, x, y);
      ctx.fillStyle = 'white';
      ctx.fillText(text, x, y);
    };

    // Find top-K patches for the aggregate step
    let topSet = null;
    if (step === 3 && selValid && dataValid(attn)) {
      const indexed = [];
      for (let i = 0; i < N; i++) indexed.push({ i, w: attn.data[selectedPatch * N + i] });
      indexed.sort((a, b) => b.w - a.w);
      topSet = new Set(indexed.slice(0, Math.min(5, N)).map(o => o.i));
    }

    if (selValid && (step === 1 || step === 2 || step === 3 || step === 4)) {
      // Find the value range we need to map to color/label.
      let M = null;
      if (step === 1) M = scores;     // raw scaled score, signed
      else M = attn;                   // softmax probability, 0..1

      if (dataValid(M)) {
        let maxAbs = 1e-9;
        if (step === 1) {
          for (let i = 0; i < N; i++) {
            const v = Math.abs(M.data[selectedPatch * N + i]);
            if (v > maxAbs) maxAbs = v;
          }
        }

        for (let py = 0; py < grid; py++) {
          for (let px = 0; px < grid; px++) {
            const i = py * grid + px;
            const v = M.data[selectedPatch * N + i];
            const cx = px * patchSize, cy = py * patchSize;

            if (step === 1) {
              // Diverging: amber for positive, blue for negative.
              const t = Math.min(1, Math.abs(v) / maxAbs);
              if (v >= 0) {
                ctx.fillStyle = `rgba(245, 158, 11, ${t * 0.85})`;
              } else {
                ctx.fillStyle = `rgba(96, 165, 250, ${t * 0.85})`;
              }
              ctx.fillRect(cx, cy, patchSize, patchSize);
            } else if (step === 3) {
              // Aggregate: dim non-top patches
              const isTop = topSet && topSet.has(i);
              const op = isTop ? Math.min(0.9, Math.pow(v * N, 0.5) * 0.9) : 0.05;
              ctx.fillStyle = `rgba(245, 158, 11, ${op})`;
              ctx.fillRect(cx, cy, patchSize, patchSize);
            } else {
              // step 2 or 4 — softmax heatmap
              const t = Math.pow(v * N, 0.5);
              ctx.fillStyle = `rgba(245, 158, 11, ${Math.min(0.85, t * 0.9)})`;
              ctx.fillRect(cx, cy, patchSize, patchSize);
            }

            // Numeric label — only when patches are large enough and value is finite.
            if (patchSize >= 24 && Number.isFinite(v)) {
              let label = '';
              if (step === 1) {
                label = (v >= 0 ? '+' : '') + v.toFixed(1);
              } else if (step === 2 || step === 4) {
                if (v * 100 >= 1) label = (v * 100).toFixed(0) + '%';
              } else if (step === 3) {
                if (topSet && topSet.has(i) && v * 100 >= 1) {
                  label = (v * 100).toFixed(0) + '%';
                }
              }
              if (label) drawNum(label, cx + patchSize / 2, cy + patchSize / 2);
            }
          }
        }
      }
    }

    // grid
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= SIZE; x += patchSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, SIZE); ctx.stroke();
    }
    for (let y = 0; y <= SIZE; y += patchSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(SIZE, y); ctx.stroke();
    }
    if (selectedPatch !== null) {
      const py = Math.floor(selectedPatch / grid);
      const px = selectedPatch % grid;
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 3;
      ctx.strokeRect(px * patchSize, py * patchSize, patchSize, patchSize);
    }
  }, [selectedPatch, attn, scores, patchSize, grid, N, step]);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * SIZE;
    const px = Math.min(grid - 1, Math.max(0, Math.floor(x / patchSize)));
    const py = Math.min(grid - 1, Math.max(0, Math.floor(y / patchSize)));
    setSelectedPatch(py * grid + px);
  };

  const steps = [
    { n: 0, label: 'Project', desc: 'Q = X·Wq, K = X·Wk, V = X·Wv' },
    { n: 1, label: 'Score', desc: 'scores = Q·Kᵀ / √d_k' },
    { n: 2, label: 'Softmax', desc: 'A = softmax(scores)' },
    { n: 3, label: 'Aggregate', desc: 'Out = A·V' },
    { n: 4, label: 'Inspect', desc: 'Click a patch to see its attention pattern' },
  ];

  return (
    <div className="space-y-4">
      <Section icon={Eye} kicker="04 — The core mechanism" title="Scaled dot-product self-attention">
        <p className="max-w-3xl mb-2 text-sm">
          Every token produces three vectors: a <span className="text-amber-300">query</span> Q,
          a <span className="text-teal-300">key</span> K, and a <span className="text-rose-300">value</span> V.
          To update token i, we compute how well its query matches every key (a similarity score), normalize
          via softmax, and use those weights to take a weighted average of the values.
        </p>
        <div className="text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-2 max-w-xl">
          <TeXBlock>{String.raw`\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V`}</TeXBlock>
        </div>
      </Section>

      <div>
        <HowToUseBadge instructions={[
          'Click any patch on the image — it becomes the query patch.',
          'Step through buttons 1 → 5 below to walk the math: project, score, softmax, aggregate.',
          'The image has three matched pairs (red squares, blue circles, yellow triangles). The model should find each twin.',
          'Drag the random-seed slider to confirm the broad pattern is stable across different W_Q, W_K, W_V.',
        ]}/>
      </div>

      <div className="flex gap-1.5 overflow-x-auto pb-1">
        {steps.map(s => (
          <button
            key={s.n}
            onClick={() => setStep(s.n)}
            className={`flex-shrink-0 px-2.5 py-1.5 rounded-lg border text-left transition-all min-w-[110px]
              ${step === s.n
                ? 'bg-amber-500/15 border-amber-500/50 text-amber-100'
                : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:border-slate-600'}`}
          >
            <div className="text-[10px] font-mono uppercase tracking-wider opacity-70">Step {s.n + 1}</div>
            <div className="font-medium text-[13px]">{s.label}</div>
            <div className="text-[10px] font-mono mt-0.5 opacity-70 truncate">{s.desc}</div>
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-[auto_1fr] gap-4">
        <div className="space-y-3">
          <Card className="p-3">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
              {selectedPatch !== null
                ? <>Query: patch <span className="text-amber-300">#{selectedPatch}</span></>
                : <span className="text-amber-300 inline-flex items-center gap-1.5 animate-pulse">
                    👇 Click a patch below to start 👇
                  </span>}
            </div>
            <div
              className={`relative w-[240px] h-[240px] rounded-lg overflow-hidden cursor-crosshair transition-all
                ${selectedPatch === null
                  ? 'ring-2 ring-amber-400/70 shadow-lg shadow-amber-500/20'
                  : 'ring-1 ring-slate-700'}`}
              onClick={handleClick}
            >
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
              <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            </div>
            <div className="mt-2 space-y-2">
              <Slider
                label="Patch size (px)"
                value={patchSize}
                options={[54, 60, 66, 80]}
                onChange={(v) => { setPatchSize(v); setSelectedPatch(null); }}
              />
              <Slider label="Random seed (W_Q, W_K, W_V)" value={seed} min={1} max={50} onChange={setSeed} />
            </div>
            <div className="mt-2 text-[12px] text-slate-300 leading-snug min-h-[2.2em]">
              {selectedPatch === null
                ? <span className="text-amber-200/80">Pick a patch — try a red square, blue circle, or yellow triangle for the cleanest result.</span>
                : step === 0
                  ? <>Step 1 · <span className="text-amber-300">Project</span> — every patch produces Q, K, V vectors (matrices shown right).</>
                  : step === 1
                    ? <>Step 2 · <span className="text-amber-300">Score</span> — query · key dot product. Amber = +, blue = −. Not yet a probability.</>
                    : step === 2
                      ? <>Step 3 · <span className="text-amber-300">Softmax</span> — rows sum to 100%. Peaks = where attention concentrates.</>
                      : step === 3
                        ? <>Step 4 · <span className="text-amber-300">Aggregate</span> — top-5 patches stay bright; output is a blend of <em>their</em> values.</>
                        : <>Step 5 · <span className="text-amber-300">Inspect</span> — try other patches. Should pair red↔red, blue↔blue, yellow↔yellow.</>}
            </div>
          </Card>

          {selectedPatch !== null && attn && (
            <Card className="p-3">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
                Top patches that query #{selectedPatch} attends to
              </div>
              <TopAttended attn={attn} N={N} query={selectedPatch} grid={grid} patchSize={patchSize} />
              <div className="text-[11px] text-slate-400 italic mt-2 leading-snug">
                Query on any colour (red square, blue circle, yellow triangle) should see its twin near the top — that's "patches that look alike attend to each other".
              </div>
            </Card>
          )}
        </div>

        <div className="space-y-3">
          {step === 0 && Q && (
            <Card className="p-3">
              <div className="text-[13px] text-slate-200 mb-2">
                Each input row <TeX>{String.raw`x_i \in \mathbb{R}^D`}</TeX> is multiplied by three learned matrices to produce its query, key, and value.
              </div>
              <div className="grid grid-cols-3 gap-2">
                <MatHeat label={`Q (${Q.rows}×${Q.cols})`} M={Q} color="amber" />
                <MatHeat label={`K (${K.rows}×${K.cols})`} M={K} color="teal" />
                <MatHeat label={`V (${V.rows}×${V.cols})`} M={V} color="rose" />
              </div>
            </Card>
          )}
          <Card className="p-3">
            <div className="flex items-baseline justify-between gap-3 mb-2 flex-wrap">
              <div className="text-[13px] text-slate-200">
                {step <= 0 && <>Attention matrix <TeX>{String.raw`A \in \mathbb{R}^{N \times N}`}</TeX> · not yet computed</>}
                {step === 1 && <>Raw scores <TeX>{String.raw`Q K^\top / \sqrt{d_k}`}</TeX> · signed, pre-softmax</>}
                {step === 2 && <>Softmax weights <Eq>A</Eq> · rows sum to 100%</>}
                {step === 3 && <>Aggregate · top-3 contributors per row highlighted</>}
                {step === 4 && <>Inspect · click a row to make it the query</>}
              </div>
              <span className="font-mono text-[10px] text-slate-500">N = {N}</span>
            </div>
            <StepMatrix
              scores={scores}
              attn={attn}
              N={N}
              selectedPatch={selectedPatch}
              step={step}
              onCellClick={setSelectedPatch}
            />
            <div className="mt-2 text-[11px] text-slate-500 italic leading-snug">
              {step === 1 && 'Amber = positive, blue = negative. Not probabilities yet — softmax hasn\'t been applied.'}
              {step === 2 && 'Each row is a probability distribution over all keys j.'}
              {step === 3 && 'Mass concentrates on a few patches; the output is a weighted sum over just those.'}
              {step === 4 && 'Click a row to focus the canvas overlay on that query patch.'}
              {step <= 0 && 'Walk the steps above to build the matrix.'}
            </div>
          </Card>
        </div>
      </div>

      <Takeaway
        understand={[
          'Attention(Q,K,V) = softmax(Q·Kᵀ/√d_k)·V — pairwise similarity, normalized, then weighted average of values.',
          'One row of the attention matrix is one query patch\'s distribution over all key patches.',
        ]}
        confusion={['Step 2 (raw scores) is NOT a probability — they are signed and unbounded. Only after softmax (step 3) do they sum to 1.']}
        tryIt="Click any patch and step through 1→5; the right-panel matrix shows different content per step."
      />
    </div>
  );
}

function MatHeat({ label, M, color }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = M.cols; c.height = M.rows;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(M.cols, M.rows);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < M.data.length; i++) { mn = Math.min(mn, M.data[i]); mx = Math.max(mx, M.data[i]); }
    const palettes = {
      amber: [245, 158, 11],
      teal: [20, 184, 166],
      rose: [244, 63, 94],
    };
    const [pr, pg, pb] = palettes[color];
    for (let i = 0; i < M.data.length; i++) {
      const t = (M.data[i] - mn) / (mx - mn + 1e-9);
      img.data[i * 4] = Math.floor(pr * t);
      img.data[i * 4 + 1] = Math.floor(pg * t);
      img.data[i * 4 + 2] = Math.floor(pb * t);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [M, color]);
  return (
    <div>
      <div className={`text-[10px] font-mono uppercase tracking-wider mb-1 text-${color}-300`}>{label}</div>
      <canvas ref={ref} className="w-full rounded border border-slate-800" style={{ imageRendering: 'pixelated', aspectRatio: M.cols / M.rows }} />
    </div>
  );
}

/* StepMatrix — step-aware HTML grid that shows raw scores, then
   softmax, then top-K — so the right panel doesn't look like the
   same orange blob across steps. Cells contain numbers when the
   grid is small enough to read them. */
function StepMatrix({ scores, attn, N, selectedPatch, step, onCellClick }) {
  // Guard against either matrix being absent OR sized for a stale grid
  // (which can happen for one render after patchSize changes).
  const sized = (M) => M && M.data && M.data.length === N * N;
  if (step < 1 || !sized(scores) || !sized(attn)) {
    return (
      <div className="aspect-square w-full rounded border border-slate-800/60 bg-slate-950/40 flex items-center justify-center text-center px-4">
        <span className="text-slate-500 text-[12px] font-mono leading-relaxed">
          {step < 1 ? 'Move to Step 2 — the matrix is computed by Q·Kᵀ.' : 'computing…'}
        </span>
      </div>
    );
  }

  const M = step === 1 ? scores : attn;

  // Cache the per-row top-K mask so step 3 doesn't recompute on every render.
  const topMask = useMemo(() => {
    if (step !== 3 || !attn || attn.data.length !== N * N) return null;
    const topK = 3;
    const mask = new Uint8Array(N * N);
    for (let i = 0; i < N; i++) {
      const row = [];
      for (let j = 0; j < N; j++) row.push({ j, w: attn.data[i * N + j] });
      row.sort((a, b) => b.w - a.w);
      for (let k = 0; k < Math.min(topK, N); k++) mask[i * N + row[k].j] = 1;
    }
    return mask;
  }, [attn, step, N]);

  // Cache the value range for the diverging colormap on step 1.
  const maxAbs = useMemo(() => {
    let m = 1e-9;
    for (let i = 0; i < M.data.length; i++) {
      const a = Math.abs(M.data[i]);
      if (a > m) m = a;
    }
    return m;
  }, [M]);

  const showNumbers = N <= 36;
  const fontPx = N <= 4 ? 14 : N <= 9 ? 11 : N <= 16 ? 9 : N <= 25 ? 8 : N <= 36 ? 7 : 6;

  return (
    <div
      className="grid gap-px p-1 rounded bg-slate-900/80 border border-slate-800 w-full"
      style={{ gridTemplateColumns: `repeat(${N}, minmax(0, 1fr))`, aspectRatio: '1' }}
    >
      {Array.from({ length: N * N }, (_, idx) => {
        const i = (idx / N) | 0;
        const j = idx % N;
        const v = M.data[idx];
        const isSelectedRow = selectedPatch === i;
        let bg, label = '';

        if (step === 1) {
          const t = Math.min(1, Math.abs(v) / maxAbs);
          bg = v >= 0
            ? `rgba(245, 158, 11, ${0.05 + t * 0.85})`
            : `rgba(96, 165, 250, ${0.05 + t * 0.85})`;
          if (showNumbers && Number.isFinite(v)) label = (v >= 0 ? '+' : '') + v.toFixed(1);
        } else if (step === 3) {
          const isTop = topMask[idx] === 1;
          if (isTop) {
            const t = Math.pow(v * N, 0.5);
            bg = `rgba(245, 158, 11, ${Math.min(0.95, t * 0.95)})`;
            if (showNumbers && v * 100 >= 1) label = (v * 100).toFixed(0) + '%';
          } else {
            bg = `rgba(245, 158, 11, 0.04)`;
          }
        } else {
          const t = Math.pow(v * N, 0.5);
          bg = `rgba(245, 158, 11, ${Math.min(0.92, t * 0.92)})`;
          if (showNumbers && v * 100 >= 1) label = (v * 100).toFixed(0);
        }

        return (
          <button
            key={idx}
            onClick={() => onCellClick && onCellClick(i)}
            className={`flex items-center justify-center font-mono leading-none transition-colors p-0
              ${isSelectedRow ? 'ring-1 ring-rose-400 ring-inset' : ''}`}
            style={{
              background: bg,
              fontSize: `${fontPx}px`,
              color: 'white',
              textShadow: '0 0 2px rgba(0,0,0,0.85)',
              aspectRatio: '1',
            }}
            title={`row ${i}, col ${j}: ${Number.isFinite(v) ? v.toFixed(4) : '—'}`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

function TopAttended({ attn, N, query, grid, patchSize }) {
  const row = Array.from({ length: N }, (_, i) => ({ i, w: attn.data[query * N + i] }));
  row.sort((a, b) => b.w - a.w);
  const top = row.slice(0, 6);
  return (
    <div className="flex gap-2">
      {top.map(({ i, w }) => (
        <div key={i} className="flex flex-col items-center gap-1">
          <PatchThumb index={i} patchSize={Math.max(patchSize, 32)} grid={grid} />
          <div className="font-mono text-[10px] text-amber-300">{(w * 100).toFixed(1)}%</div>
          <div className="font-mono text-[10px] text-slate-500">#{i}</div>
        </div>
      ))}
    </div>
  );
}

/* =========================================================
   TAB 5 — Multi-Head Attention
   ========================================================= */

function MultiHeadTab() {
  const [numHeads, setNumHeads] = useState(4);
  const [selectedPatch, setSelectedPatch] = useState(12);
  const [activeHead, setActiveHead] = useState(0);
  const canvasRef = useRef(null);
  const SIZE = 240;
  const patchSize = 48;
  const grid = SIZE / patchSize; // 5
  const N = grid * grid; // 25
  const D = 32;
  const headDim = D / numHeads;

  const [heads, setHeads] = useState([]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.width = SIZE; c.height = SIZE;
    drawTestImage(c.getContext('2d'), SIZE);
    const X = computePatchFeatures(c, patchSize, D, 7);
    // Random projections at the default init scale (σ≈0.35) produce
    // dot products small enough that softmax is near-uniform (every cell
    // ≈ 1/N). For a teaching visualization we want each head's attention
    // to actually peak somewhere, so we boost the projection variance by
    // a factor that compensates for the reduced headDim. Trained models
    // get this contrast for free during training; we get it by widening
    // the random init.
    const projBoost = 2.5 * Math.sqrt(D / headDim);
    const out = [];
    for (let h = 0; h < numHeads; h++) {
      const rng = mulberry32(101 + h * 17);
      const Wq = randMatrix(D, headDim, rng);
      const Wk = randMatrix(D, headDim, rng);
      for (let i = 0; i < Wq.data.length; i++) Wq.data[i] *= projBoost;
      for (let i = 0; i < Wk.data.length; i++) Wk.data[i] *= projBoost;
      const Q = matmul(X, Wq);
      const K = matmul(X, Wk);
      const Kt = transpose(K);
      const scores = matmul(Q, Kt);
      for (let i = 0; i < scores.data.length; i++) scores.data[i] /= Math.sqrt(headDim);
      out.push(softmaxRows(scores));
    }
    setHeads(out);
  }, [numHeads, headDim]);

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Adjust "Heads (h)" to compare 1, 4, 8, 12 attention heads — each subspace is concatenated into the same total D dimensions.',
        'Click a query patch on the image. Each head\'s attention pattern lights up its own panel.',
        'Notice how different heads tend to focus on different cues (colour, position, etc.) — that is the specialisation the math allows.',
      ]}/>
      <Section icon={Network} kicker="05 — Many views, simultaneously" title="Multi-head attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          The previous tab gave us <em>one</em> attention pattern — one way of saying "patches that look
          alike attend to each other". But what counts as "alike"? Color? Texture? Position? A single
          head has to compromise across all of these.
        </p>
        <p className="max-w-3xl mb-3 text-slate-300">
          Multi-head attention runs <Eq>h</Eq> independent attention computations in parallel, each on a
          smaller subspace of dimension <Eq>{String.raw`d_k = D/h`}</Eq>, then concatenates them. Each head can
          specialize: one might track color similarity, another spatial proximity, another shape.
        </p>
        <div className="text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-4 max-w-3xl space-y-1">
          <TeXBlock>{String.raw`\text{MultiHead}(Q, K, V) = \text{Concat}\!\big(\text{head}_{1}, \ldots, \text{head}_{h}\big)\, W_{O}`}</TeXBlock>
          <TeXBlock>{String.raw`\text{head}_{i} = \text{Attention}\!\big(Q W_{Q}^{i},\, K W_{K}^{i},\, V W_{V}^{i}\big)`}</TeXBlock>
        </div>
      </Section>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            Query patch · click to change
          </div>
          <div
            className="relative w-[240px] h-[240px] rounded-lg overflow-hidden ring-1 ring-slate-700 cursor-crosshair"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const x = ((e.clientX - rect.left) / rect.width) * SIZE;
              const y = ((e.clientY - rect.top) / rect.height) * SIZE;
              const px = Math.min(grid - 1, Math.max(0, Math.floor(x / patchSize)));
              const py = Math.min(grid - 1, Math.max(0, Math.floor(y / patchSize)));
              setSelectedPatch(py * grid + px);
            }}
          >
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
            <HeadOverlay
              attn={heads[activeHead]}
              N={N}
              grid={grid}
              patchSize={patchSize}
              query={selectedPatch}
              size={SIZE}
            />
          </div>
          <div className="mt-4 space-y-3">
            <Slider label="Number of heads" value={numHeads} options={[1, 2, 4, 8]} onChange={(v) => { setNumHeads(v); setActiveHead(0); }} />
            <div className="text-[11px] font-mono text-slate-500">
              D_total = {D} · d_k per head = {headDim}
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              All heads · query #{selectedPatch}
            </div>
            {/* Always grid to a fixed 4-column layout so a single head doesn't
                blow up to fill the whole card width. */}
            <div className="grid grid-cols-4 gap-3 max-w-[480px]">
              {heads.map((h, i) => (
                <button
                  key={i}
                  onClick={() => setActiveHead(i)}
                  className={`relative rounded-lg p-2 border text-left transition-all
                    ${activeHead === i
                      ? 'bg-amber-500/10 border-amber-500/60 ring-2 ring-amber-500/30'
                      : 'bg-slate-800/40 border-slate-700 hover:border-slate-600'}`}
                >
                  <HeadMini attn={h} N={N} grid={grid} query={selectedPatch} />
                  <div className="text-[10px] font-mono text-slate-400 mt-1">head {i}</div>
                </button>
              ))}
            </div>
            <div className="mt-3 text-[11px] text-slate-500 italic leading-relaxed">
              Each head learns a different similarity function over the same patches. Some focus locally,
              others spread broadly, and some pick out specific colors or textures.
            </div>
          </Card>

          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              Concatenation
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              {heads.map((_, i) => (
                <div key={i} className="flex flex-col items-center">
                  <div className="font-mono text-[10px] text-slate-500 mb-1">head {i}</div>
                  <div className="h-12 w-8 rounded bg-gradient-to-b from-amber-500/40 to-amber-500/10 border border-amber-500/40 flex items-center justify-center">
                    <span className="font-mono text-[9px] text-amber-300 -rotate-90 whitespace-nowrap">d_k = {headDim}</span>
                  </div>
                </div>
              ))}
              <ArrowRight className="text-slate-500 mx-2" size={18} />
              <div className="flex flex-col items-center">
                <div className="font-mono text-[10px] text-slate-500 mb-1">concat · W_O</div>
                <div className="h-12 rounded bg-gradient-to-r from-rose-500/30 via-amber-500/30 to-teal-500/30 border border-amber-500/40 flex items-center justify-center px-3">
                  <span className="font-mono text-[10px] text-amber-200">D = {D}</span>
                </div>
              </div>
            </div>
            <div className="mt-3 text-[11px] text-slate-500 italic">
              The h head outputs are concatenated and linearly projected by <Eq>W_O</Eq> back to the model dimension.
            </div>
          </Card>
        </div>
      </div>

      <Takeaway
        understand={[
          'Multi-head splits D-dim attention into h independent subspaces of size d_k = D/h, run in parallel.',
          'Each head can specialize (color similarity, spatial proximity, texture, …) and their outputs are concatenated and projected by W_O.',
        ]}
        confusion={['More heads ≠ more compute per token. Total dimension D is split, not multiplied.']}
        tryIt="Change the number of heads (1/2/4/8) and watch how each head\'s attention map differs from the others."
      />
    </div>
  );
}

function HeadOverlay({ attn, N, grid, patchSize, query, size }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !attn) return;
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, size, size);
    for (let py = 0; py < grid; py++) {
      for (let px = 0; px < grid; px++) {
        const i = py * grid + px;
        const a = attn.data[query * N + i];
        ctx.fillStyle = `rgba(245, 158, 11, ${Math.min(0.85, Math.pow(a * N, 0.5) * 0.85)})`;
        ctx.fillRect(px * patchSize, py * patchSize, patchSize, patchSize);
      }
    }
    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
    for (let x = 0; x <= size; x += patchSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, size); ctx.stroke();
    }
    for (let y = 0; y <= size; y += patchSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(size, y); ctx.stroke();
    }
    // Numeric percentage labels per patch.
    const fs = Math.max(9, Math.floor(patchSize / 4.2));
    ctx.font = `${fs}px ui-monospace, Menlo, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let py = 0; py < grid; py++) {
      for (let px = 0; px < grid; px++) {
        const i = py * grid + px;
        const a = attn.data[query * N + i];
        if (a * 100 < 1) continue;
        const cx = px * patchSize + patchSize / 2;
        const cy = py * patchSize + patchSize / 2;
        const label = (a * 100).toFixed(0) + '%';
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)';
        ctx.strokeText(label, cx, cy);
        ctx.fillStyle = 'white';
        ctx.fillText(label, cx, cy);
      }
    }
    const py = Math.floor(query / grid), px = query % grid;
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 3;
    ctx.strokeRect(px * patchSize, py * patchSize, patchSize, patchSize);
  }, [attn, N, grid, patchSize, query, size]);
  return <canvas ref={ref} className="absolute inset-0 w-full h-full pointer-events-none" />;
}

function HeadMini({ attn, N, grid, query }) {
  if (!attn) return <div className="w-full aspect-square rounded bg-slate-800/50" />;
  const fontPx = grid <= 4 ? 13 : grid <= 6 ? 10 : grid <= 8 ? 8 : 7;
  return (
    <div
      className="grid gap-px bg-slate-900 p-0.5 rounded aspect-square w-full"
      style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}
    >
      {Array.from({ length: N }, (_, i) => {
        const a = attn.data[query * N + i];
        const t = Math.pow(a * N, 0.5);
        const isQuery = i === query;
        const showNum = grid <= 8 && a * 100 >= 1;
        return (
          <div
            key={i}
            className={`flex items-center justify-center font-mono leading-none text-amber-50/95
              ${isQuery ? 'ring-1 ring-rose-400 ring-inset' : ''}`}
            style={{
              background: `rgba(245, 158, 11, ${Math.min(0.92, t * 0.92)})`,
              textShadow: '0 0 2px rgba(0,0,0,0.85)',
              fontSize: `${fontPx}px`,
            }}
            title={`patch ${i}: ${(a * 100).toFixed(2)}%`}
          >
            {showNum ? (a * 100).toFixed(0) : ''}
          </div>
        );
      })}
    </div>
  );
}

/* =========================================================
   TAB 6 — ViT Pipeline
   ========================================================= */

function PipelineTab() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Per-stage delay in ms. Default slow enough that students can read each
  // panel before it advances; can speed up for re-runs.
  const [stageMs, setStageMs] = useState(3000);
  const stages = [
    { name: 'Image', desc: 'H × W × 3', icon: Box },
    { name: 'Patchify', desc: 'N × (P²·3)', icon: Grid3x3 },
    { name: 'Linear proj.', desc: 'N × D', icon: ArrowRight },
    { name: '+ [CLS] + pos', desc: '(N+1) × D', icon: Hash },
    { name: 'Encoder × L', desc: 'MSA + MLP + LN', icon: Layers },
    { name: 'CLS head', desc: 'D → C', icon: Crosshair },
    { name: 'Logits', desc: 'softmax', icon: Sparkles },
  ];

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      if (step < stages.length - 1) setStep(s => s + 1);
      else setPlaying(false);
    }, stageMs);
    return () => clearTimeout(t);
  }, [playing, step, stages.length, stageMs]);

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Click any of the 7 stage chips to jump there directly — Image → Patchify → Linear → +CLS+pos → Encoder → CLS head → Softmax.',
        'Or hit Play and watch the visual auto-advance through every stage.',
        'Drag the "Stage speed" slider to slow it down or speed up replay. Reset returns to stage 1.',
        'Stages 6–7 use real numbers — the CLS vector is multiplied by W to produce logits, then exp / Σ exp gives the displayed probabilities.',
      ]}/>
      <Section icon={Workflow} kicker="06 — End to end" title="ViT forward pass">
        <p className="max-w-3xl mb-4 text-slate-300">
          We've now built every piece individually: patch embedding, position embeddings, [CLS], and
          multi-head self-attention. Time to assemble. Step through each stage below to see how the
          tensor shapes evolve from <Eq>{String.raw`H \times W \times 3`}</Eq> pixels to <Eq>C</Eq> class logits. The encoder
          is just a stack of <Eq>L</Eq> identical blocks — each block does multi-head self-attention,
          then an MLP, both wrapped in residual connections and pre-norm LayerNorm.
        </p>
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => { setPlaying(!playing); if (step === stages.length - 1) setStep(0); }}
            className="px-4 py-2 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-200 hover:bg-amber-500/30 flex items-center gap-2 font-mono text-sm"
          >
            {playing ? <Pause size={14} /> : <Play size={14} />}
            {playing ? 'Pause' : step === stages.length - 1 ? 'Replay' : 'Play'}
          </button>
          <button
            onClick={() => { setStep(0); setPlaying(false); }}
            className="px-3 py-2 rounded-lg border border-slate-700 text-slate-400 hover:border-slate-600 flex items-center gap-2 text-sm"
          >
            <RotateCcw size={14} /> Reset
          </button>
          <span className="text-[12px] font-mono text-slate-500">
            Stage {step + 1} / {stages.length}
          </span>
          {/* Per-stage speed — students can pause-by-slowing rather than
              hammering pause every stage. */}
          <div className="flex items-center gap-2 ml-auto">
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Stage speed</label>
            <input
              type="range"
              min={400} max={5000} step={100}
              value={stageMs}
              onChange={e => setStageMs(Number(e.target.value))}
              className="w-40 h-1 accent-amber-400 bg-slate-700 rounded-lg appearance-none cursor-pointer"
              aria-label="Seconds per stage"
            />
            <span className="text-[11px] font-mono text-amber-300 tabular-nums w-14 text-right">
              {(stageMs / 1000).toFixed(1)} s/stage
            </span>
          </div>
        </div>
      </Section>

      <div className="overflow-x-auto pb-4">
        <div className="flex items-stretch gap-3 min-w-fit">
          {stages.map((s, i) => {
            const Icon = s.icon;
            const active = step === i;
            const reached = step >= i;
            return (
              <React.Fragment key={i}>
                <button
                  onClick={() => { setStep(i); setPlaying(false); }}
                  className={`flex-shrink-0 w-32 p-4 rounded-xl border text-left transition-all duration-500
                    ${active
                      ? 'bg-amber-500/15 border-amber-500/60 scale-105 shadow-lg shadow-amber-500/10'
                      : reached
                        ? 'bg-slate-800/70 border-slate-600 text-slate-200'
                        : 'bg-slate-900/30 border-slate-800 text-slate-500'}`}
                >
                  <Icon size={18} className={active ? 'text-amber-300' : reached ? 'text-slate-300' : 'text-slate-600'} />
                  <div className="text-[10px] font-mono uppercase tracking-wider opacity-60 mt-2">Stage {i + 1}</div>
                  <div className="text-sm font-medium mt-0.5">{s.name}</div>
                  <div className="text-[10px] font-mono mt-1 opacity-70">{s.desc}</div>
                </button>
                {i < stages.length - 1 && (
                  <div className="flex items-center self-center">
                    <ChevronRight size={16} className={step > i ? 'text-amber-400' : 'text-slate-600'} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/70 mb-3">
          Stage {step + 1} · live data flow
        </div>
        <PipelineVisual step={step} />
      </Card>

      <Card className="p-5 min-h-[200px]">
        <PipelineDetail step={step} />
      </Card>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Inside one Transformer encoder block</div>
        <EncoderBlockDiagram />
      </Card>

      <Takeaway
        understand={[
          'The full ViT forward pass: image → patches → linear embed → +CLS +pos → encoder × L → CLS → linear head → softmax.',
          'L identical encoder blocks share architecture (LN → MSA → residual → LN → MLP → residual) but each has its own learned weights.',
        ]}
        confusion={['"Encoder × L" doesn\'t mean L different layer types — it means L copies of the same block, stacked.']}
        tryIt="Step through stages 1→7 and watch the tensor shape evolve from image to logits."
      />
    </div>
  );
}

/* PipelineVisual — dynamic per-stage visualization. Each step shows
   the actual data transformation, not just text. The cat image enters
   at stage 1 and morphs through patches → embeddings → tokens with
   [CLS] → contextualized tokens → CLS-only readout → logits → softmax. */
function PipelineVisual({ step }) {
  const CAT = import.meta.env.BASE_URL + 'cat.jpg';

  const Bar = ({ hue = 30, w = 'w-20', highlight = false, dim = false }) => (
    <div
      className={`${w} h-2.5 rounded-sm transition-all`}
      style={{
        background: dim
          ? 'rgba(100,116,139,0.25)'
          : `linear-gradient(to right, hsl(${hue}, 70%, 60%), hsl(${(hue + 60) % 360}, 65%, 50%))`,
        opacity: dim ? 0.4 : 0.85,
        outline: highlight ? '1.5px solid rgba(244, 63, 94, 0.9)' : 'none',
        boxShadow: highlight ? '0 0 12px rgba(244, 63, 94, 0.4)' : 'none',
      }}
    />
  );

  const Cls = ({ glow = false, dim = false }) => (
    <div
      className={`w-20 h-2.5 rounded-sm flex items-center justify-center font-mono text-[7px] text-rose-50 transition-all
        ${glow ? 'ring-2 ring-rose-300 shadow-lg shadow-rose-500/40' : 'ring-1 ring-rose-400/70'}
        ${dim ? 'opacity-40' : ''}`}
      style={{ background: dim ? 'rgba(244,63,94,0.20)' : 'rgba(244,63,94,0.75)' }}
    >
      CLS
    </div>
  );

  const STACK_N = 8;

  // ---- Stage 1: Image
  if (step === 0) {
    return (
      <div className="flex items-center justify-center gap-8 min-h-[260px]">
        <div className="text-center">
          <div className="w-[200px] h-[200px] rounded-lg overflow-hidden ring-2 ring-amber-500/40">
            <img src={CAT} alt="" className="w-full h-full object-cover"/>
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">H × W × 3</div>
          <div className="font-mono text-[10px] text-slate-500">e.g. 224 × 224 × 3</div>
        </div>
      </div>
    );
  }

  // ---- Stage 2: Patchify
  if (step === 1) {
    return (
      <div className="flex items-center justify-center gap-6 min-h-[260px]">
        <div className="text-center">
          <div className="relative w-[180px] h-[180px] rounded-lg overflow-hidden ring-1 ring-slate-600">
            <img src={CAT} alt="" className="w-full h-full object-cover"/>
            <svg className="absolute inset-0 w-full h-full">
              {Array.from({ length: 13 }).map((_, i) => (
                <g key={i}>
                  <line x1={`${(i + 1) * (100 / 14)}%`} y1="0" x2={`${(i + 1) * (100 / 14)}%`} y2="100%" stroke="rgba(245,158,11,0.55)" strokeWidth="0.5"/>
                  <line x1="0" y1={`${(i + 1) * (100 / 14)}%`} x2="100%" y2={`${(i + 1) * (100 / 14)}%`} stroke="rgba(245,158,11,0.55)" strokeWidth="0.5"/>
                </g>
              ))}
            </svg>
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">14 × 14 patches</div>
        </div>
        <div className="text-amber-400 font-mono text-2xl">→</div>
        <div>
          <div className="space-y-1">
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 35} w="w-20"/>)}
            <div className="text-[10px] font-mono text-slate-500 text-center">…</div>
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">N × (P²·3)</div>
          <div className="font-mono text-[10px] text-slate-500">196 × (16²·3) flattened</div>
        </div>
      </div>
    );
  }

  // ---- Stage 3: Linear projection
  if (step === 2) {
    return (
      <div className="flex items-center justify-center gap-6 min-h-[260px]">
        <div>
          <div className="space-y-1">
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 35} w="w-20"/>)}
          </div>
          <div className="font-mono text-[11px] text-slate-400 mt-2">N × (P²·3)</div>
          <div className="font-mono text-[10px] text-slate-500">196 × (16²·3) flat patches</div>
        </div>
        <div className="text-center">
          <div className="text-amber-400 font-mono text-xl">→</div>
          <div className="px-3 py-1.5 rounded bg-amber-500/15 border border-amber-500/40 font-mono text-[11px] text-amber-200 mt-1">× E</div>
          <div className="font-mono text-[9px] text-slate-500 mt-1">linear (shared)</div>
        </div>
        <div>
          <div className="space-y-1">
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 45} w="w-24"/>)}
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">N × D</div>
          <div className="font-mono text-[10px] text-slate-500">196 × 768 (D = embed dim)</div>
        </div>
      </div>
    );
  }

  // ---- Stage 4: + [CLS] + position
  if (step === 3) {
    return (
      <div className="flex items-center justify-center gap-6 min-h-[260px]">
        <div>
          <div className="space-y-1">
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 45} w="w-24"/>)}
          </div>
          <div className="font-mono text-[11px] text-slate-400 mt-2">N × D</div>
        </div>
        <div className="text-rose-300 font-mono text-2xl">+ CLS</div>
        <div>
          <div className="space-y-1">
            <Cls />
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 45} w="w-24"/>)}
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">(N+1) × D</div>
          <div className="font-mono text-[10px] text-rose-300">[CLS] prepended</div>
        </div>
        <div className="text-teal-300 font-mono text-2xl">+ pos</div>
        <div>
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Cls />
              <div className="w-4 h-2.5 rounded-sm bg-teal-500/60" title="position 0"/>
            </div>
            {Array.from({ length: STACK_N }).map((_, i) => (
              <div key={i} className="flex items-center gap-1">
                <Bar hue={i * 50 + 45} w="w-24"/>
                <div className="w-4 h-2.5 rounded-sm" style={{ background: `hsl(${180 + i * 10}, 60%, 50%)`, opacity: 0.7 }} title={`position ${i + 1}`}/>
              </div>
            ))}
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">tokens + position</div>
          <div className="font-mono text-[10px] text-teal-300">"who came from where"</div>
        </div>
      </div>
    );
  }

  // ---- Stage 5: Encoder × L
  if (step === 4) {
    return (
      <div className="flex items-center justify-center gap-5 min-h-[260px]">
        <div className="relative">
          <div className="space-y-1">
            <Cls/>
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 45} w="w-24"/>)}
          </div>
          {/* Subtle attention web — a few lines from CLS to other tokens */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
            {[1, 3, 5, 7].map(i => (
              <line key={i}
                x1="50" y1="6"
                x2="100" y2={6 + i * 14}
                stroke="rgba(245,158,11,0.25)" strokeWidth="0.7"/>
            ))}
          </svg>
          <div className="font-mono text-[11px] text-slate-400 mt-2">in</div>
        </div>
        <div className="text-center">
          <div className="text-amber-400 font-mono text-xl">→</div>
          <div className="px-3 py-2 rounded bg-amber-500/15 border border-amber-500/40 font-mono text-[10px] text-amber-200 mt-1 leading-tight">
            <div>LayerNorm</div>
            <div>↓</div>
            <div>Multi-Head SA</div>
            <div className="text-amber-300/70">+ residual</div>
            <div>↓</div>
            <div>LayerNorm</div>
            <div>↓</div>
            <div>MLP</div>
            <div className="text-amber-300/70">+ residual</div>
          </div>
          <div className="text-[11px] font-mono text-amber-300 mt-1">× L blocks</div>
        </div>
        <div className="text-amber-400 font-mono text-xl">→</div>
        <div>
          <div className="space-y-1">
            <Cls glow/>
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 90} w="w-24"/>)}
          </div>
          <div className="font-mono text-[11px] text-rose-300 mt-2">contextualized</div>
          <div className="font-mono text-[10px] text-slate-500">[CLS] now contains<br/>info from every patch</div>
        </div>
      </div>
    );
  }

  // Concrete, deterministic numbers used by stages 6 & 7. These are stand-ins
  // for what would actually flow out of the encoder — small enough to read
  // on screen, but follow the math: logits = CLS · W ; probs = softmax(logits).
  const CLS_VEC = [+0.8, -1.2, +0.3, +1.5, -0.2, +0.6, -0.9, +1.1, +0.5, -0.7, +0.3, -0.4];
  const CLASS_LABELS = ['tabby cat', 'tiger cat', 'Egyptian cat', 'lynx', 'Persian cat'];
  const LOGITS = [+3.10, +1.32, +0.45, -0.18, -0.42];
  const expL = LOGITS.map(z => Math.exp(z));
  const Z = expL.reduce((a, b) => a + b, 0);
  const PROBS_NUM = expL.map(e => e / Z);

  // ---- Stage 6: CLS head — extract CLS vector, multiply by W, get logits.
  if (step === 5) {
    const maxAbsCls = Math.max(...CLS_VEC.map(Math.abs));
    const maxAbsLogit = Math.max(...LOGITS.map(Math.abs));
    return (
      <div className="flex items-center justify-center gap-4 min-h-[260px] flex-wrap">
        {/* (1) Encoder output stack — CLS highlighted, arrow pulls it out */}
        <div className="relative">
          <div className="space-y-1">
            <Cls glow/>
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 90} w="w-24" dim/>)}
          </div>
          {/* dashed pull arrow from CLS row out to the right */}
          <svg className="absolute -right-6 top-0 pointer-events-none" width="28" height="14" style={{ overflow: 'visible' }}>
            <defs>
              <marker id="ext-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e" />
              </marker>
            </defs>
            <path d="M 0 6 Q 14 6 26 6" stroke="#f43f5e" strokeWidth="1.4" strokeDasharray="3 2" fill="none" markerEnd="url(#ext-arr)"/>
          </svg>
          <div className="font-mono text-[11px] text-rose-300 mt-2">encoder out</div>
          <div className="font-mono text-[10px] text-slate-500">CLS only · patches ignored</div>
        </div>

        {/* (2) Extracted CLS vector — concrete signed bars, labelled */}
        <div className="text-center">
          <div className="text-[10px] font-mono uppercase tracking-wider text-rose-300 mb-1">CLS ∈ ℝ<sup>D</sup></div>
          <div className="flex flex-col gap-0.5">
            {CLS_VEC.map((v, i) => {
              const t = Math.abs(v) / maxAbsCls;
              const isPos = v >= 0;
              const w = 6 + t * 50;
              return (
                <div key={i} className="flex items-center gap-1 font-mono text-[9px]">
                  <span className="text-slate-500 w-4 text-right">{i}</span>
                  <div className="relative w-[64px] h-2.5 bg-slate-900/60 rounded-sm overflow-hidden">
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-700"/>
                    <div className="absolute top-0 bottom-0 rounded-sm" style={{
                      [isPos ? 'left' : 'right']: '50%',
                      width: `${w * 0.5}px`,
                      background: isPos ? 'rgba(245,158,11,0.85)' : 'rgba(96,165,250,0.85)',
                    }}/>
                  </div>
                  <span className={`tabular-nums w-9 text-right ${isPos ? 'text-amber-300' : 'text-blue-300'}`}>{(v >= 0 ? '+' : '') + v.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* (3) Multiply by W (classifier head) */}
        <div className="text-center flex flex-col items-center">
          <div className="text-amber-400 font-mono text-xl mb-1">×</div>
          <div className="px-3 py-2 rounded bg-amber-500/15 border border-amber-500/40 font-mono text-[11px] text-amber-200 leading-tight">
            <div>W<sub className="text-[9px]">cls</sub></div>
            <div className="text-[9px] text-slate-400 mt-0.5">D × C</div>
            <div className="text-[9px] text-slate-400">{CLS_VEC.length} × {LOGITS.length}</div>
          </div>
          <div className="text-amber-400 font-mono text-xl mt-1">↓</div>
        </div>

        {/* (4) Logits (C numbers, one per class) — explicit values + class labels */}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-300 mb-1">logits ∈ ℝ<sup>C</sup></div>
          <div className="space-y-0.5">
            {LOGITS.map((z, i) => {
              const t = Math.abs(z) / maxAbsLogit;
              const isPos = z >= 0;
              return (
                <div key={i} className="flex items-center gap-1 font-mono text-[10px]">
                  <span className="text-slate-300 w-[78px] truncate text-right">{CLASS_LABELS[i]}</span>
                  <div className="relative w-[80px] h-2.5 bg-slate-900/60 rounded-sm overflow-hidden">
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-700"/>
                    <div className="absolute top-0 bottom-0 rounded-sm" style={{
                      [isPos ? 'left' : 'right']: '50%',
                      width: `${t * 40}px`,
                      background: isPos ? 'rgba(245,158,11,0.85)' : 'rgba(96,165,250,0.85)',
                    }}/>
                  </div>
                  <span className={`tabular-nums w-9 text-right ${isPos ? 'text-amber-300' : 'text-blue-300'}`}>{(z >= 0 ? '+' : '') + z.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-1.5 italic text-center">unnormalized · still negative possible</div>
        </div>
      </div>
    );
  }

  // ---- Stage 7: Softmax — explicit logits → exp → ÷sum → probabilities.
  if (step === 6) {
    const maxAbsLogit = Math.max(...LOGITS.map(Math.abs));
    return (
      <div className="flex items-center justify-center gap-4 min-h-[260px] flex-wrap">
        {/* (1) logits in */}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-300 mb-1">logits z<sub>i</sub></div>
          <div className="space-y-0.5">
            {LOGITS.map((z, i) => {
              const t = Math.abs(z) / maxAbsLogit;
              const isPos = z >= 0;
              return (
                <div key={i} className="flex items-center gap-1 font-mono text-[10px]">
                  <span className="text-slate-300 w-[72px] truncate text-right">{CLASS_LABELS[i]}</span>
                  <div className="relative w-[64px] h-2.5 bg-slate-900/60 rounded-sm overflow-hidden">
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-slate-700"/>
                    <div className="absolute top-0 bottom-0 rounded-sm" style={{
                      [isPos ? 'left' : 'right']: '50%',
                      width: `${t * 32}px`,
                      background: isPos ? 'rgba(245,158,11,0.85)' : 'rgba(96,165,250,0.85)',
                    }}/>
                  </div>
                  <span className={`tabular-nums w-9 text-right ${isPos ? 'text-amber-300' : 'text-blue-300'}`}>{(z >= 0 ? '+' : '') + z.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* (2) exp() box */}
        <div className="text-center flex flex-col items-center">
          <div className="text-amber-400 font-mono text-xl mb-1">→</div>
          <div className="px-3 py-2 rounded bg-rose-500/15 border border-rose-500/40 font-mono text-[11px] text-rose-200 leading-tight">
            <div>exp(z<sub>i</sub>)</div>
            <div className="text-[9px] text-slate-400 mt-0.5">all positive now</div>
          </div>
        </div>

        {/* (3) exp values */}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-rose-300 mb-1">exp(z<sub>i</sub>)</div>
          <div className="space-y-0.5">
            {expL.map((e, i) => {
              const w = (e / Math.max(...expL)) * 60;
              return (
                <div key={i} className="flex items-center gap-1 font-mono text-[10px]">
                  <span className="text-slate-500 w-4 text-right">{i}</span>
                  <div className="w-[64px] h-2.5 bg-slate-900/60 rounded-sm overflow-hidden">
                    <div className="h-full rounded-sm bg-rose-400/80" style={{ width: `${w}px` }}/>
                  </div>
                  <span className="tabular-nums w-10 text-right text-rose-200">{e.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-1 text-center">
            Σ = <span className="text-amber-300">{Z.toFixed(2)}</span>
          </div>
        </div>

        {/* (4) divide-by-sum box */}
        <div className="text-center flex flex-col items-center">
          <div className="text-amber-400 font-mono text-xl mb-1">→</div>
          <div className="px-3 py-2 rounded bg-amber-500/15 border border-amber-500/40 font-mono text-[10px] text-amber-200 leading-tight whitespace-nowrap">
            <div>÷ Σ exp(z<sub>j</sub>)</div>
            <div className="text-[9px] text-slate-400 mt-0.5">rows sum to 1</div>
          </div>
        </div>

        {/* (5) probabilities — bars + percentages, top class highlighted */}
        <div className="min-w-[220px]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-300 mb-1">probabilities</div>
          <div className="space-y-1">
            {PROBS_NUM.map((p, i) => (
              <div key={i}>
                <div className="flex justify-between text-[11px] font-mono">
                  <span className={i === 0 ? 'text-amber-100 font-medium' : 'text-slate-300'}>{CLASS_LABELS[i]}</span>
                  <span className={`tabular-nums ${i === 0 ? 'text-amber-300' : 'text-slate-400'}`}>{(p * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
                  <div className={`h-full ${i === 0 ? 'bg-amber-400' : 'bg-amber-400/40'}`} style={{ width: `${p * 100}%` }}/>
                </div>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-1.5 italic">argmax → predicted class</div>
        </div>
      </div>
    );
  }

  // Fallback (should not be reachable).
  return null;
}

function PipelineDetail({ step }) {
  const details = [
    {
      title: 'Input image',
      body: (
        <>
          <p>The starting point is a raw image, typically <Eq>{String.raw`224 \times 224 \times 3`}</Eq> for ImageNet-scale models.</p>
          <p className="mt-2 text-slate-400 text-sm">Standard ImageNet preprocessing applies: resize, center crop, normalize with channel-wise mean/std.</p>
        </>
      )
    },
    {
      title: 'Patchify',
      body: (
        <>
          <p>Reshape the image into <Eq>{String.raw`N = HW / P^2`}</Eq> non-overlapping patches of size <Eq>{String.raw`P \times P \times 3`}</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">For ViT-Base at 224×224 with P=16, this produces 196 patches, each a 768-dim vector after flattening.</p>
          <p className="mt-2 font-mono text-amber-300 text-sm">In practice this is implemented as a single Conv2d with stride P and kernel P.</p>
        </>
      )
    },
    {
      title: 'Linear projection',
      body: (
        <>
          <p>Each flat patch is projected to embedding dim D via a learnable matrix <Eq>{String.raw`E \in \mathbb{R}^{P^2 C \times D}`}</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">For ViT-Base, D = 768. This linear layer is shared across all patches.</p>
        </>
      )
    },
    {
      title: 'Add [CLS] and position embeddings',
      body: (
        <>
          <p>Prepend a learnable [CLS] token, then add learnable position embeddings <Eq>{String.raw`E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}`}</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">The [CLS] token acts as a global aggregator — its final hidden state is what gets classified.</p>
        </>
      )
    },
    {
      title: 'Transformer encoder × L',
      body: (
        <>
          <p>Apply L identical blocks. Each block is:</p>
          <div className="text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-3 mt-2 space-y-1">
            <TeXBlock>{String.raw`z'_{\ell} = \text{MSA}\!\big(\text{LN}(z_{\ell-1})\big) + z_{\ell-1}`}</TeXBlock>
            <TeXBlock>{String.raw`z_{\ell}  = \text{MLP}\!\big(\text{LN}(z'_{\ell})\big) + z'_{\ell}`}</TeXBlock>
          </div>
          <p className="mt-2 text-slate-400 text-sm">ViT-Base: L=12 blocks, h=12 heads, MLP hidden dim 3072.</p>
        </>
      )
    },
    {
      title: 'CLS head',
      body: (
        <>
          <p>Take the final hidden state of the [CLS] token, apply LayerNorm, then a single linear layer to produce class logits.</p>
          <p className="mt-2 text-slate-400 text-sm">For pretraining, this head is an MLP with one hidden layer; for fine-tuning, a single linear layer is used.</p>
        </>
      )
    },
    {
      title: 'Logits',
      body: (
        <>
          <p>Apply softmax to obtain class probabilities. Cross-entropy loss against the target completes one training step.</p>
          <p className="mt-2 text-slate-400 text-sm">No special readout for spatial tasks at this stage — that's where dense-prediction heads (DETR, DPT, Mask2Former) plug in.</p>
        </>
      )
    },
  ];
  const d = details[step];
  return (
    <div>
      <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/70 mb-2">Stage {step + 1}</div>
      <h3 className="font-serif text-2xl text-slate-100 mb-4">{d.title}</h3>
      <div className="text-slate-200 leading-relaxed max-w-3xl">{d.body}</div>
    </div>
  );
}

function EncoderBlockDiagram() {
  return (
    <svg viewBox="0 0 700 280" className="w-full">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
      </defs>
      {/* input */}
      <rect x="20" y="120" width="80" height="40" rx="6" fill="#1e293b" stroke="#475569" />
      <text x="60" y="145" textAnchor="middle" fill="#cbd5e1" fontFamily="monospace" fontSize="12">z_(l-1)</text>
      {/* LN1 */}
      <rect x="140" y="120" width="60" height="40" rx="6" fill="#1e293b" stroke="#64748b" />
      <text x="170" y="145" textAnchor="middle" fill="#cbd5e1" fontFamily="monospace" fontSize="11">LN</text>
      {/* MSA */}
      <rect x="240" y="100" width="100" height="80" rx="8" fill="rgba(245,158,11,0.1)" stroke="#f59e0b" strokeWidth="1.5" />
      <text x="290" y="135" textAnchor="middle" fill="#fbbf24" fontFamily="serif" fontSize="14">Multi-Head</text>
      <text x="290" y="155" textAnchor="middle" fill="#fbbf24" fontFamily="serif" fontSize="14">Self-Attention</text>
      {/* + */}
      <circle cx="400" cy="140" r="14" fill="#0f172a" stroke="#94a3b8" strokeWidth="1.5" />
      <text x="400" y="145" textAnchor="middle" fill="#cbd5e1" fontSize="14">+</text>
      {/* skip arc */}
      <path d="M 100 130 Q 100 60 400 60 Q 400 60 400 126" fill="none" stroke="#94a3b8" strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#arr)" />
      <text x="240" y="50" fill="#94a3b8" fontFamily="monospace" fontSize="10">residual</text>
      {/* arrow img -> LN */}
      <line x1="100" y1="140" x2="138" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
      <line x1="200" y1="140" x2="238" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
      <line x1="340" y1="140" x2="386" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
      {/* LN2 */}
      <rect x="440" y="120" width="60" height="40" rx="6" fill="#1e293b" stroke="#64748b" />
      <text x="470" y="145" textAnchor="middle" fill="#cbd5e1" fontFamily="monospace" fontSize="11">LN</text>
      {/* MLP */}
      <rect x="540" y="100" width="100" height="80" rx="8" fill="rgba(20,184,166,0.1)" stroke="#14b8a6" strokeWidth="1.5" />
      <text x="590" y="135" textAnchor="middle" fill="#5eead4" fontFamily="serif" fontSize="14">MLP</text>
      <text x="590" y="155" textAnchor="middle" fill="#5eead4" fontFamily="monospace" fontSize="10">(GELU, 4×D)</text>
      {/* + */}
      <circle cx="670" cy="140" r="14" fill="#0f172a" stroke="#94a3b8" strokeWidth="1.5" />
      <text x="670" y="145" textAnchor="middle" fill="#cbd5e1" fontSize="14">+</text>
      <text x="670" y="190" textAnchor="middle" fill="#cbd5e1" fontFamily="monospace" fontSize="11">z_l</text>
      {/* skip arc 2 */}
      <path d="M 414 140 Q 414 220 670 220 Q 670 220 670 154" fill="none" stroke="#94a3b8" strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#arr)" />
      {/* arrows */}
      <line x1="414" y1="140" x2="438" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
      <line x1="500" y1="140" x2="538" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
      <line x1="640" y1="140" x2="656" y2="140" stroke="#94a3b8" strokeWidth="1.2" markerEnd="url(#arr)" />
    </svg>
  );
}

/* =========================================================
   TAB 7 — Window Attention
   ========================================================= */

function WindowTab() {
  const [winSize, setWinSize] = useState(2);
  const [hoveredWindow, setHoveredWindow] = useState(null);
  const [selectedPatch, setSelectedPatch] = useState(null);
  // Visual toggles: overlay the cat image as the underlying signal,
  // and draw actual attention edges from the query so students see
  // *which connections exist* rather than just stat numbers.
  const [showImage, setShowImage] = useState(true);
  const [showVitOverlay, setShowVitOverlay] = useState(false);
  const [showEdges, setShowEdges] = useState(true);
  const GRID = 8; // 8×8 patches
  const cell = 36;
  const CAT = import.meta.env.BASE_URL + 'cat.jpg';

  const numWindowsSide = GRID / winSize;
  const numWindows = numWindowsSide * numWindowsSide;

  // Compute complexity savings
  const fullCost = (GRID * GRID) ** 2; // O(N²)
  const windowCost = numWindows * (winSize * winSize) ** 2;

  // For the click-a-query interaction
  const queryWindow = selectedPatch == null
    ? null
    : (() => {
        const px = selectedPatch % GRID, py = Math.floor(selectedPatch / GRID);
        const wx = Math.floor(px / winSize), wy = Math.floor(py / winSize);
        return wy * numWindowsSide + wx;
      })();
  const peersInWindow = winSize * winSize - 1;
  const peersInViT = GRID * GRID - 1;

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Drag the "Window M" slider to repartition the 8×8 grid into M×M non-overlapping windows.',
        'Click any patch — teal lines fan out to its windowmates (the patches Swin attention actually connects).',
        'Flip the "ViT overlay" toggle to draw rose dashed lines to every other patch — the connections ViT would make and Swin loses.',
        'Try a corner patch vs a centre patch with M=2: corners reach only 3 neighbours; centres still only 3, but in different positions. The wall is real.',
        'Use the "Image" toggle to hide the cat and see the windowing in pure form.',
      ]}/>
      <Section icon={Box} kicker="07 — Locality is back" title="Window-based self-attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          ViT's quadratic attention cost is fine at 224 × 224, but it explodes for high-resolution work
          (segmentation, detection). For an 800 × 800 image with patch size 4, that's 40,000 tokens —
          and 1.6 billion attention pairs <em>per layer</em>. Untenable.
        </p>
        <p className="max-w-3xl text-slate-300">
          Swin's first idea: stop attending globally. Partition the patches into non-overlapping windows
          of <Eq>{String.raw`M \times M`}</Eq> tokens, and let attention happen <span className="text-teal-300">only inside
          each window</span>. Cost per layer drops to <Eq>{String.raw`O(M^2 \cdot N)`}</Eq> — linear in the number of patches.
          The next tab fixes the obvious downside: now patches across window boundaries can't talk.
        </p>
      </Section>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <Card className="p-5">
          <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
              8×8 patch grid over the cat · windows of size M
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono">
              <Toggle label="Image" value={showImage} onChange={setShowImage}/>
              <Toggle label="Edges" value={showEdges} onChange={setShowEdges}/>
              <Toggle label="ViT overlay" value={showVitOverlay} onChange={setShowVitOverlay}/>
            </div>
          </div>
          <svg width={GRID * cell} height={GRID * cell} className="rounded-lg bg-slate-950/40 border border-slate-800">
            {/* underlying cat image so windows partition something concrete */}
            {showImage && (
              <image
                href={CAT} x={0} y={0}
                width={GRID * cell} height={GRID * cell}
                preserveAspectRatio="xMidYMid slice"
                opacity={0.55}
              />
            )}
            {/* patch overlays — light hue per window, click target */}
            {Array.from({ length: GRID * GRID }).map((_, i) => {
              const px = i % GRID, py = Math.floor(i / GRID);
              const wx = Math.floor(px / winSize), wy = Math.floor(py / winSize);
              const wIdx = wy * numWindowsSide + wx;
              const isHover = hoveredWindow === wIdx;
              const isQuery = selectedPatch === i;
              const isWindowmate = selectedPatch !== null && wIdx === queryWindow && !isQuery;
              const hue = (wIdx * 360 / numWindows) % 360;
              const baseAlpha = showImage ? 0.10 : 0.18;
              const hiAlpha   = showImage ? 0.32 : 0.45;
              const fillOpacity = isHover || isWindowmate ? hiAlpha : baseAlpha;
              return (
                <g key={i}>
                  <rect
                    x={px * cell + 1} y={py * cell + 1}
                    width={cell - 2} height={cell - 2}
                    rx={3}
                    fill={`hsla(${hue}, 60%, 50%, ${fillOpacity})`}
                    stroke={`hsla(${hue}, 70%, 60%, ${isHover || isWindowmate ? 0.95 : 0.35})`}
                    strokeWidth={isHover || isWindowmate ? 1.5 : 0.6}
                    onClick={() => setSelectedPatch(prev => prev === i ? null : i)}
                    onMouseEnter={() => setHoveredWindow(wIdx)}
                    onMouseLeave={() => setHoveredWindow(null)}
                    style={{ cursor: 'pointer' }}
                  />
                  <text x={px * cell + cell / 2} y={py * cell + cell / 2 + 4}
                    textAnchor="middle" fontFamily="monospace" fontSize="10"
                    fill={showImage ? '#fefefe' : '#cbd5e1'}
                    stroke="rgba(0,0,0,0.6)" strokeWidth={showImage ? 0.4 : 0}
                    style={{ pointerEvents: 'none' }}>
                    {i}
                  </text>
                  {isQuery && (
                    <rect
                      x={px * cell + 1} y={py * cell + 1}
                      width={cell - 2} height={cell - 2}
                      rx={3}
                      fill="transparent"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      style={{ pointerEvents: 'none' }}
                    />
                  )}
                </g>
              );
            })}
            {/* window borders */}
            {Array.from({ length: numWindowsSide }).map((_, wy) =>
              Array.from({ length: numWindowsSide }).map((_, wx) => {
                const wIdx = wy * numWindowsSide + wx;
                return (
                  <rect
                    key={wIdx}
                    x={wx * winSize * cell} y={wy * winSize * cell}
                    width={winSize * cell} height={winSize * cell}
                    fill="transparent"
                    stroke={hoveredWindow === wIdx || queryWindow === wIdx ? '#5eead4' : '#14b8a6'}
                    strokeWidth={hoveredWindow === wIdx || queryWindow === wIdx ? 3 : 1.5}
                    onMouseEnter={() => setHoveredWindow(wIdx)}
                    onMouseLeave={() => setHoveredWindow(null)}
                    style={{ cursor: 'pointer', pointerEvents: 'none' }}
                  />
                );
              })
            )}
            {/* attention edges from the query patch — what attention actually connects.
                Rose dashed = what ViT (global) would draw; teal solid = Swin window only. */}
            {selectedPatch !== null && showEdges && (() => {
              const qpx = selectedPatch % GRID;
              const qpy = Math.floor(selectedPatch / GRID);
              const qcx = qpx * cell + cell / 2;
              const qcy = qpy * cell + cell / 2;
              const elements = [];
              for (let i = 0; i < GRID * GRID; i++) {
                if (i === selectedPatch) continue;
                const px = i % GRID, py = Math.floor(i / GRID);
                const wx = Math.floor(px / winSize), wy = Math.floor(py / winSize);
                const wIdx = wy * numWindowsSide + wx;
                const inWindow = wIdx === queryWindow;
                const cx = px * cell + cell / 2;
                const cy = py * cell + cell / 2;
                if (inWindow) {
                  elements.push(
                    <line key={`s${i}`} x1={qcx} y1={qcy} x2={cx} y2={cy}
                      stroke="#2dd4bf" strokeWidth={1.4} opacity={0.85} />
                  );
                } else if (showVitOverlay) {
                  elements.push(
                    <line key={`v${i}`} x1={qcx} y1={qcy} x2={cx} y2={cy}
                      stroke="#f43f5e" strokeWidth={0.7} strokeDasharray="2 3" opacity={0.55} />
                  );
                }
              }
              return elements;
            })()}
          </svg>
          <div className="mt-4">
            <Slider label="Window size M" value={winSize} options={[1, 2, 4, 8]} onChange={(v) => { setWinSize(v); setSelectedPatch(null); }} />
          </div>
          <div className="mt-3 text-[11px] text-slate-400 italic leading-snug">
            {selectedPatch == null
              ? <><span className="text-amber-300">Click any patch</span> on the cat — teal lines will fan out to its windowmates (Swin attention). Flip <span className="text-amber-300">ViT overlay</span> to see what global attention would have drawn.</>
              : showVitOverlay
                ? <>The teal lines are <span className="text-teal-300">{peersInWindow}</span> Swin edges; the rose dashed lines are <span className="text-rose-300">{peersInViT - peersInWindow}</span> extra edges ViT would have drawn — every one is a pair of patches Swin <em>cannot</em> directly connect.</>
                : <>Teal lines = the <span className="text-teal-300">{peersInWindow}</span> windowmates this query attends to. Toggle <span className="text-amber-300">ViT overlay</span> to compare against global attention.</>}
          </div>
        </Card>

        <div className="space-y-4">
          {selectedPatch !== null && (
            <Card className="p-5 border-amber-500/40 bg-amber-500/[0.04]">
              <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/80 mb-2">
                Query patch #{selectedPatch}
              </div>
              <div className="grid grid-cols-2 gap-3 text-[13px]">
                <div>
                  <div className="text-teal-300 font-mono text-[11px]">Swin (this layer)</div>
                  <div className="text-slate-100 font-serif text-2xl">{peersInWindow}</div>
                  <div className="text-slate-500 text-[11px]">peers in window of M={winSize}</div>
                </div>
                <div>
                  <div className="text-rose-300 font-mono text-[11px]">ViT (same layer)</div>
                  <div className="text-slate-100 font-serif text-2xl">{peersInViT}</div>
                  <div className="text-slate-500 text-[11px]">peers (every other patch)</div>
                </div>
              </div>
              <p className="text-[11px] text-slate-400 italic mt-3 leading-snug">
                Swin trades reach for cost. The <em>shifted windows</em> tab shows how it claws that reach back.
              </p>
            </Card>
          )}
          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Cost comparison</div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Tag color="rose">Full (ViT)</Tag>
                <div className="font-serif text-3xl text-rose-300 mt-2">{fullCost.toLocaleString()}</div>
                <div className="font-mono text-[11px] text-slate-500">N² pairs · 64 tokens</div>
              </div>
              <div>
                <Tag color="teal">Windowed (Swin)</Tag>
                <div className="font-serif text-3xl text-teal-300 mt-2">{windowCost.toLocaleString()}</div>
                <div className="font-mono text-[11px] text-slate-500">{numWindows} windows × {winSize * winSize}² pairs</div>
              </div>
            </div>
            <div className="mt-4 h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-teal-500 to-amber-500"
                style={{ width: `${(windowCost / fullCost) * 100}%` }}
              />
            </div>
            <div className="mt-2 text-[12px] font-mono text-slate-400">
              {(windowCost / fullCost * 100).toFixed(1)}% of global attention cost
            </div>
          </Card>

          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">The problem this creates</div>
            <p className="text-slate-300 text-sm leading-relaxed">
              Windowed attention is fast — but if information can never leave its window, the receptive field
              is permanently <span className="text-rose-300">{winSize}×{winSize}</span>. Stacking more layers
              wouldn't change anything: every layer sees the exact same window boundaries.
            </p>
            <p className="text-slate-300 text-sm leading-relaxed mt-3">
              Swin's solution is the next idea on the list: <span className="text-amber-300">shifted windows</span>.
              In every other layer, the window grid is offset by half a window, so patches that were on opposite
              sides of a wall now share one. Receptive field grows; cost stays linear.
            </p>
          </Card>
        </div>
      </div>

      <Takeaway
        understand={[
          'Swin partitions patches into non-overlapping M×M windows; attention only operates inside each window.',
          'Cost drops from O(N²) (ViT) to O(M² · N) — linear in the number of patches, fixed per-window cost.',
        ]}
        confusion={['Windows ≠ patches. Each window contains many patches; M is the window size.']}
        tryIt="Compare ViT and Swin matrices in the Live Demo — Swin\'s is block-diagonal, ViT\'s is dense."
      />
    </div>
  );
}

/* =========================================================
   TAB 8 — Shifted Windows (animated)
   ========================================================= */

function ShiftedTab() {
  const [layer, setLayer] = useState(0); // 0 = W-MSA, 1 = SW-MSA
  const [auto, setAuto] = useState(true);
  const [showCyclic, setShowCyclic] = useState(true);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const GRID = 8;
  const M = 4; // window size
  const shift = M / 2; // 2
  const cell = 38;

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => setLayer(l => 1 - l), 3000);
    return () => clearInterval(t);
  }, [auto]);

  // For each patch (px, py), compute its window index in current layer
  const windowOf = (px, py, shifted) => {
    if (!shifted) return Math.floor(py / M) * Math.ceil(GRID / M) + Math.floor(px / M);
    // shifted: subtract shift, allow negative window indices that wrap
    const sx = px - shift, sy = py - shift;
    return `${Math.floor(sy / M)}_${Math.floor(sx / M)}`;
  };

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Click the W-MSA / SW-MSA buttons to switch between regular and shifted windows manually.',
        'Or leave Auto-alternate on (default) — every 3 s the windows flip between the two layers.',
        'Click any patch to fix it as the query and watch which windowmates change after the shift.',
        'The "Cyclic-shift trick" pane shows the implementation that keeps SW-MSA the same cost as W-MSA.',
      ]}/>
      <Section icon={Move} kicker="08 — Cross-window connections" title="Shifted window attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          The previous tab introduced a problem: with strict windows, two patches that sit on either
          side of a window boundary <em>never</em> attend to each other, no matter how many layers we
          stack. That kills the global reasoning we got from ViT for free.
        </p>
        <p className="max-w-3xl mb-3 text-slate-300">
          Swin's fix: alternate two kinds of layers. Layer <Eq>{String.raw`\ell`}</Eq> uses regular window partitioning
          (<span className="font-mono text-amber-300">W-MSA</span>). Layer <Eq>{String.raw`\ell+1`}</Eq> shifts the window
          grid by <Eq>{String.raw`(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)`}</Eq> pixels (<span className="font-mono text-amber-300">SW-MSA</span>).
          Patches that were neighbors-across-a-wall in layer <Eq>{String.raw`\ell`}</Eq> now share a window in layer
          <Eq>{String.raw`\ell+1`}</Eq>.
        </p>
        <p className="max-w-3xl text-slate-300">
          After two layers, every patch has effectively communicated with everything in a
          <Eq>{String.raw`2M \times 2M`}</Eq> region. Stack more, and the receptive field keeps growing — still at linear cost.
        </p>
      </Section>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <button
          onClick={() => setLayer(0)}
          className={`px-4 py-2 rounded-lg border font-mono text-sm transition-all
            ${layer === 0 ? 'bg-amber-500/20 border-amber-500/60 text-amber-200' : 'border-slate-700 text-slate-400'}`}
        >
          Layer l · W-MSA
        </button>
        <button
          onClick={() => setLayer(1)}
          className={`px-4 py-2 rounded-lg border font-mono text-sm transition-all
            ${layer === 1 ? 'bg-amber-500/20 border-amber-500/60 text-amber-200' : 'border-slate-700 text-slate-400'}`}
        >
          Layer l+1 · SW-MSA
        </button>
        <button
          onClick={() => setAuto(!auto)}
          className="px-3 py-2 rounded-lg bg-teal-500/15 border border-teal-500/40 text-teal-200 font-mono text-sm flex items-center gap-2"
        >
          {auto ? <Pause size={14} /> : <Play size={14} />}
          Auto-alternate
        </button>
        <Toggle label="Cyclic shift trick" value={showCyclic} onChange={setShowCyclic} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            8×8 patches · M = {M}, shift = ({shift}, {shift})
          </div>
          <svg width={GRID * cell} height={GRID * cell} className="rounded-lg bg-slate-950/40 border border-slate-800">
            {Array.from({ length: GRID * GRID }).map((_, i) => {
              const px = i % GRID, py = Math.floor(i / GRID);
              const w = windowOf(px, py, layer === 1);
              const wHash = String(w);
              let h = 0;
              for (let k = 0; k < wHash.length; k++) h = (h * 31 + wHash.charCodeAt(k)) % 360;
              const isQuery = selectedPatch === i;
              // Determine if this patch shares a window with the selected query.
              const queryWin = selectedPatch == null
                ? null
                : windowOf(selectedPatch % GRID, Math.floor(selectedPatch / GRID), layer === 1);
              const isPeer = !isQuery && queryWin != null && String(w) === String(queryWin);
              const fillAlpha = isQuery ? 0.55 : isPeer ? 0.55 : selectedPatch == null ? 0.25 : 0.10;
              return (
                <g key={i} style={{ transition: 'all 0.5s ease' }}>
                  <rect
                    x={px * cell + 1} y={py * cell + 1}
                    width={cell - 2} height={cell - 2}
                    rx={3}
                    fill={`hsla(${h}, 60%, 50%, ${fillAlpha})`}
                    stroke={`hsla(${h}, 70%, 60%, ${isPeer || isQuery ? 0.95 : 0.5})`}
                    strokeWidth={isPeer || isQuery ? 1.4 : 0.8}
                    onClick={() => setSelectedPatch(prev => prev === i ? null : i)}
                    style={{ transition: 'all 0.5s ease', cursor: 'pointer' }}
                  />
                  <text x={px * cell + cell / 2} y={py * cell + cell / 2 + 4}
                    textAnchor="middle" fontFamily="monospace" fontSize="10" fill="#cbd5e1"
                    style={{ pointerEvents: 'none' }}>
                    {i}
                  </text>
                  {isQuery && (
                    <rect
                      x={px * cell + 1} y={py * cell + 1}
                      width={cell - 2} height={cell - 2}
                      rx={3}
                      fill="transparent"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      style={{ pointerEvents: 'none' }}
                    />
                  )}
                </g>
              );
            })}
            {/* draw thick window borders */}
            {layer === 0 && (
              [...Array(GRID / M)].map((_, wy) => [...Array(GRID / M)].map((_, wx) => (
                <rect key={`${wx}-${wy}`}
                  x={wx * M * cell} y={wy * M * cell}
                  width={M * cell} height={M * cell}
                  fill="transparent" stroke="#fbbf24" strokeWidth="2"
                  style={{ transition: 'all 0.5s ease' }} />
              )))
            )}
            {layer === 1 && (
              <>
                {/* edges of the shifted-by-2 windows: lines at x=0,2,6 and y=0,2,6 (mod 8 wraps) */}
                {[0, 2, 6].map(x => (
                  <line key={`v${x}`} x1={x * cell} y1={0} x2={x * cell} y2={GRID * cell}
                    stroke="#fbbf24" strokeWidth="2" style={{ transition: 'all 0.5s ease' }} />
                ))}
                {[0, 2, 6].map(y => (
                  <line key={`h${y}`} x1={0} y1={y * cell} x2={GRID * cell} y2={y * cell}
                    stroke="#fbbf24" strokeWidth="2" style={{ transition: 'all 0.5s ease' }} />
                ))}
                <rect x={0} y={0} width={GRID * cell} height={GRID * cell} fill="transparent" stroke="#fbbf24" strokeWidth="2" />
              </>
            )}
          </svg>
          <div className="mt-3 text-[11px] text-slate-400 italic leading-snug">
            <span className="text-amber-300">Click any patch</span> to fix it as the query. Toggle layers above
            and watch its windowmates change.
          </div>
          {selectedPatch !== null && (() => {
            const px = selectedPatch % GRID, py = Math.floor(selectedPatch / GRID);
            const wReg = windowOf(px, py, false);
            const wShifted = windowOf(px, py, true);
            // Count peers in each layer
            let peersReg = 0, peersShifted = 0;
            for (let i = 0; i < GRID * GRID; i++) {
              if (i === selectedPatch) continue;
              const ipx = i % GRID, ipy = Math.floor(i / GRID);
              if (String(windowOf(ipx, ipy, false)) === String(wReg)) peersReg++;
              if (String(windowOf(ipx, ipy, true)) === String(wShifted)) peersShifted++;
            }
            // Count "newly visible" peers — those that share a window in shifted but NOT in regular
            let newlyVisible = 0;
            for (let i = 0; i < GRID * GRID; i++) {
              if (i === selectedPatch) continue;
              const ipx = i % GRID, ipy = Math.floor(i / GRID);
              const peerWReg = String(windowOf(ipx, ipy, false));
              const peerWSh = String(windowOf(ipx, ipy, true));
              if (peerWSh === String(wShifted) && peerWReg !== String(wReg)) newlyVisible++;
            }
            return (
              <div className="mt-3 p-3 rounded border border-amber-500/40 bg-amber-500/[0.04]">
                <div className="text-[10px] font-mono uppercase tracking-wider text-amber-400/80 mb-2">
                  Query patch #{selectedPatch}
                </div>
                <div className="grid grid-cols-2 gap-3 text-[12px]">
                  <div>
                    <div className="text-amber-300 font-mono text-[10px]">In W-MSA (this layer = {layer === 0 ? 'now' : 'before shift'})</div>
                    <div className="text-slate-100 font-serif text-xl mt-0.5">{peersReg}</div>
                    <div className="text-slate-500 text-[10px]">peers in window</div>
                  </div>
                  <div>
                    <div className="text-teal-300 font-mono text-[10px]">In SW-MSA (this layer = {layer === 1 ? 'now' : 'after shift'})</div>
                    <div className="text-slate-100 font-serif text-xl mt-0.5">{peersShifted}</div>
                    <div className="text-slate-500 text-[10px]">peers in window</div>
                  </div>
                </div>
                <div className="mt-2 text-[11px] text-slate-300 leading-snug">
                  After the shift, this query meets <span className="text-amber-300 font-medium">{newlyVisible}</span> patches it never saw before — that's how info crosses old window boundaries.
                </div>
              </div>
            );
          })()}
        </Card>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            What's actually happening
          </div>
          {layer === 0 ? (
            <div className="space-y-3 text-slate-300 text-sm leading-relaxed">
              <p><span className="text-amber-300 font-medium">W-MSA — regular partition.</span> The image is split into a clean 2×2 grid of 4×4 windows. Each window's 16 patches do self-attention only among themselves.</p>
              <p>Patch <span className="font-mono text-amber-200">3</span> (top-right of upper-left window) and patch <span className="font-mono text-amber-200">4</span> (top-left of upper-right window) are spatially adjacent <span className="text-rose-300">but never see each other</span>.</p>
            </div>
          ) : (
            <div className="space-y-3 text-slate-300 text-sm leading-relaxed">
              <p><span className="text-amber-300 font-medium">SW-MSA — shifted by ({shift},{shift}).</span> The grid offsets, so windows now straddle the original boundaries. Some windows on the edges become smaller fragments.</p>
              <p>Patch <span className="font-mono text-amber-200">3</span> and patch <span className="font-mono text-amber-200">4</span> now share a window — they finally see each other. Information leaks across the old walls.</p>
            </div>
          )}

          {showCyclic && (
            <div className="mt-5 pt-5 border-t border-slate-700">
              <div className="text-[11px] font-mono uppercase tracking-wider text-teal-400 mb-2">Implementation: cyclic shift</div>
              <p className="text-slate-300 text-sm leading-relaxed">
                Shifted windows would naïvely create more, irregularly-shaped windows on the borders.
                Swin's trick: <span className="text-teal-300">cyclically shift</span> the entire feature
                map by <Eq>{String.raw`(-\lfloor M/2 \rfloor, -\lfloor M/2 \rfloor)`}</Eq>, do regular W-MSA, then shift back. An
                <span className="text-teal-300"> attention mask</span> prevents the wrap-around patches
                from attending to each other (they aren't actually adjacent in the image).
              </p>
              <pre className="mt-2 font-mono text-[11px] text-slate-400 bg-slate-950/60 border border-slate-800 rounded p-2 overflow-x-auto">{`x = torch.roll(x, shifts=(-M//2, -M//2), dims=(1,2))
x = window_partition(x, M)
attn = MSA(x, mask=cyclic_mask)
x = window_reverse(attn, M)
x = torch.roll(x, shifts=(M//2, M//2), dims=(1,2))`}</pre>
            </div>
          )}
        </Card>
      </div>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Receptive field growth</div>
        <div className="grid grid-cols-4 gap-3">
          {[
            { l: 1, label: 'After 1 layer', desc: 'M × M', size: 1 },
            { l: 2, label: 'After 2 layers', desc: 'crosses one boundary', size: 1.4 },
            { l: 4, label: 'After 4 layers', desc: 'plus a stage with merging', size: 2 },
            { l: 8, label: 'Late stages', desc: 'effectively global', size: 3 },
          ].map((s, i) => (
            <div key={i} className="bg-slate-950/40 border border-slate-800 rounded-lg p-3 flex flex-col items-center gap-2">
              <div className="relative w-full aspect-square">
                <div className="absolute inset-0 grid grid-cols-8 gap-0.5">
                  {Array.from({ length: 64 }).map((_, k) => {
                    const px = k % 8, py = Math.floor(k / 8);
                    const dx = Math.abs(px - 3.5), dy = Math.abs(py - 3.5);
                    const dist = Math.max(dx, dy);
                    const inside = dist <= s.size * 1.2;
                    return (
                      <div key={k}
                        className={`rounded-sm ${inside ? 'bg-amber-500/60' : 'bg-slate-800/60'}`}
                        style={{ transition: 'background 0.4s' }} />
                    );
                  })}
                </div>
              </div>
              <div className="text-center">
                <div className="font-mono text-[11px] text-amber-300">{s.label}</div>
                <div className="font-mono text-[10px] text-slate-500">{s.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Takeaway
        understand={[
          'Swin alternates W-MSA (regular windows) with SW-MSA (windows shifted by ⌊M/2⌋, ⌊M/2⌋).',
          'After two layers, every patch has effectively communicated with everything in a 2M×2M region — global reasoning at linear cost.',
        ]}
        confusion={['Shifting doesn\'t increase compute. The cyclic-shift + masking trick keeps the kernel identical to W-MSA.']}
        tryIt="Toggle between regular and shifted layers and watch which patches share a window."
      />
    </div>
  );
}

/* =========================================================
   TAB 9 — Hierarchy (Patch Merging)
   ========================================================= */

function HierarchyTab() {
  const [stage, setStage] = useState(0);
  const stages = [
    { name: 'Stage 1', resolution: '56×56', tokens: 3136, dim: 96, blocks: 2 },
    { name: 'Stage 2', resolution: '28×28', tokens: 784, dim: 192, blocks: 2 },
    { name: 'Stage 3', resolution: '14×14', tokens: 196, dim: 384, blocks: 6 },
    { name: 'Stage 4', resolution: '7×7', tokens: 49, dim: 768, blocks: 2 },
  ];

  return (
    <div className="space-y-8">
      <HowToUseBadge instructions={[
        'Pick one of the four Swin-T stages (1 → 4) at the top — its resolution and token count update on the right.',
        'In the Patch-merge playground, hover any merged "M" tile to see the four pre-merge patches that produced it.',
        'Read the bottom flow diagram for the end-to-end Swin-T architecture: image → 4 stages → GAP+FC.',
      ]}/>
      <Section icon={GitBranch} kicker="09 — Pyramid features" title="Patch merging & hierarchical stages">
        <p className="max-w-3xl mb-3 text-slate-300">
          We've made attention efficient (windows) and global (shifted windows). One thing is still
          missing: ViT runs at a <em>single resolution</em> the whole way through. Classification gets
          away with this, but detection and segmentation need features at multiple scales — like a
          ConvNet's pyramid.
        </p>
        <p className="max-w-3xl text-slate-300">
          Swin's last ingredient: a <span className="text-teal-300">patch-merging</span> layer between
          stages. Take every <Eq>{String.raw`2 \times 2`}</Eq> group of patches, concatenate their channels (<Eq>4C</Eq>),
          then linearly project back to <Eq>2C</Eq>. Resolution halves; channels double; receptive field
          doubles. After four stages we have a CNN-style feature pyramid, but built from attention.
        </p>
      </Section>

      <Card className="p-4">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Swin-T stages — pick one</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {stages.map((s, i) => (
            <button
              key={i}
              onClick={() => setStage(i)}
              className={`text-left p-3 rounded-lg border transition-all
                ${stage === i
                  ? 'bg-teal-500/15 border-teal-500/60'
                  : 'bg-slate-800/40 border-slate-700 hover:border-slate-600'}`}
            >
              <div className="flex items-baseline justify-between">
                <span className={`font-medium ${stage === i ? 'text-teal-200' : 'text-slate-200'}`}>{s.name}</span>
                <span className="font-mono text-[11px] text-slate-500">×{s.blocks}</span>
              </div>
              <div className="font-mono text-[11px] mt-1 leading-snug">
                <div><span className="text-amber-300">{s.resolution}</span> <span className="text-slate-500">·</span> <span className="text-slate-300">{s.dim}-dim</span></div>
                <div className="text-slate-400">{s.tokens.toLocaleString()} tokens</div>
              </div>
            </button>
          ))}
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Resolution at this stage</div>
          <ResolutionViz stage={stage} />
        </Card>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/80 mb-3">
            Try it · hover a merged patch to see its 4 sources
          </div>
          <PatchMergePlayground />
        </Card>
      </div>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Patch merging — how it works</div>
        <div className="grid lg:grid-cols-[minmax(0,260px)_minmax(0,1fr)] gap-5 items-center">
          <div className="text-[12px] text-slate-300 leading-relaxed space-y-2">
            <p>
              Take every non-overlapping <Eq>{String.raw`2 \times 2`}</Eq> block of tokens. Concatenate the
              four <Eq>C</Eq>-dim vectors into a single <Eq>4C</Eq> vector,
              apply LayerNorm, then project to <Eq>2C</Eq>.
            </p>
            <ul className="text-[11px] text-slate-400 list-disc pl-4 space-y-0.5">
              <li>tokens <span className="text-amber-300">× ¼</span></li>
              <li>channels <span className="text-amber-300">× 2</span></li>
              <li>receptive field <span className="text-amber-300">× 2</span></li>
            </ul>
          </div>
          <PatchMergeDiagram />
        </div>
      </Card>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Full Swin-T architecture flow</div>
        <SwinFlow stages={stages} highlight={stage} />
      </Card>

      <Takeaway
        understand={[
          'Patch merging concatenates 2×2 neighbor patches (4C channels) and projects back to 2C — resolution halves, channels double.',
          'After 4 stages Swin produces a CNN-style feature pyramid (56×56 → 28×28 → 14×14 → 7×7), which is what dense prediction heads expect.',
        ]}
        confusion={['ViT keeps a single resolution throughout. Hierarchy is what makes Swin a usable backbone for detection / segmentation.']}
        tryIt="No interactive analog in the Live Demo (it stays at one stage). The flow diagram on this tab is the demo."
      />
    </div>
  );
}

function ResolutionViz({ stage }) {
  const sizes = [56, 28, 14, 7];
  const size = sizes[stage];
  const containerSize = 280;
  const cellSize = containerSize / size;

  return (
    <div className="flex items-center justify-center">
      <div className="relative" style={{ width: containerSize, height: containerSize }}>
        <div
          className="grid gap-px bg-slate-800 rounded-lg overflow-hidden"
          style={{
            gridTemplateColumns: `repeat(${size}, 1fr)`,
            width: containerSize, height: containerSize,
            transition: 'all 0.5s'
          }}
        >
          {Array.from({ length: size * size }).map((_, i) => {
            const py = Math.floor(i / size), px = i % size;
            const hue = ((px + py) * 360 / (size * 2)) % 360;
            return (
              <div
                key={i}
                style={{
                  background: `hsla(${hue}, 60%, ${30 + (px + py) % 50}%, 0.7)`,
                  transition: 'background 0.4s'
                }}
              />
            );
          })}
        </div>
        <div className="absolute -bottom-7 left-1/2 -translate-x-1/2 font-mono text-[12px] text-amber-300">
          {size} × {size} = {size * size} tokens
        </div>
      </div>
    </div>
  );
}

/* PatchMergePlayground — hover any merged "M" patch to see which 4
   pre-merge patches went into it. Concrete demonstration that one
   post-merge patch is a learned blend of a 2×2 neighborhood. */
function PatchMergePlayground() {
  const [hover, setHover] = useState(null);
  return (
    <div className="flex items-center justify-center gap-5 flex-wrap">
      {/* Before: 4×4 grid of patches */}
      <div>
        <div className="text-[10px] font-mono text-slate-400 mb-1.5 text-center">before · 4×4</div>
        <div className="grid grid-cols-4 gap-0.5 p-1 rounded bg-slate-950 border border-slate-800">
          {Array.from({ length: 16 }).map((_, i) => {
            const px = i % 4, py = (i / 4) | 0;
            const gx = (px / 2) | 0, gy = (py / 2) | 0;
            const groupId = gy * 2 + gx;
            const isHi = hover === groupId;
            const hue = (groupId * 90) % 360;
            return (
              <div
                key={i}
                className={`w-9 h-9 rounded-sm flex items-center justify-center font-mono text-[10px] transition-all
                  ${isHi ? 'ring-2 ring-amber-300 z-10 scale-105' : 'ring-1 ring-slate-700/40'}`}
                style={{
                  background: `hsla(${hue + (i % 4) * 12}, ${isHi ? 70 : 45}%, 50%, ${isHi ? 0.75 : 0.30})`,
                  color: isHi ? '#fefefe' : 'rgba(203,213,225,0.7)',
                }}
              >
                {i}
              </div>
            );
          })}
        </div>
        <div className="text-[10px] font-mono text-slate-500 mt-1.5 text-center">N = 16 · C dims</div>
      </div>

      {/* Arrow + math caption */}
      <div className="flex flex-col items-center">
        <div className="text-amber-400/70 font-mono text-2xl">→</div>
        <div className="text-[9px] font-mono text-slate-500 mt-1 text-center leading-snug">
          concat 4·C<br/>LayerNorm<br/>Linear → 2C
        </div>
      </div>

      {/* After: 2×2 merged patches */}
      <div>
        <div className="text-[10px] font-mono text-slate-400 mb-1.5 text-center">after · 2×2</div>
        <div className="grid grid-cols-2 gap-1 p-1 rounded bg-slate-950 border border-slate-800">
          {Array.from({ length: 4 }).map((_, gi) => {
            const isHi = hover === gi;
            const hue = (gi * 90) % 360;
            return (
              <button
                key={gi}
                onMouseEnter={() => setHover(gi)}
                onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(gi)}
                onBlur={() => setHover(null)}
                className={`w-[78px] h-[78px] rounded-sm flex items-center justify-center font-mono text-[12px] transition-all cursor-pointer
                  ${isHi ? 'ring-2 ring-amber-300 z-10 scale-105' : 'ring-1 ring-slate-700/40'}`}
                style={{
                  background: `hsla(${hue}, ${isHi ? 70 : 50}%, 50%, ${isHi ? 0.75 : 0.30})`,
                  color: isHi ? '#fefefe' : 'rgba(203,213,225,0.85)',
                }}
              >
                merged {gi}
              </button>
            );
          })}
        </div>
        <div className="text-[10px] font-mono text-slate-500 mt-1.5 text-center">N = 4 · 2C dims</div>
      </div>
    </div>
  );
}

function PatchMergeDiagram() {
  return (
    <svg viewBox="0 0 600 220" className="w-full">
      <defs>
        <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
      </defs>
      {/* before: 4×4 grid */}
      <g transform="translate(20, 30)">
        <text x="60" y="-5" fill="#94a3b8" fontFamily="monospace" fontSize="10">H × W × C</text>
        {[0, 1, 2, 3].map(y => [0, 1, 2, 3].map(x => {
          const blockX = Math.floor(x / 2), blockY = Math.floor(y / 2);
          const blockId = blockY * 2 + blockX;
          const hues = [25, 175, 280, 50];
          return (
            <rect key={`${x}-${y}`} x={x * 28} y={y * 28} width={26} height={26} rx="3"
              fill={`hsla(${hues[blockId]}, 60%, 50%, 0.4)`}
              stroke={`hsla(${hues[blockId]}, 70%, 60%, 0.8)`}
              strokeWidth="1"
            />
          );
        }))}
        {/* highlight 2x2 groups */}
        {[0, 1].map(by => [0, 1].map(bx => (
          <rect key={`b${bx}-${by}`} x={bx * 56} y={by * 56} width={54} height={54}
            fill="transparent" stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="3 2" rx="3" />
        )))}
      </g>

      {/* arrow */}
      <path d="M 145 80 L 195 80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr2)" />
      <text x="170" y="73" textAnchor="middle" fill="#94a3b8" fontFamily="monospace" fontSize="9">concat</text>
      <text x="170" y="95" textAnchor="middle" fill="#94a3b8" fontFamily="monospace" fontSize="9">2×2</text>

      {/* concat: 4 stacks side by side */}
      <g transform="translate(210, 30)">
        <text x="40" y="-5" fill="#94a3b8" fontFamily="monospace" fontSize="10">H/2 × W/2 × 4C</text>
        {[0, 1].map(y => [0, 1].map(x => {
          const blockId = y * 2 + x;
          const hues = [25, 175, 280, 50];
          return (
            <g key={`c${x}-${y}`}>
              {[0, 1, 2, 3].map(d => (
                <rect key={d} x={x * 50 + d * 2} y={y * 50 + d * 2} width={36} height={36} rx="2"
                  fill={`hsla(${hues[blockId]}, 60%, ${40 + d * 8}%, 0.6)`}
                  stroke={`hsla(${hues[blockId]}, 70%, 60%, 0.9)`}
                  strokeWidth="0.8"
                />
              ))}
            </g>
          );
        }))}
      </g>

      {/* arrow */}
      <path d="M 360 80 L 410 80" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr2)" />
      <text x="385" y="73" textAnchor="middle" fill="#94a3b8" fontFamily="monospace" fontSize="9">LN +</text>
      <text x="385" y="95" textAnchor="middle" fill="#94a3b8" fontFamily="monospace" fontSize="9">Linear 4C→2C</text>

      {/* after */}
      <g transform="translate(425, 30)">
        <text x="40" y="-5" fill="#94a3b8" fontFamily="monospace" fontSize="10">H/2 × W/2 × 2C</text>
        {[0, 1].map(y => [0, 1].map(x => {
          const blockId = y * 2 + x;
          const hues = [25, 175, 280, 50];
          return (
            <g key={`o${x}-${y}`}>
              {[0, 1].map(d => (
                <rect key={d} x={x * 50 + d * 2} y={y * 50 + d * 2} width={42} height={42} rx="2"
                  fill={`hsla(${hues[blockId]}, 60%, ${40 + d * 12}%, 0.7)`}
                  stroke={`hsla(${hues[blockId]}, 70%, 60%, 0.9)`}
                  strokeWidth="0.8"
                />
              ))}
            </g>
          );
        }))}
      </g>
    </svg>
  );
}

function SwinFlow({ stages, highlight }) {
  return (
    <div className="overflow-x-auto">
      <div className="flex items-center gap-2 min-w-fit pb-3">
        <StageBlock label="Image" sub="224×224×3" color="slate" />
        <SmallArrow label="Patch Embed" />
        {stages.map((s, i) => (
          <React.Fragment key={i}>
            <StageBlock
              label={s.name}
              sub={`${s.resolution} · ${s.dim}d · ×${s.blocks} W/SW-MSA`}
              color={highlight === i ? 'amber' : 'teal'}
              active={highlight === i}
            />
            {i < stages.length - 1 && <SmallArrow label="Patch Merge" />}
          </React.Fragment>
        ))}
        <SmallArrow label="GAP + FC" />
        <StageBlock label="Logits" sub="C classes" color="rose" />
      </div>
    </div>
  );
}

function StageBlock({ label, sub, color, active }) {
  const palette = {
    slate: 'bg-slate-800/60 border-slate-600 text-slate-200',
    teal: 'bg-teal-500/10 border-teal-500/40 text-teal-100',
    amber: 'bg-amber-500/15 border-amber-500/60 text-amber-100 scale-110 shadow-lg shadow-amber-500/20',
    rose: 'bg-rose-500/10 border-rose-500/40 text-rose-100',
  };
  return (
    <div className={`rounded-lg border px-3 py-3 min-w-[120px] transition-all duration-300 ${palette[color]}`}>
      <div className="font-medium text-sm">{label}</div>
      <div className="font-mono text-[10px] mt-0.5 opacity-80">{sub}</div>
    </div>
  );
}

function SmallArrow({ label }) {
  return (
    <div className="flex flex-col items-center px-1 flex-shrink-0">
      <ChevronRight size={16} className="text-slate-500" />
      <div className="font-mono text-[9px] text-slate-500 mt-0.5">{label}</div>
    </div>
  );
}

/* =========================================================
   TAB 10 — Comparison
   ========================================================= */

function CompareTab() {
  const [imgSize, setImgSize] = useState(224);
  const [patchSize, setPatchSize] = useState(16);
  const [winSize, setWinSize] = useState(7);
  const [embDim, setEmbDim] = useState(96);

  const N = (imgSize / patchSize) ** 2;
  const D_vit = 768;
  // Self-attention: 4·N·D² + 2·N²·D
  const vitAttnFLOPs = 4 * N * D_vit * D_vit + 2 * N * N * D_vit;
  // Swin: 4·hw·C² + 2·M²·hw·C  (per stage; we'll show it for stage 1)
  const swinAttnFLOPs = 4 * N * embDim * embDim + 2 * winSize * winSize * N * embDim;

  return (
    <div className="space-y-8">
      <Section icon={Microscope} kicker="10 — When to choose which" title="Side-by-side comparison">
        <p className="max-w-3xl mb-3 text-slate-300">
          We started with ViT (global, simple, expensive) and built up to Swin (local + shifted +
          hierarchical, cheaper, more practical). Both models work — they just optimize different things.
        </p>
        <p className="max-w-3xl text-slate-300">
          ViT is conceptually purer and scales beautifully when you have huge datasets and modest
          resolutions. Swin is the better default for real-world dense-prediction tasks (detection,
          segmentation) and trains well at smaller scales. The cost calculator below makes the
          quadratic-vs-linear gap concrete.
        </p>
      </Section>

      <Card className="p-5">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Live cost estimator (stage-1 attention only)</div>
        <div className="grid md:grid-cols-4 gap-4 mb-5">
          <Slider label="Image side H" value={imgSize} options={[224, 384, 512, 768, 1024]} onChange={setImgSize} />
          <Slider label="Patch P" value={patchSize} options={[4, 8, 16, 32]} onChange={setPatchSize} />
          <Slider label="Window M" value={winSize} options={[4, 7, 8, 12]} onChange={setWinSize} />
          <Slider label="Swin C₁" value={embDim} options={[48, 96, 128, 192]} onChange={setEmbDim} />
        </div>
        <div className="grid md:grid-cols-2 gap-5">
          <div className="bg-slate-950/40 border border-rose-500/30 rounded-xl p-4">
            <Tag color="rose">ViT-Base</Tag>
            <div className="font-serif text-3xl text-rose-300 mt-3">{(vitAttnFLOPs / 1e9).toFixed(2)} G</div>
            <div className="font-mono text-[11px] text-slate-500 mb-3">FLOPs · single attention layer</div>
            <div className="space-y-1 text-[11px] font-mono text-slate-400">
              <div>N = (H/P)² = {N.toLocaleString()}</div>
              <div>D = {D_vit}</div>
              <div>~ 4·N·D² + 2·N²·D</div>
            </div>
          </div>
          <div className="bg-slate-950/40 border border-teal-500/30 rounded-xl p-4">
            <Tag color="teal">Swin (stage 1)</Tag>
            <div className="font-serif text-3xl text-teal-300 mt-3">{(swinAttnFLOPs / 1e9).toFixed(2)} G</div>
            <div className="font-mono text-[11px] text-slate-500 mb-3">FLOPs · single W-MSA layer</div>
            <div className="space-y-1 text-[11px] font-mono text-slate-400">
              <div>tokens = {N.toLocaleString()}</div>
              <div>C = {embDim}, M = {winSize}</div>
              <div>~ 4·hw·C² + 2·M²·hw·C</div>
            </div>
          </div>
        </div>
        <div className="mt-4">
          <div className="font-mono text-[11px] text-slate-400 mb-1">Ratio (Swin / ViT)</div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-teal-500 to-amber-500"
              style={{ width: `${Math.min(100, (swinAttnFLOPs / vitAttnFLOPs) * 100)}%` }} />
          </div>
          <div className="font-mono text-[11px] text-slate-500 mt-1">
            {(swinAttnFLOPs / vitAttnFLOPs * 100).toFixed(1)}% — try shrinking patch size or growing image to see ViT explode.
          </div>
        </div>
      </Card>

      <div className="grid md:grid-cols-2 gap-5">
        <Card className="p-5">
          <h3 className="font-serif text-xl text-slate-100 mb-4">Pick ViT when…</h3>
          <ul className="space-y-2.5 text-slate-300 text-sm">
            <ProsItem>You have lots of pretraining data (JFT-300M scale), or strong self-supervised pretraining (DINO, MAE).</ProsItem>
            <ProsItem>The downstream task is global — image classification, retrieval, image-text alignment (CLIP-style).</ProsItem>
            <ProsItem>You want architectural simplicity — every layer looks the same, no special partitioning logic.</ProsItem>
            <ProsItem>You're running at fixed input resolution (typically 224 or 384).</ProsItem>
          </ul>
        </Card>
        <Card className="p-5">
          <h3 className="font-serif text-xl text-slate-100 mb-4">Pick Swin when…</h3>
          <ul className="space-y-2.5 text-slate-300 text-sm">
            <ProsItem>You need a feature pyramid — detection (Mask R-CNN), segmentation (UperNet), dense prediction generally.</ProsItem>
            <ProsItem>You want to train from scratch on ImageNet-scale data — locality bias helps.</ProsItem>
            <ProsItem>You need to scale to high-resolution inputs (1024+) without quadratic blowup.</ProsItem>
            <ProsItem>You want to swap into existing ConvNet pipelines (FPN, decoder heads) with minimal changes.</ProsItem>
          </ul>
        </Card>
      </div>

      <Card className="p-5">
        <h3 className="font-serif text-lg text-slate-100 mb-1">
          Spec sheet · ViT-Base/16 vs Swin-Tiny
        </h3>
        <p className="text-[12px] text-slate-400 mb-4 leading-snug">
          Two specific reference architectures benchmarked on the standard <span className="font-mono text-amber-300">224 × 224</span>
          {' '}<span className="text-slate-300">ImageNet-1K</span> setup — not tied to any particular image you've uploaded.
          ViT-Base/16 is the canonical 86M-param ViT; Swin-Tiny is the smallest Swin variant from the original paper.
          All numbers below are reported per single forward pass on one <span className="font-mono text-amber-300">224 × 224 × 3</span> image.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-2 font-mono text-[11px] uppercase tracking-wider text-slate-500">Property</th>
                <th className="text-left py-2 font-mono text-[11px] uppercase tracking-wider text-rose-400">ViT-Base/16</th>
                <th className="text-left py-2 font-mono text-[11px] uppercase tracking-wider text-teal-400">Swin-Tiny</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[12px]">
              {[
                ['Parameters (total weights)', '86 M', '28 M'],
                ['FLOPs · per 224² image', '17.6 G', '4.5 G'],
                ['Tokens · per 224² image', '197 (incl. CLS)', '3136 → 49 across stages'],
                ['Embed dim', '768 (constant)', '96 → 768 across stages'],
                ['Attention', 'Global', 'Local 7×7 window + shift'],
                ['Position', 'Absolute (learnable)', 'Relative bias'],
                ['ImageNet-1K top-1 (224² eval)', '77.9 %', '81.3 %'],
                ['Best for', 'Classification, CLIP', 'Detection, segmentation'],
              ].map(([k, a, b], i) => (
                <tr key={i} className="border-b border-slate-800/60">
                  <td className="py-2 text-slate-400">{k}</td>
                  <td className="py-2 text-slate-200">{a}</td>
                  <td className="py-2 text-slate-200">{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 text-[11px] text-slate-500 italic">
          ImageNet-1K numbers are from the original Swin paper (Liu et al., 2021), trained from scratch.
          ViT-Base improves substantially with JFT-300M pretraining.
        </div>
      </Card>

      <Takeaway
        understand={[
          'ViT: pure global attention, conceptually clean, scales beautifully with data and compute, single resolution.',
          'Swin: local windows + shifting + hierarchical merging — linear cost, multi-scale features, better default for dense prediction.',
        ]}
        confusion={['"Better" depends on the task. ViT wins big on huge-data classification; Swin wins on detection / segmentation at modest data.']}
        tryIt="Crank up the image size in the cost calculator and watch ViT FLOPs explode quadratically while Swin stays linear."
      />
    </div>
  );
}

function ProsItem({ children }) {
  return (
    <li className="flex gap-2.5 items-start">
      <ChevronRight size={14} className="text-amber-400 mt-1 flex-shrink-0" />
      <span>{children}</span>
    </li>
  );
}

/* =========================================================
   LIVE DEMO — interactive image scan + classifier
   ========================================================= */

const GALLERY = [
  { id: 'cat',      name: 'Cute Cat', src: import.meta.env.BASE_URL + 'cat.jpg' },
  { id: 'baiwan',   name: 'Baiwan',   src: import.meta.env.BASE_URL + 'baiwan.jpg' },
  { id: 'miaomiao', name: 'Miaomiao', src: import.meta.env.BASE_URL + 'miaomiao.jpg' },
];

function drawScan(canvas, img, size, patchPx, progress, mode, shifted = false, pinnedQuery = null) {
  if (!canvas || !img) return;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const crop = Math.min(w, h);
  ctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, size, size);
  ctx.fillStyle = 'rgba(10,14,26,0.40)';
  ctx.fillRect(0, 0, size, size);

  const gridN = Math.max(2, Math.floor(size / patchPx));
  const total = gridN * gridN;
  const seen = Math.min(total, Math.floor(progress * total));
  const cur = Math.max(0, seen - 1);
  // Active query: pinned overrides the moving scan cursor so all its
  // attention edges are drawn at once.
  const activeIdx = pinnedQuery != null && pinnedQuery >= 0 && pinnedQuery < total
    ? pinnedQuery
    : (seen > 0 ? cur : -1);
  const cx = activeIdx >= 0 ? activeIdx % gridN : 0;
  const cy = activeIdx >= 0 ? Math.floor(activeIdx / gridN) : 0;
  const baseColor = mode === 'vit' ? '245, 158, 11' : '20, 184, 166';

  ctx.strokeStyle = `rgba(${baseColor}, 0.18)`;
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridN; i++) {
    const p = i * patchPx;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
  }

  if (mode === 'swin') {
    const winSize = 4;
    const halfWin = shifted ? winSize / 2 : 0;
    // Window membership respects the shift offset. Two patches are in
    // the same window iff their (px+halfWin)/winSize floors agree.
    const winOf = (px, py) => `${Math.floor((px + halfWin) / winSize)}_${Math.floor((py + halfWin) / winSize)}`;

    // Window borders (offset by -halfWin so [0..M) becomes [-halfWin..winSize-halfWin) etc.)
    ctx.strokeStyle = `rgba(${baseColor}, 0.55)`;
    ctx.lineWidth = 2;
    const start = -halfWin;
    for (let col = start; col <= gridN; col += winSize) {
      if (col < 0 || col > gridN) continue;
      const p = col * patchPx;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
    }

    // Visited windows tint
    const visited = new Set();
    for (let i = 0; i < seen; i++) {
      const px = i % gridN;
      const py = Math.floor(i / gridN);
      visited.add(winOf(px, py));
    }
    visited.forEach(key => {
      const [wx, wy] = key.split('_').map(Number);
      const x = wx * winSize * patchPx - halfWin * patchPx;
      const y = wy * winSize * patchPx - halfWin * patchPx;
      ctx.fillStyle = `rgba(${baseColor}, 0.06)`;
      ctx.fillRect(x, y, winSize * patchPx, winSize * patchPx);
    });
  } else {
    for (let i = 0; i < seen; i++) {
      const px = (i % gridN) * patchPx;
      const py = Math.floor(i / gridN) * patchPx;
      ctx.fillStyle = `rgba(${baseColor}, 0.10)`;
      ctx.fillRect(px, py, patchPx, patchPx);
    }
  }

  if (activeIdx >= 0) {
    const qx = cx * patchPx + patchPx / 2;
    const qy = cy * patchPx + patchPx / 2;
    // Pinned: edges are drawn brighter and thicker so the spaghetti vs
    // cluster contrast is unmissable.
    const isPinned = pinnedQuery != null;

    if (mode === 'vit') {
      ctx.strokeStyle = `rgba(${baseColor}, ${isPinned ? 0.42 : 0.16})`;
      ctx.lineWidth = isPinned ? 0.9 : 0.5;
      for (let j = 0; j < total; j++) {
        if (j === activeIdx) continue;
        const tx = (j % gridN) * patchPx + patchPx / 2;
        const ty = Math.floor(j / gridN) * patchPx + patchPx / 2;
        ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty); ctx.stroke();
      }
    } else {
      const winSize = 4;
      const halfWin = shifted ? winSize / 2 : 0;
      const wx = Math.floor((cx + halfWin) / winSize);
      const wy = Math.floor((cy + halfWin) / winSize);
      const winX = wx * winSize * patchPx - halfWin * patchPx;
      const winY = wy * winSize * patchPx - halfWin * patchPx;
      const winSide = winSize * patchPx;
      ctx.fillStyle = `rgba(${baseColor}, 0.18)`;
      ctx.fillRect(winX, winY, winSide, winSide);
      ctx.strokeStyle = `rgba(${baseColor}, 1)`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(winX, winY, winSide, winSide);
      ctx.strokeStyle = `rgba(${baseColor}, ${isPinned ? 0.85 : 0.5})`;
      ctx.lineWidth = isPinned ? 1.1 : 0.7;
      const minPy = Math.max(0, wy * winSize - halfWin);
      const maxPy = Math.min(gridN, (wy + 1) * winSize - halfWin);
      const minPx = Math.max(0, wx * winSize - halfWin);
      const maxPx = Math.min(gridN, (wx + 1) * winSize - halfWin);
      for (let py = minPy; py < maxPy; py++) {
        for (let px = minPx; px < maxPx; px++) {
          if (px === cx && py === cy) continue;
          const tx = px * patchPx + patchPx / 2;
          const ty = py * patchPx + patchPx / 2;
          ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty); ctx.stroke();
        }
      }
    }

    ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
    ctx.fillRect(cx * patchPx, cy * patchPx, patchPx, patchPx);
    ctx.strokeStyle = 'rgba(244, 63, 94, 1)';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx * patchPx, cy * patchPx, patchPx, patchPx);
  }
}

/* Attention matrix view — fixed 8×8 patch grid (independent of scan
   patchSize) so the visual is stable. Computes pairwise attention from
   downsampled patch colors; ViT is dense, Swin masks anything outside
   a 4×4 window block. */
const MAT_N = 8;
const MAT_TOTAL = MAT_N * MAT_N;
const MAT_WIN = 4;

function computeAttention(img) {
  if (!img) return null;
  const tmp = document.createElement('canvas');
  tmp.width = MAT_N;
  tmp.height = MAT_N;
  const tctx = tmp.getContext('2d');
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const crop = Math.min(w, h);
  tctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, MAT_N, MAT_N);
  const pix = tctx.getImageData(0, 0, MAT_N, MAT_N).data;
  const feats = [];
  for (let i = 0; i < MAT_TOTAL; i++) {
    feats.push([pix[i * 4], pix[i * 4 + 1], pix[i * 4 + 2]]);
  }
  function build(modeType) {
    const A = [];
    for (let i = 0; i < MAT_TOTAL; i++) {
      const row = new Float32Array(MAT_TOTAL);
      const logits = new Float32Array(MAT_TOTAL);
      let max = -Infinity;
      const ix = i % MAT_N, iy = (i / MAT_N) | 0;
      for (let j = 0; j < MAT_TOTAL; j++) {
        const jx = j % MAT_N, jy = (j / MAT_N) | 0;
        if (modeType === 'swin' &&
            ((ix / MAT_WIN) | 0) !== ((jx / MAT_WIN) | 0) ||
            modeType === 'swin' &&
            ((iy / MAT_WIN) | 0) !== ((jy / MAT_WIN) | 0)) {
          logits[j] = -1e9;
          continue;
        }
        let d = 0;
        for (let k = 0; k < 3; k++) {
          const diff = feats[i][k] - feats[j][k];
          d += diff * diff;
        }
        const v = -d / 4000;
        logits[j] = v;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let j = 0; j < MAT_TOTAL; j++) {
        if (logits[j] <= -1e8) { row[j] = 0; continue; }
        row[j] = Math.exp(logits[j] - max);
        sum += row[j];
      }
      if (sum > 0) for (let j = 0; j < MAT_TOTAL; j++) row[j] /= sum;
      A.push(row);
    }
    return A;
  }
  return { vit: build('vit'), swin: build('swin') };
}

function drawMatrix(canvas, A, progress, mode) {
  if (!canvas || !A) return;
  const px = 480;
  canvas.width = px;
  canvas.height = px;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0a0e1a';
  ctx.fillRect(0, 0, px, px);
  const cell = px / MAT_TOTAL;
  const seenRows = Math.min(MAT_TOTAL, Math.floor(progress * MAT_TOTAL));
  const base = mode === 'vit' ? [245, 158, 11] : [20, 184, 166];

  for (let i = 0; i < MAT_TOTAL; i++) {
    const row = A[i];
    for (let j = 0; j < MAT_TOTAL; j++) {
      const v = row[j];
      let alpha;
      if (i < seenRows) {
        const intensity = Math.min(1, v * 14);
        alpha = 0.06 + intensity * 0.94;
      } else {
        alpha = 0.04;
      }
      ctx.fillStyle = `rgba(${base[0]}, ${base[1]}, ${base[2]}, ${alpha})`;
      ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
    }
  }

  // patch grid (every MAT_N cells)
  ctx.strokeStyle = `rgba(${base[0]}, ${base[1]}, ${base[2]}, 0.12)`;
  ctx.lineWidth = 1;
  for (let i = 0; i <= MAT_TOTAL; i += MAT_N) {
    const p = i * cell;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, px); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(px, p); ctx.stroke();
  }

  // Swin: show the 4×4-patch window blocks (each block = MAT_WIN×MAT_N rows in the flattened matrix)
  if (mode === 'swin') {
    ctx.strokeStyle = `rgba(${base[0]}, ${base[1]}, ${base[2]}, 0.55)`;
    ctx.lineWidth = 2;
    const wins = Math.ceil(MAT_N / MAT_WIN);
    for (let wy = 0; wy < wins; wy++) {
      for (let wx = 0; wx < wins; wx++) {
        const x = wx * MAT_WIN * MAT_N * cell;
        const y = wy * MAT_WIN * MAT_N * cell;
        const wpx = MAT_WIN * MAT_N * cell;
        ctx.strokeRect(x, y, wpx, wpx);
      }
    }
  }

  // current row highlight
  if (seenRows > 0) {
    ctx.strokeStyle = 'rgba(245, 158, 11, 1)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, (seenRows - 1) * cell, px, cell);
  }
}

/* drawClassHeatmap — given the precomputed attention matrix
   (MAT_N × MAT_N grid), render the image with a heatmap overlay
   highlighting which regions the model "looked at". We use the
   column-sums of the attention matrix as a proxy for [CLS] attention:
   patches that many queries attend TO are the ones driving the
   output representation, and therefore the prediction. */
function drawClassHeatmap(canvas, img, attentionRows, gridN, intensity = 1) {
  if (!canvas || !img || !attentionRows) return;
  const size = 320;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Draw image strongly dimmed — the heatmap needs to dominate.
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const crop = Math.min(w, h);
  ctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, size, size);
  ctx.fillStyle = 'rgba(10, 14, 26, 0.78)';
  ctx.fillRect(0, 0, size, size);

  const total = gridN * gridN;
  const colSums = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    for (let j = 0; j < total; j++) colSums[j] += attentionRows[i][j];
  }
  // Min-max normalize so the colormap stretches across the full range.
  let minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < total; i++) {
    if (colSums[i] < minV) minV = colSums[i];
    if (colSums[i] > maxV) maxV = colSums[i];
  }
  const span = Math.max(1e-9, maxV - minV);

  // Five-stop magma-ish colormap so peaks read clearly even at small sizes.
  // Stops:   t=0  → very dark (almost black)
  //          t=.25 → deep purple
  //          t=.5  → magenta
  //          t=.75 → orange
  //          t=1.0 → bright yellow
  function lerp(a, b, t) { return a + (b - a) * t; }
  function colormap(t) {
    const stops = [
      [0.00,  10,  10,  30],
      [0.25,  50,  10,  90],
      [0.50, 200,  40, 110],
      [0.75, 250, 130,  40],
      [1.00, 255, 240, 100],
    ];
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) {
        const a = stops[s - 1], b = stops[s];
        const u = (t - a[0]) / (b[0] - a[0]);
        return [lerp(a[1], b[1], u), lerp(a[2], b[2], u), lerp(a[3], b[3], u)];
      }
    }
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3]];
  }

  const cell = size / gridN;
  for (let py = 0; py < gridN; py++) {
    for (let px = 0; px < gridN; px++) {
      const v = (colSums[py * gridN + px] - minV) / span;
      const t = Math.pow(v, 2.5) * intensity;
      const [r, g, b] = colormap(t);
      // Alpha climbs aggressively so high-attention patches are nearly opaque.
      const a = 0.15 + Math.min(0.78, t * 0.85);
      ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
      ctx.fillRect(px * cell - 0.5, py * cell - 0.5, cell + 1, cell + 1);
    }
  }

  // Soft-blur for smooth-feeling edges.
  ctx.filter = 'blur(7px)';
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(canvas, 0, 0);
  ctx.filter = 'none';
  ctx.globalCompositeOperation = 'source-over';
}

/* drawClassContrib — per-class "evidence" heatmap. Starts from the
   attention column-sums (the same proxy for [CLS] attention used in the
   top heatmap) and weights it by a class-specific spatial Gaussian.
   Each class gets a slightly different focal point, so clicking through
   the top-5 reveals different "looks" — concretely demonstrates that
   classification is "where do I see evidence for THIS class". */
/* Each top-K class gets a distinctly different focal region so the
   heatmaps are clearly separable when students click between classes.
   Top-1 sits on whatever the attention rollout already highlights;
   the rest pull toward the four corners with a tight Gaussian. */
const CLASS_ANCHORS = [
  [0.50, 0.55],   // top-1: roughly where the model already looks
  [0.25, 0.30],   // top-2: upper-left quadrant
  [0.75, 0.30],   // top-3: upper-right quadrant
  [0.25, 0.75],   // top-4: lower-left quadrant
  [0.75, 0.75],   // top-5: lower-right quadrant
];

/* computeAttentionRollout — Abnar & Zuidema (2020).  Each layer's
   attention is averaged across heads, the residual stream is folded
   in by adding the identity, the row sums are renormalised to 1, and
   the per-layer matrices are multiplied together.  The CLS row of the
   final product is the per-patch attention used to render the heatmap.

   Input shape per layer: tensor with .data Float32Array, .dims
   [batch=1, heads, N+1, N+1].  Returns null if the tensor list is
   empty or malformed.

   Output: { data: Float32Array(N), gridSide: int }, with values
   min-max normalised to [0, 1] for display. */
function computeAttentionRollout(attentions) {
  if (!Array.isArray(attentions) || attentions.length === 0) return null;
  const first = attentions[0];
  if (!first || !first.dims || first.dims.length !== 4) return null;
  const numTokens = first.dims[2];           // 197
  const numPatches = numTokens - 1;          // 196
  const gridSide = Math.round(Math.sqrt(numPatches));
  if (gridSide * gridSide !== numPatches) return null;

  let rollout = new Float32Array(numTokens * numTokens);
  for (let i = 0; i < numTokens; i++) rollout[i * numTokens + i] = 1;

  const layerAvg = new Float32Array(numTokens * numTokens);

  for (const tensor of attentions) {
    if (!tensor || !tensor.data || !tensor.dims) return null;
    const [, heads, rows, cols] = tensor.dims;
    if (rows !== numTokens || cols !== numTokens) return null;
    const data = tensor.data;

    // Average across heads.
    const headStride = rows * cols;
    layerAvg.fill(0);
    for (let h = 0; h < heads; h++) {
      const base = h * headStride;
      for (let i = 0; i < headStride; i++) layerAvg[i] += data[base + i];
    }
    for (let i = 0; i < headStride; i++) layerAvg[i] /= heads;

    // Add identity (the residual stream contributes to the rollout).
    for (let i = 0; i < numTokens; i++) layerAvg[i * numTokens + i] += 1;

    // Re-normalise each row to sum to 1.
    for (let i = 0; i < numTokens; i++) {
      let rowSum = 0;
      for (let j = 0; j < numTokens; j++) rowSum += layerAvg[i * numTokens + j];
      if (rowSum > 0) {
        for (let j = 0; j < numTokens; j++) layerAvg[i * numTokens + j] /= rowSum;
      }
    }

    // rollout = layerAvg @ rollout
    const next = new Float32Array(numTokens * numTokens);
    for (let i = 0; i < numTokens; i++) {
      for (let k = 0; k < numTokens; k++) {
        const a = layerAvg[i * numTokens + k];
        if (a === 0) continue;
        for (let j = 0; j < numTokens; j++) {
          next[i * numTokens + j] += a * rollout[k * numTokens + j];
        }
      }
    }
    rollout = next;
  }

  // Take the CLS row (index 0) and drop the CLS-self entry.
  const patchAttn = new Float32Array(numPatches);
  for (let j = 0; j < numPatches; j++) patchAttn[j] = rollout[j + 1];

  // Min-max normalise for visualisation.
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < numPatches; i++) {
    if (patchAttn[i] < mn) mn = patchAttn[i];
    if (patchAttn[i] > mx) mx = patchAttn[i];
  }
  const range = mx - mn || 1;
  for (let i = 0; i < numPatches; i++) patchAttn[i] = (patchAttn[i] - mn) / range;

  return { data: patchAttn, gridSide };
}

/* drawRolloutHeatmap — paints the real attention-rollout grid over
   the cropped image. Used when transformers.js gives us actual
   per-layer attention; falls back to the synthetic per-class heatmap
   in drawClassContrib when not. */
function drawRolloutHeatmap(canvas, img, rollout) {
  if (!canvas || !img || !rollout) return;
  const size = 260;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const crop = Math.min(w, h);
  ctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, size, size);
  ctx.fillStyle = 'rgba(10, 14, 26, 0.72)';
  ctx.fillRect(0, 0, size, size);

  const { data, gridSide } = rollout;
  function lerp(a, b, t) { return a + (b - a) * t; }
  function colormap(t) {
    const stops = [
      [0.00,  10,  10,  30],
      [0.25,  50,  10,  90],
      [0.50, 200,  40, 110],
      [0.75, 250, 130,  40],
      [1.00, 255, 240, 100],
    ];
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) {
        const a = stops[s - 1], b = stops[s];
        const u = (t - a[0]) / (b[0] - a[0]);
        return [lerp(a[1], b[1], u), lerp(a[2], b[2], u), lerp(a[3], b[3], u)];
      }
    }
    return [stops[4][1], stops[4][2], stops[4][3]];
  }

  const cell = size / gridSide;
  for (let py = 0; py < gridSide; py++) {
    for (let px = 0; px < gridSide; px++) {
      const v = data[py * gridSide + px];
      const t = Math.pow(v, 1.6);
      const [r, g, b] = colormap(t);
      const a = 0.10 + Math.min(0.82, t * 0.9);
      ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
      ctx.fillRect(px * cell - 0.5, py * cell - 0.5, cell + 1, cell + 1);
    }
  }
  ctx.filter = 'blur(6px)';
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(canvas, 0, 0);
  ctx.filter = 'none';
  ctx.globalCompositeOperation = 'source-over';
}

function drawClassContrib(canvas, img, attentionRows, gridN, classIdx) {
  if (!canvas || !img || !attentionRows) return;
  const size = 260;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const crop = Math.min(w, h);
  ctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, size, size);
  ctx.fillStyle = 'rgba(10, 14, 26, 0.78)';
  ctx.fillRect(0, 0, size, size);

  const total = gridN * gridN;
  const base = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    for (let j = 0; j < total; j++) base[j] += attentionRows[i][j];
  }
  const [ax, ay] = CLASS_ANCHORS[classIdx % CLASS_ANCHORS.length];

  // Tighter Gaussian → each class focuses on a clearly different region.
  // For class 0 (top-1) we don't bias toward an anchor — let the existing
  // attention rollout show through unmodified.
  const heat = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const px = (i % gridN + 0.5) / gridN;
    const py = (Math.floor(i / gridN) + 0.5) / gridN;
    const dx = px - ax, dy = py - ay;
    const bias = classIdx === 0
      ? 1
      : Math.exp(-(dx * dx + dy * dy) / 0.025);
    heat[i] = classIdx === 0
      ? base[i]
      : base[i] * (0.05 + bias * 3.2);
  }

  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < total; i++) { if (heat[i] < mn) mn = heat[i]; if (heat[i] > mx) mx = heat[i]; }
  const span = Math.max(1e-9, mx - mn);

  function lerp(a, b, t) { return a + (b - a) * t; }
  function colormap(t) {
    const stops = [
      [0.00,  10,  10,  30],
      [0.25,  50,  10,  90],
      [0.50, 200,  40, 110],
      [0.75, 250, 130,  40],
      [1.00, 255, 240, 100],
    ];
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) {
        const a = stops[s - 1], b = stops[s];
        const u = (t - a[0]) / (b[0] - a[0]);
        return [lerp(a[1], b[1], u), lerp(a[2], b[2], u), lerp(a[3], b[3], u)];
      }
    }
    return [stops[4][1], stops[4][2], stops[4][3]];
  }

  const cell = size / gridN;
  for (let py = 0; py < gridN; py++) {
    for (let px = 0; px < gridN; px++) {
      const v = (heat[py * gridN + px] - mn) / span;
      const t = Math.pow(v, 2.3);
      const [r, g, b] = colormap(t);
      const a = 0.15 + Math.min(0.78, t * 0.85);
      ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
      ctx.fillRect(px * cell - 0.5, py * cell - 0.5, cell + 1, cell + 1);
    }
  }
  ctx.filter = 'blur(7px)';
  ctx.globalCompositeOperation = 'screen';
  ctx.drawImage(canvas, 0, 0);
  ctx.filter = 'none';
  ctx.globalCompositeOperation = 'source-over';
}

/* ClassContribution — image with a class-specific evidence heatmap +
   clickable top-5 prediction list. Concrete answer to "why this class?".
   Self-contained loading UI so it can replace the standalone predictions
   panel. */
function ClassContribution({ image, attentionRows, gridN, preds, status, statusMessage, statusProgress, rollout }) {
  const [classIdx, setClassIdx] = useState(0);
  const canvasRef = useRef(null);

  useEffect(() => { setClassIdx(0); }, [preds]);

  useEffect(() => {
    if (!canvasRef.current || !image) return;
    if (rollout) {
      drawRolloutHeatmap(canvasRef.current, image, rollout);
    } else if (attentionRows) {
      drawClassContrib(canvasRef.current, image, attentionRows, gridN, classIdx);
    }
  }, [image, attentionRows, gridN, classIdx, rollout]);

  const top5 = (preds || []).slice(0, 5);
  const loading = status === 'modelLoading' || status === 'inferring';
  const isReal = !!rollout;

  return (
    <div className="flex gap-3 items-start flex-wrap">
      <div className="flex-shrink-0">
        <canvas
          ref={canvasRef}
          className="w-[180px] h-[180px] rounded bg-slate-950 block"
        />
        <div className="text-[10px] font-mono text-slate-500 mt-1 text-center w-[180px] leading-snug">
          {isReal
            ? <span><span className="text-teal-300">attention rollout</span><br/>real ViT-B/16 · class-agnostic</span>
            : top5[classIdx]
              ? <>evidence for <span className="text-amber-300">"{top5[classIdx].label}"</span> <span className="text-slate-600">(illustrative)</span></>
              : status === 'modelLoading' ? `loading model · ${statusProgress || 0}%`
              : status === 'inferring' ? 'running ViT-Base…'
              : 'waiting for predictions'}
        </div>
      </div>

      <div className="flex-1 min-w-[180px] space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-[10px] font-mono uppercase tracking-wide text-amber-400/80">
            ViT-Base/16 · top 5
          </span>
          {loading && (
            <span className="text-[10px] font-mono text-teal-300 flex items-center gap-1">
              <Loader2 size={10} className="animate-spin"/>
              {status === 'modelLoading' ? `${statusProgress || 0}%` : '…'}
            </span>
          )}
        </div>

        {status === 'modelLoading' && top5.length === 0 && (
          <div className="text-[11px] text-slate-400 leading-snug py-1">
            <div className="mb-1">{statusMessage || 'Loading model…'}</div>
            <div className="h-1 bg-slate-800 rounded overflow-hidden">
              <div className="h-full bg-teal-400 transition-all" style={{ width: `${statusProgress || 0}%` }}/>
            </div>
          </div>
        )}

        {status === 'error' && (
          <div className="text-[11px] text-rose-300 leading-snug py-1">{statusMessage}</div>
        )}

        {top5.map((p, i) => {
          const active = classIdx === i;
          return (
            <button
              key={i}
              onClick={() => setClassIdx(i)}
              className={`w-full text-left px-2 py-1 rounded border transition-all
                ${active
                  ? 'bg-amber-500/15 border-amber-500/50 text-amber-100'
                  : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600 text-slate-300'}`}
            >
              <div className="flex justify-between items-center text-[11px] gap-2 leading-tight">
                <span className="truncate">{p.label}</span>
                <span className={`font-mono shrink-0 ${active ? 'text-amber-300' : 'text-slate-400'}`}>
                  {(p.score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1 bg-slate-800 rounded mt-0.5 overflow-hidden">
                <div
                  className={`h-full transition-all ${active ? 'bg-amber-400' : 'bg-slate-500'}`}
                  style={{ width: `${p.score * 100}%` }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* FlopsCostCard — translates the FLOPs gap into wall-clock seconds and
   dollars on a real GPU. Now driven by the scan's `progress` so all
   numbers and the rising line chart animate as the cat is scanned: at
   t=0 the bills are zero, at t=1 they hit the per-image total. Slider
   for image side controls the ceiling. */
function FlopsCostCard({ progress = 1 }) {
  const [side, setSide] = useState(768);
  const P = 16, M = 7, L = 12, D = 768;
  const N = Math.round((side / P) ** 2);
  // Attention FLOPs per image (Q,K,V,O projections + scaled-dot product). Approximate.
  const vitFlopsPeak = L * (4 * N * D * D + 2 * N * N * D);
  const swinFlopsPeak = L * (4 * N * D * D + 2 * M * M * N * D);
  const ratioPeak = swinFlopsPeak > 0 ? vitFlopsPeak / swinFlopsPeak : 0;
  // A100: ~312 TFLOPs FP16; one image throughput on a single device.
  const TFLOPs_PER_SEC = 312e12;
  const vitSecPerImgPeak = vitFlopsPeak / TFLOPs_PER_SEC;
  const swinSecPerImgPeak = swinFlopsPeak / TFLOPs_PER_SEC;
  const N_IMG = 1e6;
  const HOUR = 3600;
  const RATE = 1.5; // $/hour for a single A100 — typical cloud
  const vitCostPeak = (vitSecPerImgPeak * N_IMG / HOUR) * RATE;
  const swinCostPeak = (swinSecPerImgPeak * N_IMG / HOUR) * RATE;

  // Live values driven by scan progress. ViT grows linearly to its peak
  // across full progress; Swin reaches its (smaller) peak earlier
  // because its true work-per-image is `peak/ratioPeak` of ViT's.
  const p = Math.max(0, Math.min(1, progress));
  const vitFlops = p * vitFlopsPeak;
  const swinFlops = Math.min(swinFlopsPeak, p * vitFlopsPeak);
  const swinKneeP = Math.min(1, swinFlopsPeak / Math.max(1, vitFlopsPeak));
  const liveRatio = swinFlops > 0 ? vitFlops / swinFlops : 0;
  const vitSec = vitSecPerImgPeak * p;
  const swinSec = Math.min(swinSecPerImgPeak, vitSecPerImgPeak * p);
  const vitCost = vitCostPeak * p;
  const swinCost = Math.min(swinCostPeak, vitCostPeak * p);

  const fmtFlops = (v) => v >= 1e12 ? `${(v / 1e12).toFixed(2)} T` : v >= 1e9 ? `${(v / 1e9).toFixed(2)} G` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} M` : `${Math.round(v).toLocaleString()}`;
  const fmtCost = (v) => v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : v > 0 ? `$${v.toExponential(1)}` : '$0';
  const fmtSec = (s) => s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : `${(s * 1e6).toFixed(1)} µs`;

  return (
    <div className="space-y-3">
      {/* Image-side slider sets the ceiling; progress (from above) sets where on
          the curve we are. Stack vertically below lg so the params readout
          doesn't squash the slider on iPad-portrait-width cards. */}
      <div className="grid lg:grid-cols-[1fr_auto] gap-2 lg:gap-3 lg:items-end">
        <Slider
          label="Image side (px) — sets ceiling"
          value={side}
          options={[224, 384, 512, 768, 1024, 1536]}
          onChange={setSide}
        />
        <div className="text-[11px] font-mono text-slate-400 leading-snug">
          <span className="whitespace-nowrap">P=16 · L=12 · M=7</span>
          <span className="mx-1 text-slate-600">·</span>
          <span className="text-amber-300 whitespace-nowrap">N={N.toLocaleString()}</span>
          <span className="mx-1 text-slate-600">·</span>
          <span className="whitespace-nowrap">scan {(p * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="rounded p-2 bg-amber-500/[0.06] border border-amber-500/30">
          <div className="text-amber-300 font-mono text-[10px]">ViT-Base · live FLOPs</div>
          <div className="text-slate-100 font-serif text-2xl leading-tight tabular-nums">{fmtFlops(vitFlops)}</div>
          <div className="text-slate-500 text-[10px]">peak {fmtFlops(vitFlopsPeak)} · attn only</div>
        </div>
        <div className="rounded p-2 bg-teal-500/[0.06] border border-teal-500/30">
          <div className="text-teal-300 font-mono text-[10px]">Swin-T · live FLOPs</div>
          <div className="text-slate-100 font-serif text-2xl leading-tight tabular-nums">{fmtFlops(swinFlops)}</div>
          <div className="text-slate-500 text-[10px]">peak {fmtFlops(swinFlopsPeak)} · attn only</div>
        </div>
        <div className="rounded p-2 bg-rose-500/[0.06] border border-rose-500/40">
          <div className="text-rose-300 font-mono text-[10px]">ViT / Swin · live</div>
          <div className="text-slate-100 font-serif text-2xl leading-tight tabular-nums">
            {liveRatio > 0 ? (liveRatio < 10 ? liveRatio.toFixed(1) : Math.round(liveRatio).toLocaleString()) : '—'}×
          </div>
          <div className="text-slate-500 text-[10px]">peak {ratioPeak < 10 ? ratioPeak.toFixed(1) : Math.round(ratioPeak).toLocaleString()}×</div>
        </div>
      </div>

      {/* Rising chart — area lines grow as the cat above gets scanned. */}
      <FlopsRiseChart
        progress={p}
        vitFlopsPeak={vitFlopsPeak}
        swinFlopsPeak={swinFlopsPeak}
        swinKneeP={swinKneeP}
        fmtFlops={fmtFlops}
      />

      <div className="rounded p-2 bg-slate-800/40 border border-slate-700/60">
        <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">
          Wall-clock & money · live for one image, peak for 1M images @ A100 ${RATE.toFixed(2)}/hr
        </div>
        <div className="grid grid-cols-2 gap-2 text-center">
          <div>
            <div className="font-serif text-xl text-amber-200 tabular-nums">{fmtSec(vitSec)}</div>
            <div className="text-[10px] font-mono text-slate-500">ViT · 1 img · peak {fmtSec(vitSecPerImgPeak)}</div>
            <div className="font-serif text-base text-amber-100 mt-1 tabular-nums">{fmtCost(vitCost)}</div>
            <div className="text-[10px] font-mono text-slate-500">@ 1M images · peak {fmtCost(vitCostPeak)}</div>
          </div>
          <div>
            <div className="font-serif text-xl text-teal-200 tabular-nums">{fmtSec(swinSec)}</div>
            <div className="text-[10px] font-mono text-slate-500">Swin · 1 img · peak {fmtSec(swinSecPerImgPeak)}</div>
            <div className="font-serif text-base text-teal-100 mt-1 tabular-nums">{fmtCost(swinCost)}</div>
            <div className="text-[10px] font-mono text-slate-500">@ 1M images · peak {fmtCost(swinCostPeak)}</div>
          </div>
        </div>
      </div>

    </div>
  );
}

/* FlopsRiseChart — rising-area twin lines for the FlopsCostCard. Same
   visual language as StorageRiseChart so students recognise the
   "it grows with replay" pattern. */
function FlopsRiseChart({ progress, vitFlopsPeak, swinFlopsPeak, swinKneeP, fmtFlops }) {
  const W = 640, H = 96;
  const PAD_L = 56, PAD_R = 10, PAD_T = 6, PAD_B = 20;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const yMax = Math.max(1, vitFlopsPeak);
  const yScale = v => PAD_T + innerH - (v / yMax) * innerH;
  const xScale = p => PAD_L + Math.max(0, Math.min(1, p)) * innerW;
  const baseline = PAD_T + innerH;

  const vitX = xScale(progress);
  const vitY = yScale(progress * vitFlopsPeak);
  const kneeX = xScale(Math.min(progress, swinKneeP));
  const kneeY = yScale(Math.min(progress, swinKneeP) * vitFlopsPeak);
  const swinTipX = xScale(progress);
  const swinTipY = yScale(Math.min(swinFlopsPeak, progress * vitFlopsPeak));

  const vitArea = `M ${PAD_L} ${baseline} L ${vitX} ${vitY} L ${vitX} ${baseline} Z`;
  const swinArea = progress <= swinKneeP
    ? `M ${PAD_L} ${baseline} L ${swinTipX} ${swinTipY} L ${swinTipX} ${baseline} Z`
    : `M ${PAD_L} ${baseline} L ${kneeX} ${kneeY} L ${swinTipX} ${swinTipY} L ${swinTipX} ${baseline} Z`;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => t * yMax);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none"
      style={{ height: '110px' }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={PAD_L} y1={yScale(t)} x2={W - PAD_R} y2={yScale(t)}
            stroke="rgba(148,163,184,0.14)" strokeWidth="0.6" />
          <text x={PAD_L - 6} y={yScale(t) + 3} fontSize="9" textAnchor="end"
            fill="#64748b" fontFamily="monospace">{fmtFlops(t)}</text>
        </g>
      ))}
      <line x1={PAD_L} y1={baseline} x2={W - PAD_R} y2={baseline} stroke="#475569" strokeWidth="0.8" />
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={baseline} stroke="#475569" strokeWidth="0.8" />

      {/* Faint targets — show where each line would end at full play */}
      <line x1={PAD_L} y1={baseline} x2={xScale(1)} y2={yScale(vitFlopsPeak)}
        stroke="rgba(245,158,11,0.25)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={PAD_L} y1={baseline} x2={xScale(swinKneeP)} y2={yScale(swinFlopsPeak)}
        stroke="rgba(20,184,166,0.25)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={xScale(swinKneeP)} y1={yScale(swinFlopsPeak)} x2={xScale(1)} y2={yScale(swinFlopsPeak)}
        stroke="rgba(20,184,166,0.25)" strokeWidth="1" strokeDasharray="3 3" />

      <path d={vitArea} fill="rgba(245,158,11,0.32)" stroke="#f59e0b" strokeWidth="1.6" />
      <path d={swinArea} fill="rgba(20,184,166,0.32)" stroke="#14b8a6" strokeWidth="1.6" />

      <circle cx={vitX} cy={vitY} r="3.4" fill="#fbbf24" stroke="#0f172a" strokeWidth="1" />
      <circle cx={swinTipX} cy={swinTipY} r="3.4" fill="#2dd4bf" stroke="#0f172a" strokeWidth="1" />

      <text x={PAD_L} y={H - 6} fontSize="10" fill="#64748b" fontFamily="monospace">play 0%</text>
      <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize="10" textAnchor="middle" fill="#64748b" fontFamily="monospace">progress →</text>
      <text x={W - PAD_R} y={H - 6} fontSize="10" textAnchor="end" fill="#64748b" fontFamily="monospace">100%</text>
    </svg>
  );
}

/* SwinReachLab — click any patch on the cat and the panel shows how
   many peers that query directly exchanges with in Layer 1 (W-MSA),
   Layer 2 (SW-MSA), and the union across both. The reach map colours
   each reachable patch by which layer reached it (amber = both, dim
   amber = L1 only, teal = L2 only). Concrete artefact for class
   discussion: pick a corner patch vs a centre patch, count, debate. */
function SwinReachLab({ image, imgVersion }) {
  const SIZE = 240;
  const PATCH = 30;
  const GRID = SIZE / PATCH; // 8
  const canvasRef = useRef(null);
  const [pick, setPick] = useState(null);
  // Window size — students can drag M to see how locality vs reach trades off.
  const [M, setM] = useState(4);
  // Layer view filter — show L1 only, L2 only, or both. Discussion-driver:
  // "what does each layer alone reach? what does adding the second do?"
  const [view, setView] = useState('both'); // 'L1' | 'L2' | 'both'
  const HALF = M / 2;

  const reach = useMemo(() => {
    if (pick == null) return null;
    const px = pick % GRID;
    const py = (pick / GRID) | 0;
    const wReg = (rx, ry) => `${(rx / M) | 0}_${(ry / M) | 0}`;
    const wShift = (rx, ry) => `${((rx + HALF) / M) | 0}_${((ry + HALF) / M) | 0}`;
    const w1 = wReg(px, py);
    const w2 = wShift(px, py);
    const l1 = new Set();
    const l2 = new Set();
    for (let i = 0; i < GRID * GRID; i++) {
      if (i === pick) continue;
      const ix = i % GRID, iy = (i / GRID) | 0;
      if (wReg(ix, iy) === w1) l1.add(i);
      if (wShift(ix, iy) === w2) l2.add(i);
    }
    const combined = new Set([...l1, ...l2]);
    let overlap = 0;
    l1.forEach(i => { if (l2.has(i)) overlap++; });
    return { l1, l2, combined, overlap, total: GRID * GRID - 1 };
  }, [pick, M, HALF]);

  useEffect(() => { setPick(null); }, [imgVersion]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.width = SIZE; c.height = SIZE;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, SIZE, SIZE);
    if (image) {
      const w = image.naturalWidth || image.width;
      const h = image.naturalHeight || image.height;
      const crop = Math.min(w, h);
      ctx.drawImage(image, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, SIZE, SIZE);
      ctx.fillStyle = 'rgba(10, 14, 26, 0.7)';
      ctx.fillRect(0, 0, SIZE, SIZE);
    }
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID; i++) {
      const p = i * PATCH;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(SIZE, p); ctx.stroke();
    }
    if (reach) {
      for (let i = 0; i < GRID * GRID; i++) {
        if (i === pick) continue;
        const inL1 = reach.l1.has(i);
        const inL2 = reach.l2.has(i);
        // Apply view filter: if user picked L1 only, hide L2-only patches, etc.
        const showL1 = inL1 && (view === 'L1' || view === 'both');
        const showL2 = inL2 && (view === 'L2' || view === 'both');
        if (!showL1 && !showL2) continue;
        const px = (i % GRID) * PATCH;
        const py = ((i / GRID) | 0) * PATCH;
        const fill = (showL1 && showL2)
          ? 'rgba(245, 158, 11, 0.55)'
          : showL1
            ? 'rgba(245, 158, 11, 0.32)'
            : 'rgba(20, 184, 166, 0.55)';
        ctx.fillStyle = fill;
        ctx.fillRect(px, py, PATCH, PATCH);
      }
      const qx = (pick % GRID) * PATCH;
      const qy = ((pick / GRID) | 0) * PATCH;
      ctx.fillStyle = 'rgba(244, 63, 94, 0.55)';
      ctx.fillRect(qx, qy, PATCH, PATCH);
      ctx.strokeStyle = 'rgba(244, 63, 94, 1)';
      ctx.lineWidth = 2;
      ctx.strokeRect(qx + 1, qy + 1, PATCH - 2, PATCH - 2);
    }
  }, [image, reach, pick, view]);

  const handleClick = (e) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    if (rect.width === 0) return;
    const x = (e.clientX - rect.left) / rect.width * SIZE;
    const y = (e.clientY - rect.top) / rect.height * SIZE;
    const px = Math.min(GRID - 1, Math.max(0, (x / PATCH) | 0));
    const py = Math.min(GRID - 1, Math.max(0, (y / PATCH) | 0));
    const idx = py * GRID + px;
    setPick(prev => prev === idx ? null : idx);
  };

  return (
    <div className="grid sm:grid-cols-[auto_1fr] gap-4 items-start">
      <div>
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          className="w-[240px] h-[240px] rounded bg-slate-950 border border-slate-800 cursor-crosshair block"
        />
        <div className="text-[10px] font-mono text-slate-500 mt-1 text-center">
          {pick == null ? 'click any patch on the cat' : `picked patch #${pick}`}
        </div>
      </div>
      <div className="space-y-3">
        {/* Interactive controls — work even before a patch is picked. */}
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <div className="flex gap-1 bg-slate-800/40 rounded p-0.5 border border-slate-700/60">
            {['L1', 'L2', 'both'].map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-2.5 py-1 rounded text-[10px] font-mono transition-all
                  ${view === v ? 'bg-teal-500/25 text-teal-100' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {v === 'both' ? 'Both layers' : `${v} only`}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Window M</label>
            <div className="flex gap-1">
              {[2, 4, 8].map(opt => (
                <button
                  key={opt}
                  onClick={() => setM(opt)}
                  className={`w-8 px-1 py-0.5 rounded text-[11px] font-mono border transition-all
                    ${M === opt
                      ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                      : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:text-slate-200'}`}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
        </div>

        {!reach ? (
          <div className="text-[12px] text-slate-500 italic">
            Pick a query patch to count how many others it reaches in two layers — try a corner, then the centre. Change M and the view to compare.
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className={`rounded p-2 border transition-all
                ${view === 'L1' ? 'bg-amber-500/15 border-amber-500/60' : 'bg-amber-500/[0.06] border-amber-500/30'}`}>
                <div className="text-amber-300 font-mono text-[10px]">Layer 1 · W-MSA</div>
                <div className="text-slate-100 font-serif text-3xl leading-tight">{reach.l1.size}</div>
                <div className="text-slate-500 text-[10px]">windowmates</div>
              </div>
              <div className={`rounded p-2 border transition-all
                ${view === 'L2' ? 'bg-teal-500/15 border-teal-500/60' : 'bg-teal-500/[0.06] border-teal-500/30'}`}>
                <div className="text-teal-300 font-mono text-[10px]">Layer 2 · SW-MSA</div>
                <div className="text-slate-100 font-serif text-3xl leading-tight">{reach.l2.size}</div>
                <div className="text-slate-500 text-[10px]">windowmates</div>
              </div>
              <div className={`rounded p-2 border transition-all
                ${view === 'both' ? 'bg-rose-500/15 border-rose-500/60' : 'bg-rose-500/[0.06] border-rose-500/40'}`}>
                <div className="text-rose-300 font-mono text-[10px]">After 2 layers</div>
                <div className="text-slate-100 font-serif text-3xl leading-tight">{reach.combined.size}</div>
                <div className="text-slate-500 text-[10px]">unique / {reach.total}</div>
              </div>
            </div>
            <div className="flex flex-wrap gap-3 text-[11px] font-mono">
              <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-rose-500/60"/>query</span>
              {(view === 'both') && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-amber-500/55"/>both layers</span>}
              {(view === 'L1' || view === 'both') && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-amber-500/32"/>L1</span>}
              {(view === 'L2' || view === 'both') && <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-3 rounded-sm bg-teal-500/55"/>L2 (new via shift)</span>}
            </div>
            <p className="text-[12px] text-slate-300 leading-snug">
              {view === 'L1' && <>L1 alone reaches <span className="text-amber-300 font-medium">{reach.l1.size}</span> of {reach.total} other patches at M={M}. Without shifting that's all this query ever sees.</>}
              {view === 'L2' && <>L2 alone (the shifted layer) reaches <span className="text-teal-300 font-medium">{reach.l2.size}</span> patches — different ones than L1 because the windows moved.</>}
              {view === 'both' && <>Without the shift this query reaches <span className="text-amber-300 font-medium">{reach.l1.size}</span>; the shift adds <span className="text-teal-300 font-medium">{reach.l2.size - reach.overlap}</span> brand-new neighbours; over two layers the query touches <span className="text-rose-300 font-medium">{reach.combined.size}</span> of {reach.total} ({Math.round(reach.combined.size / reach.total * 100)}%).</>}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

/* SwinStages — four side-by-side thumbnails of the same image rendered at
   each Swin stage's grid resolution (56→28→14→7). Visualises how patch
   merging coarsens the spatial map while channels grow. Uses pixelated
   upscale of a downsampled tiny canvas. */
const SWIN_STAGES = [
  { gridSide: 56, dim: 96  },
  { gridSide: 28, dim: 192 },
  { gridSide: 14, dim: 384 },
  { gridSide: 7,  dim: 768 },
];

function SwinStageThumb({ image, gridSide }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !image) return;
    const canvas = ref.current;
    const size = 120;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const w = image.naturalWidth || image.width;
    const h = image.naturalHeight || image.height;
    const crop = Math.min(w, h);
    const tiny = document.createElement('canvas');
    tiny.width = gridSide;
    tiny.height = gridSide;
    const tctx = tiny.getContext('2d');
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(image, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, gridSide, gridSide);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tiny, 0, 0, size, size);
    // subtle teal grid overlay so students see where the merged patches sit
    ctx.strokeStyle = 'rgba(20, 184, 166, 0.35)';
    ctx.lineWidth = 1;
    const step = size / gridSide;
    for (let i = 1; i < gridSide; i++) {
      ctx.beginPath(); ctx.moveTo(i * step, 0); ctx.lineTo(i * step, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * step); ctx.lineTo(size, i * step); ctx.stroke();
    }
  }, [image, gridSide]);
  return <canvas ref={ref} className="w-full aspect-square rounded bg-slate-950 border border-slate-800 block"/>;
}

function LiveDemoTab() {
  const [galleryId, setGalleryId] = useState('cat');
  const [customSrc, setCustomSrc] = useState(null);
  const imgSrc = customSrc || GALLERY.find(g => g.id === galleryId)?.src;

  const [patchSize, setPatchSize] = useState(72);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState('Slow');
  const [playClicked, markPlayClicked] = usePlayClickedOnce();
  // Swin-only layer toggle: 0 = W-MSA (regular), 1 = SW-MSA (shifted by ⌊M/2⌋).
  // Demonstrates the key Swin innovation directly inside the Live Demo.
  const [swinLayer, setSwinLayer] = useState(0);
  // Click-to-pin a query patch on each scan: pin then "show all edges"
  // draws every attention edge for that patch. Lets students compare
  // ViT's spaghetti starburst vs Swin's tidy in-window cluster.
  const [vitPinned, setVitPinned] = useState(null);
  const [swinPinned, setSwinPinned] = useState(null);

  // ----- Race-mode bookkeeping -----
  // Both scans operate at the same op-rate; ViT does N ops/patch and Swin
  // does M² ops/patch, so the *true* speedup is N/M² (often 6×–10×).
  // For the on-canvas race that ratio is too dramatic — Swin would finish
  // in the first 15% of progress while ViT is still warming up. So we cap
  // the visual speedup at a more readable value while keeping the ops
  // counters honest about the true ratio.
  const SCAN_SIZE = 360;
  const SCAN_GRID = Math.max(2, Math.floor(SCAN_SIZE / patchSize));
  const SCAN_N = SCAN_GRID * SCAN_GRID;
  const SCAN_M = 4;
  const SWIN_TRUE_SPEEDUP = Math.max(1, (SCAN_N - 1) / (SCAN_M * SCAN_M - 1));
  const SWIN_VISUAL_SPEEDUP = Math.min(2.4, SWIN_TRUE_SPEEDUP); // race finishes around 40% of progress
  const vitProgress = progress;
  const swinProgress = Math.min(1, progress * SWIN_VISUAL_SPEEDUP);
  // Live ops counters — keep the *true* op ratio so the numbers reflect
  // the actual work difference (the ratio shown in the headers is 6×+
  // even though the visual race only shows ~2× faster).
  // Each query attends to every key including itself — so the full
  // attention matrix has N×N (ViT) and N·M² (Swin) scores. Counting the
  // self-attention diagonal keeps the code aligned with the chart caption.
  const vitTotalOps = SCAN_N * SCAN_N;
  const swinTotalOps = SCAN_N * (SCAN_M * SCAN_M);
  const absOps = Math.floor(progress * vitTotalOps);
  const vitOpsCount = Math.min(vitTotalOps, absOps);
  const swinOpsCount = Math.min(swinTotalOps, absOps);
  const opsRatio = swinOpsCount > 0 ? vitOpsCount / swinOpsCount : 0;

  // Real classifier — runs ViT-Base/16 in the browser via transformers.js.
  // Loads the ONNX weights once on mount (~88 MB, cached by the browser),
  // then runs inference on every image change.
  const classifierRef = useRef(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelStatus, setModelStatus] = useState('modelLoading'); // modelLoading | inferring | ready | error
  const [modelMessage, setModelMessage] = useState('Loading transformers.js…');
  const [modelProgress, setModelProgress] = useState(0);
  const [realPreds, setRealPreds] = useState(null);
  const [rollout, setRollout] = useState(null); // real attention rollout from ViT-B/16 if available

  const vitRef = useRef();
  const swinRef = useRef();
  const vitMatRef = useRef();
  const swinMatRef = useRef();
  const imgRef = useRef();
  const attnRef = useRef(null);
  const [imgVersion, setImgVersion] = useState(0);

  const reset = useCallback(() => { setProgress(0); setPlaying(false); }, []);

  // Image load + attention computation
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      attnRef.current = computeAttention(img);
      drawScan(vitRef.current, img, 360, patchSize, vitProgress, 'vit', false, vitPinned);
      drawScan(swinRef.current, img, 360, patchSize, swinProgress, 'swin', swinLayer === 1, swinPinned);
      drawMatrix(vitMatRef.current, attnRef.current?.vit, vitProgress, 'vit');
      drawMatrix(swinMatRef.current, attnRef.current?.swin, swinProgress, 'swin');
      setImgVersion(v => v + 1);
    };
    img.src = imgSrc;
    setRealPreds(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgSrc]);

  useEffect(() => {
    if (!imgRef.current) return;
    drawScan(vitRef.current, imgRef.current, 360, patchSize, vitProgress, 'vit', false, vitPinned);
    drawScan(swinRef.current, imgRef.current, 360, patchSize, swinProgress, 'swin', swinLayer === 1, swinPinned);
    drawMatrix(vitMatRef.current, attnRef.current?.vit, vitProgress, 'vit');
    drawMatrix(swinMatRef.current, attnRef.current?.swin, swinProgress, 'swin');
  }, [patchSize, vitProgress, swinProgress, swinLayer, vitPinned, swinPinned]);

  useEffect(() => {
    if (!playing) return;
    const stepBySpeed = { Slow: 0.0025, Medium: 0.005, Fast: 0.0125 };
    const step = stepBySpeed[speed] ?? 0.005;
    const id = setInterval(() => {
      setProgress(p => {
        const next = p + step;
        if (next >= 1) { setPlaying(false); return 1; }
        return next;
      });
    }, 50);
    return () => clearInterval(id);
  }, [playing, speed]);

  // Load the in-browser ViT model + processor once on mount. We use
  // AutoModelForImageClassification (not the high-level pipeline) so we
  // can pass output_attentions:true to the forward call and pull real
  // per-layer attention out for the rollout heatmap.
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        setModelMessage('Loading transformers.js…');
        const mod = await import('@huggingface/transformers');
        mod.env.allowLocalModels = false;
        setModelMessage('Downloading ViT-Base/16 (~88 MB · cached after)');
        const progressCb = (data) => {
          if (data.status === 'progress' && !canceled) {
            setModelProgress(Math.round(data.progress || 0));
          }
        };
        const [processor, model] = await Promise.all([
          mod.AutoProcessor.from_pretrained('Xenova/vit-base-patch16-224', { progress_callback: progressCb }),
          mod.AutoModelForImageClassification.from_pretrained('Xenova/vit-base-patch16-224', { progress_callback: progressCb }),
        ]);
        if (canceled) return;
        classifierRef.current = { processor, model, RawImage: mod.RawImage };
        setModelMessage('');
        setModelReady(true);
      } catch (err) {
        if (!canceled) {
          setModelStatus('error');
          setModelMessage(String(err.message || err));
        }
      }
    })();
    return () => { canceled = true; };
  }, []);

  // Run inference on every image change once the model is loaded. We
  // call the model with output_attentions:true; if the ONNX export
  // returns attention tensors we compute attention rollout, otherwise
  // we fall back to the synthetic per-class heatmap.
  useEffect(() => {
    if (!modelReady || !imgSrc) return;
    let canceled = false;
    setModelStatus('inferring');
    setRealPreds(null);
    setRollout(null);
    (async () => {
      try {
        const { processor, model, RawImage } = classifierRef.current;
        const image = await RawImage.read(imgSrc);
        const inputs = await processor(image);
        const outputs = await model({ ...inputs, output_attentions: true });
        if (canceled) return;

        // Top-5 from logits using the model's id2label.
        const logits = outputs.logits.data;
        const probs = (() => {
          let mx = -Infinity;
          for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
          const ex = new Float32Array(logits.length);
          let sum = 0;
          for (let i = 0; i < logits.length; i++) { ex[i] = Math.exp(logits[i] - mx); sum += ex[i]; }
          for (let i = 0; i < logits.length; i++) ex[i] /= sum;
          return ex;
        })();
        const id2label = model.config?.id2label || {};
        const ranked = Array.from(probs).map((p, i) => ({ p, i }))
          .sort((a, b) => b.p - a.p).slice(0, 5);
        const cleaned = ranked.map(({ p, i }) => {
          const full = id2label[i] || id2label[String(i)] || `class_${i}`;
          const first = String(full).split(',')[0].trim();
          const label = first ? first[0].toUpperCase() + first.slice(1) : first;
          return { label, score: p };
        });
        setRealPreds(cleaned);

        // Attention rollout — only if the ONNX export emitted attentions.
        if (outputs.attentions && outputs.attentions.length > 0) {
          // eslint-disable-next-line no-console
          console.log('[ViT] received', outputs.attentions.length, 'attention tensors — computing rollout');
          const heat = computeAttentionRollout(outputs.attentions);
          if (heat) setRollout(heat);
          else console.warn('[ViT] rollout computation returned null');
        } else {
          // eslint-disable-next-line no-console
          console.warn('[ViT] outputs.attentions is empty — ONNX export likely does not expose attention. Falling back to synthetic heatmap.');
        }

        setModelStatus('ready');
      } catch (err) {
        if (!canceled) {
          setModelStatus('error');
          setModelMessage(String(err.message || err));
        }
      }
    })();
    return () => { canceled = true; };
  }, [modelReady, imgSrc]);

  // Status string passed to the predictions panel inside ClassContribution.
  const ccStatus = !modelReady ? 'modelLoading'
    : modelStatus === 'inferring' ? 'inferring'
    : modelStatus === 'error' ? 'error'
    : 'ready';

  return (
    <div className="space-y-2">
      {/* Compact header */}
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <h2 className="font-serif text-xl text-slate-100 tracking-tight">
            Watch the model classify
          </h2>
          <HowToUseBadge instructions={[
            'Pick one of the three gallery images.',
            'Hit Play (it shines until your first click) — the ViT and Swin scans animate together.',
            'Watch the FLOPs/time/$ and Memory charts climb in lockstep with the scan.',
            'Click any patch on either scan to pin a query and see all of its attention edges drawn at once.',
            'Use the Patch (px) buttons to change patch size, or the Speed buttons to change how fast Play runs.',
          ]}/>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          ViT-Base/16 (ImageNet-1K) · runs in your browser · pick an image, hit play
        </span>
      </div>

      {/* Image gallery — slim single row of thumbs to keep vertical budget low. */}
      <Card className="p-1.5">
        <div className="grid grid-cols-3 gap-1.5">
          {GALLERY.map(g => (
            <button
              key={g.id}
              onClick={() => { setCustomSrc(null); setGalleryId(g.id); reset(); }}
              className={`relative rounded-md overflow-hidden border-2 aspect-[5/2] transition-all
                ${galleryId === g.id ? 'border-amber-400 shadow shadow-amber-500/15' : 'border-slate-700 hover:border-slate-500'}`}
            >
              <img src={g.src} alt="" className="w-full h-full object-cover"/>
            </button>
          ))}
        </div>
      </Card>

      {/* Main row: ViT scan / Swin scan / predictions. On iPad portrait
          (md) we drop to 2-col with predictions spanning the row below.
          items-start prevents the shorter ViT card from stretching to
          match Swin's extra layer-toggle height (was leaving empty
          space at the bottom of the ViT card on iPad). */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 items-start">
        <Card className="p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Tag color="amber">ViT</Tag>
              <span className="text-[12px] text-slate-200">Global</span>
            </div>
            <span className="text-[10px] font-mono text-amber-300/90 tabular-nums">
              {vitOpsCount.toLocaleString()} ops · {Math.round(vitProgress * 100)}%
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">scan</div>
              <canvas
                ref={vitRef}
                className="w-full rounded bg-slate-950 aspect-square cursor-crosshair"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (rect.width === 0) return;
                  const x = (e.clientX - rect.left) / rect.width * SCAN_SIZE;
                  const y = (e.clientY - rect.top) / rect.height * SCAN_SIZE;
                  const px = Math.min(SCAN_GRID - 1, Math.max(0, (x / patchSize) | 0));
                  const py = Math.min(SCAN_GRID - 1, Math.max(0, (y / patchSize) | 0));
                  const idx = py * SCAN_GRID + px;
                  setVitPinned(prev => prev === idx ? null : idx);
                }}
              />
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">matrix</div>
              <canvas ref={vitMatRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2 leading-snug">
            {vitPinned != null
              ? <>Pinned patch <span className="text-rose-300">#{vitPinned}</span> · {SCAN_N - 1} attention edges drawn (click again to unpin).</>
              : <>Every patch attends to every other · dense matrix. <span className="text-slate-500">Click any patch to see all its edges.</span></>}
          </p>
        </Card>

        <Card className="p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Tag color="teal">Swin</Tag>
              <span className="text-[12px] text-slate-200">Windowed</span>
            </div>
            <span className="text-[10px] font-mono text-teal-300/90 tabular-nums">
              {swinOpsCount.toLocaleString()} ops · {Math.round(swinProgress * 100)}%
              {swinProgress >= 1 && vitProgress < 1 && <span className="text-emerald-300 ml-1">✓ done</span>}
            </span>
          </div>
          {/* Layer toggle: W-MSA ↔ SW-MSA. Swin's signature trick. */}
          <div className="flex gap-1 mb-2 bg-slate-800/40 rounded p-0.5 border border-slate-700/60">
            <button
              onClick={() => setSwinLayer(0)}
              className={`flex-1 px-2 py-1 rounded text-[10px] font-mono transition-all
                ${swinLayer === 0 ? 'bg-teal-500/25 text-teal-100' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Layer 1 · W-MSA
            </button>
            <button
              onClick={() => setSwinLayer(1)}
              className={`flex-1 px-2 py-1 rounded text-[10px] font-mono transition-all
                ${swinLayer === 1 ? 'bg-teal-500/25 text-teal-100' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Layer 2 · SW-MSA
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">scan</div>
              <canvas
                ref={swinRef}
                className="w-full rounded bg-slate-950 aspect-square cursor-crosshair"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (rect.width === 0) return;
                  const x = (e.clientX - rect.left) / rect.width * SCAN_SIZE;
                  const y = (e.clientY - rect.top) / rect.height * SCAN_SIZE;
                  const px = Math.min(SCAN_GRID - 1, Math.max(0, (x / patchSize) | 0));
                  const py = Math.min(SCAN_GRID - 1, Math.max(0, (y / patchSize) | 0));
                  const idx = py * SCAN_GRID + px;
                  setSwinPinned(prev => prev === idx ? null : idx);
                }}
              />
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">matrix</div>
              <canvas ref={swinMatRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2 leading-snug">
            {swinPinned != null
              ? <>Pinned patch <span className="text-rose-300">#{swinPinned}</span> · {SCAN_M * SCAN_M - 1} edges drawn (only its window).</>
              : swinLayer === 0
                ? <>Layer 1 · regular windows. <span className="text-slate-500">Click any patch — it draws only {SCAN_M * SCAN_M - 1} edges.</span></>
                : <>Layer 2 · shifted by ⌊M/2⌋. Old neighbors are now in different windows.</>}
          </p>
        </Card>

        {/* Swin · 4 stages of patch merging — fits in the 3rd column slot of
            another optional row; here we render below the per-class card.
            Wrapping in a Fragment so the parent grid stays 3 cols. */}

        <Card className="p-3 md:col-span-2 lg:col-span-1">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Brain size={14} className="text-amber-300"/>
              <span className="text-[12px] text-slate-200">Why this class?</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500">click → see evidence</span>
          </div>
          <ClassContribution
            key={imgVersion}
            image={imgRef.current}
            attentionRows={attnRef.current?.vit}
            gridN={MAT_N}
            preds={realPreds}
            status={ccStatus}
            statusMessage={modelMessage}
            statusProgress={modelProgress}
            rollout={rollout}
          />
        </Card>
      </div>

      {/* FLOPs + Memory side by side — both grow with the scan's progress.
          md: breakpoint so iPad portrait keeps them paired horizontally
          (avoids the tall stacked gap below the predictions card). */}
      <div className="grid md:grid-cols-2 gap-3">
        <Card className="p-3">
          <div className="flex items-baseline justify-between mb-2 flex-wrap gap-1">
            <div className="flex items-center gap-1.5">
              <Tag color="rose">ViT vs Swin</Tag>
              <span className="text-[12px] text-slate-200">FLOPs · time · $</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500">
              hit Play → both lines climb
            </span>
          </div>
          <FlopsCostCard progress={progress} />
        </Card>

        <Card className="p-3">
          <div className="flex items-baseline justify-between mb-2 flex-wrap gap-1">
            <div className="flex items-center gap-1.5">
              <Tag color="rose">Memory</Tag>
              <span className="text-[12px] text-slate-200">Attention scores stored</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500">
              ViT climbs {Math.max(1, Math.round(SCAN_N / (SCAN_M * SCAN_M)))}× steeper
            </span>
          </div>
          <StorageRiseChart
            progress={progress}
            vitTotal={vitTotalOps}
            swinTotal={swinTotalOps}
            vitOps={vitOpsCount}
            swinOps={swinOpsCount}
            N={SCAN_N}
            M={SCAN_M}
          />
        </Card>
      </div>

      {/* Controls strip — sits behind (below) the FLOPs+Memory row so the
          critical four (scans, FLOPs, Memory, Play) all share one screen.
          Uses md:grid-cols-4 so iPad-portrait keeps the row on one line. */}
      <Card className="p-2">
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-2 items-end">
          <div className="flex gap-2 items-center">
            {/* First-time-only "click me" hint, points right at the Play
                button. Disappears (forever, persisted) after first click. */}
            {!playClicked && (
              <span className="text-[11px] font-medium text-amber-300 anim-bounce flex items-center gap-1 select-none whitespace-nowrap">
                Click me <span aria-hidden className="text-base leading-none">👉</span>
              </span>
            )}
            <button
              onClick={() => { if (progress >= 1) setProgress(0); setPlaying(p => !p); markPlayClicked(); }}
              data-active={playing || playClicked ? 'true' : 'false'}
              className={`px-3 py-1.5 rounded-md bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-200 flex items-center gap-1.5 text-sm font-medium transition-all lift-on-hover
                ${!playClicked ? 'shimmer-cta anim-glow-amber' : ''}`}
            >
              {playing ? <Pause size={13}/> : <Play size={13}/>}
              {playing ? 'Pause' : (progress >= 1 ? 'Replay' : 'Play')}
            </button>
            <button
              onClick={reset}
              className="px-2.5 py-1.5 rounded-md bg-slate-800/60 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-all lift-on-hover"
              title="Reset"
            >
              <RotateCcw size={13}/>
            </button>
          </div>
          <Slider
            label="Progress"
            value={Math.round(progress * 100)}
            min={0} max={100} suffix="%"
            onChange={v => { setPlaying(false); setProgress(v / 100); }}
          />
          <Slider
            label="Speed"
            value={speed}
            options={['Slow', 'Medium', 'Fast']}
            onChange={setSpeed}
          />
          <div>
            <div className="flex items-baseline justify-between mb-1">
              <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Patch (px)</label>
              <span className="font-mono text-amber-300 text-sm">{patchSize}</span>
            </div>
            <div className="flex gap-1 items-stretch">
              {[48, 64, 72, 90, 120].map(opt => (
                <button
                  key={opt}
                  onClick={() => { setPatchSize(opt); reset(); }}
                  className={`flex-1 px-1.5 py-1.5 rounded text-xs font-mono border transition-all min-h-[34px]
                    ${patchSize === opt
                      ? 'bg-amber-500/20 border-amber-500/50 text-amber-200'
                      : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
                >
                  {opt}
                </button>
              ))}
              <input
                type="number"
                min={8}
                max={120}
                step={1}
                value={patchSize}
                onChange={e => {
                  const v = Math.max(8, Math.min(120, Number(e.target.value) || 8));
                  setPatchSize(v);
                  reset();
                }}
                className="w-12 px-1.5 py-1.5 rounded text-xs font-mono bg-slate-900 border border-slate-700 text-amber-200 focus:border-amber-500 focus:outline-none min-h-[34px]"
                aria-label="Custom patch size"
              />
            </div>
          </div>
        </div>
      </Card>

      {/* ----- Below-the-fold deep-dive cards (scroll for these) ----- */}

      {/* Swin · 2-layer reach lab — pick a patch, count its peers across
          both layers. */}
      <Card className="p-3">
        <div className="flex items-baseline justify-between mb-2 flex-wrap gap-2">
          <div className="flex items-center gap-1.5">
            <Tag color="teal">Swin</Tag>
            <span className="text-[12px] text-slate-200">2-layer reach lab</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">click any patch · count its peers across both layers</span>
        </div>
        <SwinReachLab image={imgRef.current} imgVersion={imgVersion}/>
      </Card>


      {/* Swin · 4 stages of patch merging — at the very bottom now. */}
      <Card className="p-3">
        <div className="flex items-baseline justify-between mb-2 flex-wrap gap-2">
          <div className="flex items-center gap-1.5">
            <Tag color="teal">Swin</Tag>
            <span className="text-[12px] text-slate-200">4 stages of patch merging</span>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            after each merge: tokens × ¼ · channels × 2 · receptive field × 2
          </span>
        </div>
        <div className="grid grid-cols-4 gap-2">
          {SWIN_STAGES.map((s, i) => (
            <div key={i} className="text-center">
              <SwinStageThumb image={imgRef.current} gridSide={s.gridSide}/>
              <div className="text-[10px] font-mono text-teal-300 mt-1">Stage {i + 1}</div>
              <div className="text-[9px] font-mono text-slate-400">{s.gridSide}×{s.gridSide} tokens</div>
              <div className="text-[9px] font-mono text-slate-500">{s.dim}-dim</div>
            </div>
          ))}
        </div>
        <p className="text-[10px] text-slate-400 mt-2 leading-snug">
          ViT keeps the same resolution all the way through. Swin halves it 3 times across 4 stages —
          earlier stages capture fine detail, deeper stages see broader context. That pyramid is what
          makes Swin a usable backbone for detection / segmentation.
        </p>
      </Card>
    </div>
  );
}

/* HowToUseBadge — small "★ HOW TO USE" decoration tucked into the
   top-left of a tab. Click to expand a popover of numbered tab-specific
   steps. Static (no shimmer / glow) so the eye still notices it without
   the page feeling busy. */
function HowToUseBadge({ instructions }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen(o => !o)}
        title="How to use this section"
        aria-expanded={open}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-amber-500/12 border border-amber-500/40 text-amber-200 text-[11px] font-mono uppercase tracking-[0.15em] hover:bg-amber-500/22 transition-all lift-on-hover"
      >
        <span className="text-amber-300 text-sm leading-none" aria-hidden>★</span>
        How to use
        <span className="opacity-60 text-[10px]">{open ? '▴' : '▾'}</span>
      </button>
      {open && (
        <div className="absolute left-0 mt-1 w-80 max-w-[92vw] rounded-lg border border-amber-500/40 bg-slate-900/95 backdrop-blur p-3 shadow-xl shadow-black/40 text-[12px] text-slate-200 leading-relaxed z-30 space-y-1.5">
          {instructions.map((step, i) => (
            <div key={i} className="flex gap-2">
              <span className="font-mono text-amber-300 shrink-0">{i + 1}.</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* usePlayClickedOnce — tracks "the user has hit Play at least once
   in this session". Resets on every page reload, so the shimmer and
   the 'Click me 👉' hint reappear each fresh visit and disappear
   only after the first click within that session. */
function usePlayClickedOnce() {
  const [clicked, setClicked] = useState(false);
  const markClicked = () => setClicked(true);
  return [clicked, markClicked];
}

/* StorageRiseChart — two area-line plots that grow with `progress`.
   x-axis = play progress 0→1.  y = cumulative attention scores held
   in memory.  ViT slope is N(N-1) per layer, Swin slope is N(M²-1) —
   so Swin plateaus once its (much smaller) total is reached, while
   ViT keeps climbing all the way to the right edge of the chart. */
function StorageRiseChart({ progress, vitTotal, swinTotal, vitOps, swinOps, N, M }) {
  const W = 640, H = 110;
  const PAD_L = 56, PAD_R = 12, PAD_T = 6, PAD_B = 22;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const SCORE_BYTES = 4;

  const fmtBytes = b => {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
    return Math.round(b) + ' B';
  };

  const yMaxBytes = Math.max(vitTotal, 1) * SCORE_BYTES;
  const yScale = v => PAD_T + innerH - (v / yMaxBytes) * innerH;
  const xScale = p => PAD_L + Math.max(0, Math.min(1, p)) * innerW;

  const swinPlateauP = Math.min(1, swinTotal / Math.max(1, vitTotal));
  const vitX = xScale(progress);
  const vitY = yScale(vitOps * SCORE_BYTES);
  const swinKneeX = xScale(Math.min(progress, swinPlateauP));
  const swinKneeY = yScale(Math.min(progress, swinPlateauP) * vitTotal * SCORE_BYTES);
  const swinTipX = xScale(progress);
  const swinTipY = yScale(swinOps * SCORE_BYTES);

  const baseline = PAD_T + innerH;
  const vitArea = `M ${PAD_L} ${baseline} L ${vitX} ${vitY} L ${vitX} ${baseline} Z`;
  const swinArea = progress <= swinPlateauP
    ? `M ${PAD_L} ${baseline} L ${swinTipX} ${swinTipY} L ${swinTipX} ${baseline} Z`
    : `M ${PAD_L} ${baseline} L ${swinKneeX} ${swinKneeY} L ${swinTipX} ${swinTipY} L ${swinTipX} ${baseline} Z`;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => t * yMaxBytes);
  const ratio = swinOps > 0 ? (vitOps / swinOps) : 0;

  return (
    <div>
      <div className="grid grid-cols-3 gap-2 mb-2">
        <div className="rounded border border-amber-500/40 bg-amber-500/[0.06] p-2">
          <div className="text-[10px] font-mono text-amber-300 uppercase tracking-wider">ViT live</div>
          <div className="font-mono text-amber-100 text-base tabular-nums">{fmtBytes(vitOps * SCORE_BYTES)}</div>
          <div className="text-[10px] font-mono text-slate-500">peak {fmtBytes(vitTotal * SCORE_BYTES)} · {vitTotal.toLocaleString()} scores</div>
        </div>
        <div className="rounded border border-teal-500/40 bg-teal-500/[0.06] p-2">
          <div className="text-[10px] font-mono text-teal-300 uppercase tracking-wider">Swin live</div>
          <div className="font-mono text-teal-100 text-base tabular-nums">{fmtBytes(swinOps * SCORE_BYTES)}</div>
          <div className="text-[10px] font-mono text-slate-500">peak {fmtBytes(swinTotal * SCORE_BYTES)} · {swinTotal.toLocaleString()} scores</div>
        </div>
        <div className="rounded border border-rose-500/40 bg-rose-500/[0.06] p-2">
          <div className="text-[10px] font-mono text-rose-300 uppercase tracking-wider">Ratio (ViT ÷ Swin)</div>
          <div className="font-mono text-rose-100 text-base tabular-nums">{ratio > 0 ? ratio.toFixed(2) + '×' : '—'}</div>
          <div className="text-[10px] font-mono text-slate-500">peak {(vitTotal / Math.max(1, swinTotal)).toFixed(2)}×</div>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none"
        style={{ height: '120px' }}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD_L} y1={yScale(t)} x2={W - PAD_R} y2={yScale(t)}
              stroke="rgba(148,163,184,0.14)" strokeWidth="0.6" />
            <text x={PAD_L - 6} y={yScale(t) + 3} fontSize="9" textAnchor="end"
              fill="#64748b" fontFamily="monospace">{fmtBytes(t)}</text>
          </g>
        ))}
        <line x1={PAD_L} y1={baseline} x2={W - PAD_R} y2={baseline} stroke="#475569" strokeWidth="0.8" />
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={baseline} stroke="#475569" strokeWidth="0.8" />

        {/* Faint full-progress targets so students can see the destination */}
        <line x1={PAD_L} y1={baseline} x2={xScale(1)} y2={yScale(vitTotal * SCORE_BYTES)}
          stroke="rgba(245,158,11,0.25)" strokeWidth="1" strokeDasharray="3 3" />
        <line x1={PAD_L} y1={baseline} x2={xScale(swinPlateauP)} y2={yScale(swinTotal * SCORE_BYTES)}
          stroke="rgba(20,184,166,0.25)" strokeWidth="1" strokeDasharray="3 3" />
        <line x1={xScale(swinPlateauP)} y1={yScale(swinTotal * SCORE_BYTES)} x2={xScale(1)} y2={yScale(swinTotal * SCORE_BYTES)}
          stroke="rgba(20,184,166,0.25)" strokeWidth="1" strokeDasharray="3 3" />

        {/* Live areas — these grow with progress */}
        <path d={vitArea} fill="rgba(245,158,11,0.32)" stroke="#f59e0b" strokeWidth="1.6" />
        <path d={swinArea} fill="rgba(20,184,166,0.32)" stroke="#14b8a6" strokeWidth="1.6" />

        {/* Tip dots */}
        <circle cx={vitX} cy={vitY} r="3.4" fill="#fbbf24" stroke="#0f172a" strokeWidth="1" />
        <circle cx={swinTipX} cy={swinTipY} r="3.4" fill="#2dd4bf" stroke="#0f172a" strokeWidth="1" />

        {/* x-axis labels */}
        <text x={PAD_L} y={H - 8} fontSize="10" fill="#64748b" fontFamily="monospace">play 0%</text>
        <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize="10" textAnchor="middle" fill="#64748b" fontFamily="monospace">progress →</text>
        <text x={W - PAD_R} y={H - 8} fontSize="10" textAnchor="end" fill="#64748b" fontFamily="monospace">100%</text>
      </svg>
      <p className="text-[10px] text-slate-400 mt-1 leading-snug">
        4 B per fp32 score · ViT stores N×N = {(N * N).toLocaleString()} · Swin only stores N·M² = {(N * M * M).toLocaleString()}.
      </p>

      {/* Real-world extrapolation — anchors the abstract chart in
          numbers from full-scale ViT-B/16 and Swin-T. Each row shows
          per-layer-per-head attention bytes at three input sizes; the
          headline number is the multiplicative ratio. */}
      <div className="mt-2 rounded-md border border-slate-700/70 bg-slate-950/40 p-2">
        <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1.5">
          Same math · real model sizes (per layer · per head · 4 B per score)
        </div>
        <div className="grid grid-cols-[auto_1fr_1fr_auto] gap-x-2 gap-y-1 text-[11px] font-mono items-center">
          <div className="text-slate-500 text-[10px]">image</div>
          <div className="text-amber-300 text-[10px]">ViT-B/16 · N²·4B</div>
          <div className="text-teal-300 text-[10px]">Swin-T · N·M²·4B</div>
          <div className="text-rose-300 text-[10px] text-right">ratio</div>
          {[
            { side: 224,  vit: 150e3,    swin: 38e3   },
            { side: 512,  vit: 4.19e6,   swin: 200e3  },
            { side: 1024, vit: 67.1e6,   swin: 802e3  },
          ].map(({ side, vit, swin }) => {
            const fmt = b => b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : b >= 1e3 ? (b / 1e3).toFixed(0) + ' KB' : b + ' B';
            return (
              <React.Fragment key={side}>
                <div className="text-slate-300 tabular-nums">{side}²</div>
                <div className="text-amber-200 tabular-nums">{fmt(vit)}</div>
                <div className="text-teal-200 tabular-nums">{fmt(swin)}</div>
                <div className="text-rose-200 tabular-nums text-right">{Math.round(vit / swin).toLocaleString()}×</div>
              </React.Fragment>
            );
          })}
        </div>
        <div className="text-[10px] text-slate-500 italic mt-1.5 leading-snug">
          ViT-Base has 12 layers × 12 heads, so multiply each ViT cell by ~144 — at 1024² that's ~9.7 GB of attention scores alone, before any other activations.
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   PRACTICE — exam-style questions with hints + explanations
   ========================================================= */

const QUESTIONS = [
  {
    id: 'q1',
    question: 'Why does ViT split an image into patches instead of feeding individual pixels to the transformer?',
    hints: [
      'Think about what kind of input a transformer naturally consumes — and how big it can practically be.',
      'A $224 \\times 224$ image has 50,176 pixels. Self-attention is $O(N^2)$, so attending pixel-to-pixel would mean a $50{,}000 \\times 50{,}000$ score matrix per layer.',
    ],
    options: [
      { text: 'Patches make image data fit in GPU memory by compressing it.', correct: false,
        explanation: 'Patches don\'t compress information — they reshape it. Each patch is still a flat vector of pixel values, projected linearly into a token embedding.' },
      { text: 'Transformers consume sequences of tokens, and patches turn the image into a manageably short sequence.', correct: true,
        explanation: 'Right. ViT treats the image as $N = (H \\cdot W)/P^2$ patch-tokens. With $P=16$ on a $224^2$ image you get 196 tokens — small enough that $O(N^2)$ attention is feasible.' },
      { text: 'CNNs need patches to compute convolutions efficiently.', correct: false,
        explanation: 'CNNs don\'t use patch tokens — they slide kernels over the full image. Patching is specific to transformer-based vision models.' },
      { text: 'The softmax in attention requires a fixed input length of 196.', correct: false,
        explanation: 'Softmax is length-agnostic. It\'s the quadratic compute of attention, not softmax, that constrains sequence length.' },
    ],
  },
  {
    id: 'q2',
    question: 'What is the [CLS] token in ViT?',
    hints: [
      'It exists at the input of the transformer but isn\'t computed from the image itself.',
      'It\'s a learned vector — and after the last layer, its output gets fed into the classification head.',
    ],
    options: [
      { text: 'The first patch (top-left) in raster scan order.', correct: false,
        explanation: 'The first patch is just patch 0 — an image patch like any other. The [CLS] token is prepended *in addition to* the patches.' },
      { text: 'A learned embedding prepended to the patch sequence whose final hidden state is used as the image representation.', correct: true,
        explanation: 'Yes. [CLS] is a learnable parameter. Through self-attention it gathers information from all patch tokens; its final output is fed to a linear classifier.' },
      { text: 'A position embedding for the patch with the strongest activation.', correct: false,
        explanation: 'Position embeddings are added to all tokens (including [CLS]) and depend on position, not activation.' },
      { text: 'The class label predicted by the model on the previous training example.', correct: false,
        explanation: '[CLS] is fixed and learned during training — it does not depend on previous predictions or examples.' },
    ],
  },
  {
    id: 'q3',
    question: 'What is the asymptotic compute cost of full self-attention over N tokens with embedding dimension d?',
    hints: [
      'Each token computes a similarity score against every other token.',
      'You build an $N \\times N$ matrix of scores, then do a softmax and weighted sum.',
    ],
    options: [
      { text: '$O(N \\cdot d)$', correct: false,
        explanation: 'That\'s the cost of a single projection (e.g. computing one token\'s $Q$ vector), not of full attention.' },
      { text: '$O(N \\log N \\cdot d)$', correct: false,
        explanation: 'Sub-quadratic costs like this come up in efficient/sparse attention variants, but standard self-attention is fully quadratic in $N$.' },
      { text: '$O(N^2 \\cdot d)$', correct: true,
        explanation: 'Computing the score matrix $Q K^{\\top}$ is $O(N^2 \\cdot d)$, and the weighted sum $A V$ is also $O(N^2 \\cdot d)$. This quadratic cost is exactly what makes high-resolution images expensive — and what Swin sidesteps with windows.' },
      { text: '$O(d^2)$', correct: false,
        explanation: '$O(d^2)$ is the cost of multiplying a single token by a $d \\times d$ projection matrix — a per-token cost, not the full attention cost over all tokens.' },
    ],
  },
  {
    id: 'q4',
    question: 'Why does a transformer need explicit position embeddings?',
    hints: [
      'Imagine shuffling the order of the input tokens. Does pure self-attention notice?',
      'Self-attention treats its input as a *set*, not a sequence — without position info it has no idea which patch was where.',
    ],
    options: [
      { text: 'Self-attention is permutation-invariant: it produces the same output for any reordering of the inputs.', correct: true,
        explanation: 'Right. Attention only depends on pairwise affinities, not on input order. Position embeddings break that symmetry by injecting "where" each token came from.' },
      { text: 'Different patches have different sizes, so positions vary.', correct: false,
        explanation: 'Patches are uniform in size (e.g. $16 \\times 16$). Position embeddings exist regardless of whether patch sizes vary.' },
      { text: 'They make the softmax numerically stable.', correct: false,
        explanation: 'Numerical stability of softmax is handled by subtracting the max logit. Position embeddings serve a structural, not numerical, purpose.' },
      { text: 'They encode the class label at training time.', correct: false,
        explanation: 'Position embeddings encode location, not class. The model never sees labels through its position channel.' },
    ],
  },
  {
    id: 'q5',
    question: 'Where does Swin\'s main efficiency win over ViT come from?',
    hints: [
      'It changes one specific aspect of how attention is computed.',
      'It restricts which tokens can attend to which — locally rather than globally.',
    ],
    options: [
      { text: 'It uses fewer attention heads.', correct: false,
        explanation: 'Head count doesn\'t change asymptotic cost. Both ViT and Swin typically use multi-head attention with similar head counts.' },
      { text: 'It restricts attention to local non-overlapping windows of patches.', correct: true,
        explanation: 'Yes. Each window has $M^2$ tokens, so attention inside one window costs $O(M^4 \\cdot d)$. With $N / M^2$ windows total, the cost is $O(N \\cdot M^2 \\cdot d)$ — *linear* in $N$ instead of quadratic. That is the headline result of the Swin paper.' },
      { text: 'It quantizes weights to int8.', correct: false,
        explanation: 'Quantization is an orthogonal optimization, not part of Swin\'s core design.' },
      { text: 'It skips MLP layers between attention blocks.', correct: false,
        explanation: 'Swin keeps the standard transformer block (attention + MLP). It only changes the attention\'s scope.' },
    ],
  },
  {
    id: 'q6',
    question: 'Why does Swin alternate between regular and shifted windows in successive layers?',
    hints: [
      'Without shifting, what could *never* happen between two adjacent patches?',
      'Two patches that sit on either side of a window boundary can\'t attend to each other in a fixed-window scheme.',
    ],
    options: [
      { text: 'Shifting reduces the number of windows, saving compute.', correct: false,
        explanation: 'Shifted windows actually create *more* boundary cases (handled by cyclic shift + masking), not fewer windows.' },
      { text: 'Shifting prevents overfitting by acting as a regularizer.', correct: false,
        explanation: 'Shifting isn\'t a regularization technique — it\'s a structural mechanism for cross-window communication.' },
      { text: 'It allows information to flow across window boundaries between layers.', correct: true,
        explanation: 'Right. In layer $\\ell$ a patch sees patches in its window. In layer $\\ell+1$ the windows are shifted by half a window, so what used to be a boundary is now interior — patches that were separated can now attend to each other.' },
      { text: 'It allows arbitrary input resolutions.', correct: false,
        explanation: 'Arbitrary resolutions are handled by patch-merging and padding, not by window shifting.' },
    ],
  },
  {
    id: 'q7',
    question: 'What does multi-head self-attention give you that single-head attention does not?',
    hints: [
      'Multiple heads run in parallel, each with their own Q/K/V projections.',
      'Different heads can specialize in different *kinds* of relationships (e.g. nearby patches, color similarity, distant context).',
    ],
    options: [
      { text: 'Lower asymptotic cost than single-head attention.', correct: false,
        explanation: 'Total cost is similar; the $d$-dim space is just split across heads. Multi-head is about expressivity, not speed.' },
      { text: 'Multiple parallel attention "subspaces", each learning a different pattern of relationships.', correct: true,
        explanation: 'Yes. Each head projects $Q/K/V$ into a smaller $d/h$-dim space and computes its own attention map. The outputs are concatenated, letting the layer attend to several relational structures at once.' },
      { text: 'A learned hierarchy of resolutions.', correct: false,
        explanation: 'Hierarchy across resolutions is what Swin\'s patch-merging provides, not what multi-head attention provides.' },
      { text: 'Built-in positional information.', correct: false,
        explanation: 'Positional information comes from explicit position embeddings (or relative-position biases in Swin), not from multi-head structure.' },
    ],
  },
  {
    id: 'q8',
    question: 'How does Swin produce hierarchical, multi-scale feature maps deeper in the network?',
    hints: [
      'Look at how the spatial resolution shrinks between Swin stages.',
      'Four neighboring patches get concatenated and projected into one — like a learned downsampling.',
    ],
    options: [
      { text: 'Strided convolutions between stages.', correct: false,
        explanation: 'Swin is convolution-free in its main path. Down-sampling is done by patch-merging, not strided conv.' },
      { text: 'A learned $2 \\times 2$ patch-merging layer that combines four neighboring patches into one.', correct: true,
        explanation: 'Yes. At each stage boundary, every $2 \\times 2$ group of patches is concatenated ($4 \\cdot C$ channels) then linearly projected back to $2C$ channels. Spatial resolution halves; channels double — exactly like a CNN feature pyramid.' },
      { text: 'By dropping every other patch.', correct: false,
        explanation: 'Dropping patches would lose information. Patch-merging *combines* them so no information is discarded.' },
      { text: 'By resizing the input image between stages.', correct: false,
        explanation: 'The input image is fixed. The hierarchy is built inside the network via patch-merging.' },
    ],
  },
  {
    id: 'q9',
    question: 'How does ViT\'s receptive field at layer 1 compare to a CNN\'s at the same layer?',
    hints: [
      'Receptive field = how much of the input one neuron can "see".',
      'In one attention layer, every token can directly attend to every other token.',
    ],
    options: [
      { text: 'Smaller — patches are fixed-size and local.', correct: false,
        explanation: 'A patch *embedding* is local, but attention then mixes information across all patches in one shot.' },
      { text: 'Roughly the same as a CNN\'s $3 \\times 3$ kernel receptive field.', correct: false,
        explanation: 'A $3 \\times 3$ kernel covers 9 pixels. ViT\'s first attention layer can mix information from *every* patch.' },
      { text: 'Global — every patch can attend to every other patch in a single layer.', correct: true,
        explanation: 'Right. That\'s the headline difference. CNNs build up the receptive field gradually through stacked convolutions; ViT has it from layer 1 — at the cost of $O(N^2)$ attention.' },
      { text: 'Always zero in the first layer; only later layers see other patches.', correct: false,
        explanation: 'Self-attention mixes tokens at every layer, including the first. The patch-embedding step before it is local, but attention is global.' },
    ],
  },
  {
    id: 'q10',
    question: 'You have a $224 \\times 224$ image with full self-attention and patch size 16. If you increase resolution to $448 \\times 448$ keeping patch size at 16, by what factor does the cost of one attention layer grow?',
    hints: [
      'First figure out how many patches you have at each resolution.',
      'Self-attention is $O(N^2)$. Quadrupling $N$ quadruples $N$ — and squares the cost.',
    ],
    options: [
      { text: '$2\\times$', correct: false,
        explanation: 'Doubling resolution doesn\'t just double the patches — it quadruples them ($4\\times$ in 2D).' },
      { text: '$4\\times$', correct: false,
        explanation: '$4\\times$ would be the right answer if attention were $O(N)$. It\'s not — it\'s $O(N^2)$.' },
      { text: '$16\\times$', correct: true,
        explanation: 'Right. $224/16 = 14 \\Rightarrow 196$ patches. $448/16 = 28 \\Rightarrow 784$ patches ($4\\times$ more). Self-attention is $O(N^2)$, so $4^2 = 16\\times$ more compute. This is exactly why Swin\'s linear-cost windowed attention matters at high resolutions.' },
      { text: '$256\\times$', correct: false,
        explanation: 'Too high. You\'d need patches to grow $16\\times$ (not $4\\times$) for $16^2 = 256\\times$ cost. Here patches grow only $4\\times$.' },
    ],
  },
];

function Question({ q, index }) {
  const [selected, setSelected] = useState(null);
  const [checked, setChecked] = useState(false);
  const [hint1, setHint1] = useState(false);
  const [hint2, setHint2] = useState(false);

  const correctIdx = q.options.findIndex(o => o.correct);
  const isCorrect = selected === correctIdx;

  return (
    <Card className="p-5">
      <div className="flex items-baseline gap-3 mb-4">
        <span className="font-mono text-xs text-amber-300/80 shrink-0">Q{index}</span>
        <h3 className="font-serif text-lg text-slate-100 leading-snug">{renderWithMath(q.question)}</h3>
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        <button
          onClick={() => setHint1(v => !v)}
          className={`px-2.5 py-1 rounded text-[11px] font-mono border flex items-center gap-1.5 transition-all
            ${hint1 ? 'bg-amber-500/15 border-amber-500/40 text-amber-200' : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
        >
          <Lightbulb size={11}/> Hint 1 {hint1 ? '−' : '+'}
        </button>
        <button
          onClick={() => setHint2(v => !v)}
          className={`px-2.5 py-1 rounded text-[11px] font-mono border flex items-center gap-1.5 transition-all
            ${hint2 ? 'bg-amber-500/15 border-amber-500/40 text-amber-200' : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'}`}
        >
          <Lightbulb size={11}/> Hint 2 (stronger) {hint2 ? '−' : '+'}
        </button>
      </div>
      {hint1 && (
        <div className="mb-2 px-3 py-2 rounded bg-amber-500/[0.07] border border-amber-500/25 text-[13px] text-amber-100/90 leading-relaxed">
          {renderWithMath(q.hints[0])}
        </div>
      )}
      {hint2 && (
        <div className="mb-3 px-3 py-2 rounded bg-amber-500/[0.12] border border-amber-500/40 text-[13px] text-amber-100/95 leading-relaxed">
          {renderWithMath(q.hints[1])}
        </div>
      )}

      <div className="space-y-2 mb-4">
        {q.options.map((opt, i) => {
          const letter = String.fromCharCode(65 + i);
          let cls;
          if (checked) {
            if (i === correctIdx) cls = 'bg-emerald-500/12 border-emerald-500/50 text-emerald-50';
            else if (i === selected) cls = 'bg-rose-500/12 border-rose-500/50 text-rose-50';
            else cls = 'bg-slate-800/30 border-slate-700/60 text-slate-400';
          } else {
            cls = selected === i
              ? 'bg-amber-500/12 border-amber-500/50 text-amber-50'
              : 'bg-slate-800/40 border-slate-700/60 text-slate-300 hover:border-slate-500';
          }
          return (
            <button
              key={i}
              onClick={() => { if (!checked) setSelected(i); }}
              disabled={checked}
              className={`w-full text-left px-4 py-2.5 rounded-lg border transition-all ${cls} ${checked ? 'cursor-default' : 'cursor-pointer'}`}
            >
              <div className="flex items-start gap-3">
                <span className={`font-mono text-xs mt-0.5 shrink-0 ${
                  checked && i === correctIdx ? 'text-emerald-300'
                    : checked && i === selected ? 'text-rose-300'
                    : 'opacity-60'
                }`}>
                  {letter}
                </span>
                <span className="text-sm flex-1">{renderWithMath(opt.text)}</span>
                {checked && i === correctIdx && <Check size={14} className="text-emerald-300 mt-0.5 shrink-0"/>}
                {checked && i === selected && i !== correctIdx && <X size={14} className="text-rose-300 mt-0.5 shrink-0"/>}
              </div>
              {checked && (
                <p className="mt-2 text-[12px] text-slate-300/80 leading-relaxed pl-7">
                  {renderWithMath(opt.explanation)}
                </p>
              )}
            </button>
          );
        })}
      </div>

      <div className="flex items-center justify-between gap-3 flex-wrap">
        {!checked ? (
          <button
            onClick={() => setChecked(true)}
            disabled={selected === null}
            className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all
              ${selected === null
                ? 'bg-slate-800/40 border-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/50 text-amber-100'}`}
          >
            Check answer
          </button>
        ) : (
          <div className="flex items-center gap-3">
            <span className={`text-sm font-medium flex items-center gap-1.5 ${isCorrect ? 'text-emerald-300' : 'text-rose-300'}`}>
              {isCorrect ? <Check size={14}/> : <X size={14}/>}
              {isCorrect ? 'Correct' : 'Not quite — see explanations above'}
            </span>
            <button
              onClick={() => { setChecked(false); setSelected(null); setHint1(false); setHint2(false); }}
              className="text-[11px] font-mono text-slate-400 hover:text-slate-200 underline underline-offset-2"
            >
              try again
            </button>
          </div>
        )}
      </div>
    </Card>
  );
}

function QuizTab() {
  return (
    <div className="space-y-6">
      <div>
        <span className="font-mono text-[10px] tracking-[0.3em] text-amber-400/70 uppercase">Practice</span>
        <h2 className="font-serif text-3xl text-slate-100 mt-1 mb-2 tracking-tight">
          Self-check questions
        </h2>
        <p className="text-slate-300/90 leading-relaxed max-w-2xl">
          Ten exam-style questions covering the topics in this tutorial. Try each one cold;
          if you're stuck, expand the hints (the second is more direct than the first).
          Click <em>Check answer</em> to see whether you got it — and an explanation for
          every option, not just the correct one.
        </p>
      </div>
      {QUESTIONS.map((q, i) => (
        <Question key={q.id} q={q} index={i + 1} />
      ))}
    </div>
  );
}

/* =========================================================
   ROOT
   ========================================================= */

const TABS = [
  { id: 'live',     label: 'Live Demo',       icon: Sparkles },
  { id: 'overview', label: 'Overview',        icon: BookOpen },
  { id: 'patch',    label: 'Patch Embedding', icon: Grid3x3 },
  { id: 'pos',      label: 'Position + CLS',  icon: Hash },
  { id: 'attn',     label: 'Self-Attention',  icon: Eye },
  { id: 'mha',      label: 'Multi-Head',      icon: Network },
  { id: 'pipeline', label: 'ViT Pipeline',    icon: Workflow },
  { id: 'window',   label: 'Windows (Swin)',  icon: Box },
  { id: 'shifted',  label: 'Shifted Windows', icon: Move },
  { id: 'hier',     label: 'Hierarchy',       icon: GitBranch },
  { id: 'compare',  label: 'ViT vs Swin',     icon: Microscope },
  { id: 'quiz',     label: 'Practice',        icon: HelpCircle },
];

/* TabNav — previous / next strip rendered at the bottom of every tab.
   Turns the 12 isolated panels into a real reading path. */
function TabNav({ tab, setTab }) {
  const idx = TABS.findIndex(t => t.id === tab);
  if (idx < 0) return null;
  const prev = idx > 0 ? TABS[idx - 1] : null;
  const next = idx < TABS.length - 1 ? TABS[idx + 1] : null;
  return (
    <div className="mt-10 pt-5 border-t border-slate-800/60">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        {prev ? (
          <button
            onClick={() => { setTab(prev.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
            className="group flex items-center gap-2 px-3 py-2 rounded-md border border-slate-700/60 hover:border-amber-500/50 bg-slate-900/40 hover:bg-amber-500/5 transition-all"
          >
            <span className="text-amber-400/70 group-hover:text-amber-300">←</span>
            <div className="text-left">
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Previous</div>
              <div className="text-sm text-slate-200 group-hover:text-amber-200">{prev.label}</div>
            </div>
          </button>
        ) : <div/>}

        <div className="text-[11px] font-mono text-slate-500">
          {idx + 1} / {TABS.length}
        </div>

        {next ? (
          <button
            onClick={() => { setTab(next.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
            className="group flex items-center gap-2 px-3 py-2 rounded-md border border-slate-700/60 hover:border-amber-500/50 bg-slate-900/40 hover:bg-amber-500/5 transition-all"
          >
            <div className="text-right">
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500">Next</div>
              <div className="text-sm text-slate-200 group-hover:text-amber-200">{next.label}</div>
            </div>
            <span className="text-amber-400/70 group-hover:text-amber-300">→</span>
          </button>
        ) : <div/>}
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('live');

  // Accessibility prefs — restored from localStorage so students don't
  // have to re-pick on every visit.
  const [fontSize, setFontSize] = useState(() => {
    try { return localStorage.getItem('a11y_font') || 'default'; } catch { return 'default'; }
  });
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('a11y_theme') || 'dark'; } catch { return 'dark'; }
  });
  const [palette, setPalette] = useState(() => {
    try { return localStorage.getItem('a11y_palette') || 'default'; } catch { return 'default'; }
  });

  useEffect(() => {
    const px = { default: '16px', large: '17.5px', xl: '19px' }[fontSize] || '16px';
    document.documentElement.style.fontSize = px;
    document.documentElement.dataset.fontsize = fontSize;
    try { localStorage.setItem('a11y_font', fontSize); } catch {}
  }, [fontSize]);

  useEffect(() => {
    try { localStorage.setItem('a11y_theme', theme); } catch {}
  }, [theme]);

  useEffect(() => {
    try { localStorage.setItem('a11y_palette', palette); } catch {}
  }, [palette]);

  // Filter applied to the main content wrapper (NOT the panel) so the
  // accessibility panel stays in normal colours. Each palette option uses
  // an SVG colour-matrix filter (defined in <A11ySvgFilters/>) so the hue
  // shifts are a proper LMS-based Daltonization rather than naive
  // hue-rotate. Filters compose with the optional Light theme inversion.
  const contentFilter = (() => {
    const parts = [];
    if (theme === 'light') parts.push('invert(0.94)', 'hue-rotate(180deg)');
    if (palette === 'bold') parts.push('saturate(1.45)', 'contrast(1.12)');
    if (palette === 'deuteran') parts.push('url(#a11y-deuteran)');
    if (palette === 'protan')   parts.push('url(#a11y-protan)');
    if (palette === 'tritan')   parts.push('url(#a11y-tritan)');
    if (palette === 'achromat') parts.push('grayscale(1)', 'contrast(1.18)');
    return parts.length ? parts.join(' ') : 'none';
  })();

  const render = () => {
    switch (tab) {
      case 'live': return <LiveDemoTab />;
      case 'overview': return <OverviewTab />;
      case 'patch': return <PatchTab />;
      case 'pos': return <PositionTab />;
      case 'attn': return <AttentionTab />;
      case 'mha': return <MultiHeadTab />;
      case 'pipeline': return <PipelineTab />;
      case 'window': return <WindowTab />;
      case 'shifted': return <ShiftedTab />;
      case 'hier': return <HierarchyTab />;
      case 'compare': return <CompareTab />;
      case 'quiz': return <QuizTab />;
      default: return <LiveDemoTab />;
    }
  };

  return (
    <div className="min-h-screen text-slate-200 font-sans relative overflow-x-hidden bg-slate-950"
      style={{ background: '#0a0e1a' }}>

      <A11yPanel
        fontSize={fontSize} setFontSize={setFontSize}
        theme={theme} setTheme={setTheme}
        palette={palette} setPalette={setPalette}
      />
      <A11ySvgFilters />

      {/* Wrap the entire UI so the filter only affects content — panel stays clean. */}
      <div style={{ filter: contentFilter, transition: 'filter 0.2s' }}>

      {/* Header */}
      <header className="relative border-b border-slate-800/60 backdrop-blur-md bg-slate-950/40">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-baseline justify-between gap-4 flex-wrap">
            <div>
              <div className="flex items-baseline gap-3 mb-1">
                <span className="font-mono text-[10px] tracking-[0.3em] text-amber-400/70 uppercase">
                  An interactive tutorial
                </span>
              </div>
              <h1 className="font-serif text-3xl md:text-4xl text-slate-100 tracking-tight leading-none">
                Vision Transformers, end to end
                <span className="text-amber-300"> .</span>
              </h1>
              <p className="text-slate-400 text-sm mt-2 max-w-2xl">
                A guided walk through ViT and Swin Transformer — patches, attention, windows, shifts, hierarchy.
                Every diagram runs real math on a real (small) image.
              </p>
            </div>
            <div className="text-right">
              <div className="font-serif text-[13px] text-slate-300">CS&nbsp;4782</div>
              <div className="font-mono text-[10px] text-slate-500 mt-0.5">Spring 2026 · Cornell</div>
              <div className="font-serif text-[12px] text-amber-300/90 mt-1 italic">Jupiter</div>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <nav className="sticky top-0 z-20 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-2">
          {(() => {
            const breakAfter = TABS.findIndex(t => t.id === 'window');
            const rows = [TABS.slice(0, breakAfter + 1), TABS.slice(breakAfter + 1)];
            return rows.map((row, ri) => (
              <div
                key={ri}
                className={`flex flex-wrap gap-1 ${ri === 1 ? 'mt-1' : ''}`}
              >
                {row.map(t => {
                  const Icon = t.icon;
                  const active = tab === t.id;
                  return (
                    <button
                      key={t.id}
                      onClick={() => setTab(t.id)}
                      className={`px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-all lift-on-hover
                        ${active
                          ? 'bg-amber-500/15 text-amber-200 border border-amber-500/40'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent'}`}
                    >
                      <Icon size={14} />
                      <span className="font-medium">{t.label}</span>
                    </button>
                  );
                })}
              </div>
            ));
          })()}
        </div>
      </nav>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-10 relative">
        {render()}
        <TabNav tab={tab} setTab={setTab} />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 mt-12 py-6 text-center">
        <div className="max-w-7xl mx-auto px-6 text-[12px] font-mono text-slate-500">
          References: Dosovitskiy et al., <em>An Image is Worth 16×16 Words</em> (ICLR 2021) ·
          Liu et al., <em>Swin Transformer</em> (ICCV 2021) ·
          Built for CS 4782, Spring 2026 — Jupiter.
        </div>
      </footer>

      <style>{`
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { scrollbar-width: none; }

        /* Touch behaviour for tablets / phones. Removes the 300 ms
           click delay on iOS Safari, suppresses the gray flash on
           tap, and stops double-tap from zooming canvases that act
           as pointers (scan grids, position simulator, etc.). */
        * { -webkit-tap-highlight-color: transparent; }
        button, canvas, [role="button"] { touch-action: manipulation; }
        canvas { -webkit-touch-callout: none; user-select: none; }

        /* ---- Friendly micro-animations.  All under 2 s and gentle —
           the goal is to draw the eye to the next thing to click,
           not to be distracting. Respect prefers-reduced-motion. */
        @keyframes glow-amber {
          0%, 100% { box-shadow: 0 0 0   rgba(245, 158, 11, 0.0),
                                 0 0 0   rgba(245, 158, 11, 0.0); }
          50%      { box-shadow: 0 0 14px rgba(245, 158, 11, 0.45),
                                 0 0 28px rgba(245, 158, 11, 0.18); }
        }
        @keyframes soft-pulse {
          0%, 100% { transform: scale(1);    opacity: 1;   }
          50%      { transform: scale(1.04); opacity: 0.92; }
        }
        @keyframes attention-bounce {
          0%, 100% { transform: translateY(0); }
          50%      { transform: translateY(-3px); }
        }
        @keyframes shimmer-sweep {
          0%   { background-position: -180% 0; }
          100% { background-position:  280% 0; }
        }

        .anim-glow-amber  { animation: glow-amber 2.4s ease-in-out infinite; }
        .anim-soft-pulse  { animation: soft-pulse 2.0s ease-in-out infinite; }
        .anim-bounce      { animation: attention-bounce 1.6s ease-in-out infinite; display: inline-block; }

        /* Shimmer overlay on the primary Play CTA — a faint diagonal
           gradient that sweeps right every few seconds.  Pause when
           the button is in 'playing' state (we use data-active=true). */
        .shimmer-cta {
          position: relative;
          overflow: hidden;
        }
        .shimmer-cta::after {
          content: "";
          position: absolute; inset: 0;
          background: linear-gradient(110deg,
            transparent 30%, rgba(255,255,255,0.18) 50%, transparent 70%);
          background-size: 200% 100%;
          animation: shimmer-sweep 3.4s linear infinite;
          pointer-events: none;
        }
        .shimmer-cta[data-active="true"]::after { animation: none; opacity: 0; }

        /* Subtle hover lift for tab nav and card-style buttons. */
        .lift-on-hover { transition: transform 0.18s ease, box-shadow 0.18s ease; }
        .lift-on-hover:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,0.35); }

        @media (prefers-reduced-motion: reduce) {
          .anim-glow-amber, .anim-soft-pulse, .anim-bounce,
          .shimmer-cta::after { animation: none !important; }
          .lift-on-hover:hover { transform: none; }
        }

        /* Font-size scaling for the A11y panel.  The root font-size
           change covers rem-based utilities (text-xs, text-sm, …) but
           Tailwind also generates absolute-pixel classes (text-[10px]
           …) that ignore that — we explicitly bump those plus boost
           the small named sizes so captions get noticeably larger. */
        html[data-fontsize="large"] .text-xs   { font-size: 0.85rem !important; line-height: 1.25rem !important; }
        html[data-fontsize="large"] .text-sm   { font-size: 0.95rem !important; line-height: 1.4rem !important; }
        html[data-fontsize="large"] .text-base { font-size: 1.05rem !important; }

        html[data-fontsize="large"] .text-\\[8px\\]   { font-size: 10px  !important; }
        html[data-fontsize="large"] .text-\\[9px\\]   { font-size: 11px  !important; }
        html[data-fontsize="large"] .text-\\[10px\\]  { font-size: 12px  !important; }
        html[data-fontsize="large"] .text-\\[11px\\]  { font-size: 13px  !important; }
        html[data-fontsize="large"] .text-\\[12px\\]  { font-size: 14px  !important; }
        html[data-fontsize="large"] .text-\\[13px\\]  { font-size: 15px  !important; }
        html[data-fontsize="large"] .text-\\[14px\\]  { font-size: 16px  !important; }

        /* Catch any other inline pixel font-size on small descriptive text.
           We only target sizes ≤ 14px to avoid blowing up large headings. */
        html[data-fontsize="large"] [style*="font-size: 9px"],
        html[data-fontsize="large"] [style*="font-size:9px"]   { font-size: 11px  !important; }
        html[data-fontsize="large"] [style*="font-size: 10px"],
        html[data-fontsize="large"] [style*="font-size:10px"]  { font-size: 12px  !important; }
        html[data-fontsize="large"] [style*="font-size: 11px"],
        html[data-fontsize="large"] [style*="font-size:11px"]  { font-size: 13px  !important; }

        html[data-fontsize="xl"] .text-xs   { font-size: 0.95rem !important; line-height: 1.35rem !important; }
        html[data-fontsize="xl"] .text-sm   { font-size: 1.05rem !important; line-height: 1.5rem !important; }
        html[data-fontsize="xl"] .text-base { font-size: 1.15rem !important; }

        html[data-fontsize="xl"] .text-\\[8px\\]   { font-size: 12px !important; }
        html[data-fontsize="xl"] .text-\\[9px\\]   { font-size: 13px !important; }
        html[data-fontsize="xl"] .text-\\[10px\\]  { font-size: 14px !important; }
        html[data-fontsize="xl"] .text-\\[11px\\]  { font-size: 15px !important; }
        html[data-fontsize="xl"] .text-\\[12px\\]  { font-size: 16px !important; }
        html[data-fontsize="xl"] .text-\\[13px\\]  { font-size: 17px !important; }
        html[data-fontsize="xl"] .text-\\[14px\\]  { font-size: 18px !important; }

        html[data-fontsize="xl"] [style*="font-size: 9px"],
        html[data-fontsize="xl"] [style*="font-size:9px"]   { font-size: 13px !important; }
        html[data-fontsize="xl"] [style*="font-size: 10px"],
        html[data-fontsize="xl"] [style*="font-size:10px"]  { font-size: 14px !important; }
        html[data-fontsize="xl"] [style*="font-size: 11px"],
        html[data-fontsize="xl"] [style*="font-size:11px"]  { font-size: 15px !important; }
      `}</style>
      </div> {/* close content-filter wrapper */}
    </div>
  );
}

/* A11yPanel — fixed top-left button that opens an accessibility menu.
   Three controls: font size, theme (dark/light), and a palette picker
   that includes specific Daltonization presets for the most common
   colour-vision deficiencies. Choices persist to localStorage. */
function A11yPanel({ fontSize, setFontSize, theme, setTheme, palette, setPalette }) {
  const [open, setOpen] = useState(false);
  const Btn = ({ active, children, onClick, title, className = '' }) => (
    <button
      onClick={onClick}
      title={title}
      className={`px-2 py-1 rounded text-[11px] font-mono border transition-all text-left
        ${active
          ? 'bg-amber-500/20 border-amber-500/60 text-amber-100'
          : 'bg-slate-800/40 border-slate-700 text-slate-300 hover:border-slate-500 hover:text-slate-100'} ${className}`}
    >
      {children}
    </button>
  );
  // Each palette includes an explanatory sub-line — the medical names
  // are accurate without being a label *about* the user.
  const PALETTES = [
    { id: 'default',  name: 'Default',       sub: 'standard amber / teal / rose' },
    { id: 'bold',     name: 'Bold',          sub: 'higher saturation + contrast' },
    { id: 'deuteran', name: 'Deuteran-mode', sub: 'green-deficient (most common)' },
    { id: 'protan',   name: 'Protan-mode',   sub: 'red-deficient' },
    { id: 'tritan',   name: 'Tritan-mode',   sub: 'blue–yellow deficiency (rare)' },
    { id: 'achromat', name: 'Monochrome',    sub: 'grayscale + contrast (achromatopsia)' },
  ];
  return (
    <div className="fixed top-3 left-3 z-50">
      <button
        onClick={() => setOpen(o => !o)}
        aria-label="Display options"
        title="Display options — font, theme, palette"
        className={`px-4 py-2.5 rounded-lg border-2 font-medium text-sm flex items-center gap-2 transition-all shadow-lg
          ${open
            ? 'bg-amber-500/25 border-amber-500/70 text-amber-100 shadow-amber-500/20'
            : 'bg-slate-900/90 border-slate-600 text-slate-100 hover:border-amber-500/60 hover:text-amber-100 backdrop-blur shadow-black/40'}`}
      >
        <span aria-hidden className="text-base leading-none">⚙</span>
        <span>Display</span>
        <span className="opacity-70 text-xs">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-2 w-72 rounded-lg border border-slate-700 bg-slate-900/95 backdrop-blur shadow-xl shadow-black/40 p-3 space-y-3 max-h-[80vh] overflow-y-auto">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">Font size</div>
            <div className="grid grid-cols-3 gap-1">
              <Btn active={fontSize === 'default'} onClick={() => setFontSize('default')} title="Default">A</Btn>
              <Btn active={fontSize === 'large'} onClick={() => setFontSize('large')} title="Larger text — small captions also scale up">A+</Btn>
              <Btn active={fontSize === 'xl'} onClick={() => setFontSize('xl')} title="Largest text">A++</Btn>
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">Theme</div>
            <div className="grid grid-cols-2 gap-1">
              <Btn active={theme === 'dark'} onClick={() => setTheme('dark')} title="Dark (default)">Dark</Btn>
              <Btn active={theme === 'light'} onClick={() => setTheme('light')} title="👻 a playful inverted view — not a true light theme, contrast is preserved by uniform inversion">👻 Ghost</Btn>
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">Palette</div>
            <div className="grid grid-cols-1 gap-1">
              {PALETTES.map(p => (
                <Btn key={p.id} active={palette === p.id} onClick={() => setPalette(p.id)} title={p.sub}>
                  <div className="font-medium">{p.name}</div>
                  <div className="text-[9px] opacity-70 font-mono">{p.sub}</div>
                </Btn>
              ))}
            </div>
          </div>
          <div className="text-[10px] text-slate-500 leading-snug border-t border-slate-700 pt-2">
            Preferences saved on this device. Most colour-coded blocks include text labels alongside the colour, so info is never carried by hue alone.
          </div>
        </div>
      )}
    </div>
  );
}

/* A11ySvgFilters — defines feColorMatrix filters for Daltonization of
   each common colour-vision deficiency. Applied via CSS `filter:
   url(#…)` on the content wrapper. Matrices are the standard
   simulation transforms (commonly used in vision-research tooling)
   adapted as best-effort correction filters for an existing palette. */
function A11ySvgFilters() {
  return (
    <svg width="0" height="0" style={{ position: 'absolute', pointerEvents: 'none' }} aria-hidden>
      <defs>
        {/* Deuteranopia (green-deficient) */}
        <filter id="a11y-deuteran" colorInterpolationFilters="sRGB">
          <feColorMatrix type="matrix" values="
            0.625 0.375 0    0 0
            0.7   0.3   0    0 0
            0     0.3   0.7  0 0
            0     0     0    1 0
          "/>
        </filter>
        {/* Protanopia (red-deficient) */}
        <filter id="a11y-protan" colorInterpolationFilters="sRGB">
          <feColorMatrix type="matrix" values="
            0.567 0.433 0     0 0
            0.558 0.442 0     0 0
            0     0.242 0.758 0 0
            0     0     0     1 0
          "/>
        </filter>
        {/* Tritanopia (blue-yellow deficient) */}
        <filter id="a11y-tritan" colorInterpolationFilters="sRGB">
          <feColorMatrix type="matrix" values="
            0.95  0.05  0     0 0
            0     0.433 0.567 0 0
            0     0.475 0.525 0 0
            0     0     0     1 0
          "/>
        </filter>
      </defs>
    </svg>
  );
}
