import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Grid3x3, Eye, Layers, Play, Pause, RotateCcw, ChevronRight,
  Box, Network, GitBranch, Move, BookOpen,
  ArrowRight, Hash, Crosshair, Workflow, Microscope, Sparkles,
  Upload, Loader2, Brain, Cpu, HelpCircle, Lightbulb, Check, X
} from 'lucide-react';

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
   has obvious "answers". Two pairs of objects on a 5×5 grid:
       red square  ┐                          ┌ blue circle
                   │                          │
                   │   (background)           │
                   │                          │
       blue circle ┘                          └ red square
   Diagonal twins share color — a query patch on a red square should
   attend strongly to the OTHER red square, and similarly for the
   blue circles. Designed to align with patchSize=48 / SIZE=240. */
function drawPairsDemoImage(ctx, size) {
  const bg = ctx.createLinearGradient(0, 0, 0, size);
  bg.addColorStop(0, '#1e293b');
  bg.addColorStop(1, '#0f172a');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  const cx1 = size * 0.22, cy1 = size * 0.22;
  const cx2 = size * 0.78, cy2 = size * 0.22;
  const cx3 = size * 0.22, cy3 = size * 0.78;
  const cx4 = size * 0.78, cy4 = size * 0.78;
  const r = size * 0.11;

  // Pair A — red squares (top-left & bottom-right, diagonal twins)
  ctx.fillStyle = '#ef4444';
  ctx.fillRect(cx1 - r, cy1 - r, r * 2, r * 2);
  ctx.fillRect(cx4 - r, cy4 - r, r * 2, r * 2);

  // Pair B — blue circles (top-right & bottom-left, diagonal twins)
  ctx.fillStyle = '#3b82f6';
  ctx.beginPath(); ctx.arc(cx2, cy2, r, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(cx3, cy3, r, 0, Math.PI * 2); ctx.fill();

  // small reference dot in the center to break symmetry visually
  ctx.fillStyle = '#fbbf24';
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, size * 0.025, 0, Math.PI * 2);
  ctx.fill();

  // very subtle noise
  const img = ctx.getImageData(0, 0, size, size);
  const rng = mulberry32(11);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (rng() - 0.5) * 6;
    img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
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

const Tag = ({ children, color = 'amber' }) => {
  const palette = {
    amber: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    teal: 'bg-teal-500/10 text-teal-300 border-teal-500/30',
    rose: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
    slate: 'bg-slate-500/10 text-slate-300 border-slate-500/30',
    violet: 'bg-violet-500/10 text-violet-300 border-violet-500/30',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-mono tracking-wide ${palette[color]}`}>
      {children}
    </span>
  );
};

const Eq = ({ children }) => (
  <span className="font-mono text-amber-200/90 bg-amber-500/5 border border-amber-500/20 rounded px-1.5 py-0.5 text-[13px]">
    {children}
  </span>
);

const Section = ({ icon: Icon, title, kicker, children }) => (
  <section className="mb-12">
    <div className="flex items-baseline gap-3 mb-1">
      {kicker && (
        <span className="font-mono text-[11px] tracking-[0.2em] text-amber-400/70 uppercase">
          {kicker}
        </span>
      )}
    </div>
    <div className="flex items-center gap-3 mb-5">
      {Icon && (
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-500/20 to-rose-500/10 border border-amber-500/30 flex items-center justify-center">
          <Icon size={18} className="text-amber-300" />
        </div>
      )}
      <h2 className="text-2xl font-serif text-slate-100 tracking-tight">{title}</h2>
    </div>
    <div className="text-slate-300/90 leading-relaxed">{children}</div>
  </section>
);

const Card = ({ children, className = '' }) => (
  <div className={`bg-slate-900/60 border border-slate-700/60 rounded-xl backdrop-blur-sm ${className}`}>
    {children}
  </div>
);

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
        <Card className="p-6 relative overflow-hidden">
          <div className="absolute -top-20 -right-20 w-48 h-48 bg-amber-500/10 rounded-full blur-3xl" />
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <Tag color="amber">ViT · 2020</Tag>
              <Tag color="slate">Dosovitskiy et al.</Tag>
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

        <Card className="p-6 relative overflow-hidden">
          <div className="absolute -top-20 -right-20 w-48 h-48 bg-teal-500/10 rounded-full blur-3xl" />
          <div className="relative">
            <div className="flex items-center gap-2 mb-3">
              <Tag color="teal">Swin · 2021</Tag>
              <Tag color="slate">Liu et al.</Tag>
            </div>
            <h3 className="font-serif text-xl text-slate-100 mb-3">Swin Transformer</h3>
            <p className="text-slate-300 text-sm mb-4 leading-relaxed">
              Restrict attention to non-overlapping local windows of size <Eq>M×M</Eq>, then alternate
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

      <Card className="p-6">
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
      <Section icon={Grid3x3} kicker="02 — From image to sequence" title="Patch embedding">
        <p className="max-w-3xl mb-3">
          A Transformer expects a sequence of vectors. ViT produces this sequence the simplest way imaginable:
          slice the image into a grid of <Eq>P × P</Eq> patches, flatten each patch into a vector of
          length <Eq>P²·C</Eq>, and project it linearly to dimension <Eq>D</Eq>.
        </p>
        <p className="max-w-3xl text-slate-400 text-sm">
          The standard ViT-Base uses <Eq>P=16</Eq> on 224×224 inputs, giving 196 tokens. Drag the
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
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">
              What happens to one patch
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex flex-col items-center gap-1">
                <div className="w-16 h-16 rounded ring-1 ring-amber-500/40 overflow-hidden">
                  <PatchThumb index={hoveredPatch ?? 0} patchSize={patchSize} grid={grid} />
                </div>
                <div className="text-[10px] font-mono text-slate-500">P × P × 3</div>
              </div>
              <div className="text-amber-400/70 font-mono">→</div>
              <div className="flex flex-col items-center gap-1">
                <div className="px-3 py-1.5 rounded bg-slate-800/60 border border-slate-700 text-[11px] font-mono text-slate-300">flatten</div>
                <div className="text-[10px] font-mono text-slate-500">vector ∈ ℝ^(P²·3)</div>
              </div>
              <div className="text-amber-400/70 font-mono">→</div>
              <div className="flex flex-col items-center gap-1">
                <div className="px-3 py-1.5 rounded bg-amber-500/15 border border-amber-500/40 text-[11px] font-mono text-amber-200">× E (linear)</div>
                <div className="text-[10px] font-mono text-slate-500">learned matrix</div>
              </div>
              <div className="text-amber-400/70 font-mono">→</div>
              <div className="flex flex-col items-center gap-1">
                <div className="h-12 w-2 bg-gradient-to-b from-amber-400 via-rose-400 to-teal-400 rounded-sm shadow shadow-amber-500/30" title="D-dim patch embedding" />
                <div className="text-[10px] font-mono text-slate-500">embedding ∈ ℝ^D</div>
              </div>
            </div>
            <div className="text-[12px] text-slate-400 mt-4 leading-relaxed">
              Every patch goes through the <em>same</em> linear projection — the matrix <Eq>E</Eq> is
              shared across all N patches and is what the model learns. After this step we have N
              embeddings of dimension D, ready to feed to the transformer. Hover a patch on the image
              to see which one is being projected.
            </div>
          </Card>
        </div>
      </div>
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

function PositionTab() {
  const [posOn, setPosOn] = useState(true);
  const [shuffle, setShuffle] = useState(false);
  const embDim = 64;
  const N = 16;

  // Learnable position embeddings approximated by a 2D sinusoidal pattern
  const posMatrix = useMemo(() => {
    const grid = Math.sqrt(N);
    const data = new Float32Array(N * embDim);
    for (let i = 0; i < N; i++) {
      const py = Math.floor(i / grid), px = i % grid;
      for (let d = 0; d < embDim; d++) {
        const freq = Math.pow(10000, -2 * Math.floor(d / 4) / embDim);
        if (d % 4 === 0) data[i * embDim + d] = Math.sin(px * freq);
        else if (d % 4 === 1) data[i * embDim + d] = Math.cos(px * freq);
        else if (d % 4 === 2) data[i * embDim + d] = Math.sin(py * freq);
        else data[i * embDim + d] = Math.cos(py * freq);
      }
    }
    return data;
  }, []);

  // Cosine similarity between position embeddings — illustrates spatial nearness
  const simMatrix = useMemo(() => {
    const sim = new Float32Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let dot = 0, na = 0, nb = 0;
        for (let d = 0; d < embDim; d++) {
          const a = posMatrix[i * embDim + d], b = posMatrix[j * embDim + d];
          dot += a * b; na += a * a; nb += b * b;
        }
        sim[i * N + j] = dot / (Math.sqrt(na * nb) + 1e-8);
      }
    }
    return sim;
  }, [posMatrix]);

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
          {' '}<Eq>E_pos[i]</Eq> to each token to mark "this came from position i".
        </p>

        <div className="grid lg:grid-cols-2 gap-5">
          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              Try it
            </div>
            <div className="flex items-center gap-1 mb-4 flex-wrap">
              {order.map(i => (
                <div
                  key={i}
                  className="w-10 h-10 rounded border flex flex-col items-center justify-center font-mono text-amber-200 transition-all"
                  style={{
                    background: posOn ? 'rgba(245,158,11,0.10)' : 'rgba(100,116,139,0.10)',
                    borderColor: posOn ? 'rgba(245,158,11,0.40)' : 'rgba(100,116,139,0.40)',
                  }}
                >
                  <span className="text-xs leading-none">{i}</span>
                  {posOn && <span className="text-[8px] text-teal-300 leading-none mt-0.5">@{i}</span>}
                </div>
              ))}
            </div>
            <div className="space-y-2">
              <Toggle label="Shuffle patches" value={shuffle} onChange={setShuffle} />
              <Toggle label="Add position embeddings" value={posOn} onChange={setPosOn} />
            </div>
            <div className="mt-4 text-[12px] leading-relaxed">
              {!posOn && shuffle && <span className="text-rose-300">No position info + shuffled → the model treats this as the same input as the original. Spatial structure is lost.</span>}
              {!posOn && !shuffle && <span className="text-slate-400">No position info. It happens to look "right" because we didn't shuffle, but the model isn't actually using the order.</span>}
              {posOn && shuffle && <span className="text-amber-200">Shuffled, but each patch carries its position tag (the small <span className="text-teal-300">@i</span>). The model can recover the original spatial structure.</span>}
              {posOn && !shuffle && <span className="text-teal-300">Each token now carries both content (patch) <em>and</em> location (the <span className="text-teal-300">@i</span> tag).</span>}
            </div>
          </Card>

          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              Position similarity (4×4 grid)
            </div>
            <SimHeatmap data={simMatrix} N={N} />
            <div className="mt-3 text-[12px] text-slate-400 leading-relaxed">
              Each cell shows how similar the position embedding for patch <em>i</em> is to patch <em>j</em>.
              Bright cells cluster around the diagonal — <strong>spatially-nearby patches have similar
              position embeddings</strong>. That's how attention learns the concept of "near" vs "far".
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
          per image. ViT prepends a single learnable <Eq>[CLS]</Eq> embedding (a learned parameter, not
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

/* =========================================================
   TAB 4 — Self-Attention
   ========================================================= */

function AttentionTab() {
  const [patchSize, setPatchSize] = useState(48);
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
    <div className="space-y-8">
      <Section icon={Eye} kicker="04 — The core mechanism" title="Scaled dot-product self-attention">
        <p className="max-w-3xl mb-3">
          Every token produces three vectors: a <span className="text-amber-300">query</span> Q,
          a <span className="text-teal-300">key</span> K, and a <span className="text-rose-300">value</span> V.
          To update token i, we compute how well its query matches every key (a similarity score), normalize
          via softmax, and use those weights to take a weighted average of the values.
        </p>
        <div className="font-mono text-[14px] text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-4 max-w-2xl">
          Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
        </div>
      </Section>

      {/* How to use — clear three-step instruction so the demo isn't a guessing game */}
      <Card className="p-4 bg-amber-500/[0.04] border-amber-500/30">
        <div className="text-[11px] font-mono tracking-[0.2em] text-amber-400/80 uppercase mb-2">How to use this demo</div>
        <ol className="grid sm:grid-cols-3 gap-3 text-[13px] text-slate-200">
          <li className="flex gap-2">
            <span className="font-mono text-amber-300 shrink-0">1.</span>
            <span><span className="text-amber-300">Click any patch</span> on the image to set it as the <em>query</em> — the patch you're asking "what should I look at?".</span>
          </li>
          <li className="flex gap-2">
            <span className="font-mono text-amber-300 shrink-0">2.</span>
            <span>Step through buttons <strong>1 → 5</strong> below to walk the math: project, score, softmax, aggregate.</span>
          </li>
          <li className="flex gap-2">
            <span className="font-mono text-amber-300 shrink-0">3.</span>
            <span>The image has two pairs of identical objects (red squares, blue circles). Watch the model find them.</span>
          </li>
        </ol>
      </Card>

      <div className="flex gap-2 mb-2 overflow-x-auto pb-2">
        {steps.map(s => (
          <button
            key={s.n}
            onClick={() => setStep(s.n)}
            className={`flex-shrink-0 px-3 py-2 rounded-lg border text-left transition-all min-w-[140px]
              ${step === s.n
                ? 'bg-amber-500/15 border-amber-500/50 text-amber-100'
                : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:border-slate-600'}`}
          >
            <div className="text-[10px] font-mono uppercase tracking-wider opacity-70">Step {s.n + 1}</div>
            <div className="font-medium text-sm">{s.label}</div>
            <div className="text-[10px] font-mono mt-1 opacity-70">{s.desc}</div>
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <div className="space-y-4">
          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              {selectedPatch !== null
                ? <>Query: patch <span className="text-amber-300">#{selectedPatch}</span></>
                : <span className="text-amber-300 animate-pulse">↓ Click a patch below to start ↓</span>}
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
            <div className="mt-4 space-y-3">
              <Slider
                label="Patch size (px)"
                value={patchSize}
                options={[30, 48, 60]}
                onChange={(v) => { setPatchSize(v); setSelectedPatch(null); }}
              />
              <Slider label="Random seed (W_Q, W_K, W_V)" value={seed} min={1} max={50} onChange={setSeed} />
            </div>
            <div className="mt-3 text-[12px] text-slate-300 leading-relaxed min-h-[2.5em]">
              {selectedPatch === null
                ? <span className="text-amber-200/80">Pick a patch — try one of the red squares or blue circles for the cleanest result.</span>
                : step === 0
                  ? <>Step 1 · <span className="text-amber-300">Project</span> — every patch produces three vectors: Q (query), K (key), V (value). The matrices on the right show all three.</>
                  : step === 1
                    ? <>Step 2 · <span className="text-amber-300">Score</span> — your query's dot product with every key. Amber = positive (similar), blue = negative (dissimilar). <em>Not</em> a probability yet.</>
                    : step === 2
                      ? <>Step 3 · <span className="text-amber-300">Softmax</span> — scores normalised so they sum to 100%. The peaks are where attention concentrates.</>
                      : step === 3
                        ? <>Step 4 · <span className="text-amber-300">Aggregate</span> — only the top-5 contributing patches stay bright. The output for your query is a weighted blend of <em>their</em> values.</>
                        : <>Step 5 · <span className="text-amber-300">Inspect</span> — try other patches. The model should pair red↔red and blue↔blue.</>}
            </div>
          </Card>

          {/* Top-attended patches — pulled into the LEFT column so it's visible
              right next to the image, even when the matrix card is tall. */}
          {selectedPatch !== null && attn && (
            <Card className="p-5">
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
                Top patches that query #{selectedPatch} attends to
              </div>
              <TopAttended attn={attn} N={N} query={selectedPatch} grid={grid} patchSize={patchSize} />
              <div className="text-[11px] text-slate-400 italic mt-3 leading-relaxed">
                If your query was on a red square, you should see the OTHER red square near the top of this list.
                That's "patches that look alike attend to each other".
              </div>
            </Card>
          )}
        </div>

        <div className="space-y-4">
          {step === 0 && Q && (
            <Card className="p-5">
              <div className="text-sm text-slate-200 mb-3">
                Each input row <Eq>x_i ∈ ℝ^D</Eq> is multiplied by three learned matrices to produce its query, key, and value.
              </div>
              <div className="grid grid-cols-3 gap-3">
                <MatHeat label={`Q (${Q.rows}×${Q.cols})`} M={Q} color="amber" />
                <MatHeat label={`K (${K.rows}×${K.cols})`} M={K} color="teal" />
                <MatHeat label={`V (${V.rows}×${V.cols})`} M={V} color="rose" />
              </div>
            </Card>
          )}
          <Card className="p-5">
            <div className="flex items-baseline justify-between gap-3 mb-3 flex-wrap">
              <div className="text-sm text-slate-200">
                {step <= 0 && <>Attention matrix <Eq>A ∈ ℝ^(N×N)</Eq> · not yet computed</>}
                {step === 1 && <>Raw scores <Eq>Q·Kᵀ / √d</Eq> · signed values, before softmax</>}
                {step === 2 && <>Softmax weights <Eq>A</Eq> · each row sums to 100%</>}
                {step === 3 && <>Aggregate · top-3 contributors per row highlighted</>}
                {step === 4 && <>Inspect · click any row to set that as the query</>}
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
            <div className="mt-3 text-[11px] text-slate-500 italic leading-relaxed">
              {step === 1 && 'Amber = positive, blue = negative. These are not probabilities yet — softmax hasn\'t been applied.'}
              {step === 2 && 'Each row is a probability distribution: how patch i splits its attention over all keys j.'}
              {step === 3 && 'Most of the attention mass usually concentrates on a handful of patches; the output is a weighted sum of just those.'}
              {step === 4 && 'Click a row to focus the canvas overlay on that query patch.'}
              {step <= 0 && 'Walk the steps above to build the matrix.'}
            </div>
          </Card>
        </div>
      </div>
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
      <Section icon={Network} kicker="05 — Many views, simultaneously" title="Multi-head attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          The previous tab gave us <em>one</em> attention pattern — one way of saying "patches that look
          alike attend to each other". But what counts as "alike"? Color? Texture? Position? A single
          head has to compromise across all of these.
        </p>
        <p className="max-w-3xl mb-3 text-slate-300">
          Multi-head attention runs <Eq>h</Eq> independent attention computations in parallel, each on a
          smaller subspace of dimension <Eq>d_k = D/h</Eq>, then concatenates them. Each head can
          specialize: one might track color similarity, another spatial proximity, another shape.
        </p>
        <div className="font-mono text-[13px] text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-4 max-w-3xl">
          MultiHead(Q, K, V) = Concat(head₁, …, head_h)·W_O,&nbsp;&nbsp;
          head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
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
            <div className={`grid gap-3`} style={{ gridTemplateColumns: `repeat(${Math.min(4, numHeads)}, minmax(0,1fr))` }}>
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
    }, 900);
    return () => clearTimeout(t);
  }, [playing, step, stages.length]);

  return (
    <div className="space-y-8">
      <Section icon={Workflow} kicker="06 — End to end" title="ViT forward pass">
        <p className="max-w-3xl mb-4 text-slate-300">
          We've now built every piece individually: patch embedding, position embeddings, [CLS], and
          multi-head self-attention. Time to assemble. Step through each stage below to see how the
          tensor shapes evolve from <Eq>H × W × 3</Eq> pixels to <Eq>C</Eq> class logits. The encoder
          is just a stack of <Eq>L</Eq> identical blocks — each block does multi-head self-attention,
          then an MLP, both wrapped in residual connections and pre-norm LayerNorm.
        </p>
        <div className="flex items-center gap-3">
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

      <Card className="p-6">
        <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400/70 mb-3">
          Stage {step + 1} · live data flow
        </div>
        <PipelineVisual step={step} />
      </Card>

      <Card className="p-6 min-h-[200px]">
        <PipelineDetail step={step} />
      </Card>

      <Card className="p-6">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Inside one Transformer encoder block</div>
        <EncoderBlockDiagram />
      </Card>
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
          <div className="font-mono text-[10px] text-slate-500">196 × 768 (flattened)</div>
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
          <div className="font-mono text-[10px] text-slate-500">flat patches</div>
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
          <div className="font-mono text-[10px] text-slate-500">196 × 768</div>
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

  // ---- Stage 6: CLS head
  if (step === 5) {
    return (
      <div className="flex items-center justify-center gap-6 min-h-[260px]">
        <div>
          <div className="space-y-1">
            <Cls glow/>
            {Array.from({ length: STACK_N }).map((_, i) => <Bar key={i} hue={i * 50 + 90} w="w-24" dim/>)}
          </div>
          <div className="font-mono text-[11px] text-rose-300 mt-2">extract [CLS]</div>
          <div className="font-mono text-[10px] text-slate-500">patches: ignored</div>
        </div>
        <div className="text-rose-300 font-mono text-2xl">→</div>
        <div className="text-center">
          <div className="px-4 py-3 rounded bg-amber-500/15 border border-amber-500/40 font-mono text-[11px] text-amber-200">
            <div>Linear</div>
            <div className="text-[9px] text-slate-400 mt-0.5">D → C</div>
          </div>
          <div className="font-mono text-[10px] text-slate-500 mt-1">classifier head</div>
        </div>
        <div className="text-amber-400 font-mono text-2xl">→</div>
        <div>
          <div className="space-y-0.5">
            {[60, 28, 22, 18, 14, 12, 10, 9].map((w, i) => (
              <div key={i} className="h-2.5 rounded-sm bg-amber-400/70" style={{ width: `${w}px` }}/>
            ))}
          </div>
          <div className="font-mono text-[11px] text-amber-300 mt-2">logits (C)</div>
          <div className="font-mono text-[10px] text-slate-500">unnormalized</div>
        </div>
      </div>
    );
  }

  // ---- Stage 7: Softmax → probabilities
  const PROBS = [
    { label: 'tabby cat',     score: 0.74 },
    { label: 'tiger cat',     score: 0.13 },
    { label: 'Egyptian cat',  score: 0.06 },
    { label: 'lynx',          score: 0.04 },
    { label: 'Persian cat',   score: 0.03 },
  ];
  return (
    <div className="flex flex-col items-center justify-center gap-3 min-h-[260px] py-4">
      <div className="font-mono text-[11px] text-slate-400">softmax(logits) → class probabilities</div>
      <div className="space-y-2 w-full max-w-md px-4">
        {PROBS.map((p, i) => (
          <div key={i}>
            <div className="flex justify-between text-[12px] mb-1">
              <span className="text-slate-200">{p.label}</span>
              <span className="font-mono text-amber-300">{(p.score * 100).toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-slate-800 rounded overflow-hidden">
              <div className={`h-full ${i === 0 ? 'bg-amber-400' : 'bg-amber-400/40'}`} style={{ width: `${p.score * 100}%` }}/>
            </div>
          </div>
        ))}
      </div>
      <div className="font-mono text-[10px] text-slate-500 mt-1">cross-entropy loss vs. one-hot target finishes one training step</div>
    </div>
  );
}

function PipelineDetail({ step }) {
  const details = [
    {
      title: 'Input image',
      body: (
        <>
          <p>The starting point is a raw image, typically <Eq>224×224×3</Eq> for ImageNet-scale models.</p>
          <p className="mt-2 text-slate-400 text-sm">Standard ImageNet preprocessing applies: resize, center crop, normalize with channel-wise mean/std.</p>
        </>
      )
    },
    {
      title: 'Patchify',
      body: (
        <>
          <p>Reshape the image into <Eq>N = HW/P²</Eq> non-overlapping patches of size <Eq>P × P × 3</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">For ViT-Base at 224×224 with P=16, this produces 196 patches, each a 768-dim vector after flattening.</p>
          <p className="mt-2 font-mono text-amber-300 text-sm">In practice this is implemented as a single Conv2d with stride P and kernel P.</p>
        </>
      )
    },
    {
      title: 'Linear projection',
      body: (
        <>
          <p>Each flat patch is projected to embedding dim D via a learnable matrix <Eq>E ∈ ℝ^(P²C × D)</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">For ViT-Base, D = 768. This linear layer is shared across all patches.</p>
        </>
      )
    },
    {
      title: 'Add [CLS] and position embeddings',
      body: (
        <>
          <p>Prepend a learnable [CLS] token, then add learnable position embeddings <Eq>E_pos ∈ ℝ^((N+1)×D)</Eq>.</p>
          <p className="mt-2 text-slate-400 text-sm">The [CLS] token acts as a global aggregator — its final hidden state is what gets classified.</p>
        </>
      )
    },
    {
      title: 'Transformer encoder × L',
      body: (
        <>
          <p>Apply L identical blocks. Each block is:</p>
          <pre className="font-mono text-[12px] text-amber-200 bg-slate-950/60 border border-amber-500/20 rounded-lg p-3 mt-2 overflow-x-auto">{`z'_l = MSA(LN(z_{l-1})) + z_{l-1}     # multi-head self-attention + residual
z_l  = MLP(LN(z'_l))     + z'_l       # 2-layer MLP w/ GELU + residual`}</pre>
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
  const GRID = 8; // 8×8 patches
  const cell = 36;

  const numWindowsSide = GRID / winSize;
  const numWindows = numWindowsSide * numWindowsSide;

  // Compute complexity savings
  const fullCost = (GRID * GRID) ** 2; // O(N²)
  const windowCost = numWindows * (winSize * winSize) ** 2;

  return (
    <div className="space-y-8">
      <Section icon={Box} kicker="07 — Locality is back" title="Window-based self-attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          ViT's quadratic attention cost is fine at 224 × 224, but it explodes for high-resolution work
          (segmentation, detection). For an 800 × 800 image with patch size 4, that's 40,000 tokens —
          and 1.6 billion attention pairs <em>per layer</em>. Untenable.
        </p>
        <p className="max-w-3xl text-slate-300">
          Swin's first idea: stop attending globally. Partition the patches into non-overlapping windows
          of <Eq>M × M</Eq> tokens, and let attention happen <span className="text-teal-300">only inside
          each window</span>. Cost per layer drops to <Eq>O(M² · N)</Eq> — linear in the number of patches.
          The next tab fixes the obvious downside: now patches across window boundaries can't talk.
        </p>
      </Section>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            8×8 patch grid · windows of size M
          </div>
          <svg width={GRID * cell} height={GRID * cell} className="rounded-lg bg-slate-950/40 border border-slate-800">
            {/* patches */}
            {Array.from({ length: GRID * GRID }).map((_, i) => {
              const px = i % GRID, py = Math.floor(i / GRID);
              const wx = Math.floor(px / winSize), wy = Math.floor(py / winSize);
              const wIdx = wy * numWindowsSide + wx;
              const isHover = hoveredWindow === wIdx;
              const hue = (wIdx * 360 / numWindows) % 360;
              return (
                <g key={i}>
                  <rect
                    x={px * cell + 1} y={py * cell + 1}
                    width={cell - 2} height={cell - 2}
                    rx={3}
                    fill={`hsla(${hue}, 60%, 50%, ${isHover ? 0.45 : 0.18})`}
                    stroke={`hsla(${hue}, 70%, 60%, ${isHover ? 0.9 : 0.4})`}
                    strokeWidth={isHover ? 1.5 : 0.6}
                  />
                  <text x={px * cell + cell / 2} y={py * cell + cell / 2 + 4}
                    textAnchor="middle" fontFamily="monospace" fontSize="10" fill="#cbd5e1">
                    {i}
                  </text>
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
                    stroke={hoveredWindow === wIdx ? '#5eead4' : '#14b8a6'}
                    strokeWidth={hoveredWindow === wIdx ? 3 : 1.5}
                    onMouseEnter={() => setHoveredWindow(wIdx)}
                    onMouseLeave={() => setHoveredWindow(null)}
                    style={{ cursor: 'pointer' }}
                  />
                );
              })
            )}
          </svg>
          <div className="mt-4">
            <Slider label="Window size M" value={winSize} options={[1, 2, 4, 8]} onChange={setWinSize} />
          </div>
          <div className="mt-3 text-[11px] text-slate-500 italic">
            Hover a window to highlight it. Same color = same window. Each window does its own self-attention independently.
          </div>
        </Card>

        <div className="space-y-4">
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
    </div>
  );
}

/* =========================================================
   TAB 8 — Shifted Windows (animated)
   ========================================================= */

function ShiftedTab() {
  const [layer, setLayer] = useState(0); // 0 = W-MSA, 1 = SW-MSA
  const [auto, setAuto] = useState(false);
  const [showCyclic, setShowCyclic] = useState(false);
  const GRID = 8;
  const M = 4; // window size
  const shift = M / 2; // 2
  const cell = 38;

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => setLayer(l => 1 - l), 1500);
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
      <Section icon={Move} kicker="08 — Cross-window connections" title="Shifted window attention">
        <p className="max-w-3xl mb-3 text-slate-300">
          The previous tab introduced a problem: with strict windows, two patches that sit on either
          side of a window boundary <em>never</em> attend to each other, no matter how many layers we
          stack. That kills the global reasoning we got from ViT for free.
        </p>
        <p className="max-w-3xl mb-3 text-slate-300">
          Swin's fix: alternate two kinds of layers. Layer <Eq>l</Eq> uses regular window partitioning
          (<span className="font-mono text-amber-300">W-MSA</span>). Layer <Eq>l+1</Eq> shifts the window
          grid by <Eq>(⌊M/2⌋, ⌊M/2⌋)</Eq> pixels (<span className="font-mono text-amber-300">SW-MSA</span>).
          Patches that were neighbors-across-a-wall in layer <Eq>l</Eq> now share a window in layer
          <Eq>l+1</Eq>.
        </p>
        <p className="max-w-3xl text-slate-300">
          After two layers, every patch has effectively communicated with everything in a
          <Eq>2M × 2M</Eq> region. Stack more, and the receptive field keeps growing — still at linear cost.
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
              return (
                <g key={i} style={{ transition: 'all 0.5s ease' }}>
                  <rect
                    x={px * cell + 1} y={py * cell + 1}
                    width={cell - 2} height={cell - 2}
                    rx={3}
                    fill={`hsla(${h}, 60%, 50%, 0.25)`}
                    stroke={`hsla(${h}, 70%, 60%, 0.7)`}
                    strokeWidth={0.8}
                    style={{ transition: 'all 0.5s ease' }}
                  />
                  <text x={px * cell + cell / 2} y={py * cell + cell / 2 + 4}
                    textAnchor="middle" fontFamily="monospace" fontSize="10" fill="#cbd5e1">
                    {i}
                  </text>
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
          <div className="mt-3 text-[11px] text-slate-500 italic">
            Same color = same window in this layer. Watch how patches change neighborhoods between W-MSA and SW-MSA.
          </div>
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
                map by <Eq>(-⌊M/2⌋, -⌊M/2⌋)</Eq>, do regular W-MSA, then shift back. An
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

      <Card className="p-6">
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
      <Section icon={GitBranch} kicker="09 — Pyramid features" title="Patch merging & hierarchical stages">
        <p className="max-w-3xl mb-3 text-slate-300">
          We've made attention efficient (windows) and global (shifted windows). One thing is still
          missing: ViT runs at a <em>single resolution</em> the whole way through. Classification gets
          away with this, but detection and segmentation need features at multiple scales — like a
          ConvNet's pyramid.
        </p>
        <p className="max-w-3xl text-slate-300">
          Swin's last ingredient: a <span className="text-teal-300">patch-merging</span> layer between
          stages. Take every <Eq>2×2</Eq> group of patches, concatenate their channels (<Eq>4C</Eq>),
          then linearly project back to <Eq>2C</Eq>. Resolution halves; channels double; receptive field
          doubles. After four stages we have a CNN-style feature pyramid, but built from attention.
        </p>
      </Section>

      <div className="grid lg:grid-cols-[auto_1fr] gap-6">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Swin-T stages</div>
          <div className="space-y-2">
            {stages.map((s, i) => (
              <button
                key={i}
                onClick={() => setStage(i)}
                className={`w-full text-left p-3 rounded-lg border transition-all
                  ${stage === i
                    ? 'bg-teal-500/15 border-teal-500/60'
                    : 'bg-slate-800/40 border-slate-700 hover:border-slate-600'}`}
              >
                <div className="flex items-baseline justify-between">
                  <span className={`font-medium ${stage === i ? 'text-teal-200' : 'text-slate-200'}`}>{s.name}</span>
                  <span className="font-mono text-[11px] text-slate-500">×{s.blocks}</span>
                </div>
                <div className="font-mono text-[11px] mt-1">
                  <span className="text-amber-300">{s.resolution}</span>
                  <span className="text-slate-500 mx-2">·</span>
                  <span className="text-slate-300">{s.tokens.toLocaleString()} tokens</span>
                  <span className="text-slate-500 mx-2">·</span>
                  <span className="text-slate-300">{s.dim}-dim</span>
                </div>
              </button>
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Resolution at this stage</div>
            <ResolutionViz stage={stage} />
          </Card>

          <Card className="p-5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">Patch merging — how it works</div>
            <PatchMergeDiagram />
            <div className="mt-3 text-[11px] text-slate-500 italic leading-relaxed">
              For each non-overlapping 2×2 block of tokens, concatenate the four <Eq>C</Eq>-dim vectors
              into a <Eq>4C</Eq> vector, apply LayerNorm, then project to <Eq>2C</Eq>.
              The number of tokens drops by 4×; the channel dimension doubles.
            </div>
          </Card>
        </div>
      </div>

      <Card className="p-6">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Full Swin-T architecture flow</div>
        <SwinFlow stages={stages} highlight={stage} />
      </Card>
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

      <Card className="p-6">
        <h3 className="font-serif text-lg text-slate-100 mb-4">Quick reference — ViT-B vs Swin-T</h3>
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
                ['Parameters', '86 M', '28 M'],
                ['FLOPs (224²)', '17.6 G', '4.5 G'],
                ['Tokens', '197 (incl. CLS)', '3136 → 49 across stages'],
                ['Embed dim', '768 (constant)', '96 → 768 across stages'],
                ['Attention', 'Global', 'Local 7×7 window + shift'],
                ['Position', 'Absolute (learnable)', 'Relative bias'],
                ['ImageNet-1K top-1', '77.9 %', '81.3 %'],
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

const SIM_PREDS = {
  cat: [
    { label: 'tabby cat',    score: 0.72 },
    { label: 'tiger cat',    score: 0.18 },
    { label: 'Egyptian cat', score: 0.06 },
    { label: 'lynx',         score: 0.02 },
    { label: 'Persian cat',  score: 0.02 },
  ],
  baiwan: [
    { label: 'tabby cat',    score: 0.55 },
    { label: 'Egyptian cat', score: 0.22 },
    { label: 'tiger cat',    score: 0.13 },
    { label: 'Persian cat',  score: 0.06 },
    { label: 'Siamese cat',  score: 0.04 },
  ],
  miaomiao: [
    { label: 'tabby cat',    score: 0.61 },
    { label: 'tiger cat',    score: 0.19 },
    { label: 'Egyptian cat', score: 0.11 },
    { label: 'lynx',         score: 0.05 },
    { label: 'Persian cat',  score: 0.04 },
  ],
};

const SIM_FALLBACK = [
  { label: 'unknown',     score: 0.40 },
  { label: 'tabby cat',   score: 0.20 },
  { label: 'tiger cat',   score: 0.15 },
  { label: 'lynx',        score: 0.10 },
  { label: 'Persian cat', score: 0.05 },
];

function drawScan(canvas, img, size, patchPx, progress, mode) {
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
  const cx = cur % gridN;
  const cy = Math.floor(cur / gridN);
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
    const wins = Math.ceil(gridN / winSize);
    ctx.strokeStyle = `rgba(${baseColor}, 0.55)`;
    ctx.lineWidth = 2;
    for (let i = 0; i <= wins; i++) {
      const p = Math.min(i * winSize * patchPx, size);
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
    }
    const visited = new Set();
    for (let i = 0; i < seen; i++) {
      const px = i % gridN;
      const py = Math.floor(i / gridN);
      visited.add(Math.floor(py / winSize) * wins + Math.floor(px / winSize));
    }
    visited.forEach(wid => {
      const wxx = wid % wins;
      const wyy = Math.floor(wid / wins);
      ctx.fillStyle = `rgba(${baseColor}, 0.06)`;
      ctx.fillRect(wxx * winSize * patchPx, wyy * winSize * patchPx, winSize * patchPx, winSize * patchPx);
    });
  } else {
    for (let i = 0; i < seen; i++) {
      const px = (i % gridN) * patchPx;
      const py = Math.floor(i / gridN) * patchPx;
      ctx.fillStyle = `rgba(${baseColor}, 0.10)`;
      ctx.fillRect(px, py, patchPx, patchPx);
    }
  }

  if (seen > 0) {
    const qx = cx * patchPx + patchPx / 2;
    const qy = cy * patchPx + patchPx / 2;

    if (mode === 'vit') {
      ctx.strokeStyle = `rgba(${baseColor}, 0.16)`;
      ctx.lineWidth = 0.5;
      for (let j = 0; j < total; j++) {
        if (j === cur) continue;
        const tx = (j % gridN) * patchPx + patchPx / 2;
        const ty = Math.floor(j / gridN) * patchPx + patchPx / 2;
        ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty); ctx.stroke();
      }
    } else {
      const winSize = 4;
      const wx = Math.floor(cx / winSize);
      const wy = Math.floor(cy / winSize);
      ctx.fillStyle = `rgba(${baseColor}, 0.18)`;
      ctx.fillRect(wx * winSize * patchPx, wy * winSize * patchPx, winSize * patchPx, winSize * patchPx);
      ctx.strokeStyle = `rgba(${baseColor}, 1)`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(wx * winSize * patchPx, wy * winSize * patchPx, winSize * patchPx, winSize * patchPx);
      ctx.strokeStyle = `rgba(${baseColor}, 0.5)`;
      ctx.lineWidth = 0.7;
      for (let py = wy * winSize; py < (wy + 1) * winSize && py < gridN; py++) {
        for (let px = wx * winSize; px < (wx + 1) * winSize && px < gridN; px++) {
          if (px === cx && py === cy) continue;
          const tx = px * patchPx + patchPx / 2;
          const ty = py * patchPx + patchPx / 2;
          ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty); ctx.stroke();
        }
      }
    }

    ctx.fillStyle = 'rgba(245, 158, 11, 0.35)';
    ctx.fillRect(cx * patchPx, cy * patchPx, patchPx, patchPx);
    ctx.strokeStyle = 'rgba(245, 158, 11, 1)';
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
const CLASS_ANCHORS = [
  [0.50, 0.55],
  [0.40, 0.50],
  [0.55, 0.40],
  [0.60, 0.62],
  [0.38, 0.62],
];

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

  const heat = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const px = (i % gridN + 0.5) / gridN;
    const py = (Math.floor(i / gridN) + 0.5) / gridN;
    const dx = px - ax, dy = py - ay;
    const bias = Math.exp(-(dx * dx + dy * dy) / 0.05);
    heat[i] = base[i] * (0.1 + bias * 2.2);
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
function ClassContribution({ image, attentionRows, gridN, preds, status, statusMessage, statusProgress }) {
  const [classIdx, setClassIdx] = useState(0);
  const canvasRef = useRef(null);

  useEffect(() => { setClassIdx(0); }, [preds]);

  useEffect(() => {
    if (!canvasRef.current || !image || !attentionRows) return;
    drawClassContrib(canvasRef.current, image, attentionRows, gridN, classIdx);
  }, [image, attentionRows, gridN, classIdx]);

  const top5 = (preds || []).slice(0, 5);
  const loading = status === 'modelLoading' || status === 'inferring';

  return (
    <div className="flex gap-3 items-start flex-wrap">
      <div className="flex-shrink-0">
        <canvas
          ref={canvasRef}
          className="w-[180px] h-[180px] rounded bg-slate-950 block"
        />
        <div className="text-[10px] font-mono text-slate-500 mt-1 text-center w-[180px] truncate">
          {top5[classIdx]
            ? <>evidence for <span className="text-amber-300">"{top5[classIdx].label}"</span></>
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

/* ClassifierFlow — full neural-network forward-pass demo.
   Shows tokens → encoder × L → CLS extraction → real matrix multiplication
   CLS · W = logits → softmax → top-K probabilities. Values are
   deterministic per image so the visual is reproducible.

   This is "how the matrix becomes a prediction" with actual math made
   visible: the CLS vector and W matrix have signed entries (amber+,
   teal−); the W·CLS dot products produce the logit bars; softmax
   normalizes them to the probability bars at the bottom. */
function ClassifierFlow({ progress, preds, seedKey }) {
  const stage = progress >= 0.92 ? 5
              : progress >= 0.78 ? 4
              : progress >= 0.62 ? 3
              : progress >= 0.45 ? 2
              : progress >= 0.25 ? 1 : 0;
  const D_show = 12;
  const C_show = 5;

  // Deterministic per-image CLS, W, and logits.
  const computed = useMemo(() => {
    const seedNum = (typeof seedKey === 'string' ? seedKey : String(seedKey || 'default'))
      .split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
    const rng = mulberry32(7919 + seedNum);
    const cls = Array.from({ length: D_show }, () => (rng() - 0.5) * 1.8);
    const W = Array.from({ length: D_show * C_show }, () => (rng() - 0.5) * 1.3);
    const logits = Array(C_show).fill(0);
    for (let c = 0; c < C_show; c++) {
      let s = 0;
      for (let d = 0; d < D_show; d++) s += cls[d] * W[d * C_show + c];
      logits[c] = s;
    }
    // softmax
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    const exps = logits.map(v => Math.exp(v - mx));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sum);
    return { cls, W, logits, probs };
  }, [seedKey]);

  const labels = preds && preds.length >= C_show
    ? preds.slice(0, C_show).map(p => p.label)
    : ['tabby cat', 'tiger cat', 'Egyptian cat', 'lynx', 'Persian cat'];

  // Use the real (or simulated) probabilities for display so the
  // bottom row matches the predictions panel below; keep our toy logits
  // for the visualization of the matmul mechanics.
  const realProbs = preds && preds.length >= C_show
    ? preds.slice(0, C_show).map(p => p.score)
    : computed.probs;

  // ----- Sub-elements -----

  const SignedBar = ({ v, max = 1.5, w = 'w-3', orient = 'vertical', active = true }) => {
    const t = Math.min(1, Math.abs(v) / max);
    const color = v >= 0 ? `rgba(245,158,11,${active ? 0.85 : 0.25})` : `rgba(20,184,166,${active ? 0.85 : 0.25})`;
    if (orient === 'horizontal') {
      const len = Math.max(2, Math.round(t * 36));
      return (
        <div className="h-2 flex" style={{ width: 36 }}>
          <div className="flex-1 flex justify-end">
            {v < 0 && <div className="h-full rounded-l-sm" style={{ width: len, background: color }} />}
          </div>
          <div className="w-px bg-slate-700"/>
          <div className="flex-1 flex justify-start">
            {v >= 0 && <div className="h-full rounded-r-sm" style={{ width: len, background: color }} />}
          </div>
        </div>
      );
    }
    const len = Math.max(2, Math.round(t * 28));
    return (
      <div className={`${w} flex flex-col items-center`} style={{ height: 60 }}>
        <div className="flex-1 w-full flex items-end justify-center">
          {v >= 0 && <div className={`${w} rounded-t-sm`} style={{ height: len, background: color }} />}
        </div>
        <div className="w-full h-px bg-slate-700"/>
        <div className="flex-1 w-full flex items-start justify-center">
          {v < 0 && <div className={`${w} rounded-b-sm`} style={{ height: len, background: color }} />}
        </div>
      </div>
    );
  };

  const EncoderGraph = ({ active }) => {
    const NODES = 5;
    const W = 70, H = 60;
    const inX = 8, outX = W - 8;
    const ys = Array.from({ length: NODES }, (_, i) => 6 + i * (H - 12) / (NODES - 1));
    return (
      <svg width={W} height={H} className="block">
        {active && ys.flatMap((y1, i) =>
          ys.map((y2, j) => (
            <line key={`e-${i}-${j}`} x1={inX} y1={y1} x2={outX} y2={y2}
              stroke="rgba(245,158,11,0.20)" strokeWidth="0.4"/>
          ))
        )}
        {ys.map((y, i) => <circle key={`l-${i}`} cx={inX} cy={y} r={2.4}
          fill={active ? '#fbbf24' : '#475569'}/>)}
        {ys.map((y, i) => <circle key={`r-${i}`} cx={outX} cy={y} r={2.4}
          fill={active ? '#fbbf24' : '#475569'}/>)}
      </svg>
    );
  };

  return (
    <div className="space-y-5">
      {/* TOP: token flow through encoder → extract CLS */}
      <div className="flex items-end justify-center gap-2 flex-wrap">
        <div className="flex flex-col items-center">
          <div className="text-[9px] font-mono uppercase tracking-wide text-amber-400/80 mb-1">patches</div>
          <div className="space-y-0.5">
            {[0, 1, 2, 3, 4].map(i => (
              <div key={i} className="w-9 h-1.5 rounded-sm"
                style={{ background: stage >= 0 ? `linear-gradient(to right, hsl(${i*60}, 65%, 55%), hsl(${i*60+50}, 60%, 50%))` : 'rgba(100,116,139,0.2)' }}/>
            ))}
          </div>
          <div className="text-[8px] font-mono text-slate-500 mt-1">N × D</div>
        </div>

        <div className={`text-base font-mono pb-3 ${stage >= 1 ? 'text-amber-400' : 'text-slate-700'}`}>→</div>

        <div className="flex flex-col items-center">
          <div className={`text-[9px] font-mono uppercase tracking-wide mb-1 ${stage >= 1 ? 'text-amber-400/80' : 'text-slate-500'}`}>self-attention</div>
          <div className={`rounded-md border px-1.5 py-1 ${stage >= 1 ? 'bg-amber-500/10 border-amber-500/50' : 'bg-slate-800/30 border-slate-700/60'}`}>
            <EncoderGraph active={stage >= 1}/>
          </div>
          <div className={`text-[8px] font-mono mt-1 ${stage >= 1 ? 'text-amber-300/80' : 'text-slate-500'}`}>× L = 12</div>
        </div>

        <div className={`text-base font-mono pb-3 ${stage >= 2 ? 'text-amber-400' : 'text-slate-700'}`}>→</div>

        <div className="flex flex-col items-center">
          <div className={`text-[9px] font-mono uppercase tracking-wide mb-1 ${stage >= 2 ? 'text-rose-400/90' : 'text-slate-500'}`}>extract [CLS]</div>
          <div className="space-y-0.5">
            <div className={`w-9 h-1.5 rounded-sm ring-1 ring-rose-300 ${stage >= 2 ? 'shadow shadow-rose-500/50' : ''}`}
              style={{ background: stage >= 2 ? 'rgba(244,63,94,0.85)' : 'rgba(244,63,94,0.30)' }}/>
            {[0, 1, 2, 3].map(i => (
              <div key={i} className="w-9 h-1.5 rounded-sm bg-slate-700/30 opacity-50"/>
            ))}
          </div>
          <div className="text-[8px] font-mono text-slate-500 mt-1">1 × D</div>
        </div>
      </div>

      {/* MIDDLE: real matrix multiplication CLS · W = logits */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-amber-400/70 mb-3 text-center">
          linear classifier · logits = CLS · W
        </div>
        <div className="flex items-center justify-center gap-3 flex-wrap">
          {/* CLS vector — signed bars */}
          <div className="flex flex-col items-center">
            <div className="text-[10px] font-mono text-rose-300 mb-1">[CLS]</div>
            <div className={`flex gap-0.5 px-1.5 py-1 rounded border transition-all
              ${stage >= 2 ? 'bg-rose-500/5 border-rose-500/40' : 'bg-slate-800/30 border-slate-700/60'}`}>
              {computed.cls.map((v, i) => (
                <SignedBar key={i} v={v} active={stage >= 2}/>
              ))}
            </div>
            <div className="text-[8px] font-mono text-slate-500 mt-1">D = 768 (12 dims shown)</div>
          </div>

          <div className={`text-2xl font-mono pb-2 ${stage >= 3 ? 'text-amber-400' : 'text-slate-700'}`}>·</div>

          {/* W matrix — D × C colored grid */}
          <div className="flex flex-col items-center">
            <div className="text-[10px] font-mono text-amber-300 mb-1">W</div>
            <div className={`p-1 rounded border transition-all
              ${stage >= 3 ? 'bg-amber-500/5 border-amber-500/40' : 'bg-slate-800/30 border-slate-700/60'}`}>
              <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${D_show}, minmax(0, 1fr))` }}>
                {computed.W.map((v, i) => {
                  const c = i % C_show;
                  const d = Math.floor(i / C_show);
                  // arrange row-major by class so each row is one class
                  const idx = c * D_show + d;
                  return null;
                })}
                {Array.from({ length: C_show }).flatMap((_, c) =>
                  Array.from({ length: D_show }).map((_, d) => {
                    const v = computed.W[d * C_show + c];
                    const t = Math.min(1, Math.abs(v) / 1.3);
                    const bg = v >= 0
                      ? `rgba(245,158,11,${stage >= 3 ? 0.15 + t * 0.8 : 0.1})`
                      : `rgba(20,184,166,${stage >= 3 ? 0.15 + t * 0.8 : 0.1})`;
                    return (
                      <div key={`${c}-${d}`} className="w-3 h-3 rounded-[1px]"
                        title={`W[d=${d}, class=${c}] = ${v.toFixed(2)}`}
                        style={{ background: bg, gridRow: c + 1, gridColumn: d + 1 }}/>
                    );
                  })
                )}
              </div>
            </div>
            <div className="text-[8px] font-mono text-slate-500 mt-1">D × C (each row = one class)</div>
          </div>

          <div className={`text-2xl font-mono pb-2 ${stage >= 4 ? 'text-amber-400' : 'text-slate-700'}`}>=</div>

          {/* Logits — signed bars per class */}
          <div className="flex flex-col items-center">
            <div className="text-[10px] font-mono text-teal-300 mb-1">logits</div>
            <div className={`flex flex-col gap-1 px-2 py-1 rounded border transition-all
              ${stage >= 4 ? 'bg-teal-500/5 border-teal-500/40' : 'bg-slate-800/30 border-slate-700/60'}`}>
              {computed.logits.map((v, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-[8px] font-mono text-slate-500 w-4 text-right">{i}</span>
                  <SignedBar v={v} max={3.5} orient="horizontal" active={stage >= 4}/>
                  <span className={`text-[9px] font-mono w-8 ${v >= 0 ? 'text-amber-300' : 'text-teal-300'}`}>
                    {v.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
            <div className="text-[8px] font-mono text-slate-500 mt-1">unnormalized scores</div>
          </div>
        </div>
      </div>

      {/* BOTTOM: softmax → probabilities */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
        <div className="flex items-center gap-3 mb-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-400/70">
            softmax(logits) → class probabilities
          </div>
          <div className="text-[10px] font-mono text-slate-500">P(class) = exp(logit) / Σ exp(logits)</div>
        </div>
        <div className="space-y-1.5 max-w-2xl">
          {realProbs.map((p, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-slate-300 w-32 truncate text-right">{labels[i]}</span>
              <div className="flex-1 h-2.5 bg-slate-800 rounded overflow-hidden">
                <div className={`h-full transition-all duration-500 ${i === 0 ? 'bg-teal-400' : 'bg-teal-400/40'}`}
                  style={{ width: `${(stage >= 5 ? p : p * Math.min(1, (stage / 5))) * 100}%` }}/>
              </div>
              <span className="text-[11px] font-mono text-teal-300 w-12">
                {((stage >= 5 ? p : p * Math.min(1, (stage / 5))) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* MiniArch — concrete neural-network sketch. Tokens are rendered as
   colored bars so it's clear they're vectors. The encoder is a tiny
   bipartite graph showing every token attending to every token (which
   IS what self-attention does). Then CLS gets extracted, multiplied by
   the classifier weight matrix, and turned into class probabilities. */
function MiniArch({ progress, topPred }) {
  const stage = progress >= 0.95
    ? 4
    : progress >= 0.7 ? 3
    : progress >= 0.4 ? 2
    : progress >= 0.1 ? 1 : 0;

  const TokenBar = ({ hue, dim = false, cls = false, glow = false }) => (
    <div
      className={`w-9 h-1.5 rounded-sm transition-all
        ${cls ? 'ring-1 ring-rose-300' : ''}
        ${glow ? 'shadow shadow-rose-500/50' : ''}`}
      style={{
        background: dim
          ? 'rgba(100,116,139,0.18)'
          : cls
            ? 'rgba(244,63,94,0.85)'
            : `linear-gradient(to right, hsl(${hue}, 65%, 55%), hsl(${(hue + 50) % 360}, 60%, 50%))`,
        opacity: dim ? 0.5 : 1,
      }}
    />
  );
  const Arr = ({ on }) => (
    <div className={`text-base font-mono shrink-0 ${on ? 'text-amber-400' : 'text-slate-700'}`}>→</div>
  );
  const Label = ({ children, color = 'slate' }) => (
    <div className={`text-[9px] font-mono uppercase tracking-wide mb-1 text-${color}-400/80 text-center whitespace-nowrap`}>
      {children}
    </div>
  );

  // Compact encoder block: 5-node bipartite graph in a small SVG.
  const Encoder = ({ active }) => {
    const NODES = 5;
    const W = 64, H = 56;
    const inX = 8, outX = W - 8;
    const ySpacing = (H - 12) / (NODES - 1);
    const ys = Array.from({ length: NODES }, (_, i) => 6 + i * ySpacing);
    return (
      <div
        className={`rounded-md border px-1.5 py-1 transition-all
          ${active
            ? 'bg-amber-500/10 border-amber-500/50'
            : 'bg-slate-800/30 border-slate-700/60'}`}
      >
        <svg width={W} height={H} className="block">
          {active && ys.flatMap((y1, i) =>
            ys.map((y2, j) => (
              <line
                key={`e-${i}-${j}`}
                x1={inX} y1={y1} x2={outX} y2={y2}
                stroke="rgba(245,158,11,0.20)"
                strokeWidth="0.4"
              />
            ))
          )}
          {ys.map((y, i) => (
            <circle key={`l-${i}`} cx={inX} cy={y} r={2.2}
              fill={active ? '#fbbf24' : '#475569'} opacity={active ? 0.9 : 0.6}/>
          ))}
          {ys.map((y, i) => (
            <circle key={`r-${i}`} cx={outX} cy={y} r={2.2}
              fill={active ? '#fbbf24' : '#475569'} opacity={active ? 0.9 : 0.6}/>
          ))}
        </svg>
        <div className={`text-[8px] font-mono text-center leading-none ${active ? 'text-amber-300/70' : 'text-slate-600'}`}>× L</div>
      </div>
    );
  };

  const Linear = ({ active }) => (
    <div
      className={`rounded-md border px-1.5 py-1 transition-all flex flex-col items-center
        ${active
          ? 'bg-amber-500/10 border-amber-500/50'
          : 'bg-slate-800/30 border-slate-700/60'}`}
    >
      <div className="grid grid-cols-4 gap-px">
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} className="w-1.5 h-1.5 rounded-[1px]"
            style={{
              background: active ? `hsl(${(i * 25) % 360}, 60%, 55%)` : 'rgba(100,116,139,0.3)',
              opacity: active ? 0.85 : 0.4,
            }}/>
        ))}
      </div>
      <div className={`text-[8px] font-mono mt-1 leading-none ${active ? 'text-amber-200' : 'text-slate-500'}`}>
        W
      </div>
    </div>
  );

  return (
    <div className="flex items-end justify-between gap-1.5">
      <div className="flex flex-col items-center">
        <Label color={stage >= 0 ? 'amber' : 'slate'}>patches</Label>
        <div className="space-y-0.5">
          {[0, 1, 2, 3, 4].map(i => <TokenBar key={i} hue={i * 60}/>)}
        </div>
      </div>

      <Arr on={stage >= 1}/>

      <div className="flex flex-col items-center">
        <Label color={stage >= 1 ? 'amber' : 'slate'}>self-attn</Label>
        <Encoder active={stage >= 1}/>
      </div>

      <Arr on={stage >= 2}/>

      <div className="flex flex-col items-center">
        <Label color={stage >= 2 ? 'rose' : 'slate'}>[CLS]</Label>
        <div className="space-y-0.5">
          <TokenBar cls glow={stage >= 2} dim={stage < 2}/>
          {[0, 1, 2, 3].map(i => <TokenBar key={i} hue={0} dim/>)}
        </div>
      </div>

      <Arr on={stage >= 3}/>

      <div className="flex flex-col items-center">
        <Label color={stage >= 3 ? 'amber' : 'slate'}>classifier</Label>
        <Linear active={stage >= 3}/>
      </div>

      <Arr on={stage >= 4}/>

      <div className="flex flex-col items-center">
        <Label color={stage >= 4 ? 'teal' : 'slate'}>top class</Label>
        <div
          className={`px-2 py-1 rounded-md border text-center min-w-[80px] transition-all
            ${stage >= 4
              ? 'bg-teal-500/15 border-teal-500/50 text-teal-100'
              : 'bg-slate-800/30 border-slate-700/60 text-slate-500'}`}
        >
          <div className="text-[11px] font-mono leading-tight truncate">
            {stage >= 4 && topPred ? topPred.label : '—'}
          </div>
          {stage >= 4 && topPred && (
            <div className="text-[9px] font-mono opacity-80 leading-none">
              {(topPred.score * 100).toFixed(0)}%
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PredictionBars({ preds, accent }) {
  const max = (preds && preds[0]?.score) || 1;
  return (
    <div className="space-y-2">
      {(preds || []).map((p, i) => (
        <div key={`${p.label}-${i}`}>
          <div className="flex justify-between text-[13px] mb-1">
            <span className="text-slate-200 truncate pr-2">{p.label}</span>
            <span className={`font-mono ${accent === 'teal' ? 'text-teal-300' : 'text-amber-300'}`}>
              {(p.score * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                i === 0
                  ? (accent === 'teal' ? 'bg-teal-400' : 'bg-amber-400')
                  : (accent === 'teal' ? 'bg-teal-400/40' : 'bg-amber-400/40')
              }`}
              style={{ width: `${(p.score / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function LiveDemoTab() {
  const [galleryId, setGalleryId] = useState('cat');
  const [customSrc, setCustomSrc] = useState(null);
  const imgSrc = customSrc || GALLERY.find(g => g.id === galleryId)?.src;

  const [patchSize, setPatchSize] = useState(36);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState('Slow');

  // Real classifier — calls HuggingFace Inference API.
  // No local model download; each image fires off an HTTP request.
  // Token is optional but recommended (anonymous calls are heavily rate-limited).
  const [hfToken, setHfToken] = useState(() => {
    try { return localStorage.getItem('hfToken') || ''; } catch { return ''; }
  });
  const [modelStatus, setModelStatus] = useState('idle'); // idle | inferring | ready | error
  const [modelMessage, setModelMessage] = useState('');
  const [modelProgress, setModelProgress] = useState(0);
  const [realPreds, setRealPreds] = useState(null);

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
      drawScan(vitRef.current, img, 360, patchSize, progress, 'vit');
      drawScan(swinRef.current, img, 360, patchSize, progress, 'swin');
      drawMatrix(vitMatRef.current, attnRef.current?.vit, progress, 'vit');
      drawMatrix(swinMatRef.current, attnRef.current?.swin, progress, 'swin');
      setImgVersion(v => v + 1);
    };
    img.src = imgSrc;
    setRealPreds(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgSrc]);

  useEffect(() => {
    if (!imgRef.current) return;
    drawScan(vitRef.current, imgRef.current, 360, patchSize, progress, 'vit');
    drawScan(swinRef.current, imgRef.current, 360, patchSize, progress, 'swin');
    drawMatrix(vitMatRef.current, attnRef.current?.vit, progress, 'vit');
    drawMatrix(swinMatRef.current, attnRef.current?.swin, progress, 'swin');
  }, [patchSize, progress]);

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

  // Persist HF token to localStorage so users don't re-enter it.
  useEffect(() => {
    try {
      if (hfToken) localStorage.setItem('hfToken', hfToken);
      else localStorage.removeItem('hfToken');
    } catch { /* localStorage may be unavailable */ }
  }, [hfToken]);

  // Run inference on every image change by calling the HF Inference API.
  useEffect(() => {
    if (!imgSrc) return;
    let canceled = false;
    setModelStatus('inferring');
    setModelMessage('Calling HuggingFace Inference API…');
    setRealPreds(null);

    (async () => {
      try {
        // Fetch the image bytes (works for same-origin and uploaded blobs).
        const imgResp = await fetch(imgSrc);
        if (!imgResp.ok) throw new Error(`Could not load image (${imgResp.status})`);
        const blob = await imgResp.blob();

        const apiResp = await fetch(
          'https://api-inference.huggingface.co/models/google/vit-base-patch16-224',
          {
            method: 'POST',
            headers: hfToken ? { 'Authorization': `Bearer ${hfToken}` } : {},
            body: blob,
          }
        );
        if (canceled) return;

        if (!apiResp.ok) {
          const txt = await apiResp.text().catch(() => '');
          if (apiResp.status === 401 || apiResp.status === 403) {
            setModelStatus('error');
            setModelMessage('API requires a HuggingFace token — paste one above.');
            return;
          }
          if (apiResp.status === 429) {
            setModelStatus('error');
            setModelMessage('Rate limited (anonymous quota). Add a HF token for higher limits.');
            return;
          }
          if (apiResp.status === 503) {
            setModelStatus('error');
            setModelMessage('Model is warming up on HF servers (cold start). Try again in ~20 s.');
            return;
          }
          throw new Error(`HF API ${apiResp.status}: ${txt.slice(0, 140)}`);
        }

        const out = await apiResp.json();
        if (canceled) return;
        if (!Array.isArray(out)) throw new Error('Unexpected API response shape');

        // Clean up ImageNet synonym lists, keep top 5.
        const cleaned = out.slice(0, 5).map(p => {
          const first = String(p.label || '').split(',')[0].trim();
          const label = first ? first[0].toUpperCase() + first.slice(1) : first;
          return { ...p, label };
        });
        setRealPreds(cleaned);
        setModelStatus('ready');
        setModelMessage('');
      } catch (err) {
        if (!canceled) {
          setModelStatus('error');
          setModelMessage(String(err.message || err));
        }
      }
    })();
    return () => { canceled = true; };
  }, [imgSrc, hfToken]);

  // Status string passed to the predictions panel inside ClassContribution.
  const ccStatus = modelStatus === 'inferring' ? 'inferring'
    : modelStatus === 'error' ? 'error'
    : modelStatus === 'ready' ? 'ready'
    : 'inferring';

  return (
    <div className="space-y-3">
      {/* Compact header */}
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <h2 className="font-serif text-2xl text-slate-100 tracking-tight">
          Watch the model classify
        </h2>
        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span className="text-slate-400 hidden sm:inline">
            ViT-Base/16 (ImageNet-1K) · HuggingFace Inference API
          </span>
          <input
            type="password"
            value={hfToken}
            onChange={e => setHfToken(e.target.value.trim())}
            placeholder="hf_… (optional token)"
            spellCheck={false}
            className="w-44 px-2 py-1 rounded bg-slate-900 border border-slate-700 text-amber-200 text-[11px] focus:border-amber-500 focus:outline-none"
            aria-label="HuggingFace API token"
          />
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noreferrer"
            className="text-amber-300 underline underline-offset-2 hover:text-amber-200"
          >
            get one
          </a>
        </div>
      </div>

      {/* Image gallery — compact single row */}
      <Card className="p-2">
        <div className="grid grid-cols-4 gap-2">
          {GALLERY.map(g => (
            <button
              key={g.id}
              onClick={() => { setCustomSrc(null); setGalleryId(g.id); reset(); }}
              className={`relative rounded-md overflow-hidden border-2 aspect-[3/2] transition-all
                ${!customSrc && galleryId === g.id ? 'border-amber-400 shadow shadow-amber-500/15' : 'border-slate-700 hover:border-slate-500'}`}
            >
              <img src={g.src} alt="" className="w-full h-full object-cover"/>
            </button>
          ))}
          <label className={`relative rounded-md overflow-hidden border-2 transition-all flex items-center justify-center cursor-pointer aspect-[3/2]
            ${customSrc ? 'border-amber-400 shadow shadow-amber-500/15' : 'border-dashed border-slate-700 hover:border-slate-500 bg-slate-800/30'}`}>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => {
                const f = e.target.files?.[0];
                if (!f) return;
                const url = URL.createObjectURL(f);
                setCustomSrc(url);
                reset();
              }}
            />
            {customSrc ? (
              <img src={customSrc} alt="" className="w-full h-full object-cover"/>
            ) : (
              <div className="text-center">
                <Upload size={16} className="mx-auto mb-0.5 text-slate-400"/>
                <span className="text-[10px] font-mono text-slate-400">Upload</span>
              </div>
            )}
          </label>
        </div>
      </Card>

      {/* Three-column main row: ViT scan/matrix · Swin scan/matrix · per-class predictions */}
      <div className="grid lg:grid-cols-3 gap-3">
        <Card className="p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Tag color="amber">ViT</Tag>
              <span className="text-[12px] text-slate-200">Global</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500 uppercase">O(N²)</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">scan</div>
              <canvas ref={vitRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">matrix</div>
              <canvas ref={vitMatRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2 leading-snug">
            Every patch attends to every other · dense matrix.
          </p>
        </Card>

        <Card className="p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Tag color="teal">Swin</Tag>
              <span className="text-[12px] text-slate-200">Windowed</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500 uppercase">O(N)</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">scan</div>
              <canvas ref={swinRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 uppercase mb-0.5">matrix</div>
              <canvas ref={swinMatRef} className="w-full rounded bg-slate-950 aspect-square"/>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2 leading-snug">
            Attention stays inside windows · block-diagonal matrix.
          </p>
        </Card>

        <Card className="p-3">
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
          />
        </Card>
      </div>

      {/* Compact controls row */}
      <Card className="p-3">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 items-end">
          <div className="flex gap-2">
            <button
              onClick={() => { if (progress >= 1) setProgress(0); setPlaying(p => !p); }}
              className="px-3 py-1.5 rounded-md bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-200 flex items-center gap-1.5 text-sm font-medium transition-all"
            >
              {playing ? <Pause size={13}/> : <Play size={13}/>}
              {playing ? 'Pause' : (progress >= 1 ? 'Replay' : 'Play')}
            </button>
            <button
              onClick={reset}
              className="px-2.5 py-1.5 rounded-md bg-slate-800/60 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-all"
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
              {[16, 32, 48, 64].map(opt => (
                <button
                  key={opt}
                  onClick={() => { setPatchSize(opt); reset(); }}
                  className={`flex-1 px-1.5 py-1 rounded text-xs font-mono border transition-all
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
                className="w-12 px-1.5 py-1 rounded text-xs font-mono bg-slate-900 border border-slate-700 text-amber-200 focus:border-amber-500 focus:outline-none"
                aria-label="Custom patch size"
              />
            </div>
          </div>
        </div>
      </Card>
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
      'A 224×224 image has 50,176 pixels. Self-attention is O(N²), so attending pixel-to-pixel would mean a 50k×50k score matrix per layer.',
    ],
    options: [
      { text: 'Patches make image data fit in GPU memory by compressing it.', correct: false,
        explanation: 'Patches don\'t compress information — they reshape it. Each patch is still a flat vector of pixel values, projected linearly into a token embedding.' },
      { text: 'Transformers consume sequences of tokens, and patches turn the image into a manageably short sequence.', correct: true,
        explanation: 'Right. ViT treats the image as N = (H·W)/P² patch-tokens. With P=16 on a 224² image you get 196 tokens — small enough that O(N²) attention is feasible.' },
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
      'You build an N×N matrix of scores, then do a softmax and weighted sum.',
    ],
    options: [
      { text: 'O(N · d)', correct: false,
        explanation: 'That\'s the cost of a single projection (e.g. computing one token\'s Q vector), not of full attention.' },
      { text: 'O(N · log N · d)', correct: false,
        explanation: 'Sub-quadratic costs like this come up in efficient/sparse attention variants, but standard self-attention is fully quadratic in N.' },
      { text: 'O(N² · d)', correct: true,
        explanation: 'Computing the score matrix QKᵀ is O(N²·d), and the weighted sum AV is also O(N²·d). This quadratic cost is exactly what makes high-resolution images expensive — and what Swin sidesteps with windows.' },
      { text: 'O(d²)', correct: false,
        explanation: 'O(d²) is the cost of multiplying a single token by a d×d projection matrix — a per-token cost, not the full attention cost over all tokens.' },
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
        explanation: 'Patches are uniform in size (e.g. 16×16). Position embeddings exist regardless of whether patch sizes vary.' },
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
        explanation: 'Yes. Within a window of M×M patches, attention is O(M²) per window. Total cost is *linear* in N (number of patches) instead of quadratic — the headline result of the Swin paper.' },
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
        explanation: 'Right. In layer ℓ a patch sees patches in its window. In layer ℓ+1 the windows are shifted by half a window, so what used to be a boundary is now interior — patches that were separated can now attend to each other.' },
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
        explanation: 'Total cost is similar; the d-dim space is just split across heads. Multi-head is about expressivity, not speed.' },
      { text: 'Multiple parallel attention "subspaces", each learning a different pattern of relationships.', correct: true,
        explanation: 'Yes. Each head projects Q/K/V into a smaller d/h-dim space and computes its own attention map. The outputs are concatenated, letting the layer attend to several relational structures at once.' },
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
      { text: 'A learned 2×2 patch-merging layer that combines four neighboring patches into one.', correct: true,
        explanation: 'Yes. At each stage boundary, every 2×2 group of patches is concatenated (4·C channels) then linearly projected back to 2C channels. Spatial resolution halves; channels double — exactly like a CNN feature pyramid.' },
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
      { text: 'Roughly the same as a CNN\'s 3×3 kernel receptive field.', correct: false,
        explanation: 'A 3×3 kernel covers 9 pixels. ViT\'s first attention layer can mix information from *every* patch.' },
      { text: 'Global — every patch can attend to every other patch in a single layer.', correct: true,
        explanation: 'Right. That\'s the headline difference. CNNs build up the receptive field gradually through stacked convolutions; ViT has it from layer 1 — at the cost of O(N²) attention.' },
      { text: 'Always zero in the first layer; only later layers see other patches.', correct: false,
        explanation: 'Self-attention mixes tokens at every layer, including the first. The patch-embedding step before it is local, but attention is global.' },
    ],
  },
  {
    id: 'q10',
    question: 'You have a 224×224 image with full self-attention and patch size 16. If you increase resolution to 448×448 keeping patch size at 16, by what factor does the cost of one attention layer grow?',
    hints: [
      'First figure out how many patches you have at each resolution.',
      'Self-attention is O(N²). Quadrupling N quadruples N — and squares the cost.',
    ],
    options: [
      { text: '2×', correct: false,
        explanation: 'Doubling resolution doesn\'t just double the patches — it quadruples them (4× in 2D).' },
      { text: '4×', correct: false,
        explanation: '4× would be the right answer if attention were O(N). It\'s not — it\'s O(N²).' },
      { text: '16×', correct: true,
        explanation: 'Right. 224/16 = 14 → 196 patches. 448/16 = 28 → 784 patches (4× more). Self-attention is O(N²), so 4² = 16× more compute. This is exactly why Swin\'s linear-cost windowed attention matters at high resolutions.' },
      { text: '256×', correct: false,
        explanation: 'Too high. You\'d need patches to grow 16× (not 4×) for 16² = 256× cost. Here patches grow only 4×.' },
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
        <h3 className="font-serif text-lg text-slate-100 leading-snug">{q.question}</h3>
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
          {q.hints[0]}
        </div>
      )}
      {hint2 && (
        <div className="mb-3 px-3 py-2 rounded bg-amber-500/[0.12] border border-amber-500/40 text-[13px] text-amber-100/95 leading-relaxed">
          {q.hints[1]}
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
                <span className="text-sm flex-1">{opt.text}</span>
                {checked && i === correctIdx && <Check size={14} className="text-emerald-300 mt-0.5 shrink-0"/>}
                {checked && i === selected && i !== correctIdx && <X size={14} className="text-rose-300 mt-0.5 shrink-0"/>}
              </div>
              {checked && (
                <p className="mt-2 text-[12px] text-slate-300/80 leading-relaxed pl-7">
                  {opt.explanation}
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

export default function App() {
  const [tab, setTab] = useState('live');

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
    <div className="min-h-screen text-slate-200 font-sans relative overflow-x-hidden"
      style={{
        background:
          'radial-gradient(ellipse at top right, rgba(245,158,11,0.08), transparent 50%), ' +
          'radial-gradient(ellipse at bottom left, rgba(20,184,166,0.06), transparent 50%), ' +
          '#0a0e1a',
      }}>
      {/* subtle grain */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] mix-blend-overlay"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
        }}
      />

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
                      className={`px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-all
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
      `}</style>
    </div>
  );
}
