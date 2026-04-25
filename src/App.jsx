import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Grid3x3, Eye, Layers, Play, Pause, RotateCcw, ChevronRight,
  Box, Network, GitBranch, Move, BookOpen,
  ArrowRight, Hash, Crosshair, Workflow, Microscope, Sparkles,
  Upload, Loader2, Brain, Cpu
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
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
              Flattened patch sequence — N = {numPatches}
            </div>
            <div
              className="grid gap-1 p-2 bg-slate-950/40 rounded border border-slate-800 max-h-[280px] overflow-auto"
              style={{ gridTemplateColumns: `repeat(${Math.min(grid, 16)}, minmax(0,1fr))` }}
            >
              {patchSequence.map(i => (
                <PatchThumb
                  key={i}
                  index={i}
                  patchSize={patchSize}
                  grid={grid}
                  highlighted={i === hoveredPatch}
                  onHover={() => setHoveredPatch(i)}
                />
              ))}
            </div>
            <div className="text-[11px] text-slate-500 mt-3 italic">
              Hover the image or any thumbnail — the grid loses no information about
              order on its own. Position embeddings (next tab) will tell the model who came from where.
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
  const [showCls, setShowCls] = useState(true);
  const [shuffle, setShuffle] = useState(false);
  const [embDim] = useState(64);
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
    <div className="space-y-8">
      <Section icon={Hash} kicker="03 — Spatial order, restored" title="Position embeddings & [CLS] token">
        <p className="max-w-3xl mb-3">
          Self-attention is permutation-equivariant: scramble the input tokens and the output
          scrambles the same way. That means the raw patch sequence carries no spatial information.
          ViT fixes this by adding a learned <Eq>E_pos ∈ ℝ^(N+1)×D</Eq> to each token before the
          first encoder block.
        </p>
        <p className="max-w-3xl">
          A learnable <span className="text-rose-300">[CLS]</span> token is prepended to the sequence;
          its final-layer representation is what the classification head reads. Below: a stylized
          2D-sinusoidal position scheme — tokens close in space have high cosine similarity in their
          position embeddings.
        </p>
      </Section>

      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            Token sequence (4×4 = 16 patches)
          </div>
          <div className="flex items-center gap-1 mb-4 flex-wrap">
            {showCls && (
              <div className="w-9 h-9 rounded bg-rose-500/20 border-2 border-rose-400 flex items-center justify-center font-mono text-[10px] text-rose-300">
                CLS
              </div>
            )}
            {order.map(i => (
              <div key={i} className="w-9 h-9 rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center font-mono text-xs text-amber-200">
                {i}
              </div>
            ))}
          </div>
          <div className="space-y-2">
            <Toggle label="Add position embeddings" value={posOn} onChange={setPosOn} />
            <Toggle label="Prepend [CLS] token" value={showCls} onChange={setShowCls} />
            <Toggle label="Randomly shuffle patches" value={shuffle} onChange={setShuffle} />
          </div>
          <div className="mt-4 text-[12px] text-slate-400 leading-relaxed">
            {!posOn && <span className="text-rose-300">Without position embeddings, shuffling has zero effect on the model.</span>}
            {posOn && shuffle && <span className="text-amber-300">With position embeddings, the model can detect that the order is wrong — each patch carries its identity.</span>}
            {posOn && !shuffle && <span className="text-teal-300">Each token now carries both content (patch) and location (position) information.</span>}
          </div>
        </Card>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            Position embedding matrix · 16 × 64
          </div>
          <PosHeatmap data={posMatrix} rows={N} cols={embDim} />
          <div className="mt-3 text-[11px] text-slate-500 leading-relaxed">
            Each row is the position embedding of one patch (top-left to bottom-right).
            Brighter = larger value. The diagonal banding reflects the 2D sin/cos basis.
          </div>
        </Card>

        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            Cosine similarity between positions
          </div>
          <SimHeatmap data={simMatrix} N={N} />
          <div className="mt-3 text-[11px] text-slate-500 leading-relaxed">
            Cell (i, j) shows how similar position i's embedding is to position j's. Spatial neighbors
            cluster together — this is what gives self-attention its sense of "near" and "far".
          </div>
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
  const [patchSize, setPatchSize] = useState(32);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [seed, setSeed] = useState(7);
  const [step, setStep] = useState(4);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const SIZE = 256;
  const grid = SIZE / patchSize;
  const N = grid * grid;
  const D = 16;

  const [computed, setComputed] = useState({ Q: null, K: null, V: null, attn: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = SIZE; canvas.height = SIZE;
    drawTestImage(canvas.getContext('2d'), SIZE);
    const X = computePatchFeatures(canvas, patchSize, D, seed);
    const rng = mulberry32(seed);
    const Wq = randMatrix(D, D, rng);
    const Wk = randMatrix(D, D, rng);
    const Wv = randMatrix(D, D, rng);
    const Q = matmul(X, Wq);
    const K = matmul(X, Wk);
    const V = matmul(X, Wv);
    const Kt = transpose(K);
    const scores = matmul(Q, Kt);
    for (let i = 0; i < scores.data.length; i++) scores.data[i] /= Math.sqrt(D);
    const attn = softmaxRows(scores);
    setComputed({ Q, K, V, attn });
  }, [patchSize, seed]);

  const { Q, K, V, attn } = computed;

  // Draw overlay: highlight selected, draw attention as heat
  useEffect(() => {
    const c = overlayRef.current; if (!c) return;
    c.width = SIZE; c.height = SIZE;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, SIZE, SIZE);

    if (step >= 1 && selectedPatch !== null && attn) {
      // draw attention heat for selected query
      for (let py = 0; py < grid; py++) {
        for (let px = 0; px < grid; px++) {
          const i = py * grid + px;
          const a = attn.data[selectedPatch * N + i];
          const t = Math.pow(a * N, 0.5); // amplify
          ctx.fillStyle = `rgba(245, 158, 11, ${Math.min(0.85, t * 0.9)})`;
          ctx.fillRect(px * patchSize, py * patchSize, patchSize, patchSize);
        }
      }
    }
    // grid
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
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
  }, [selectedPatch, attn, patchSize, grid, N, step]);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * SIZE;
    const px = Math.floor(x / patchSize);
    const py = Math.floor(y / patchSize);
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
        <Card className="p-5">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-3">
            {selectedPatch !== null ? `Query: patch #${selectedPatch}` : 'Click a patch to set the query'}
          </div>
          <div
            className="relative w-[256px] h-[256px] rounded-lg overflow-hidden ring-1 ring-slate-700 cursor-crosshair"
            onClick={handleClick}
          >
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
            <canvas ref={overlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
          </div>
          <div className="mt-4 space-y-3">
            <Slider label="Patch size" value={patchSize} options={[16, 32, 64]} onChange={setPatchSize} />
            <Slider label="Random seed (W_Q, W_K, W_V)" value={seed} min={1} max={50} onChange={setSeed} />
          </div>
          <div className="mt-3 text-[11px] text-slate-500 leading-relaxed italic">
            Brighter = higher attention weight. Try the seed slider — different random projections produce
            different attention heads.
          </div>
        </Card>

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
          {step >= 1 && attn && (
            <Card className="p-5">
              <div className="text-sm text-slate-200 mb-3">
                Attention matrix <Eq>A ∈ ℝ^(N×N)</Eq> · row i = how much patch i attends to every patch.
              </div>
              <AttentionMatrix attn={attn} N={N} highlight={selectedPatch} onCellClick={setSelectedPatch} />
              <div className="mt-3 text-[11px] text-slate-500 italic">
                Each row sums to 1. Click a cell to set that row as the query.
              </div>
            </Card>
          )}
          {step >= 4 && selectedPatch !== null && attn && (
            <Card className="p-5">
              <div className="text-sm text-slate-200 mb-3">
                Top-attended patches for query #{selectedPatch}
              </div>
              <TopAttended attn={attn} N={N} query={selectedPatch} grid={grid} patchSize={patchSize} />
            </Card>
          )}
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

function AttentionMatrix({ attn, N, highlight, onCellClick }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = N; c.height = N;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(N, N);
    let mx = 0;
    for (let i = 0; i < attn.data.length; i++) mx = Math.max(mx, attn.data[i]);
    for (let i = 0; i < attn.data.length; i++) {
      const t = Math.pow(attn.data[i] / mx, 0.6);
      img.data[i * 4] = Math.floor(245 * t);
      img.data[i * 4 + 1] = Math.floor(158 * t);
      img.data[i * 4 + 2] = Math.floor(11 * t);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [attn, N]);
  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * N);
    onCellClick(row);
  };
  return (
    <div className="relative">
      <canvas
        ref={ref}
        className="w-full aspect-square rounded border border-slate-800 cursor-pointer"
        style={{ imageRendering: 'pixelated' }}
        onClick={handleClick}
      />
      {highlight !== null && (
        <div
          className="absolute pointer-events-none border-2 border-rose-400"
          style={{
            left: '0%', width: '100%',
            top: `${(highlight / N) * 100}%`,
            height: `${100 / N}%`,
          }}
        />
      )}
      <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
        <span>← keys (j) →</span>
      </div>
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
  const [selectedPatch, setSelectedPatch] = useState(28);
  const [activeHead, setActiveHead] = useState(0);
  const canvasRef = useRef(null);
  const SIZE = 256;
  const patchSize = 32;
  const grid = SIZE / patchSize; // 8
  const N = grid * grid; // 64
  const D = 32;
  const headDim = D / numHeads;

  const [heads, setHeads] = useState([]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.width = SIZE; c.height = SIZE;
    drawTestImage(c.getContext('2d'), SIZE);
    const X = computePatchFeatures(c, patchSize, D, 7);
    const out = [];
    for (let h = 0; h < numHeads; h++) {
      const rng = mulberry32(101 + h * 17);
      const Wq = randMatrix(D, headDim, rng);
      const Wk = randMatrix(D, headDim, rng);
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
        <p className="max-w-3xl mb-3">
          A single attention head must compromise across many possible relationships — texture, color,
          spatial proximity, semantic similarity. Multi-head attention runs <Eq>h</Eq> independent
          attention computations in parallel, each on a smaller subspace of dimension <Eq>d_k = D/h</Eq>,
          and concatenates the results.
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
            className="relative w-[256px] h-[256px] rounded-lg overflow-hidden ring-1 ring-slate-700 cursor-crosshair"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const x = ((e.clientX - rect.left) / rect.width) * SIZE;
              const y = ((e.clientY - rect.top) / rect.height) * SIZE;
              setSelectedPatch(Math.floor(y / patchSize) * grid + Math.floor(x / patchSize));
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
    const py = Math.floor(query / grid), px = query % grid;
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 3;
    ctx.strokeRect(px * patchSize, py * patchSize, patchSize, patchSize);
  }, [attn, N, grid, patchSize, query, size]);
  return <canvas ref={ref} className="absolute inset-0 w-full h-full pointer-events-none" />;
}

function HeadMini({ attn, N, grid, query }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !attn) return;
    c.width = grid; c.height = grid;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(grid, grid);
    for (let i = 0; i < N; i++) {
      const a = attn.data[query * N + i];
      const t = Math.pow(a * N, 0.5);
      img.data[i * 4] = Math.floor(245 * t);
      img.data[i * 4 + 1] = Math.floor(158 * t);
      img.data[i * 4 + 2] = Math.floor(11 * t);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    // mark query
    const py = Math.floor(query / grid), px = query % grid;
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 0.4;
    ctx.strokeRect(px, py, 1, 1);
  }, [attn, N, grid, query]);
  return <canvas ref={ref} className="w-full aspect-square rounded" style={{ imageRendering: 'pixelated' }} />;
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
        <p className="max-w-3xl mb-4">
          The full pipeline composes the pieces from the last few tabs. Step through each stage to see the
          tensor shapes evolve. The encoder is a stack of <Eq>L</Eq> identical blocks — each one applies
          multi-head self-attention, then an MLP, both wrapped in residual connections and pre-norm LayerNorm.
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

      <Card className="p-6 min-h-[300px]">
        <PipelineDetail step={step} />
      </Card>

      <Card className="p-6">
        <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-4">Inside one Transformer encoder block</div>
        <EncoderBlockDiagram />
      </Card>
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
        <p className="max-w-3xl mb-3">
          Global self-attention costs <Eq>O(N²)</Eq> in the number of tokens. For an 800×800 image with
          patch size 4, that's 40,000 tokens — and 1.6 billion attention pairs per layer. Untenable.
        </p>
        <p className="max-w-3xl">
          Swin's first move: partition the patches into non-overlapping windows of <Eq>M × M</Eq> tokens,
          and let attention happen <span className="text-teal-300">only inside each window</span>. Cost
          per layer drops to <Eq>O(M² · N)</Eq> — linear in the number of patches.
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
        <p className="max-w-3xl mb-3">
          Swin alternates two layers. Layer <Eq>l</Eq> uses regular window partitioning <span className="font-mono text-amber-300">W-MSA</span>.
          Layer <Eq>l+1</Eq> shifts the window grid by <Eq>(⌊M/2⌋, ⌊M/2⌋)</Eq> pixels — call this <span className="font-mono text-amber-300">SW-MSA</span>.
        </p>
        <p className="max-w-3xl">
          Patches that were neighbors-across-a-wall in layer <Eq>l</Eq> now share a window in layer <Eq>l+1</Eq>.
          After two layers, every patch has effectively communicated with everything in a <Eq>2M × 2M</Eq> region.
          Stack more, and the receptive field keeps growing — at linear cost.
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
        <p className="max-w-3xl mb-3">
          ViT operates at a single resolution — usually <Eq>14×14</Eq> — for the whole network.
          That's enough for image classification, but disastrous for dense prediction tasks like
          detection and segmentation, which need fine-grained features.
        </p>
        <p className="max-w-3xl">
          Swin builds a feature pyramid like a ConvNet. Between stages, a <span className="text-teal-300">patch
          merging</span> layer concatenates each <Eq>2×2</Eq> neighborhood of patches and projects
          back to <Eq>2C</Eq> dimensions. Resolution halves; channels double; receptive field doubles.
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
        <p className="max-w-3xl">
          Both models work — they just optimize different things. ViT is conceptually purer and scales
          beautifully on huge datasets. Swin is more practical for real-world dense-prediction tasks
          and trains well at smaller scales. Knowing the trade-offs helps you pick the right backbone.
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
  const imgName = customSrc ? 'your image' : (GALLERY.find(g => g.id === galleryId)?.name || 'image');

  const [patchSize, setPatchSize] = useState(36);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState('simulated');

  const [realPreds, setRealPreds] = useState(null);
  const [realStatus, setRealStatus] = useState('idle');
  const [realProgress, setRealProgress] = useState(0);
  const [realMessage, setRealMessage] = useState('');
  const classifierRef = useRef(null);

  const vitRef = useRef();
  const swinRef = useRef();
  const imgRef = useRef();

  const reset = useCallback(() => { setProgress(0); setPlaying(false); }, []);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      drawScan(vitRef.current, img, 360, patchSize, progress, 'vit');
      drawScan(swinRef.current, img, 360, patchSize, progress, 'swin');
    };
    img.src = imgSrc;
    setRealPreds(null);
    setRealStatus('idle');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imgSrc]);

  useEffect(() => {
    if (!imgRef.current) return;
    drawScan(vitRef.current, imgRef.current, 360, patchSize, progress, 'vit');
    drawScan(swinRef.current, imgRef.current, 360, patchSize, progress, 'swin');
  }, [patchSize, progress]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setProgress(p => {
        const next = p + 0.012;
        if (next >= 1) { setPlaying(false); return 1; }
        return next;
      });
    }, 50);
    return () => clearInterval(id);
  }, [playing]);

  const baseSim = customSrc ? SIM_FALLBACK : (SIM_PREDS[galleryId] || SIM_FALLBACK);
  const animatedSim = baseSim.map(p => ({
    ...p,
    score: progress < 0.05 ? 0 : p.score * Math.min(1, (progress - 0.05) / 0.95),
  }));

  async function runReal() {
    if (realStatus === 'loading') return;
    setRealStatus('loading');
    setRealPreds(null);
    setRealProgress(0);
    try {
      if (!classifierRef.current) {
        setRealMessage('Loading transformers.js…');
        const mod = await import('@huggingface/transformers');
        mod.env.allowLocalModels = false;
        setRealMessage('Downloading ViT-Base/16 weights (~88 MB, cached after first run)…');
        classifierRef.current = await mod.pipeline(
          'image-classification',
          'Xenova/vit-base-patch16-224',
          {
            progress_callback: (data) => {
              if (data.status === 'progress') {
                setRealProgress(Math.round(data.progress || 0));
              }
            },
          }
        );
      }
      setRealMessage('Running inference…');
      const out = await classifierRef.current(imgSrc, { topk: 5 });
      setRealPreds(out);
      setRealStatus('ready');
      setRealMessage('');
    } catch (err) {
      console.error(err);
      setRealStatus('error');
      setRealMessage(String(err.message || err));
    }
  }

  const showPreds = mode === 'real' ? realPreds : animatedSim;

  return (
    <div className="space-y-6">
      <div>
        <span className="font-mono text-[10px] tracking-[0.3em] text-amber-400/70 uppercase">Live demo</span>
        <h2 className="font-serif text-3xl text-slate-100 mt-1 mb-2 tracking-tight">
          Watch the model look at a cat
        </h2>
        <p className="text-slate-300/90 leading-relaxed max-w-2xl">
          Pick an image, hit play, and see how Vision Transformer (ViT) and Swin Transformer
          scan it patch by patch. ViT lets every patch attend to every other; Swin keeps
          attention inside local windows. The probability bars below show the model's confidence
          in each class.
        </p>
      </div>

      <Card className="p-5">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {GALLERY.map(g => (
            <button
              key={g.id}
              onClick={() => { setCustomSrc(null); setGalleryId(g.id); reset(); }}
              className={`relative rounded-lg overflow-hidden border-2 aspect-square transition-all
                ${!customSrc && galleryId === g.id ? 'border-amber-400 shadow-lg shadow-amber-500/10' : 'border-slate-700 hover:border-slate-500'}`}
            >
              <img src={g.src} alt={g.name} className="w-full h-full object-cover" />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/95 to-transparent px-2 py-1.5 text-left">
                <span className="text-[11px] font-mono text-slate-100">{g.name}</span>
              </div>
            </button>
          ))}
          <label className={`relative rounded-lg overflow-hidden border-2 transition-all flex items-center justify-center cursor-pointer aspect-square
            ${customSrc ? 'border-amber-400 shadow-lg shadow-amber-500/10' : 'border-dashed border-slate-700 hover:border-slate-500 bg-slate-800/30'}`}>
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
              <>
                <img src={customSrc} alt="custom" className="w-full h-full object-cover" />
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/95 to-transparent px-2 py-1.5 text-left">
                  <span className="text-[11px] font-mono text-slate-100">Your image</span>
                </div>
              </>
            ) : (
              <div className="text-center">
                <Upload size={20} className="mx-auto mb-1 text-slate-400" />
                <span className="text-[11px] font-mono text-slate-400">Upload</span>
              </div>
            )}
          </label>
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card className="p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Tag color="amber">ViT</Tag>
              <span className="text-sm text-slate-200">Global Attention</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500 uppercase">O(N²)</span>
          </div>
          <canvas ref={vitRef} className="w-full rounded-lg bg-slate-950 aspect-square" />
          <p className="text-[12px] text-slate-400 mt-3 leading-relaxed">
            Lines from the bright patch reach <em>every</em> other patch. Expensive, but lets
            distant pixels talk directly.
          </p>
        </Card>
        <Card className="p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Tag color="teal">Swin</Tag>
              <span className="text-sm text-slate-200">Windowed Attention</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500 uppercase">O(N)</span>
          </div>
          <canvas ref={swinRef} className="w-full rounded-lg bg-slate-950 aspect-square" />
          <p className="text-[12px] text-slate-400 mt-3 leading-relaxed">
            Attention stays inside the bright 4×4 window. Cheaper at high resolution; later
            layers <em>shift</em> windows to mix neighbours.
          </p>
        </Card>
      </div>

      <Card className="p-5">
        <div className="flex flex-wrap items-end gap-5">
          <div className="flex gap-2">
            <button
              onClick={() => { if (progress >= 1) setProgress(0); setPlaying(p => !p); }}
              className="px-3.5 py-2 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-200 flex items-center gap-1.5 text-sm font-medium transition-all"
            >
              {playing ? <Pause size={14}/> : <Play size={14}/>}
              {playing ? 'Pause' : (progress >= 1 ? 'Replay scan' : 'Play scan')}
            </button>
            <button
              onClick={reset}
              className="px-3.5 py-2 rounded-lg bg-slate-800/60 hover:bg-slate-800 border border-slate-700 text-slate-300 flex items-center gap-1.5 text-sm transition-all"
            >
              <RotateCcw size={14}/> Reset
            </button>
          </div>
          <div className="flex-1 min-w-[200px]">
            <Slider
              label="Scan Progress"
              value={Math.round(progress * 100)}
              min={0} max={100} suffix="%"
              onChange={v => { setPlaying(false); setProgress(v / 100); }}
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <Slider
              label="Patch Size (px)"
              value={patchSize}
              options={[24, 36, 60]}
              onChange={v => { setPatchSize(v); reset(); }}
            />
          </div>
        </div>
      </Card>

      <Card className="p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <Brain size={18} className="text-amber-300"/>
            <h3 className="font-serif text-lg text-slate-100">Predictions</h3>
            <span className="text-[11px] font-mono text-slate-500">on {imgName}</span>
          </div>
          <div className="flex gap-1 bg-slate-800/60 rounded-lg p-1 border border-slate-700/60">
            <button
              onClick={() => setMode('simulated')}
              className={`px-3 py-1.5 rounded-md text-[11px] font-mono tracking-wider transition-all
                ${mode === 'simulated' ? 'bg-amber-500/25 text-amber-200 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              SIMULATED
            </button>
            <button
              onClick={() => setMode('real')}
              className={`px-3 py-1.5 rounded-md text-[11px] font-mono tracking-wider transition-all flex items-center gap-1
                ${mode === 'real' ? 'bg-teal-500/25 text-teal-200 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              <Cpu size={11}/> REAL CLASSIFIER
            </button>
          </div>
        </div>

        {mode === 'real' && (
          <div className="mb-4 p-3 rounded-lg bg-slate-800/40 border border-slate-700/60">
            {realStatus === 'idle' && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-[13px] text-slate-300">
                  Runs <span className="text-teal-300 font-medium">ViT-Base/16</span> in your browser via transformers.js.
                  First load downloads ~88 MB, cached afterward.
                </p>
                <button onClick={runReal}
                  className="px-3 py-1.5 rounded-lg bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/50 text-teal-100 text-sm whitespace-nowrap transition-all">
                  Run on this image
                </button>
              </div>
            )}
            {realStatus === 'loading' && (
              <div>
                <div className="flex items-center gap-2 text-[13px] text-slate-300 mb-2">
                  <Loader2 size={14} className="animate-spin text-teal-300"/>
                  <span>{realMessage}</span>
                </div>
                <div className="h-1.5 bg-slate-700/60 rounded overflow-hidden">
                  <div className="h-full bg-teal-400 transition-all" style={{ width: `${realProgress}%` }}/>
                </div>
              </div>
            )}
            {realStatus === 'ready' && (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-[13px] text-teal-200">Real ViT-Base/16 prediction · model cached.</p>
                <button onClick={runReal}
                  className="px-3 py-1.5 rounded-lg bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/50 text-teal-100 text-sm whitespace-nowrap transition-all">
                  Re-run
                </button>
              </div>
            )}
            {realStatus === 'error' && (
              <p className="text-[13px] text-rose-300">Error: {realMessage}</p>
            )}
          </div>
        )}

        {mode === 'real' && !realPreds ? (
          <p className="text-sm text-slate-500 text-center py-6">
            {realStatus === 'idle' ? 'Click "Run on this image" to get a real ViT prediction.' : ''}
          </p>
        ) : (
          <PredictionBars preds={showPreds} accent={mode === 'real' ? 'teal' : 'amber'} />
        )}

        <p className="text-[11px] font-mono text-slate-500 mt-4">
          {mode === 'simulated'
            ? '// Simulated probabilities — illustrative only, scaled by scan progress.'
            : '// Real predictions from Xenova/vit-base-patch16-224 (ImageNet-1K).'}
        </p>
      </Card>
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
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto py-2 -mx-2 px-2 no-scrollbar">
            {TABS.map(t => {
              const Icon = t.icon;
              const active = tab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex-shrink-0 px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-all
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
