// Deterministic RNG + statistics utilities (pure JS).

// Mulberry32 seeded RNG for reproducibility.
export function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
export function sample(arr, k, rand) { return shuffle(arr, rand).slice(0, k); }

// ---- vector ops ----
export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function normalize(a) { const n = norm(a) || 1; return a.map(x => x / n); }
export function cosine(a, b) { return dot(a, b) / ((norm(a) * norm(b)) || 1); }
export function add(a, b) { return a.map((x, i) => x + b[i]); }
export function scale(a, s) { return a.map(x => x * s); }
export const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
export function std(a) { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); }
export function sigmoid(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
export function softplus(x) { return x > 20 ? x : Math.log1p(Math.exp(x)); }

export function quantile(arr, q) {
  const a = arr.slice().sort((x, y) => x - y);
  if (!a.length) return NaN;
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? a[lo] : a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

// ---- ranking / correlation ----
function ranks(x) {
  const idx = x.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(x.length);
  let i = 0;
  while (i < idx.length) {
    let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
    i = j + 1;
  }
  return r;
}
export function pearson(x, y) {
  const mx = mean(x), my = mean(y);
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < x.length; i++) { const a = x[i] - mx, b = y[i] - my; num += a * b; dx += a * a; dy += b * b; }
  return num / (Math.sqrt(dx * dy) || 1);
}
export function spearman(x, y) { return pearson(ranks(x), ranks(y)); }
// Kendall's tau-b: tie-adjusted so that many tied values (e.g. domains with
// visibility 0) do not deflate the coefficient toward zero.
export function kendallTau(x, y) {
  let nc = 0, nd = 0, n1 = 0, n2 = 0; const n = x.length;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const dx = Math.sign(x[i] - x[j]), dy = Math.sign(y[i] - y[j]);
    const tx = dx === 0, ty = dy === 0;
    if (tx) n1++;            // pair tied in x
    if (ty) n2++;            // pair tied in y
    if (!tx && !ty) { if (dx === dy) nc++; else nd++; }
  }
  const n0 = 0.5 * n * (n - 1);
  const denom = Math.sqrt((n0 - n1) * (n0 - n2));
  return denom ? (nc - nd) / denom : 0;
}

// ---- bootstrap CI ----
export function bootstrapCI(values, stat = mean, { nboot = 2000, alpha = 0.05, seed = 7 } = {}) {
  const rand = rng(seed), n = values.length, stats = [];
  for (let b = 0; b < nboot; b++) {
    const s = new Array(n);
    for (let i = 0; i < n; i++) s[i] = values[Math.floor(rand() * n)];
    stats.push(stat(s));
  }
  stats.sort((a, b) => a - b);
  return { est: stat(values), lo: quantile(stats, alpha / 2), hi: quantile(stats, 1 - alpha / 2) };
}

// Paired bootstrap for difference of a statistic between two aligned arrays.
export function pairedBootstrapDiff(a, b, stat = mean, { nboot = 5000, seed = 11 } = {}) {
  const rand = rng(seed), n = a.length, diffs = [];
  for (let k = 0; k < nboot; k++) {
    const sa = new Array(n), sb = new Array(n);
    for (let i = 0; i < n; i++) { const j = Math.floor(rand() * n); sa[i] = a[j]; sb[i] = b[j]; }
    diffs.push(stat(sa) - stat(sb));
  }
  diffs.sort((x, y) => x - y);
  const est = stat(a) - stat(b);
  // two-sided p via fraction crossing zero
  let below = 0; for (const d of diffs) if (d <= 0) below++;
  const p = 2 * Math.min(below / nboot, 1 - below / nboot);
  return { est, lo: quantile(diffs, 0.025), hi: quantile(diffs, 0.975), p: Math.max(p, 1 / nboot) };
}

// Holm-Bonferroni step-down correction. Returns adjusted p-values in input order.
export function holm(pvals) {
  const idx = pvals.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const m = pvals.length, adj = new Array(m);
  let running = 0;
  for (let k = 0; k < m; k++) {
    const [p, i] = idx[k];
    running = Math.max(running, Math.min(1, (m - k) * p));
    adj[i] = running;
  }
  return adj;
}

// ---- AUC (Mann-Whitney) ----
export function auc(scores, labels) {
  // labels 1 = positive. AUC = P(score_pos > score_neg).
  const pos = [], neg = [];
  for (let i = 0; i < scores.length; i++) (labels[i] ? pos : neg).push(scores[i]);
  if (!pos.length || !neg.length) return 0.5;
  const all = scores.map((s, i) => [s, labels[i]]).sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  for (let i = 0; i < all.length; i++) if (all[i][1]) rankSum += (i + 1);
  const U = rankSum - pos.length * (pos.length + 1) / 2;
  return U / (pos.length * neg.length);
}

// Krippendorff's alpha for nominal binary ratings.
// data: array of items, each an array of ratings (0/1), missing allowed as null.
export function krippendorffAlpha(data) {
  const values = new Set();
  for (const item of data) for (const r of item) if (r != null) values.add(r);
  const vals = [...values];
  // observed disagreement
  let Do_num = 0, nTotal = 0;
  const coincidence = {};
  for (const v of vals) for (const w of vals) coincidence[v + ',' + w] = 0;
  const marginal = {}; for (const v of vals) marginal[v] = 0;
  for (const item of data) {
    const present = item.filter(r => r != null);
    const mu = present.length;
    if (mu < 2) continue;
    for (let i = 0; i < mu; i++) for (let j = 0; j < mu; j++) if (i !== j) {
      coincidence[present[i] + ',' + present[j]] += 1 / (mu - 1);
    }
  }
  let n = 0;
  for (const v of vals) { let row = 0; for (const w of vals) row += coincidence[v + ',' + w]; marginal[v] = row; n += row; }
  // metric delta: nominal -> 1 if v!=w else 0
  for (const v of vals) for (const w of vals) if (v !== w) Do_num += coincidence[v + ',' + w];
  const Do = Do_num / n;
  let De_num = 0;
  for (const v of vals) for (const w of vals) if (v !== w) De_num += marginal[v] * marginal[w];
  const De = De_num / (n * (n - 1));
  return De === 0 ? 1 : 1 - Do / De;
}
