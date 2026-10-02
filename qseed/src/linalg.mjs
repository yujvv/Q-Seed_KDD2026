// Clustering, PCA, and diversity (Vendi) on embedding vectors.
import { Matrix, EigenvalueDecomposition } from 'ml-matrix';
import { rng, normalize, dot } from './util.mjs';

// Spherical k-means (cosine) on L2-normalized vectors.
// Returns { assign, centroids, k }.
export function sphericalKMeans(vectors, k, { iters = 60, seed = 1, restarts = 3 } = {}) {
  const X = vectors.map(normalize);
  const n = X.length, dim = X[0].length;
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const rand = rng(seed + r * 101);
    // k-means++ init on cosine distance
    const cIdx = [Math.floor(rand() * n)];
    while (cIdx.length < k) {
      const d2 = X.map(x => {
        let best = -1; for (const ci of cIdx) best = Math.max(best, dot(x, X[ci]));
        return Math.max(1e-9, 1 - best);
      });
      const sum = d2.reduce((a, b) => a + b, 0);
      let t = rand() * sum, pick = 0;
      for (let i = 0; i < n; i++) { t -= d2[i]; if (t <= 0) { pick = i; break; } }
      cIdx.push(pick);
    }
    let centroids = cIdx.map(i => X[i].slice());
    let assign = new Array(n).fill(0);
    for (let it = 0; it < iters; it++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let bc = 0, bs = -Infinity;
        for (let c = 0; c < k; c++) { const s = dot(X[i], centroids[c]); if (s > bs) { bs = s; bc = c; } }
        if (assign[i] !== bc) { assign[i] = bc; changed = true; }
      }
      const sums = Array.from({ length: k }, () => new Float64Array(dim));
      const cnt = new Array(k).fill(0);
      for (let i = 0; i < n; i++) { const c = assign[i]; cnt[c]++; const s = sums[c]; const x = X[i]; for (let d = 0; d < dim; d++) s[d] += x[d]; }
      for (let c = 0; c < k; c++) {
        if (cnt[c] === 0) { centroids[c] = X[Math.floor(rand() * n)].slice(); continue; }
        centroids[c] = normalize(Array.from(sums[c]));
      }
      if (!changed && it > 0) break;
    }
    // inertia = sum of cosine sim to own centroid (higher better)
    let inertia = 0; for (let i = 0; i < n; i++) inertia += dot(X[i], centroids[assign[i]]);
    if (!best || inertia > best.inertia) best = { assign, centroids, k, inertia };
  }
  return best;
}

// Silhouette-lite: mean(cosine to own centroid) - used only to pick k coarsely.
export function meanCentroidSim(vectors, res) {
  const X = vectors.map(normalize);
  let s = 0; for (let i = 0; i < X.length; i++) s += dot(X[i], res.centroids[res.assign[i]]);
  return s / X.length;
}

// Standard (Euclidean) k-means for low-dimensional feature vectors.
export function euclideanKMeans(vectors, k, { iters = 80, seed = 1, restarts = 4 } = {}) {
  const n = vectors.length, dim = vectors[0].length;
  const d2 = (a, b) => { let s = 0; for (let i = 0; i < dim; i++) { const t = a[i] - b[i]; s += t * t; } return s; };
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const rand = rng(seed + r * 131);
    const cIdx = [Math.floor(rand() * n)];
    while (cIdx.length < k) {
      const dist = vectors.map(x => { let m = Infinity; for (const ci of cIdx) m = Math.min(m, d2(x, vectors[ci])); return m; });
      const sum = dist.reduce((a, b) => a + b, 0) || 1;
      let t = rand() * sum, pick = 0;
      for (let i = 0; i < n; i++) { t -= dist[i]; if (t <= 0) { pick = i; break; } }
      cIdx.push(pick);
    }
    let cent = cIdx.map(i => vectors[i].slice());
    const assign = new Array(n).fill(0);
    for (let it = 0; it < iters; it++) {
      let changed = false;
      for (let i = 0; i < n; i++) { let bc = 0, bd = Infinity; for (let c = 0; c < k; c++) { const dd = d2(vectors[i], cent[c]); if (dd < bd) { bd = dd; bc = c; } } if (assign[i] !== bc) { assign[i] = bc; changed = true; } }
      const sums = Array.from({ length: k }, () => new Float64Array(dim)); const cnt = new Array(k).fill(0);
      for (let i = 0; i < n; i++) { const c = assign[i]; cnt[c]++; for (let j = 0; j < dim; j++) sums[c][j] += vectors[i][j]; }
      for (let c = 0; c < k; c++) { if (!cnt[c]) { cent[c] = vectors[Math.floor(rand() * n)].slice(); continue; } for (let j = 0; j < dim; j++) cent[c][j] = sums[c][j] / cnt[c]; }
      if (!changed && it > 0) break;
    }
    let inertia = 0; for (let i = 0; i < n; i++) inertia += d2(vectors[i], cent[assign[i]]);
    if (!best || inertia < best.inertia) best = { assign, centroids: cent, k, inertia };
  }
  return best;
}

// PCA via eigendecomposition of covariance. Returns projector fn -> reduced dims.
export function pcaFit(vectors, dims = 40) {
  const n = vectors.length, d = vectors[0].length;
  const meanv = new Float64Array(d);
  for (const v of vectors) for (let i = 0; i < d; i++) meanv[i] += v[i];
  for (let i = 0; i < d; i++) meanv[i] /= n;
  const Xc = vectors.map(v => v.map((x, i) => x - meanv[i]));
  // covariance d x d  (d=1536 is large; use Gram trick: eigvecs of X X^T then map)
  // For n < d, compute n x n Gram matrix G = Xc Xc^T, eigenvectors u -> principal dirs w = Xc^T u.
  const G = new Array(n);
  for (let i = 0; i < n; i++) { G[i] = new Float64Array(n); for (let j = 0; j <= i; j++) { let s = 0; const a = Xc[i], b = Xc[j]; for (let t = 0; t < d; t++) s += a[t] * b[t]; G[i][j] = s; if (i !== j) G[j] = G[j] || new Float64Array(n); } }
  // symmetric fill
  const Gm = new Array(n); for (let i = 0; i < n; i++) { Gm[i] = new Array(n); for (let j = 0; j < n; j++) Gm[i][j] = j <= i ? G[i][j] : G[j][i]; }
  const evd = new EigenvalueDecomposition(new Matrix(Gm));
  const evals = evd.realEigenvalues, evecs = evd.eigenvectorMatrix.to2DArray();
  // sort desc
  const order = evals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, dims).map(x => x[1]);
  // principal directions w_k = Xc^T u_k / sqrt(lambda_k), each length d
  const dirs = order.map(k => {
    const uk = evecs.map(row => row[k]); // length n
    const w = new Float64Array(d);
    for (let i = 0; i < n; i++) { const c = uk[i]; const xi = Xc[i]; for (let t = 0; t < d; t++) w[t] += c * xi[t]; }
    const lam = Math.max(evals[k], 1e-9); const nrm = Math.sqrt(lam);
    for (let t = 0; t < d; t++) w[t] /= nrm;
    return w;
  });
  const project = (v) => {
    const vc = v.map((x, i) => x - meanv[i]);
    return dirs.map(w => { let s = 0; for (let t = 0; t < d; t++) s += vc[t] * w[t]; return s; });
  };
  return { project, dims: dirs.length };
}

// Vendi Score: exp(Shannon entropy of eigenvalues of normalized similarity matrix K/n).
// Effective number of distinct modes in the set. Higher = more diverse.
export function vendiScore(vectors, { maxN = 400 } = {}) {
  let X = vectors.map(normalize);
  if (X.length > maxN) {
    // deterministic uniform (Fisher-Yates) subsample
    const rand = rng(3); const idx = X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    X = idx.slice(0, maxN).map(i => X[i]);
  }
  const n = X.length;
  const K = new Array(n);
  for (let i = 0; i < n; i++) { K[i] = new Array(n); for (let j = 0; j < n; j++) K[i][j] = j < i ? K[j][i] : dot(X[i], X[j]) / n; }
  const evd = new EigenvalueDecomposition(new Matrix(K));
  const ev = evd.realEigenvalues.map(v => Math.max(v, 0));
  const s = ev.reduce((a, b) => a + b, 0) || 1;
  let H = 0; for (const v of ev) { const p = v / s; if (p > 1e-12) H -= p * Math.log(p); }
  return Math.exp(H);
}
