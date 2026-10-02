// S4 - Realism calibration (STYLE-based, topic-invariant).
// Extract stylistic features, fit a logistic-regression discriminator on those
// features (real WildChat search prompts vs synthetic candidates), output
// r(q)=P(real|q), and measure stylistic distributional distance via MAUVE (over
// standardised style vectors) + held-out discriminator AUC. Embeddings are
// deliberately NOT used here: they let the discriminator separate on topic
// rather than the phrasing realism S4 is meant to calibrate.
import { euclideanKMeans } from './linalg.mjs';
import { rng, shuffle, auc, mean, std } from './util.mjs';

const STOP_Q = ['what', 'which', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'should', 'can', 'do', 'does'];
const CONSTRAINT = ['under', 'less than', 'below', 'over', 'best', 'vs', 'versus', 'than', 'without', 'with', 'for', 'near me', 'cheap', 'budget', 'top', 'compare', 'difference', 'between', 'good', 'affordable'];

export function styleFeatures(q) {
  const s = q.toLowerCase();
  const words = s.split(/\s+/).filter(Boolean);
  const wc = words.length;
  const hasQ = /\?/.test(q) ? 1 : 0;
  const firstPerson = /\b(i|my|me|we|our|i'm|i've)\b/.test(s) ? 1 : 0;
  let constraints = 0; for (const c of CONSTRAINT) if (s.includes(c)) constraints++;
  const nums = (s.match(/\d+/g) || []).length;
  const ner = (q.match(/\b[A-Z][a-z]{2,}/g) || []).length; // capitalized-word proxy
  const avgWordLen = wc ? words.reduce((a, w) => a + w.length, 0) / wc : 0;
  const startsQ = STOP_Q.includes(words[0]) ? 1 : 0;
  return [wc, q.length, hasQ, firstPerson, constraints, nums, ner, avgWordLen, startsQ];
}

// Logistic regression with standardization + L2, gradient descent.
export function fitLogReg(X, y, { iters = 400, lr = 0.3, l2 = 1e-3, seed = 3 } = {}) {
  const n = X.length, d = X[0].length;
  const mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  for (let j = 0; j < d; j++) { mu[j] = mean(X.map(r => r[j])); sd[j] = std(X.map(r => r[j])) || 1; }
  const Z = X.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
  const w = new Array(d).fill(0); let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = new Array(d).fill(0); let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b; for (let j = 0; j < d; j++) z += w[j] * Z[i][j];
      const p = 1 / (1 + Math.exp(-z));
      const e = p - y[i];
      for (let j = 0; j < d; j++) gw[j] += e * Z[i][j];
      gb += e;
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]);
    b -= lr * (gb / n);
  }
  const predict = (row) => {
    let z = b; for (let j = 0; j < d; j++) z += w[j] * ((row[j] - mu[j]) / sd[j]);
    return 1 / (1 + Math.exp(-z));
  };
  return { predict, w, b, mu, sd };
}

// MAUVE over arbitrary feature vectors: cluster the joint set (Euclidean),
// form per-cluster histograms for the two populations, integrate the
// KL-divergence frontier. Returns a value in (0,1]; higher = closer.
export function mauve(realVecs, synthVecs, { k = 0, c = 5, seed = 9 } = {}) {
  const R = realVecs, S = synthVecs;
  const all = R.concat(S);
  const kk = k || Math.max(6, Math.min(50, Math.floor(all.length / 15)));
  const res = euclideanKMeans(all, kk, { seed, restarts: 2, iters: 40 });
  const p = new Array(kk).fill(0), qd = new Array(kk).fill(0);
  for (let i = 0; i < R.length; i++) p[res.assign[i]]++;
  for (let i = 0; i < S.length; i++) qd[res.assign[R.length + i]]++;
  const ps = p.map(v => v / R.length), qs = qd.map(v => v / S.length);
  const pts = [];
  for (let t = 1; t <= 24; t++) {
    const lam = t / 25;
    let klPM = 0, klQM = 0;
    for (let i = 0; i < kk; i++) {
      const m = lam * ps[i] + (1 - lam) * qs[i];
      if (ps[i] > 0 && m > 0) klPM += ps[i] * Math.log(ps[i] / m);
      if (qs[i] > 0 && m > 0) klQM += qs[i] * Math.log(qs[i] / m);
    }
    pts.push([Math.exp(-c * klQM), Math.exp(-c * klPM)]);
  }
  pts.sort((a, b) => a[0] - b[0]);
  // area under curve (trapezoid), x in [0,1]
  let area = 0;
  const xs = [0, ...pts.map(p => p[0]), 1];
  const ys = [pts[0][1], ...pts.map(p => p[1]), pts[pts.length - 1][1]];
  for (let i = 1; i < xs.length; i++) area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2;
  return Math.max(0, Math.min(1, area));
}

// Standardise a set of style-feature rows against real+synth pooled stats.
function standardize(rows, mu, sd) { return rows.map(r => r.map((v, j) => (v - mu[j]) / sd[j])); }

// Calibrate stylistic realism for a candidate set against a real-query pool.
// Returns r(q) per candidate + held-out discriminator AUC + style MAUVE.
export function calibrateRealism(candidateQueries, realPool, { seed = 4 } = {}) {
  const rand = rng(seed);
  const realSample = shuffle(realPool, rand).slice(0, Math.min(1200, realPool.length));
  const synthSample = candidateQueries;

  const Xreal0 = realSample.map(styleFeatures);
  const Xsyn0 = synthSample.map(styleFeatures);
  // pooled standardisation (so MAUVE clustering is scale-free)
  const d = Xreal0[0].length, pooled = Xreal0.concat(Xsyn0);
  const mu = new Array(d), sd = new Array(d);
  for (let j = 0; j < d; j++) { mu[j] = mean(pooled.map(r => r[j])); sd[j] = std(pooled.map(r => r[j])) || 1; }
  const Zreal = standardize(Xreal0, mu, sd), Zsyn = standardize(Xsyn0, mu, sd);

  // held-out discriminator AUC
  const idxR = shuffle(realSample.map((_, i) => i), rand);
  const idxS = shuffle(synthSample.map((_, i) => i), rand);
  const cutR = Math.floor(idxR.length * 0.7), cutS = Math.floor(idxS.length * 0.7);
  const trainX = [], trainY = [];
  idxR.slice(0, cutR).forEach(i => { trainX.push(Xreal0[i]); trainY.push(1); });
  idxS.slice(0, cutS).forEach(i => { trainX.push(Xsyn0[i]); trainY.push(0); });
  const clf = fitLogReg(trainX, trainY);
  const testScores = [], testLabels = [];
  idxR.slice(cutR).forEach(i => { testScores.push(clf.predict(Xreal0[i])); testLabels.push(1); });
  idxS.slice(cutS).forEach(i => { testScores.push(clf.predict(Xsyn0[i])); testLabels.push(0); });
  const discAUC = auc(testScores, testLabels);

  // r(q) = P(real | q) from discriminator on all data
  const clfAll = fitLogReg(Xreal0.map(x => x).concat(Xsyn0), Xreal0.map(() => 1).concat(Xsyn0.map(() => 0)));
  const rq = synthSample.map((q, i) => clfAll.predict(Xsyn0[i]));

  const mauveScore = mauve(Zreal, Zsyn);
  return { rq, discAUC, mauve: mauveScore };
}
