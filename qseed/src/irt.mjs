// 2PL Item Response Theory.
// P(y_qd = 1) = sigmoid(a_q (theta_d - b_q))
//   theta_d : domain "GEO ability"     b_q : query difficulty   a_q : query discrimination
// Joint MLE by gradient ascent; a_q = softplus(alpha_q) keeps discrimination > 0
// (which also fixes the reflection indeterminacy so higher theta = more cited).
import { sigmoid, softplus, mean, std } from './util.mjs';

// succ[q][d] successes, trials[q][d] trial counts (Binomial per cell).
// Joint MLE with Adam; a_q = softplus(alpha_q) > 0, theta standardized per step.
export function fit2PL(succ, trials, { iters = 3000, lr = 0.05 } = {}) {
  const Q = succ.length, D = succ[0].length;
  let alpha = new Float64Array(Q);
  let b = new Float64Array(Q);
  let theta = new Float64Array(D).map((_, d) => (d % 2 ? 0.1 : -0.1));
  const st = {
    mA: new Float64Array(Q), vA: new Float64Array(Q),
    mB: new Float64Array(Q), vB: new Float64Array(Q),
    mT: new Float64Array(D), vT: new Float64Array(D),
  };
  const b1 = 0.9, b2 = 0.999, eps = 1e-8, priB = 0.05, priAlpha = 0.02;
  for (let it = 1; it <= iters; it++) {
    const gA = new Float64Array(Q), gB = new Float64Array(Q), gT = new Float64Array(D);
    for (let q = 0; q < Q; q++) {
      const a = softplus(alpha[q]); const dA = sigmoid(alpha[q]);
      for (let d = 0; d < D; d++) {
        const n = trials[q][d]; if (!n) continue;
        const p = sigmoid(a * (theta[d] - b[q]));
        const resid = succ[q][d] - n * p;
        gT[d] += a * resid; gB[q] += -a * resid; gA[q] += (theta[d] - b[q]) * resid * dA;
      }
      gB[q] -= priB * b[q]; gA[q] -= priAlpha * alpha[q];
    }
    const upd = (p, g, m, v, i) => {
      m[i] = b1 * m[i] + (1 - b1) * g; v[i] = b2 * v[i] + (1 - b2) * g * g;
      return lr * (m[i] / (1 - Math.pow(b1, it))) / (Math.sqrt(v[i] / (1 - Math.pow(b2, it))) + eps);
    };
    for (let q = 0; q < Q; q++) { alpha[q] += upd(alpha[q], gA[q], st.mA, st.vA, q); b[q] += upd(b[q], gB[q], st.mB, st.vB, q); }
    for (let d = 0; d < D; d++) theta[d] += upd(theta[d], gT[d], st.mT, st.vT, d);
    let mt = 0; for (let d = 0; d < D; d++) mt += theta[d]; mt /= D;
    let s2 = 0; for (let d = 0; d < D; d++) s2 += (theta[d] - mt) ** 2; const sd = Math.sqrt(s2 / D) || 1;
    for (let d = 0; d < D; d++) theta[d] = (theta[d] - mt) / sd;
  }
  return { a: Array.from(alpha.map(softplus)), b: Array.from(b), theta: Array.from(theta) };
}

// Fisher information of query q at ability theta: a^2 p(1-p).
export function itemInfo(a, b, thetaVal) {
  const p = sigmoid(a * (thetaVal - b));
  return a * a * p * (1 - p);
}

// Total test information across a theta grid for a set of items.
export function testInfoCurve(items, grid) {
  return grid.map(t => items.reduce((s, it) => s + itemInfo(it.a, it.b, t), 0));
}
