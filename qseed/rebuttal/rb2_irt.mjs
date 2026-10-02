// Rebuttal E2 (Reviewer EPyX W5/Q5): stability of the 2PL item parameters.
//  (a) refit on the independent retest round  -> test-retest of a_q, b_q
//  (b) bootstrap over domains (the "test-takers") -> per-item rank stability
//  (c) random split of the domains into two halves -> split-half of a_q, b_q
// Free: probes are served from the released cache.
import { loadVertical, VERTICALS, ENGINES, CONFIG, save } from './common.mjs';
import { buildIRTMatrices } from '../src/probe.mjs';
import { fit2PL } from '../src/irt.mjs';
import { spearman, rng, shuffle, mean, quantile } from '../src/util.mjs';

function pooled(recs, domains) {
  const mats = buildIRTMatrices(recs, domains, ENGINES);
  const Q = recs.length, D = domains.length;
  const succ = Array.from({ length: Q }, () => new Array(D).fill(0));
  const trials = Array.from({ length: Q }, () => new Array(D).fill(0));
  for (const e of ENGINES) for (let q = 0; q < Q; q++) for (let d = 0; d < D; d++) { succ[q][d] += mats[e.id].succ[q][d]; trials[q][d] += mats[e.id].trials[q][d]; }
  return { succ, trials };
}
const cols = (M, idx) => M.map(row => idx.map(d => row[d]));
const topQ = (a) => { const t = quantile(a, 0.75); return new Set(a.map((x, i) => [x, i]).filter(([x]) => x >= t).map(([, i]) => i)); };
const jac = (A, B) => { let k = 0; for (const x of A) if (B.has(x)) k++; return k / (A.size + B.size - k); };

const NBOOT = Number(process.env.NBOOT || 200), NSPLIT = 50;
const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v);
  const m1 = pooled(S.recsQ0, S.domains), m2 = pooled(S.recsQ0r2, S.domains);
  const f1 = fit2PL(m1.succ, m1.trials), f2 = fit2PL(m2.succ, m2.trials);
  const D = S.domains.length, Q = S.recsQ0.length;
  const res = {
    Q, D,
    retest: { a: spearman(f1.a, f2.a), b: spearman(f1.b, f2.b), theta: spearman(f1.theta, f2.theta), topQuartileJaccard: jac(topQ(f1.a), topQ(f2.a)) },
  };
  const rand = rng(CONFIG.seed + 4242);
  const bootA = [], bootB = [], bootJ = [], perItem = Array.from({ length: Q }, () => []);
  for (let k = 0; k < NBOOT; k++) {
    const idx = Array.from({ length: D }, () => Math.floor(rand() * D));
    const f = fit2PL(cols(m1.succ, idx), cols(m1.trials, idx), { iters: 1500 });
    bootA.push(spearman(f.a, f1.a)); bootB.push(spearman(f.b, f1.b)); bootJ.push(jac(topQ(f.a), topQ(f1.a)));
    f.a.forEach((x, q) => perItem[q].push(x));
  }
  // rank-interval width of a_q: how far an item's discrimination moves across replicates
  const iqrRatio = perItem.map(xs => (quantile(xs, 0.75) - quantile(xs, 0.25)) / (quantile(xs, 0.5) || 1));
  res.bootstrap = {
    n: NBOOT, aRho: mean(bootA), aRhoCI: [quantile(bootA, 0.025), quantile(bootA, 0.975)],
    bRho: mean(bootB), bRhoCI: [quantile(bootB, 0.025), quantile(bootB, 0.975)],
    topQuartileJaccard: mean(bootJ), medianItemIQRoverMedian: quantile(iqrRatio, 0.5),
    // heterogeneity claim of the paper: top-quartile / bottom-quartile mean discrimination
    spreadRatio: (() => { const rs = []; for (let k = 0; k < NBOOT; k++) { const a = perItem.map(xs => xs[k]).sort((x, y) => x - y); const q = Math.floor(Q / 4); rs.push(mean(a.slice(-q)) / (mean(a.slice(0, q)) || 1e-9)); } return { mean: mean(rs), lo: quantile(rs, 0.025), hi: quantile(rs, 0.975) }; })(),
  };
  const sa = [], sb = [];
  for (let k = 0; k < NSPLIT; k++) {
    const idx = shuffle(Array.from({ length: D }, (_, i) => i), rng(CONFIG.seed + 7000 + k));
    const h = Math.floor(D / 2), A = idx.slice(0, h), Bh = idx.slice(h, 2 * h);
    const fa = fit2PL(cols(m1.succ, A), cols(m1.trials, A), { iters: 1500 }), fb = fit2PL(cols(m1.succ, Bh), cols(m1.trials, Bh), { iters: 1500 });
    sa.push(spearman(fa.a, fb.a)); sb.push(spearman(fa.b, fb.b));
  }
  res.domainSplitHalf = { a: mean(sa), b: mean(sb) };
  out[v] = res;
  console.error(v, JSON.stringify(res));
}
save('e2_irt_stability.json', out);
const avg = (f) => mean(VERTICALS.map(v => f(out[v])));
console.log('MEAN retest rho: a', avg(x => x.retest.a).toFixed(2), 'b', avg(x => x.retest.b).toFixed(2), 'theta', avg(x => x.retest.theta).toFixed(2), 'topQ-J', avg(x => x.retest.topQuartileJaccard).toFixed(2));
console.log('MEAN bootstrap rho: a', avg(x => x.bootstrap.aRho).toFixed(2), 'b', avg(x => x.bootstrap.bRho).toFixed(2), 'topQ-J', avg(x => x.bootstrap.topQuartileJaccard).toFixed(2), 'spread', avg(x => x.bootstrap.spreadRatio.mean).toFixed(1));
console.log('MEAN domain split-half rho: a', avg(x => x.domainSplitHalf.a).toFixed(2), 'b', avg(x => x.domainSplitHalf.b).toFixed(2));
