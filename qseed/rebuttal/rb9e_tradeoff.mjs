// Coverage granularity (M) x realism weight (lambda): can one bank have both
// fine-grained independent coverage and S4-space realism? Free (cached).
import { loadVertical, VERTICALS, CONFIG, save, freeEmbed, embedN } from './common.mjs';
import { sphericalKMeans } from '../src/linalg.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { calibrateRealism } from '../src/calibrate.mjs';
import { dot, mean } from '../src/util.mjs';
const ENC = ['nvidia/nemotron-3-embed-1b:free', 'liquid/lfm-2.5-embedding-350m:free'], KS = [12, 18, 24, 30, 40];
const GRID = [[18, 0.35], [40, 0.35], [40, 2], [40, 5], [40, 15], [0, 2], [0, 5], [0, 15]];
const assign = (e, cents) => { let b = 0, s = -2; cents.forEach((c, j) => { const x = dot(e, c); if (x > s) { s = x; b = j; } }); return b; };
const acc = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const trainE = await embedN(S.train.map(k => k.kw));
  const totW = S.train.reduce((s, k) => s + k.logvol, 0), scale = S.tree.clusters.reduce((s, c) => s + c.weight, 0) / totW;
  const indep = [];
  for (const enc of ENC) { const kwE = await freeEmbed(S.bank.keywords.map(k => k.kw), enc); for (const k of KS) for (const seed of [1, 2, 3]) { const km = sphericalKMeans(kwE, k, { seed: 1000 * seed + k, restarts: 2 }); const w = new Array(k).fill(0); S.bank.keywords.forEach((kw, i) => { w[km.assign[i]] += kw.logvol; }); indep.push({ enc, cents: km.centroids, w, tot: w.reduce((x, y) => x + y, 0) }); } }
  for (const [M, lam] of GRID) {
    let clusters;
    if (M === 0) clusters = S.train.map((k, i) => ({ centroid: trainE[i], weight: k.logvol * scale }));
    else { const km = sphericalKMeans(trainE, M, { seed: CONFIG.seed, restarts: 3 }); const w = new Array(M).fill(0); S.train.forEach((k, i) => { w[km.assign[i]] += k.logvol; }); clusters = km.centroids.map((c, j) => ({ centroid: c, weight: w[j] * scale })); }
    const qs = lazyGreedy(S.keptItems, clusters, CONFIG.B, lam).selectedIds.map(S.qById);
    const a = (acc[(M || 'N') + '|' + lam] ||= { cov: [], auc: [] });
    a.auc.push(mean([4, 5, 6, 7, 8].map(seed => calibrateRealism(qs, S.realPool, { seed }).discAUC)));
    const E = {}; for (const enc of ENC) E[enc] = await freeEmbed(qs, enc);
    a.cov.push(mean(indep.map(c => { const hit = new Set(E[c.enc].map(e => assign(e, c.cents))); let s = 0; for (const j of hit) s += c.w[j]; return s / c.tot; })));
  }
}
const out = Object.fromEntries(Object.entries(acc).map(([k, a]) => [k, { indepCoverage: mean(a.cov), auc: mean(a.auc) }]));
save('e9e_tradeoff.json', out);
console.log('M | lambda | independent coverage (mean over 2 encoders, k=12..40) | S4-space AUC');
for (const [k, o] of Object.entries(out)) console.log(k.replace('|', ' | '), '|', (100 * o.indepCoverage).toFixed(1), '|', o.auc.toFixed(3));
