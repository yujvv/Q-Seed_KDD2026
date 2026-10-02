// Is the independent-space coverage gap a property of the method or of the
// granularity M it was run at? Re-select the bank from the SAME candidate pool
// with the same objective F, but with the intent space cut finer (M' clusters of
// the training keywords, up to one "intent" per keyword). Selection is local and
// free; evaluation uses the two independent encoders (cached embeddings).
import { loadVertical, VERTICALS, CONFIG, save, freeEmbed, embedN } from './common.mjs';
import { sphericalKMeans } from '../src/linalg.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { calibrateRealism } from '../src/calibrate.mjs';
import * as BL from '../src/baselines.mjs';
import { dot, mean } from '../src/util.mjs';
const ENC = ['nvidia/nemotron-3-embed-1b:free', 'liquid/lfm-2.5-embedding-350m:free'], KS = [12, 18, 24, 30, 40];
const MS = [18, 30, 40, 60, 0]; // 0 = one intent per training keyword
const assign = (e, cents) => { let b = 0, s = -2; cents.forEach((c, j) => { const x = dot(e, c); if (x > s) { s = x; b = j; } }); return b; };
const acc = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const trainE = await embedN(S.train.map(k => k.kw));
  const sets = { 'Naive Paraphrase': await BL.naiveParaphrase(S.bank), 'Forward Fan-out': await BL.forwardFanout(S.bank), 'Q-SEED (paper, M=18 labelled clusters)': S.finalQueries };
  const totW = S.train.reduce((s, k) => s + k.logvol, 0);
  for (const M of MS) {
    let clusters;
    if (M === 0) clusters = S.train.map((k, i) => ({ centroid: trainE[i], weight: k.logvol }));
    else { const km = sphericalKMeans(trainE, M, { seed: CONFIG.seed, restarts: 3 }); const w = new Array(M).fill(0); S.train.forEach((k, i) => { w[km.assign[i]] += k.logvol; }); clusters = km.centroids.map((c, j) => ({ centroid: c, weight: w[j] })); }
    // keep the coverage/realism balance of the paper: total coverage weight is unchanged, lambda unchanged
    const scale = S.tree.clusters.reduce((s, c) => s + c.weight, 0) / totW;
    clusters = clusters.map(c => ({ ...c, weight: c.weight * scale }));
    const ids = lazyGreedy(S.keptItems, clusters, CONFIG.B, CONFIG.lambda).selectedIds;
    sets['re-selected, ' + (M ? 'M=' + M : 'M=N (per keyword)')] = ids.map(S.qById);
  }
  for (const [n, qs] of Object.entries(sets)) {
    const a = (acc[n] ||= { cov: {}, auc: [], sameAsPaper: [] });
    a.auc.push(mean([4, 5, 6, 7, 8].map(seed => calibrateRealism(qs, S.realPool, { seed }).discAUC)));
    a.sameAsPaper.push(qs.filter(q => S.finalQueries.includes(q)).length);
    for (const enc of ENC) {
      const kwE = await freeEmbed(S.bank.keywords.map(k => k.kw), enc), E = await freeEmbed(qs, enc);
      for (const k of KS) for (const seed of [1, 2, 3]) {
        const km = sphericalKMeans(kwE, k, { seed: 1000 * seed + k, restarts: 2 });
        const w = new Array(k).fill(0); S.bank.keywords.forEach((kw, i) => { w[km.assign[i]] += kw.logvol; }); const tot = w.reduce((x, y) => x + y, 0);
        const hit = new Set(E.map(e => assign(e, km.centroids))); let c = 0; for (const j of hit) c += w[j]; (a.cov[k] ||= []).push(c / tot);
      }
    }
  }
}
const out = Object.fromEntries(Object.entries(acc).map(([n, a]) => [n, { cov: Object.fromEntries(Object.entries(a.cov).map(([k, xs]) => [k, mean(xs)])), auc: mean(a.auc), sharedWithPaperBank: mean(a.sameAsPaper) }]));
save('e9d_granularity.json', out);
console.log('bank | ' + KS.map(k => 'k=' + k).join(' | ') + ' | mean | S4-space AUC | queries shared with paper bank');
for (const [n, o] of Object.entries(out)) console.log(n, '|', KS.map(k => (100 * o.cov[k]).toFixed(1)).join(' | '), '|', (100 * mean(KS.map(k => o.cov[k]))).toFixed(1), '|', o.auc.toFixed(3), '|', o.sharedWithPaperBank.toFixed(0));
