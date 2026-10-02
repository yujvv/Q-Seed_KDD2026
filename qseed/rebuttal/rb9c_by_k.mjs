// Coverage in the independent encoders, broken down by the granularity k of the
// independent clustering (cached embeddings only).
import { loadVertical, VERTICALS, save, freeEmbed } from './common.mjs';
import { sphericalKMeans } from '../src/linalg.mjs';
import * as BL from '../src/baselines.mjs';
import { dot, mean } from '../src/util.mjs';
const ENC = ['nvidia/nemotron-3-embed-1b:free', 'liquid/lfm-2.5-embedding-350m:free'], KS = [8, 12, 18, 24, 30, 40];
const assign = (e, cents) => { let b = 0, s = -2; cents.forEach((c, j) => { const x = dot(e, c); if (x > s) { s = x; b = j; } }); return b; };
const acc = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const banks = { 'KW-as-Query': BL.kwAsQuery(S.bank), 'Naive Paraphrase': await BL.naiveParaphrase(S.bank), 'Forward Fan-out': await BL.forwardFanout(S.bank), 'GEO-bench-like': await BL.geoBenchLike(S.bank), 'Q-SEED': S.finalQueries };
  for (const enc of ENC) {
    const kwE = await freeEmbed(S.bank.keywords.map(k => k.kw), enc);
    const E = {}; for (const [n, qs] of Object.entries(banks)) E[n] = await freeEmbed(qs, enc);
    for (const k of KS) for (const seed of [1, 2, 3]) {
      const km = sphericalKMeans(kwE, k, { seed: 1000 * seed + k, restarts: 2 });
      const w = new Array(k).fill(0); S.bank.keywords.forEach((kw, i) => { w[km.assign[i]] += kw.logvol; }); const tot = w.reduce((a, b) => a + b, 0);
      for (const n of Object.keys(banks)) { const hit = new Set(E[n].map(e => assign(e, km.centroids))); let c = 0; for (const j of hit) c += w[j]; ((acc[n] ||= {})[k] ||= []).push(c / tot); }
    }
  }
}
const out = Object.fromEntries(Object.entries(acc).map(([n, byK]) => [n, Object.fromEntries(Object.entries(byK).map(([k, xs]) => [k, mean(xs)]))]));
save('e9c_coverage_by_k.json', out);
console.log('method | ' + KS.map(k => 'k=' + k).join(' | '));
for (const [n, byK] of Object.entries(out)) console.log(n, '|', KS.map(k => (100 * byK[k]).toFixed(1)).join(' | '));
