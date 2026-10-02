// Rebuttal E9b: the independent-encoder coverage test of rb9, on the REAL keyword
// banks (rb6). Everything except the free encoder calls is served from cache.
import { VERTICALS, CONFIG, save, freeEmbed } from './common.mjs';
import { runBank, realName } from './rb6_realbank.mjs';
import { cacheGet } from '../src/cache.mjs';
import { sphericalKMeans } from '../src/linalg.mjs';
import * as BL from '../src/baselines.mjs';
import { dot, rng, shuffle, mean } from '../src/util.mjs';

const ENCODERS = (process.env.RB_ENC || 'nvidia/nemotron-3-embed-1b:free,liquid/lfm-2.5-embedding-350m:free').split(',');
const VERTS = (process.env.RB_VERTS || 'consumer_electronics,personal_finance').split(',');
const KS = [18, 24, 30, 40];
const assign = (e, cents) => { let b = 0, s = -2; cents.forEach((c, j) => { const x = dot(e, c); if (x > s) { s = x; b = j; } }); return b; };

const out = {};
for (const v of VERTS) {
  const bank = cacheGet('kwbank', realName(v) + ':v2');
  const R = await runBank(bank, { withBaselines: false });
  const kws = bank.keywords;
  const banks = {
    'KW-as-Query': BL.kwAsQuery(bank), 'PAA-Template': BL.paaTemplate(bank), 'Naive Paraphrase': await BL.naiveParaphrase(bank),
    'Forward Fan-out': await BL.forwardFanout(bank), 'GEO-bench-like': await BL.geoBenchLike(bank), 'Q-SEED': R.finalQueries,
  };
  const poolQ = R.filt.kept.map(c => c.q);
  const res = {};
  for (const enc of ENCODERS) {
    const kwE = await freeEmbed(kws.map(k => k.kw), enc);
    const bankE = {}; for (const [n, qs] of Object.entries(banks)) bankE[n] = await freeEmbed(qs, enc);
    const poolE = await freeEmbed(poolQ, enc);
    const r = { atB: Object.fromEntries(Object.keys(banks).map(n => [n, []])), tight: {} };
    for (const k of KS) for (const seed of [1, 2, 3]) {
      const km = sphericalKMeans(kwE, k, { seed: 1000 * seed + k, restarts: 2, iters: 40 });
      const w = new Array(k).fill(0); kws.forEach((kw, i) => { w[km.assign[i]] += kw.logvol; });
      const tot = w.reduce((a, b) => a + b, 0);
      const cov = (E) => { const hit = new Set(E.map(e => assign(e, km.centroids))); let c = 0; for (const j of hit) c += w[j]; return c / tot; };
      for (const n of Object.keys(banks)) r.atB[n].push(cov(bankE[n]));
      for (const n of [10, 15, 20, 25, 50]) {
        const t = (r.tight[n] ||= { greedy: [], random: [] });
        t.greedy.push(cov(bankE['Q-SEED'].slice(0, n)));
        for (let s = 0; s < 10; s++) t.random.push(cov(shuffle(poolE, rng(CONFIG.seed + 50 * s + k)).slice(0, n)));
      }
    }
    res[enc] = { atB: Object.fromEntries(Object.entries(r.atB).map(([n, xs]) => [n, mean(xs)])), tight: Object.fromEntries(Object.entries(r.tight).map(([n, t]) => [n, { greedy: mean(t.greedy), random: mean(t.random) }])) };
    console.error(v, enc, JSON.stringify(res[enc]));
  }
  out[v] = res; save('e9b_coverage_encoders_real.json', out);
}
console.log('REAL banks | method | ' + ENCODERS.join(' | '));
for (const n of ['KW-as-Query', 'PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED']) console.log([n, ...ENCODERS.map(e => (100 * mean(VERTS.map(v => out[v][e].atB[n]))).toFixed(1))].join(' | '));
for (const e of ENCODERS) console.log('tight', e, [10, 15, 20, 25, 50].map(n => 'B=' + n + ': greedy ' + (100 * mean(VERTS.map(v => out[v][e].tight[n].greedy))).toFixed(0) + ' vs random ' + (100 * mean(VERTS.map(v => out[v][e].tight[n].random))).toFixed(0)).join(' ; '));
