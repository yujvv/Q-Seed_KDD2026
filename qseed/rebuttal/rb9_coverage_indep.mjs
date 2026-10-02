// Rebuttal E9 (Reviewer EPyX W3/Q3): intent coverage measured in representations
// the method never optimised.
//  A. Independent encoders (different model families from text-embedding-3-small):
//     re-cluster the FULL keyword bank in that space, at several k the method
//     never used, and score every bank by nearest-cluster assignment.
//  B. (RB_TAX=1) A curated taxonomy with no embedding at all: the 40 human-chosen
//     head terms of each vertical. An LLM that took no part in the pipeline
//     assigns each query to the head term(s) it is about; coverage =
//     volume-weighted share of head-term families reached.
// Both parts use free-tier models only.
import { loadVertical, VERTICALS, CONFIG, save, freeChat, freeEmbed } from './common.mjs';
import { sphericalKMeans } from '../src/linalg.mjs';
import * as BL from '../src/baselines.mjs';
import { SEEDS } from '../src/seeds.mjs';
import { dot, rng, shuffle, mean } from '../src/util.mjs';

const ENCODERS = (process.env.RB_ENC || 'nvidia/nemotron-3-embed-1b:free,liquid/lfm-2.5-embedding-350m:free').split(',');
const KS = [12, 18, 24, 30];
const JUDGE = process.env.RB_TAXJUDGE || 'qwen/qwen3.8-27b:free';
const DO_TAX = process.env.RB_TAX === '1';
const BATCH = 25;
const assign = (e, cents) => { let b = 0, s = -2; cents.forEach((c, j) => { const x = dot(e, c); if (x > s) { s = x; b = j; } }); return b; };

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const kws = S.bank.keywords;
  const banks = {
    'KW-as-Query': BL.kwAsQuery(S.bank), 'PAA-Template': BL.paaTemplate(S.bank), 'Naive Paraphrase': await BL.naiveParaphrase(S.bank),
    'Forward Fan-out': await BL.forwardFanout(S.bank), 'GEO-bench-like': await BL.geoBenchLike(S.bank), 'Q-SEED': S.finalQueries,
  };
  const poolQ = S.filt.kept.map(c => c.q);
  const res = { encoders: {}, taxonomy: {} };
  // ---- A: independent encoders ----
  for (const enc of ENCODERS) {
    const kwE = await freeEmbed(kws.map(k => k.kw), enc);
    const bankE = {}; for (const [n, qs] of Object.entries(banks)) bankE[n] = await freeEmbed(qs, enc);
    const poolE = await freeEmbed(poolQ, enc);
    const r = { atB: Object.fromEntries(Object.keys(banks).map(n => [n, []])), tight: {} };
    for (const k of KS) for (const seed of [1, 2, 3]) {
      const km = sphericalKMeans(kwE, k, { seed: 1000 * seed + k, restarts: 2 });
      const w = new Array(k).fill(0); kws.forEach((kw, i) => { w[km.assign[i]] += kw.logvol; });
      const tot = w.reduce((a, b) => a + b, 0);
      const cov = (E) => { const hit = new Set(E.map(e => assign(e, km.centroids))); let c = 0; for (const j of hit) c += w[j]; return c / tot; };
      for (const n of Object.keys(banks)) r.atB[n].push(cov(bankE[n]));
      for (const n of [10, 15, 20, 25]) {
        const t = (r.tight[n] ||= { greedy: [], random: [] });
        t.greedy.push(cov(bankE['Q-SEED'].slice(0, n)));
        for (let s = 0; s < 10; s++) t.random.push(cov(shuffle(poolE, rng(CONFIG.seed + 50 * s + k)).slice(0, n)));
      }
    }
    res.encoders[enc] = { atB: Object.fromEntries(Object.entries(r.atB).map(([n, xs]) => [n, mean(xs)])), tight: Object.fromEntries(Object.entries(r.tight).map(([n, t]) => [n, { greedy: mean(t.greedy), random: mean(t.random) }])) };
    console.error(v, enc, JSON.stringify(res.encoders[enc]));
  }
  // ---- B: curated head-term taxonomy, LLM-assigned in batches ----
  if (DO_TAX) {
    const heads = SEEDS[v];
    const wFam = heads.map(h => kws.filter(k => k.seed === h).reduce((s, k) => s + k.logvol, 0));
    const totFam = wFam.reduce((a, b) => a + b, 0);
    const list = heads.map((h, i) => (i + 1) + '. ' + h).join('\n');
    const prompt = (qs) => 'Here is a numbered list of ' + heads.length + ' topics:\n' + list +
      '\n\nFor each query below, say which topic(s) from the list it is about: at most two topic numbers, or 0 if it is about none of them.\n' +
      qs.map((q, i) => 'Q' + (i + 1) + ': ' + q).join('\n') +
      '\n\nReply with ONLY a JSON array with one entry per query, in order; each entry is an array of topic numbers, e.g. [[3],[0],[12,7]].';
    for (const [n, qs] of Object.entries(banks)) {
      const ans = [];
      for (let i = 0; i < qs.length; i += BATCH) {
        const chunk = qs.slice(i, i + BATCH); let arr = null;
        try { const raw = await freeChat(prompt(chunk), { model: JUDGE, maxTokens: 3000 }); const m = raw.match(/\[[\s\S]*\]/); arr = m ? JSON.parse(m[0]) : null; } catch (e) { console.error('tax fail', String(e).slice(0, 120)); }
        for (let k = 0; k < chunk.length; k++) ans.push(Array.isArray(arr) && arr.length === chunk.length ? [].concat(arr[k]).map(Number).filter(x => x >= 1 && x <= heads.length).slice(0, 2) : null);
      }
      const ok = ans.filter(a => a !== null), hit = new Set(); ok.forEach(a => a.forEach(i => hit.add(i - 1)));
      res.taxonomy[n] = { judged: ok.length, weighted: [...hit].reduce((s, i) => s + wFam[i], 0) / totFam, families: hit.size, offTaxonomy: ok.filter(a => !a.length).length / (ok.length || 1) };
      console.error(v, 'taxonomy', n, JSON.stringify(res.taxonomy[n]));
    }
  }
  out[v] = res;
  save(DO_TAX ? 'e9_coverage_indep.json' : 'e9_coverage_encoders.json', out);
}
const names = ['KW-as-Query', 'PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED'];
console.log('method | ' + ENCODERS.join(' | ') + (DO_TAX ? ' | head-term taxonomy (weighted) | families / 40 | off-taxonomy | judged' : ''));
for (const n of names) console.log([n, ...ENCODERS.map(e => (100 * mean(VERTICALS.map(v => out[v].encoders[e].atB[n]))).toFixed(1)), ...(DO_TAX ? [(100 * mean(VERTICALS.map(v => out[v].taxonomy[n].weighted))).toFixed(1), mean(VERTICALS.map(v => out[v].taxonomy[n].families)).toFixed(1), (100 * mean(VERTICALS.map(v => out[v].taxonomy[n].offTaxonomy))).toFixed(0) + '%', mean(VERTICALS.map(v => out[v].taxonomy[n].judged)).toFixed(0)] : [])].join(' | '));
for (const e of ENCODERS) console.log('tight budgets', e, [10, 15, 20, 25].map(n => 'B=' + n + ': greedy ' + (100 * mean(VERTICALS.map(v => out[v].encoders[e].tight[n].greedy))).toFixed(0) + ' vs random ' + (100 * mean(VERTICALS.map(v => out[v].encoders[e].tight[n].random))).toFixed(0)).join(' ; '));
