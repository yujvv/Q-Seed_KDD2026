// Rebuttal E6 (Reviewers uXKA, epgg W2, JZBt W1/Q1, EPyX W1/Q1): run the unchanged
// Q-SEED pipeline on REAL keyword banks. Keyphrases and popularity values come
// verbatim from a public Bing keyword inventory (rb0_fetch_real_keywords.mjs);
// nothing in the bank is LLM-generated and no volume is modelled.
// Stages S1-S5, the five baselines and the ablations use the released code and
// the released CONFIG; only the keyword bank changes.
import fs from 'node:fs';
import path from 'node:path';
import { VERTICALS, CONFIG, save, chargeExternal, embedN } from './common.mjs';
import { ROOT } from '../src/config.mjs';
import { VERTICAL_LABEL, SEEDS } from '../src/seeds.mjs';
import { cacheSet } from '../src/cache.mjs';
import { embed, stats } from '../src/llm.mjs';
import { buildKeywordBank, splitHoldout, buildReferenceSets } from '../src/data.mjs';
import { buildIntentTree } from '../src/intent.mjs';
import { generateCandidates } from '../src/generate.mjs';
import { fanoutFilter } from '../src/filter.mjs';
import { calibrateRealism } from '../src/calibrate.mjs';
import { lazyGreedy } from '../src/select.mjs';
import * as BL from '../src/baselines.mjs';
import * as MET from '../src/metrics.mjs';
import { vendiScore } from '../src/linalg.mjs';
import { dot, normalize, rng, shuffle, mean, quantile } from '../src/util.mjs';

const PER_SEED = 50;
export const realName = (v) => v + '_real';

function buildRealBank(v) {
  const rows = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'bing_real', v + '.json'), 'utf8'));
  const bySeed = {};
  rows.forEach((r, i) => (bySeed[r.seed] ||= []).push({ ...r, i }));
  const seen = new Set(), kws = [];
  for (const seed of SEEDS[v]) {
    const fam = (bySeed[seed] || []).sort((a, b) => b.pop - a.pop || a.i - b.i); // most popular first, inventory order on ties
    let n = 0;
    for (const r of fam) {
      if (n >= PER_SEED) break;
      if (seen.has(r.kw) || r.kw.length > 60) continue;
      seen.add(r.kw); n++;
      kws.push({ kw: r.kw, intent: 'I', volume: r.pop, seed });
    }
  }
  // popularity is a 0-10 log-scaled index, so it plays the role of log(1+volume) directly
  kws.forEach((k, i) => { k.id = i; k.logvol = 1 + k.volume; });
  return { vertical: realName(v), label: VERTICAL_LABEL[v], keywords: kws };
}

const gini = (xs) => { const a = xs.slice().sort((x, y) => x - y), n = a.length, s = a.reduce((p, c) => p + c, 0); let g = 0; a.forEach((x, i) => { g += (2 * (i + 1) - n - 1) * x; }); return g / (n * s); };
async function clusterStats(tree, bank, train) {
  const emb = await embedN(train.map(k => k.kw));
  const idOf = Object.fromEntries(train.map((k, i) => [k.id, i]));
  const coh = [];
  for (const cl of tree.clusters) { const c = normalize(cl.centroid); for (const id of cl.memberIds) coh.push(dot(emb[idOf[id]], c)); }
  const sizes = tree.clusters.map(c => c.memberIds.length);
  return { k: tree.k, cohesion: mean(coh), sizeMin: Math.min(...sizes), sizeMax: Math.max(...sizes), weightGini: gini(tree.clusters.map(c => c.weight)), facets: tree.clusters.reduce((h, c) => (h[c.facet] = (h[c.facet] || 0) + 1, h), {}), types: tree.clusters.reduce((h, c) => (h[c.type] = (h[c.type] || 0) + 1, h), {}) };
}
const SEEDS10 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
export async function quality(queries, tree, hold, realPool) {
  const e = await embedN(queries);
  const cals = SEEDS10.map(seed => calibrateRealism(queries, realPool, { seed }));
  return {
    n: queries.length, coverage: MET.clusterCoverage(e, tree.clusters), recall: MET.intentRecall(e, hold.emb, hold.w, tree.clusters),
    vendi: vendiScore(e), mauve: mean(cals.map(c => c.mauve)), auc: mean(cals.map(c => c.discAUC)), aucSeed4: cals[0].discAUC, mauveSeed4: cals[0].mauve,
  };
}

export async function runBank(bank, { withBaselines = true } = {}) {
  const { train, holdout } = splitHoldout(bank);
  const tree = await buildIntentTree(bank, train);
  const hold = { emb: await embed(holdout.map(k => k.kw)), w: holdout.map(k => k.logvol) };
  const gen = await generateCandidates(bank, tree);
  const cq = gen.candidates.map(c => c.q), ce = await embedN(cq);
  const refsets = await buildReferenceSets([bank]);
  const realPool = refsets.globalRealPool;
  const cal = calibrateRealism(cq, realPool);
  const filt = await fanoutFilter(bank, gen, tree);
  const mk = (cands) => cands.map(c => ({ id: c.id, emb: ce[c.id], r: cal.rq[c.id], cid: c.cid }));
  const kept = mk(filt.kept), all = mk(gen.candidates);
  const order = lazyGreedy(kept, tree.clusters, CONFIG.B, CONFIG.lambda).selectedIds;
  const q = (ids) => ids.map(id => gen.candidates[id].q);
  const finalQueries = q(order);
  const methods = {};
  methods['Q-SEED'] = await quality(finalQueries, tree, hold, realPool);
  if (withBaselines) {
    methods['KW-as-Query'] = await quality(BL.kwAsQuery(bank), tree, hold, realPool);
    methods['PAA-Template'] = await quality(BL.paaTemplate(bank), tree, hold, realPool);
    methods['Naive Paraphrase'] = await quality(await BL.naiveParaphrase(bank), tree, hold, realPool);
    methods['Forward Fan-out'] = await quality(await BL.forwardFanout(bank), tree, hold, realPool);
    methods['GEO-bench-like'] = await quality(await BL.geoBenchLike(bank), tree, hold, realPool);
    methods['w/o S3'] = await quality(q(lazyGreedy(all, tree.clusters, CONFIG.B, CONFIG.lambda).selectedIds), tree, hold, realPool);
    methods['w/o S4'] = await quality(q(lazyGreedy(kept, tree.clusters, CONFIG.B, 0).selectedIds), tree, hold, realPool);
  }
  // coverage at tight budgets: greedy vs random (20 draws) vs keyword order
  const kwSorted = bank.keywords.slice().sort((a, b) => b.volume - a.volume).map(k => k.kw), kwEmb = await embedN(kwSorted.slice(0, 50));
  const cov = {};
  for (const n of [10, 15, 20, 25, 50]) {
    const rnd = []; for (let s = 0; s < 20; s++) rnd.push(MET.clusterCoverage(shuffle(kept, rng(CONFIG.seed + 900 + s)).slice(0, n).map(it => it.emb), tree.clusters));
    cov[n] = { greedy: MET.clusterCoverage(order.slice(0, n).map(id => ce[id]), tree.clusters), random: mean(rnd), kw: MET.clusterCoverage(kwEmb.slice(0, n), tree.clusters) };
  }
  const sc = filt.consistencyScores;
  return {
    bank, tree, gen, filt, cal, finalQueries, refBank: refsets.perVertical[bank.vertical],
    summary: {
      vertical: bank.vertical, N: bank.keywords.length, popHist: bank.keywords.reduce((h, k) => (h[k.volume] = (h[k.volume] || 0) + 1, h), {}),
      clusters: await clusterStats(tree, bank, train), nCandidates: cq.length,
      S3: { dropRate: filt.dropRate, keptMeanSim: mean(sc.filter(s => s >= CONFIG.tau)), dropMeanSim: mean(sc.filter(s => s < CONFIG.tau)) },
      pool: { auc: cal.discAUC, mauve: cal.mauve }, methods, cov,
      examples: finalQueries.slice(0, 4), sampleKeywords: shuffle(bank.keywords, rng(7)).slice(0, 14).map(k => `${k.kw} (${k.volume})`),
    },
  };
}

const isMain = process.argv[1] && process.argv[1].endsWith('rb6_realbank.mjs');
if (isMain) {
  const out = {};
  const VERTS = process.env.RB_VERTS ? process.env.RB_VERTS.split(',') : VERTICALS;
  for (const v of VERTS) {
    const real = buildRealBank(v);
    cacheSet('kwbank', real.vertical + ':v2', real);
    const c0 = stats();
    const R = await runBank(real);
    // semi-synthetic bank, re-scored under the same 10-seed protocol for a like-for-like table
    const Ssyn = await runBank(await buildKeywordBank(v));
    out[v] = { real: R.summary, synthetic: Ssyn.summary };
    save('e6_realbank_' + v + '.json', { finalQueries: R.finalQueries, refBank: R.refBank, keywords: real.keywords });
    console.error(v, 'N', real.keywords.length, 'k', R.summary.clusters.k, 'drop', R.summary.S3.dropRate.toFixed(3), 'cost so far $' + stats().cost.toFixed(3));
    save('e6_realbank.json', out);
  }
  chargeExternal('realbank_build', stats().cost, stats().liveCalls);
  const M = ['KW-as-Query', 'PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED', 'w/o S3', 'w/o S4'];
  for (const kind of ['real', 'synthetic']) {
    console.log('\n== ' + kind + ' banks (mean over verticals): method | Cov | Recall | Vendi | MAUVE | AUC(10 seeds)');
    for (const m of M) { const a = (f) => mean(VERTS.map(v => f(out[v][kind].methods[m]))); console.log(m, '|', (100 * a(x => x.coverage)).toFixed(1), '|', (100 * a(x => x.recall)).toFixed(1), '|', a(x => x.vendi).toFixed(1), '|', (100 * a(x => x.mauve)).toFixed(1), '|', a(x => x.auc).toFixed(3)); }
    console.log('cov@budget greedy/random/kw:', [10, 15, 20, 25].map(n => n + ': ' + ['greedy', 'random', 'kw'].map(k => (100 * mean(VERTS.map(v => out[v][kind].cov[n][k]))).toFixed(0)).join('/')).join('  '));
    console.log('clusters:', VERTS.map(v => JSON.stringify({ N: out[v][kind].N, k: out[v][kind].clusters.k, coh: +out[v][kind].clusters.cohesion.toFixed(3), size: out[v][kind].clusters.sizeMin + '-' + out[v][kind].clusters.sizeMax, wGini: +out[v][kind].clusters.weightGini.toFixed(2), drop: +out[v][kind].S3.dropRate.toFixed(3), poolAUC: +out[v][kind].pool.auc.toFixed(3) })).join(' '));
  }
  for (const v of VERTS) { console.log('\n' + v, 'keywords:', out[v].real.sampleKeywords.join(' | ')); console.log(' queries:', out[v].real.examples.join(' || ')); console.log(' pop hist', JSON.stringify(out[v].real.popHist), 'per-vertical AUC', Object.entries(out[v].real.methods).map(([m, x]) => m + '=' + x.auc.toFixed(3)).join(' ')); }
  console.log('live calls', stats().liveCalls, 'cost $' + stats().cost.toFixed(3));
}
