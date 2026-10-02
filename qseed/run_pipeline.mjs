// Q-SEED end-to-end driver. Runs S1-S5 + baselines + ablations + probing +
// IRT + all metrics for every vertical, and writes results/<vertical>.json.
import { CONFIG, VERTICALS, ENGINES } from './src/config.mjs';
import { embed, stats } from './src/llm.mjs';
import { normalize, rng, shuffle, mean } from './src/util.mjs';
import { buildKeywordBank, splitHoldout, buildReferenceSets } from './src/data.mjs';
import { buildIntentTree } from './src/intent.mjs';
import { generateCandidates } from './src/generate.mjs';
import { fanoutFilter } from './src/filter.mjs';
import { calibrateRealism } from './src/calibrate.mjs';
import { lazyGreedy, randomSelect, irtDistill } from './src/select.mjs';
import { fit2PL } from './src/irt.mjs';
import { probeQueries, buildDomainUniverse, buildIRTMatrices, domainVisibility } from './src/probe.mjs';
import * as BL from './src/baselines.mjs';
import * as MET from './src/metrics.mjs';
import { saveJSON } from './src/cache.mjs';

const log = (...a) => console.error('[qseed]', ...a);
const sigmoidP = (a, b, t) => { const z = a * (t - b); const p = z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); return p * (1 - p); };

async function embedN(texts) { return (await embed(texts)).map(normalize); }

// Quality metrics for a set of query strings.
async function qualityMetrics(queries, tree, holdout, realPool) {
  const qEmbN = await embedN(queries);
  const cov = MET.clusterCoverage(qEmbN, tree.clusters);
  const rec = MET.intentRecall(qEmbN, holdout.emb, holdout.weights, tree.clusters);
  const div = MET.diversity(await embed(queries));
  // realism: MAUVE + discriminator AUC on this specific set
  const cal = await calibrateRealism(queries, realPool);
  return { n: queries.length, coverage: cov, intentRecall: rec, vendi: div, mauve: cal.mauve, discAUC: cal.discAUC };
}

async function runVertical(vertical) {
  log('=== vertical:', vertical, '===');
  const bank = await buildKeywordBank(vertical);
  log('keyword bank size', bank.keywords.length);
  const { train, holdout } = splitHoldout(bank);

  // S1
  const tree = await buildIntentTree(bank, train);
  log('S1 clusters', tree.k);

  // holdout embeddings + weights (for Intent-Recall)
  const holdEmb = await embed(holdout.map(k => k.kw));
  const holdout2 = { emb: holdEmb, weights: holdout.map(k => k.logvol) };

  // S2
  const gen = await generateCandidates(bank, tree);
  log('S2 candidates', gen.candidates.length);
  const candQueries = gen.candidates.map(c => c.q);
  const candEmbN = await embedN(candQueries);

  // reference real corpus for this vertical (built once globally, passed in via cache)
  const refsets = await buildReferenceSets(BANKS);
  const realPool = refsets.globalRealPool;
  const refBank = refsets.perVertical[vertical];

  // S4 realism (score all candidates once)
  const cal = await calibrateRealism(candQueries, realPool);
  log('S4 candidate-pool discAUC', cal.discAUC.toFixed(3), 'mauve', cal.mauve.toFixed(3));
  const rqById = {};
  gen.candidates.forEach((c, i) => { rqById[c.id] = cal.rq[i]; });

  // S3 reverse fan-out filter
  const filt = await fanoutFilter(bank, gen, tree);
  log('S3 kept', filt.nKept, '/', filt.nIn, 'dropRate', filt.dropRate.toFixed(3));
  const keptIds = new Set(filt.kept.map(c => c.id));

  // Build selection item sets
  const embById = {}; gen.candidates.forEach((c, i) => { embById[c.id] = candEmbN[i]; });
  const qById = {}; gen.candidates.forEach(c => { qById[c.id] = c.q; });
  const cidById = {}; gen.candidates.forEach(c => { cidById[c.id] = c.cid; });
  const mkItems = (cands) => cands.map(c => ({ id: c.id, emb: embById[c.id], r: rqById[c.id], cid: c.cid }));

  const keptItems = mkItems(filt.kept);
  const allItems = mkItems(gen.candidates);
  const Q0size = CONFIG.Q0factor * CONFIG.B;

  // ---- S5 greedy: main pipeline selects Q0 (2B) from kept + realism ----
  const g0 = lazyGreedy(keptItems, tree.clusters, Q0size, CONFIG.lambda);
  const Q0ids = g0.selectedIds;
  const Q0queries = Q0ids.map(id => qById[id]);
  log('S5 greedy Q0', Q0queries.length);

  // ---------- PROBING ----------
  log('probing Q0 (round 1) ...');
  const recsQ0 = await probeQueries(Q0queries, { label: vertical + '/Q0', repeats: CONFIG.probeRepeats });
  const domains = buildDomainUniverse(recsQ0, ENGINES, { minQueries: 3, maxD: 60 });
  log('domain universe |D| =', domains.length);

  // IRT per engine + pooled (concatenate engine trials into one matrix)
  const mats = buildIRTMatrices(recsQ0, domains, ENGINES);
  const perEngineIRT = {};
  for (const e of ENGINES) perEngineIRT[e.id] = fit2PL(mats[e.id].succ, mats[e.id].trials);
  // pooled matrix: sum successes and trials across engines
  const Q = recsQ0.length, D = domains.length;
  const succP = Array.from({ length: Q }, () => new Array(D).fill(0));
  const trialP = Array.from({ length: Q }, () => new Array(D).fill(0));
  for (const e of ENGINES) for (let q = 0; q < Q; q++) for (let d = 0; d < D; d++) { succP[q][d] += mats[e.id].succ[q][d]; trialP[q][d] += mats[e.id].trials[q][d]; }
  const irtPooled = fit2PL(succP, trialP);
  log('IRT pooled: mean a', mean(irtPooled.a).toFixed(3), 'a>0.15 frac', (irtPooled.a.filter(x => x > 0.15).length / Q).toFixed(2));

  // Q0 item list aligned with IRT arrays
  const q0Items = Q0ids.map(id => ({ id, cid: cidById[id], q: qById[id] }));

  // ---- Final Q-SEED bank = coverage-optimal greedy first-B (submodular).
  // (IRT is used below as a measurement-analysis lens, not a selection step:
  // its discrimination-based distillation trades coverage without an efficiency
  // gain -- reported as the +IRT-distill ablation.)
  const finalIds = Q0ids.slice(0, CONFIG.B);
  const finalQueries = finalIds.map(id => qById[id]);
  log('final Q (greedy B)', finalQueries.length);
  const distill = irtDistill(q0Items, irtPooled, CONFIG.B);
  const distillQueries = distill.selectedIds.map(id => qById[id]);

  // retest round 2 (fresh calls -> different rep indices) on Q0
  log('probing Q0 (round 2 / retest) ...');
  const recsQ0r2 = await probeQueries(Q0queries, { label: vertical + '/Q0r2', repeats: CONFIG.probeRepeats, repOffset: 100 });

  // reference bank probe (independent real queries) for external validity
  log('probing reference bank ...');
  const recsRef = await probeQueries(refBank, { label: vertical + '/ref', repeats: CONFIG.refBankRepeats });

  // ---------- ABLATION QUERY SETS ----------
  const rand = rng(CONFIG.seed + 77);
  // w/o S3: greedy to Q0 on ALL candidates, distill via IRT ordering restricted to overlap not possible w/o probing;
  //         evaluate quality on greedy-to-B of all candidates.
  const abl_woS3_B = lazyGreedy(allItems, tree.clusters, CONFIG.B, CONFIG.lambda).selectedIds.map(id => qById[id]);
  // w/o S4: greedy to B on kept, lambda=0
  const abl_woS4_B = lazyGreedy(keptItems, tree.clusters, CONFIG.B, 0).selectedIds.map(id => qById[id]);
  // random selection from kept
  const abl_rand_B = randomSelect(keptItems, CONFIG.B, rand).selectedIds.map(id => qById[id]);
  // +IRT-distill: greedy Q0 then IRT discrimination distillation to B
  const abl_irtDistill_B = distillQueries;
  // full method quality uses finalQueries (greedy first-B)

  // w/o S3 also needs its OWN probe for external validity (H3). Probe its B set.
  log('probing w/o-S3 bank (H3) ...');
  const recsWoS3 = await probeQueries(abl_woS3_B, { label: vertical + '/woS3', repeats: CONFIG.refBankRepeats });

  // ---------- METRICS ----------
  log('computing quality metrics ...');
  const methods = {};
  methods['Q-SEED'] = await qualityMetrics(finalQueries, tree, holdout2, realPool);
  methods['KW-as-Query'] = await qualityMetrics(BL.kwAsQuery(bank), tree, holdout2, realPool);
  methods['NaiveParaphrase'] = await qualityMetrics(await BL.naiveParaphrase(bank), tree, holdout2, realPool);
  methods['PAA-Template'] = await qualityMetrics(BL.paaTemplate(bank), tree, holdout2, realPool);
  methods['ForwardFanout'] = await qualityMetrics(await BL.forwardFanout(bank), tree, holdout2, realPool);
  methods['GEO-bench-like'] = await qualityMetrics(await BL.geoBenchLike(bank), tree, holdout2, realPool);
  methods['Q-SEED wo/S3'] = await qualityMetrics(abl_woS3_B, tree, holdout2, realPool);
  methods['Q-SEED wo/S4'] = await qualityMetrics(abl_woS4_B, tree, holdout2, realPool);
  methods['Q-SEED +IRTdistill'] = await qualityMetrics(abl_irtDistill_B, tree, holdout2, realPool);
  methods['Q-SEED random-sel'] = await qualityMetrics(abl_rand_B, tree, holdout2, realPool);

  // Restrict the probed rows to the FINAL B-query deliverable bank (a subset of
  // Q0, already probed in both rounds) so reliability/external validity describe
  // the actual bank and the H3 comparison is size-matched with w/o-S3 (both B).
  const q0Index = Object.fromEntries(Q0ids.map((id, i) => [id, i]));
  const recsFinal = finalIds.map(id => recsQ0[q0Index[id]]);
  const recsFinalR2 = finalIds.map(id => recsQ0r2[q0Index[id]]);

  // reliability (split-half + test-retest) on the final B bank
  const splitHalf = MET.splitHalfReliability(recsFinal, domains, ENGINES);
  const retest = MET.testRetest(recsFinal, recsFinalR2, domains, ENGINES);
  // Per-domain round-2 visibility (round 1 is `visQ0`, computed below). Together
  // they let emit_results.mjs bootstrap the reliability/validity coefficients
  // over DOMAINS (n=|D|), the unit that actually carries sampling noise -- with
  // only 3 verticals a vertical-level bootstrap is uninformative.
  const visR2 = domainVisibility(recsFinalR2, domains, ENGINES).map(v => v.rate);

  // efficiency curve: discrimination-prioritized order vs random order, scored
  // against the INDEPENDENT round-2 full-bank ranking (fair efficiency test).
  const gtRanks = domainVisibility(recsQ0r2, domains, ENGINES).map(v => v.rate);
  const irtOrder = Q0ids.map((_, i) => i).sort((x, y) => irtPooled.a[y] - irtPooled.a[x]); // descending a_q
  const steps = [10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100].filter(n => n <= Q0queries.length);
  const effIRT = MET.efficiencyCurve(recsQ0, irtOrder, domains, ENGINES, steps, gtRanks);
  // random ordering averaged over many seeds (single draw is high variance)
  const randCurves = [];
  for (let s = 0; s < 25; s++) {
    const ro = shuffle(q0Items.map((_, i) => i), rng(CONFIG.seed + 200 + s));
    randCurves.push(MET.efficiencyCurve(recsQ0, ro, domains, ENGINES, steps, gtRanks));
  }
  const effRand = steps.map((n, i) => ({ n, kendall: mean(randCurves.map(c => c[i].kendall)), spearman: mean(randCurves.map(c => c[i].spearman)) }));

  // external validity: final-bank domain visibility vs reference-bank visibility
  // (size-matched B-vs-B against the w/o-S3 bank for H3).
  const visQ0 = domainVisibility(recsFinal, domains, ENGINES).map(v => v.rate);
  const visRef = domainVisibility(recsRef, domains, ENGINES).map(v => v.rate);
  const visWoS3 = domainVisibility(recsWoS3, domains, ENGINES).map(v => v.rate);
  const extMain = MET.externalCorr(visQ0, visRef);
  const extWoS3 = MET.externalCorr(visWoS3, visRef);

  // cross-engine agreement: how similarly do the two engines cite domains?
  const perEngVis = {};
  for (const e of ENGINES) perEngVis[e.id] = domainVisibility(recsQ0, domains, [e]).map(v => v.rate);
  const crossVis = MET.externalCorr(perEngVis[ENGINES[0].id], perEngVis[ENGINES[1].id]);
  const engCited = Object.fromEntries(ENGINES.map(e => [e.id, perEngVis[e.id].filter(x => x > 0).length]));
  // cross-engine IRT parameter correlation (query difficulty / discrimination)
  const be = ENGINES.map(e => perEngineIRT[e.id]);
  const crossB = MET.externalCorr(be[0].b, be[1].b);
  const crossA = MET.externalCorr(be[0].a, be[1].a);

  // coverage-vs-budget curves (selection quality, RQ1): greedy vs random vs KW
  const covSteps = [5, 10, 15, 20, 25, 30, 40, 50];
  const greedyOrderIds = g0.selectedIds; // greedy incremental order
  const kwSorted = bank.keywords.slice().sort((a, b) => b.volume - a.volume);
  const randOrderIds = shuffle(keptItems.map(it => it.id), rng(CONFIG.seed + 55));
  const covCurve = async (idsOrKw, isKw) => {
    const out = [];
    for (const n of covSteps) {
      const subset = isKw ? kwSorted.slice(0, n).map(k => k.kw) : idsOrKw.slice(0, n).map(id => qById[id]);
      const em = await embedN(subset);
      out.push(MET.clusterCoverage(em, tree.clusters));
    }
    return out;
  };
  const coverageBudget = {
    steps: covSteps,
    greedy: await covCurve(greedyOrderIds, false),
    random: await covCurve(randOrderIds, false),
    kw: await covCurve(null, true),
  };

  // ---- lambda sensitivity: the coverage/realism trade-off in F(Q) ----
  // Re-select the size-B bank at several lambda and re-measure quality. This is
  // free: selection is local, and every candidate query is already embedded and
  // cached, so no engine or API call is issued. It answers the obvious reviewer
  // question -- why lambda=0.35? -- with the actual trade-off curve rather than
  // an assertion. (lambda=0 is exactly the w/o-S4 ablation.)
  log('lambda sweep ...');
  const lambdaGrid = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0];
  const lambdaSweep = [];
  for (const lam of lambdaGrid) {
    const ids = lazyGreedy(keptItems, tree.clusters, CONFIG.B, lam).selectedIds;
    const qs = ids.map(id => qById[id]);
    const m = await qualityMetrics(qs, tree, holdout2, realPool);
    // mean realism of the selected bank, i.e. the modular term greedy optimizes
    const meanR = mean(ids.map(id => rqById[id]));
    lambdaSweep.push({
      lambda: lam, coverage: m.coverage, intentRecall: m.intentRecall,
      mauve: m.mauve, discAUC: m.discAUC, vendi: m.vendi, meanR,
    });
  }

  // IRT test-information curves: final Q vs random B subset of Q0, over theta grid
  const thetaGrid = []; for (let t = -3; t <= 3.0001; t += 0.25) thetaGrid.push(+t.toFixed(3));
  const finalIdxInQ0 = finalIds.map(id => Q0ids.indexOf(id)).filter(i => i >= 0);
  const randBIdx = shuffle(q0Items.map((_, i) => i), rng(CONFIG.seed + 63)).slice(0, CONFIG.B);
  const infoItems = (idxs) => idxs.map(i => ({ a: irtPooled.a[i], b: irtPooled.b[i] }));
  const infoFinal = thetaGrid.map(t => infoItems(finalIdxInQ0).reduce((s, it) => s + it.a * it.a * sigmoidP(it.a, it.b, t), 0));
  const infoRand = thetaGrid.map(t => infoItems(randBIdx).reduce((s, it) => s + it.a * it.a * sigmoidP(it.a, it.b, t), 0));

  // theta (domain ability) table
  const domainTable = domains.map((d, i) => ({ domain: d, theta: irtPooled.theta[i], visibility: visQ0[i] }))
    .sort((a, b) => b.theta - a.theta);

  const nProbeCalls = ENGINES.length * (
    CONFIG.probeRepeats * (Q0queries.length + Q0queries.length) +
    CONFIG.refBankRepeats * (refBank.length + abl_woS3_B.length));

  const result = {
    vertical,
    probeRepeats: CONFIG.probeRepeats,
    nProbeCalls,
    bankSize: bank.keywords.length,
    nClusters: tree.k,
    nCandidates: gen.candidates.length,
    S3: {
      kept: filt.nKept, in: filt.nIn, dropRate: filt.dropRate, tau: CONFIG.tau,
      keptMeanSim: mean(filt.consistencyScores.filter(s => s >= CONFIG.tau)),
      dropMeanSim: (() => { const d = filt.consistencyScores.filter(s => s < CONFIG.tau); return d.length ? mean(d) : 0; })(),
    },
    S4: { candPoolDiscAUC: cal.discAUC, candPoolMauve: cal.mauve },
    domains, nDomains: domains.length,
    methods,
    irt: {
      pooled: irtPooled,
      meanA: mean(irtPooled.a),
      informativeFrac: irtPooled.a.filter(x => x > 0.15).length / Q,
      perEngine: Object.fromEntries(ENGINES.map(e => [e.id, perEngineIRT[e.id]])),
    },
    reliability: { splitHalf, retest },
    // Aligned per-domain rate vectors (index-matched to `domains`) that underlie
    // the reliability / convergent-validity / cross-engine coefficients. Stored
    // so every correlation in the paper can be re-derived and bootstrapped
    // without re-probing the engines.
    visVectors: { r1: visQ0, r2: visR2, ref: visRef, woS3: visWoS3 },
    efficiency: { steps, irt: effIRT, random: effRand },
    lambdaSweep,
    coverageBudget,
    testInfo: { thetaGrid, final: infoFinal, random: infoRand },
    external: { main: extMain, woS3: extWoS3 },
    crossEngine: { visibility: crossVis, b: crossB, a: crossA, engCited, perEngVis, engIds: ENGINES.map(e => e.id) },
    domainTable,
    consistencyScores: filt.consistencyScores,
    finalQueries, Q0queries, refBank,
    apiStats: stats(),
  };
  saveJSON(vertical + '.json', result);
  log('saved', vertical, 'apiCost so far $' + stats().cost.toFixed(3));
  return result;
}

let BANKS = [];
async function main() {
  const only = process.argv[2];
  const verts = only ? [only] : VERTICALS;
  // build all banks first (needed for reference sets)
  BANKS = [];
  for (const v of VERTICALS) BANKS.push(await buildKeywordBank(v));
  await buildReferenceSets(BANKS);
  const all = {};
  for (const v of verts) all[v] = await runVertical(v);
  saveJSON('_index.json', { verticals: verts, config: CONFIG, engines: ENGINES, apiStats: stats() });
  log('ALL DONE. total API cost $' + stats().cost.toFixed(3), 'live calls', stats().liveCalls);
}
main().catch(e => { console.error(e); process.exit(1); });
