// Rebuttal E3 (Reviewer 5mFN; uXKA W1): what the cross-engine disagreement is
// made of. Free: decomposes the cached probes of the two submitted engines.
//  - noise ceiling: per-engine test-retest vs cross-engine agreement (disattenuated)
//  - per-query cited-domain Jaccard: same engine across rounds vs across engines
//  - support vs ordering: do the engines cite different domain SETS, or rank a
//    shared set differently?
//  - citations per answer
import { loadVertical, VERTICALS, ENGINES, CONFIG, save } from './common.mjs';
import { domainVisibility } from '../src/probe.mjs';
import { probe } from '../src/llm.mjs';
import { spearman, mean } from '../src/util.mjs';

const vis = (recs, domains, engs) => domainVisibility(recs, domains, engs).map(v => v.rate);
const setOf = (cell) => new Set(Object.keys(cell.domainCounts));
const jac = (A, B) => { if (!A.size && !B.size) return null; let k = 0; for (const x of A) if (B.has(x)) k++; return k / (A.size + B.size - k); };
const mjac = (xs) => mean(xs.filter(x => x !== null));

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v);
  const [A, Bn] = ENGINES, D = S.domains;
  const r = {};
  // per-engine retest (final bank) and cross-engine (same rows)
  const rel = {};
  for (const e of ENGINES) rel[e.id] = spearman(vis(S.recsFinal, D, [e]), vis(S.recsFinalR2, D, [e]));
  const cross1 = spearman(vis(S.recsFinal, D, [A]), vis(S.recsFinal, D, [Bn]));
  const cross2 = spearman(vis(S.recsFinalR2, D, [A]), vis(S.recsFinalR2, D, [Bn]));
  const crossAcross = mean([spearman(vis(S.recsFinal, D, [A]), vis(S.recsFinalR2, D, [Bn])), spearman(vis(S.recsFinalR2, D, [A]), vis(S.recsFinal, D, [Bn]))]);
  r.retestPerEngine = rel;
  r.crossEngine = { round1: cross1, round2: cross2, acrossRounds: crossAcross };
  r.disattenuated = crossAcross / Math.sqrt(rel[A.id] * rel[Bn.id]);
  // Q0-level (as in the paper's Table 4)
  r.crossEngineQ0 = spearman(vis(S.recsQ0, D, [A]), vis(S.recsQ0, D, [Bn]));
  // per-query Jaccard of cited-domain sets
  const within = { [A.id]: [], [Bn.id]: [] }, across = [];
  S.recsQ0.forEach((rec, i) => {
    for (const e of ENGINES) within[e.id].push(jac(setOf(rec[e.id]), setOf(S.recsQ0r2[i][e.id])));
    across.push(jac(setOf(rec[A.id]), setOf(rec[Bn.id])));
  });
  r.perQueryJaccard = { withinA: mjac(within[A.id]), withinB: mjac(within[Bn.id]), across: mjac(across) };
  // support vs ordering over ALL cited domains (not the top-60 universe)
  const tot = { [A.id]: {}, [Bn.id]: {} };
  for (const recs of [S.recsQ0, S.recsQ0r2]) for (const rec of recs) for (const e of ENGINES) for (const [d, c] of Object.entries(rec[e.id].domainCounts)) tot[e.id][d] = (tot[e.id][d] || 0) + c;
  const sA = new Set(Object.keys(tot[A.id])), sB = new Set(Object.keys(tot[Bn.id]));
  const shared = [...sA].filter(d => sB.has(d));
  const mass = (t, set) => { let s = 0, a = 0; for (const [d, c] of Object.entries(t)) { a += c; if (set.has(d)) s += c; } return s / a; };
  r.support = {
    nA: sA.size, nB: sB.size, shared: shared.length, jaccard: jac(sA, sB),
    massOfAOnShared: mass(tot[A.id], new Set(shared)), massOfBOnShared: mass(tot[Bn.id], new Set(shared)),
    rhoOnShared: spearman(shared.map(d => tot[A.id][d]), shared.map(d => tot[Bn.id][d])),
  };
  // within the 60-domain universe: jointly cited only
  const vA = vis(S.recsQ0, D, [A]), vB = vis(S.recsQ0, D, [Bn]);
  const both = D.map((_, i) => i).filter(i => vA[i] > 0 && vB[i] > 0);
  r.universe = { D: D.length, citedA: vA.filter(x => x > 0).length, citedB: vB.filter(x => x > 0).length, both: both.length, rhoBoth: spearman(both.map(i => vA[i]), both.map(i => vB[i])) };
  // citations per answer (distinct domains per probe), from the probe cache
  const per = { [A.id]: [], [Bn.id]: [] }, empty = { [A.id]: 0, [Bn.id]: 0 };
  for (const q of S.Q0queries) for (const e of ENGINES) for (const rep of [0, 1, 2, 100, 101, 102]) {
    const p = await probe(q, e.model, rep); per[e.id].push(p.domains.length); if (!p.domains.length) empty[e.id]++;
  }
  r.domainsPerAnswer = Object.fromEntries(ENGINES.map(e => [e.id, { mean: mean(per[e.id]), max: Math.max(...per[e.id]), emptyFrac: empty[e.id] / per[e.id].length, hist: per[e.id].reduce((h, n) => (h[n] = (h[n] || 0) + 1, h), {}) }]));
  // agreement by intent facet of the query's source cluster
  const byFacet = {};
  S.Q0ids.forEach((id, i) => { const cl = S.tree.clusters.find(c => c.cid === S.gen.candidates[id].cid); const j = across[i]; if (j !== null) (byFacet[cl.facet] ||= []).push(j); });
  r.acrossJaccardByFacet = Object.fromEntries(Object.entries(byFacet).map(([k, xs]) => [k, { n: xs.length, mean: mean(xs) }]));
  out[v] = r;
  console.error(v, JSON.stringify(r, null, 0));
}
save('e3_engines_decomp.json', out);
const avg = (f) => mean(VERTICALS.map(v => f(out[v])));
console.log('retest per engine (rho):', ENGINES.map(e => e.id + '=' + avg(x => x.retestPerEngine[e.id]).toFixed(2)).join(' '));
console.log('cross-engine rho (final bank, across rounds):', avg(x => x.crossEngine.acrossRounds).toFixed(2), ' disattenuated:', avg(x => x.disattenuated).toFixed(2));
console.log('per-query Jaccard within A/B/across:', avg(x => x.perQueryJaccard.withinA).toFixed(2), avg(x => x.perQueryJaccard.withinB).toFixed(2), avg(x => x.perQueryJaccard.across).toFixed(2));
console.log('support Jaccard:', avg(x => x.support.jaccard).toFixed(2), 'mass on shared A/B:', avg(x => x.support.massOfAOnShared).toFixed(2), avg(x => x.support.massOfBOnShared).toFixed(2), 'rho on shared:', avg(x => x.support.rhoOnShared).toFixed(2));
console.log('domains/answer:', ENGINES.map(e => e.id + '=' + avg(x => x.domainsPerAnswer[e.id].mean).toFixed(2)).join(' '));
