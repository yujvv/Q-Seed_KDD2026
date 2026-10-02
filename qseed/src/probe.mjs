// Generative-engine probing harness + response -> item-response matrix.
import { probe, pool } from './llm.mjs';
import { CONFIG, ENGINES } from './config.mjs';

// Probe a list of queries across engines x repeats. Returns raw records:
//   rec[queryIdx][engineId] = { domainCounts: {domain: hits}, trials }
export async function probeQueries(queries, { engines = ENGINES, repeats = CONFIG.probeRepeats, label = 'probe', concurrency = 20, repOffset = 0 } = {}) {
  const tasks = [];
  queries.forEach((q, qi) => {
    for (const eng of engines) {
      for (let r = 0; r < repeats; r++) tasks.push({ qi, q, eng, r: r + repOffset });
    }
  });
  let done = 0;
  const res = await pool(tasks, async (t) => {
    const out = await probe(t.q, t.eng.model, t.r);
    return { qi: t.qi, engId: t.eng.id, domains: out.domains || [], n: out.nCitations || 0 };
  }, concurrency, (d, tot) => { done = d; if (d % Math.max(1, Math.floor(tot / 10)) === 0) process.stderr.write(`  [${label}] ${d}/${tot}\n`); });

  const recs = queries.map(() => Object.fromEntries(engines.map(e => [e.id, { domainCounts: {}, trials: 0 }])));
  for (const r of res) {
    if (!r || r.__error) continue;
    const cell = recs[r.qi][r.engId];
    cell.trials += 1;
    for (const d of r.domains) cell.domainCounts[d] = (cell.domainCounts[d] || 0) + 1;
  }
  return recs;
}

// Build the domain universe: domains cited for >= minQueries distinct queries
// (across engines/repeats), capped to the top-D most frequent.
export function buildDomainUniverse(recs, engines, { minQueries = 3, maxD = 60 } = {}) {
  const qCount = {};
  for (const rec of recs) {
    const seen = new Set();
    for (const e of engines) for (const d of Object.keys(rec[e.id].domainCounts)) seen.add(d);
    for (const d of seen) qCount[d] = (qCount[d] || 0) + 1;
  }
  return Object.entries(qCount).filter(([, c]) => c >= minQueries)
    .sort((a, b) => b[1] - a[1]).slice(0, maxD).map(([d]) => d);
}

// Build succ/trials item-response matrices per engine for the given domain set.
//   succ[q][d] = # repeats (this engine) where domain d cited for query q
//   trials[q][d] = # repeats for query q on this engine
export function buildIRTMatrices(recs, domains, engines) {
  const out = {};
  const dIndex = Object.fromEntries(domains.map((d, i) => [d, i]));
  for (const e of engines) {
    const Q = recs.length, D = domains.length;
    const succ = Array.from({ length: Q }, () => new Array(D).fill(0));
    const trials = Array.from({ length: Q }, () => new Array(D).fill(0));
    for (let q = 0; q < Q; q++) {
      const cell = recs[q][e.id];
      for (let d = 0; d < D; d++) trials[q][d] = cell.trials;
      for (const [dom, c] of Object.entries(cell.domainCounts)) {
        if (dom in dIndex) succ[q][dIndex[dom]] = c;
      }
    }
    out[e.id] = { succ, trials };
  }
  return out;
}

// Per-domain citation rate over a set of query rows (mean success prob), pooled
// across engines. Used as a simple visibility signal + IRT sanity anchor.
export function domainVisibility(recs, domains, engines) {
  const D = domains.length;
  const dIndex = Object.fromEntries(domains.map((d, i) => [d, i]));
  const num = new Array(D).fill(0), den = new Array(D).fill(0);
  for (const rec of recs) {
    for (const e of engines) {
      const cell = rec[e.id];
      for (let d = 0; d < D; d++) den[d] += cell.trials;
      for (const [dom, c] of Object.entries(cell.domainCounts)) if (dom in dIndex) num[dIndex[dom]] += c;
    }
  }
  return domains.map((d, i) => ({ domain: d, rate: den[i] ? num[i] / den[i] : 0 }));
}
