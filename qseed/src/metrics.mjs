// Evaluation metrics: coverage / intent-recall, diversity, reliability,
// measurement efficiency, and external validity.
import { cosine, normalize, spearman, kendallTau, rng, shuffle } from './util.mjs';
import { vendiScore } from './linalg.mjs';
import { domainVisibility } from './probe.mjs';
import { CONFIG } from './config.mjs';

// Assign a (normalized) vector to its nearest intent cluster (argmax cosine).
// Assignment-based coverage is modality-invariant: it does not penalize the
// keyword->conversational embedding gap the way an absolute-similarity threshold
// would (raw keywords sit at cluster centroids and would trivially win).
function assignCluster(embN, cents) {
  let bc = 0, bs = -2;
  for (let c = 0; c < cents.length; c++) { const s = cosine(embN, cents[c]); if (s > bs) { bs = s; bc = c; } }
  return bc;
}

// Weighted intent coverage: fraction of intent-cluster weight whose cluster
// receives at least one bank query (by nearest-cluster assignment).
export function clusterCoverage(queryEmbN, clusters) {
  const cents = clusters.map(cl => normalize(cl.centroid));
  const covered = new Set(queryEmbN.map(q => assignCluster(q, cents)));
  let cov = 0, total = 0;
  clusters.forEach((cl, j) => { total += cl.weight; if (covered.has(j)) cov += cl.weight; });
  return total ? cov / total : 0;
}

// Intent-Recall@B on held-out real intents: a held-out keyword is recalled if
// its nearest cluster is covered by the bank. Weighted by log-volume.
export function intentRecall(queryEmbN, holdoutEmb, holdoutWeights, clusters) {
  const cents = clusters.map(cl => normalize(cl.centroid));
  const covered = new Set(queryEmbN.map(q => assignCluster(q, cents)));
  let rec = 0, total = 0;
  for (let i = 0; i < holdoutEmb.length; i++) {
    const c = assignCluster(normalize(holdoutEmb[i]), cents);
    total += holdoutWeights[i];
    if (covered.has(c)) rec += holdoutWeights[i];
  }
  return total ? rec / total : 0;
}

export function diversity(queryEmb) {
  return vendiScore(queryEmb);
}

// Rank domains by pooled visibility over a subset of probe rows.
function rankVector(recSubset, domains, engines) {
  const vis = domainVisibility(recSubset, domains, engines);
  return vis.map(v => v.rate);
}

// Split-half reliability with Spearman-Brown correction.
export function splitHalfReliability(recs, domains, engines, { seed = 31 } = {}) {
  const rand = rng(seed);
  const idx = shuffle(recs.map((_, i) => i), rand);
  const half = Math.floor(idx.length / 2);
  const A = idx.slice(0, half).map(i => recs[i]);
  const B = idx.slice(half, 2 * half).map(i => recs[i]);
  const ra = rankVector(A, domains, engines), rb = rankVector(B, domains, engines);
  const r = spearman(ra, rb);
  const sb = (2 * r) / (1 + r); // Spearman-Brown
  return { half: r, spearmanBrown: Math.max(-1, Math.min(1, sb)) };
}

// Test-retest reliability between two probe rounds (Kendall tau of domain ranks).
export function testRetest(recsR1, recsR2, domains, engines) {
  const r1 = rankVector(recsR1, domains, engines);
  const r2 = rankVector(recsR2, domains, engines);
  return { kendall: kendallTau(r1, r2), spearman: spearman(r1, r2) };
}

// Measurement-efficiency curve: agreement of the domain ranking estimated from
// the first-n queries (given an ordering, from round-1 probes) with an
// INDEPENDENT ground-truth ranking (gtRanks, e.g. the round-2 full-bank
// visibility). Using independent ground truth is what makes this a fair
// efficiency test: random sub-sampling is unbiased for the same-data ranking,
// so only against independent data can discrimination-prioritized ordering show
// a benefit.
export function efficiencyCurve(recs, order, domains, engines, steps, gtRanks) {
  const out = [];
  for (const n of steps) {
    if (n > order.length) break;
    const subset = order.slice(0, n).map(i => recs[i]);
    const rv = rankVector(subset, domains, engines);
    out.push({ n, kendall: kendallTau(rv, gtRanks), spearman: spearman(rv, gtRanks) });
  }
  return out;
}

// External validity: correlate a visibility ranking with an independent signal.
export function externalCorr(rankA, rankB) {
  return { spearman: spearman(rankA, rankB), kendall: kendallTau(rankA, rankB) };
}
