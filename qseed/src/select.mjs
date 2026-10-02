// S5 - Submodular distillation + IRT discrimination reweighting.
//   F(Q) = sum_j w_j * max_{q in Q} sim(q, I_j)   [facility location, submodular]
//        + lambda * sum_{q in Q} r(q)             [realism, modular]
// Lazy greedy on F gives a (1-1/e) guarantee. Then probe Q0=2B, fit 2PL-IRT,
// distil to B by keeping high-discrimination items spread across difficulty.
import { cosine, normalize } from './util.mjs';
import { CONFIG } from './config.mjs';

// Lazy (accelerated) greedy for facility-location + modular realism.
// items: [{id, emb(normalized), r}], clusters: [{centroid(normalized), weight}]
export function lazyGreedy(items, clusters, B, lambda) {
  const M = clusters.length, n = items.length;
  const cw = clusters.map(c => c.weight);
  const cCent = clusters.map(c => normalize(c.centroid));
  // precompute sim(item, cluster)
  const sim = items.map(it => cCent.map(c => Math.max(0, cosine(it.emb, c))));
  const best = new Array(M).fill(0); // current max coverage per cluster
  const selected = [];
  const inSel = new Array(n).fill(false);

  // marginal gain given current 'best'
  const gain = (i) => {
    let g = 0; const s = sim[i];
    for (let j = 0; j < M; j++) { const d = s[j] - best[j]; if (d > 0) g += cw[j] * d; }
    return g + lambda * items[i].r;
  };

  // priority queue via array of [upperBound, itemIdx, lastUpdatedIter]
  let heap = items.map((_, i) => [gain(i), i, 0]);
  heap.sort((a, b) => b[0] - a[0]);

  let iter = 0;
  const objTrace = [];
  while (selected.length < B && heap.length) {
    // lazy: recompute top until its bound is fresh and still max
    let top = heap[0];
    while (top[2] !== iter) {
      const g = gain(top[1]);
      top[0] = g; top[2] = iter;
      // reinsert by simple down-sift: resort small region (n small enough)
      heap.sort((a, b) => b[0] - a[0]);
      top = heap[0];
    }
    heap.shift();
    const i = top[1];
    if (inSel[i]) continue;
    inSel[i] = true; selected.push(i);
    const s = sim[i];
    for (let j = 0; j < M; j++) if (s[j] > best[j]) best[j] = s[j];
    iter++;
    let obj = lambda * selected.reduce((a, k) => a + items[k].r, 0);
    for (let j = 0; j < M; j++) obj += cw[j] * best[j];
    objTrace.push(obj);
  }
  return { selectedIds: selected.map(i => items[i].id), selectedIdx: selected, objTrace };
}

// Random-selection baseline (for the "greedy vs random" ablation).
export function randomSelect(items, B, rand) {
  const idx = items.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const sel = idx.slice(0, B);
  return { selectedIds: sel.map(i => items[i].id), selectedIdx: sel };
}

// IRT distillation: from Q0 (probed) choose B items maximizing discrimination
// while keeping difficulty spread. Removes near-zero-a (uninformative) items,
// then greedily fills difficulty strata by descending a_q.
export function irtDistill(q0Items, irt, B, nStrata = 6) {
  // q0Items: [{id, cid, ...}] aligned with irt.a / irt.b arrays (same order)
  const items = q0Items.map((it, i) => ({ ...it, a: irt.a[i], b: irt.b[i] }));
  const aThresh = 0.15; // drop uninformative
  const informative = items.filter(it => it.a > aThresh);
  const pool = informative.length >= B ? informative : items.slice();
  // difficulty strata by quantiles of b
  const bs = pool.map(it => it.b).sort((x, y) => x - y);
  const edges = [];
  for (let s = 1; s < nStrata; s++) edges.push(bs[Math.floor((s / nStrata) * (bs.length - 1))]);
  const strataOf = (b) => { let s = 0; while (s < edges.length && b > edges[s]) s++; return s; };
  const buckets = Array.from({ length: nStrata }, () => []);
  for (const it of pool) buckets[strataOf(it.b)].push(it);
  for (const bk of buckets) bk.sort((x, y) => y.a - x.a); // best discrimination first
  const chosen = [];
  let round = 0;
  while (chosen.length < B && chosen.length < pool.length) {
    let added = false;
    for (let s = 0; s < nStrata && chosen.length < B; s++) {
      if (buckets[s][round]) { chosen.push(buckets[s][round]); added = true; }
    }
    round++;
    if (!added) break;
  }
  return { selectedIds: chosen.map(it => it.id), items: chosen };
}
