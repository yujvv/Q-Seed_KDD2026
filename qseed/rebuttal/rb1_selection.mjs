// Rebuttal E1 (Reviewer EPyX W4/Q4): stronger selection baselines over the SAME
// post-S3 candidate pool. Free: uses cached embeddings, no API call.
import { loadVertical, VERTICALS, CONFIG, save } from './common.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { calibrateRealism } from '../src/calibrate.mjs';
import { sphericalKMeans, vendiScore } from '../src/linalg.mjs';
import { clusterCoverage, intentRecall } from '../src/metrics.mjs';
import { embed } from '../src/llm.mjs';
import { dot, normalize, rng, shuffle, mean } from '../src/util.mjs';

const B = CONFIG.B, LAM = CONFIG.lambda;
const BUDGETS = [10, 15, 20, 50];

function relVec(items, clusters) {
  const cents = clusters.map(c => normalize(c.centroid));
  return items.map(it => Math.max(...cents.map(c => dot(it.emb, c))));
}
// Maximal Marginal Relevance; quality = rel (+ lam * r when withR).
function mmr(items, clusters, n, alpha, withR) {
  const rel = relVec(items, clusters);
  const qual = items.map((it, i) => rel[i] + (withR ? LAM * it.r : 0));
  const sel = [], maxSim = new Array(items.length).fill(0), used = new Array(items.length).fill(false);
  while (sel.length < n) {
    let bi = -1, bs = -Infinity;
    for (let i = 0; i < items.length; i++) {
      if (used[i]) continue;
      const s = alpha * qual[i] - (1 - alpha) * maxSim[i];
      if (s > bs) { bs = s; bi = i; }
    }
    used[bi] = true; sel.push(bi);
    for (let i = 0; i < items.length; i++) { const s = dot(items[i].emb, items[bi].emb); if (s > maxSim[i]) maxSim[i] = s; }
  }
  return sel;
}
// k-center (farthest-first traversal): pure diversity.
function kCenter(items, clusters, n) {
  const rel = relVec(items, clusters);
  let first = 0; for (let i = 1; i < items.length; i++) if (rel[i] > rel[first]) first = i;
  const sel = [first], maxSim = items.map(it => dot(it.emb, items[first].emb));
  while (sel.length < n) {
    let bi = -1, bs = Infinity;
    for (let i = 0; i < items.length; i++) if (!sel.includes(i) && maxSim[i] < bs) { bs = maxSim[i]; bi = i; }
    sel.push(bi);
    for (let i = 0; i < items.length; i++) { const s = dot(items[i].emb, items[bi].emb); if (s > maxSim[i]) maxSim[i] = s; }
  }
  return sel;
}
// k-means over the candidate pool (k = n), one representative per cluster:
// the medoid, or the most realistic member when withR.
function kMeansRep(items, n, withR) {
  const res = sphericalKMeans(items.map(it => it.emb), n, { seed: CONFIG.seed, restarts: 2, iters: 40 });
  const sel = [];
  for (let c = 0; c < n; c++) {
    let bi = -1, bs = -Infinity;
    for (let i = 0; i < items.length; i++) {
      if (res.assign[i] !== c) continue;
      const s = withR ? items[i].r : dot(items[i].emb, res.centroids[c]);
      if (s > bs) { bs = s; bi = i; }
    }
    if (bi >= 0) sel.push(bi);
  }
  return sel;
}
// Stratified round-robin over the S1 intent clusters (by weight), taking the
// most realistic unused candidate GENERATED FOR that cluster each time.
function stratified(items, clusters, n) {
  const order = clusters.map((c, j) => j).sort((a, b) => clusters[b].weight - clusters[a].weight);
  const buckets = clusters.map(c => items.map((it, i) => i).filter(i => items[i].cid === c.cid).sort((a, b) => items[b].r - items[a].r));
  const sel = []; let round = 0;
  while (sel.length < n) {
    let added = false;
    for (const j of order) { if (sel.length >= n) break; if (buckets[j][round] !== undefined) { sel.push(buckets[j][round]); added = true; } }
    if (!added) break; round++;
  }
  return sel;
}
function objective(items, clusters, sel) {
  const cents = clusters.map(c => normalize(c.centroid));
  let f = 0;
  cents.forEach((c, j) => { f += clusters[j].weight * Math.max(0, ...sel.map(i => dot(items[i].emb, c))); });
  return f + LAM * sel.reduce((s, i) => s + items[i].r, 0);
}

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const items = S.keptItems, clusters = S.tree.clusters;
  const holdEmb = await embed(S.holdout.map(k => k.kw)), holdW = S.holdout.map(k => k.logvol);
  const idxOf = Object.fromEntries(items.map((it, i) => [it.id, i]));
  const methods = {};
  // incremental selectors return an ORDER (prefix = smaller budget); k-means re-solves per budget
  const greedy = lazyGreedy(items, clusters, B, LAM).selectedIds.map(id => idxOf[id]);
  methods['Q-SEED greedy (ours)'] = (n) => greedy.slice(0, n);
  const m5 = mmr(items, clusters, B, 0.5, false), m7 = mmr(items, clusters, B, 0.7, false);
  const m5r = mmr(items, clusters, B, 0.5, true), m7r = mmr(items, clusters, B, 0.7, true);
  methods['MMR (a=0.5)'] = (n) => m5.slice(0, n);
  methods['MMR (a=0.7)'] = (n) => m7.slice(0, n);
  methods['MMR (a=0.5) + realism'] = (n) => m5r.slice(0, n);
  methods['MMR (a=0.7) + realism'] = (n) => m7r.slice(0, n);
  const kc = kCenter(items, clusters, B);
  methods['k-center (max-min diversity)'] = (n) => kc.slice(0, n);
  methods['k-means medoids'] = (n) => kMeansRep(items, n, false);
  methods['k-means + realism'] = (n) => kMeansRep(items, n, true);
  const st = stratified(items, clusters, B);
  methods['S1-stratified + realism'] = (n) => st.slice(0, n);

  const evalSel = (sel, full) => {
    const embs = sel.map(i => items[i].emb);
    const r = { n: sel.length, coverage: clusterCoverage(embs, clusters) };
    if (full) {
      const qs = sel.map(i => S.qById(items[i].id));
      const cals = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13].map(seed => calibrateRealism(qs, S.realPool, { seed }));
      Object.assign(r, {
        recall: intentRecall(embs, holdEmb, holdW, clusters),
        F: objective(items, clusters, sel), meanR: mean(sel.map(i => items[i].r)),
        vendi: vendiScore(embs), auc: mean(cals.map(c => c.discAUC)), aucSeed4: cals[0].discAUC, mauve: cals[0].mauve,
      });
    }
    return r;
  };
  const res = {};
  for (const [name, f] of Object.entries(methods)) {
    res[name] = { atB: evalSel(f(B), true), cov: Object.fromEntries(BUDGETS.map(n => [n, evalSel(f(n), false).coverage])) };
  }
  // random: mean over 20 draws
  const rr = [];
  for (let s = 0; s < 20; s++) {
    const o = shuffle(items.map((_, i) => i), rng(CONFIG.seed + 900 + s));
    rr.push({ atB: evalSel(o.slice(0, B), s < 5), cov: Object.fromEntries(BUDGETS.map(n => [n, evalSel(o.slice(0, n), false).coverage])) });
  }
  const full = rr.filter(x => x.atB.auc !== undefined);
  res['random'] = {
    atB: Object.fromEntries(['coverage', 'recall', 'F', 'meanR', 'vendi', 'auc', 'mauve'].map(k => [k, mean((k === 'coverage' ? rr : full).map(x => x.atB[k]))])),
    cov: Object.fromEntries(BUDGETS.map(n => [n, mean(rr.map(x => x.cov[n]))])),
  };
  const Fg = res['Q-SEED greedy (ours)'].atB.F;
  for (const k of Object.keys(res)) res[k].atB.Frel = res[k].atB.F / Fg;
  out[v] = res;
  console.error('done', v);
}
save('e1_selection.json', out);

const names = Object.keys(out[VERTICALS[0]]);
const avg = (name, f) => mean(VERTICALS.map(v => f(out[v][name])));
console.log('method | cov@10 | cov@15 | cov@20 | cov@50 | F/F_greedy | meanR | AUC(10 seeds) | MAUVE | Vendi');
for (const n of names) {
  console.log([n, ...[10, 15, 20, 50].map(b => (100 * avg(n, x => x.cov[b])).toFixed(1)),
    avg(n, x => x.atB.Frel).toFixed(3), avg(n, x => x.atB.meanR).toFixed(3), avg(n, x => x.atB.auc).toFixed(3),
    (100 * avg(n, x => x.atB.mauve)).toFixed(1), avg(n, x => x.atB.vendi).toFixed(1)].join(' | '));
}
