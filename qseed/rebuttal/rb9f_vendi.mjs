// Vendi score of the per-keyword-facility bank vs the paper bank (cached, free).
import { loadVertical, VERTICALS, CONFIG, save, embedN } from './common.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { vendiScore } from '../src/linalg.mjs';
import { mean } from '../src/util.mjs';
const r = { paper: [], perKeyword: [] };
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const trainE = await embedN(S.train.map(k => k.kw));
  const scale = S.tree.clusters.reduce((s, c) => s + c.weight, 0) / S.train.reduce((s, k) => s + k.logvol, 0);
  const clusters = S.train.map((k, i) => ({ centroid: trainE[i], weight: k.logvol * scale }));
  const ids = lazyGreedy(S.keptItems, clusters, CONFIG.B, 5).selectedIds;
  r.paper.push(vendiScore(S.finalIds.map(id => S.candEmbN[id]))); r.perKeyword.push(vendiScore(ids.map(id => S.candEmbN[id])));
}
save('e9f_vendi.json', r); console.log('Vendi paper bank', mean(r.paper).toFixed(1), r.paper.map(x => x.toFixed(1)).join('/'), '| per-keyword lambda=5', mean(r.perKeyword).toFixed(1), r.perKeyword.map(x => x.toFixed(1)).join('/'));
