// Rebuttal E5 (Reviewer epgg Q5): does selecting for style change WHICH domains
// the engines cite? Free. Within the 100 probed queries of each vertical, split
// by realism score r(q) (top half vs bottom half) and compare the two halves'
// domain-visibility rankings against the null of random half-splits.
// Also: split-half reliability of the reference bank (for disattenuation).
import { loadVertical, VERTICALS, ENGINES, CONFIG, save } from './common.mjs';
import { domainVisibility } from '../src/probe.mjs';
import { styleFeatures } from '../src/calibrate.mjs';
import { spearman, rng, shuffle, mean, quantile } from '../src/util.mjs';

const vis = (recs, D, engs = ENGINES) => domainVisibility(recs, D, engs).map(v => v.rate);
const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v);
  const D = S.domains, n = S.recsQ0.length, h = n / 2;
  const r = S.Q0ids.map(id => S.cal.rq[id]);
  const halves = (order) => spearman(vis(order.slice(0, h).map(i => S.recsQ0[i]), D), vis(order.slice(h).map(i => S.recsQ0[i]), D));
  const byR = r.map((x, i) => i).sort((a, b) => r[b] - r[a]);
  const obs = halves(byR);
  const nul = []; for (let k = 0; k < 2000; k++) nul.push(halves(shuffle(r.map((_, i) => i), rng(CONFIG.seed + 31000 + k))));
  const pct = nul.filter(x => x <= obs).length / nul.length;
  // same test on the two strongest single style features the discriminator uses
  const feat = (j) => { const f = S.Q0queries.map(q => styleFeatures(q)[j]); const o = f.map((_, i) => i).sort((a, b) => f[b] - f[a]); return halves(o); };
  // citations per answer vs realism
  const nCited = S.recsQ0.map(rec => mean(ENGINES.map(e => Object.keys(rec[e.id].domainCounts).length)));
  // reference-bank split-half (Spearman-Brown), for disattenuating convergent validity
  const sh = [];
  for (let k = 0; k < 200; k++) {
    const o = shuffle(S.recsRef.map((_, i) => i), rng(CONFIG.seed + 52000 + k)); const m = Math.floor(o.length / 2);
    const c = spearman(vis(o.slice(0, m).map(i => S.recsRef[i]), D), vis(o.slice(m, 2 * m).map(i => S.recsRef[i]), D)); sh.push(2 * c / (1 + c));
  }
  const shQ = [];
  for (let k = 0; k < 200; k++) {
    const o = shuffle(S.recsFinal.map((_, i) => i), rng(CONFIG.seed + 53000 + k)); const m = Math.floor(o.length / 2);
    const c = spearman(vis(o.slice(0, m).map(i => S.recsFinal[i]), D), vis(o.slice(m, 2 * m).map(i => S.recsFinal[i]), D)); shQ.push(2 * c / (1 + c));
  }
  const conv = spearman(vis(S.recsFinal, D), vis(S.recsRef, D));
  out[v] = {
    highVsLowRealism: obs, randomSplit: { mean: mean(nul), lo: quantile(nul, 0.025), hi: quantile(nul, 0.975) }, percentileOfObserved: pct,
    meanR: { high: mean(byR.slice(0, h).map(i => r[i])), low: mean(byR.slice(h).map(i => r[i])) },
    splitByWordCount: feat(0), splitByFirstPerson: feat(3),
    rhoRealismVsDomainsCited: spearman(r, nCited),
    refSplitHalfSB: mean(sh), bankSplitHalfSB_200splits: mean(shQ), convergent: conv, convergentDisattenuated: conv / Math.sqrt(mean(sh) * mean(shQ)),
  };
  console.error(v, JSON.stringify(out[v]));
}
save('e5_style_bias.json', out);
const avg = (f) => mean(VERTICALS.map(v => f(out[v])));
console.log('high-r vs low-r halves rho:', avg(x => x.highVsLowRealism).toFixed(3), '| random halves:', avg(x => x.randomSplit.mean).toFixed(3), '[', avg(x => x.randomSplit.lo).toFixed(3), avg(x => x.randomSplit.hi).toFixed(3), ']');
console.log('convergent', avg(x => x.convergent).toFixed(2), 'ref SB', avg(x => x.refSplitHalfSB).toFixed(2), 'bank SB(200)', avg(x => x.bankSplitHalfSB_200splits).toFixed(2), 'disattenuated', avg(x => x.convergentDisattenuated).toFixed(2));
