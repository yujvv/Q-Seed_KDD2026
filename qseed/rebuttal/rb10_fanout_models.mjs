// Rebuttal E10 (Reviewers JZBt Q3, uXKA): sensitivity of S3 to the model that
// simulates the fan-out. Re-run the identical fan-out prompt with two other
// model families on every candidate and compare scores, keep/drop decisions and
// the resulting delivered bank with the submitted run (gemini-2.5-flash-lite).
import { loadVertical, VERTICALS, CONFIG, save, rbChat, pmap, embedN, chargeExternal, spent } from './common.mjs';
import { fanoutPrompt, cachedFanouts } from './rb4_s3.mjs';
import { parseJSON, stats } from '../src/llm.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { dot, normalize, spearman, mean } from '../src/util.mjs';

const ALT = (process.env.RB_FANOUT || 'openai/gpt-4o-mini,meta-llama/llama-3.3-70b-instruct').split(',');
const tau = CONFIG.tau;
const kappa = (a, b) => { const n = a.length; let both = 0, pa = 0, pb = 0; for (let i = 0; i < n; i++) { if (a[i] === b[i]) both++; pa += a[i]; pb += b[i]; } const po = both / n, pe = (pa / n) * (pb / n) + (1 - pa / n) * (1 - pb / n); return (po - pe) / (1 - pe); };

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const cands = S.gen.candidates;
  const cents = Object.fromEntries(S.tree.clusters.map(c => [c.cid, normalize(c.centroid)]));
  const score = async (fo) => {
    const flat = fo.flat(), E = await embedN(flat); let p = 0;
    return fo.map((f, i) => { const e = E.slice(p, p + f.length); p += f.length; return e.length ? Math.max(...e.map(x => dot(x, cents[cands[i].cid]))) : -1; });
  };
  const fos = { 'google/gemini-2.5-flash-lite': await cachedFanouts(S) };
  for (const m of ALT) {
    const raw = await pmap(cands, c => rbChat(fanoutPrompt(c.q), { model: m, temperature: 0.3, maxTokens: 200, tag: 'fanout_alt' }), 16);
    fos[m] = raw.map(r => { let a = parseJSON(r, []); if (!Array.isArray(a)) a = []; return a.filter(s => typeof s === 'string' && s.length > 1).slice(0, 6).map(s => s.trim()); });
    console.error(v, m, 'fan-outs done, empty:', fos[m].filter(f => !f.length).length, 'spent $' + spent().toFixed(3));
  }
  const sc = {}; for (const m of Object.keys(fos)) sc[m] = await score(fos[m]);
  const base = 'google/gemini-2.5-flash-lite', keepB = sc[base].map(s => +(s >= tau));
  const mk = (keep) => cands.filter((c, i) => keep[i]).map(c => ({ id: c.id, emb: S.candEmbN[c.id], r: S.cal.rq[c.id], cid: c.cid }));
  const bankOf = (keep) => new Set(lazyGreedy(mk(keep), S.tree.clusters, CONFIG.B, CONFIG.lambda).selectedIds);
  const bankB = bankOf(keepB);
  const res = {};
  for (const m of Object.keys(fos)) {
    const valid = sc[m].map((s, i) => i).filter(i => sc[m][i] > -1 && sc[base][i] > -1);
    const keep = sc[m].map(s => +(s >= tau));
    const bank = bankOf(keep);
    res[m] = {
      subQueriesPerCandidate: mean(fos[m].map(f => f.length)),
      dropRate: 1 - mean(keep), keptMeanSim: mean(sc[m].filter(s => s >= tau)), dropMeanSim: mean(sc[m].filter(s => s < tau && s > -1)),
      rhoWithSubmitted: spearman(valid.map(i => sc[m][i]), valid.map(i => sc[base][i])),
      decisionAgreement: mean(keep.map((k, i) => +(k === keepB[i]))), kappa: kappa(keep, keepB),
      bankOverlapWithSubmitted: [...bank].filter(id => bankB.has(id)).length,
    };
  }
  out[v] = res; console.error(v, JSON.stringify(res));
}
chargeExternal('fanout_alt_embed', stats().cost, stats().liveCalls);
save('e10_fanout_models.json', out);
for (const m of Object.keys(out[VERTICALS[0]])) { const a = (f) => mean(VERTICALS.map(v => f(out[v][m]))); console.log(m, '| drop', (100 * a(x => x.dropRate)).toFixed(1) + '%', '| kept/dropped sim', a(x => x.keptMeanSim).toFixed(2), a(x => x.dropMeanSim).toFixed(2), '| rho', a(x => x.rhoWithSubmitted).toFixed(2), '| agree', (100 * a(x => x.decisionAgreement)).toFixed(1) + '%', 'kappa', a(x => x.kappa).toFixed(2), '| bank overlap /50', a(x => x.bankOverlapWithSubmitted).toFixed(1)); }
