// Rebuttal E4 (Reviewers uXKA, epgg W4, JZBt Q3): what S3 does and does not do.
// Free part (cached fan-outs + embeddings):
//  - does S3 change the delivered bank?
//  - does the fan-out test add anything over a plain sim(q, k_src) filter?
//  - detection power: rejection rate when a candidate is paired with a WRONG
//    source cluster (controlled drift), fan-out test vs plain-similarity test.
import { loadVertical, VERTICALS, CONFIG, save, embedN } from './common.mjs';
import { chat, parseJSON } from '../src/llm.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { MODELS } from '../src/config.mjs';
import { dot, normalize, spearman, mean, quantile } from '../src/util.mjs';

export const fanoutPrompt = (q) => `A generative search engine answers a user query by decomposing it into 3-5 focused retrieval sub-queries (query fan-out), then retrieving documents for each.
User query: "${q}"
Output ONLY a JSON array of 3-5 short retrieval sub-queries (keyword-style search strings) the engine would issue.`;

export async function cachedFanouts(S) {
  const fo = [];
  for (const c of S.gen.candidates) {
    const raw = await chat([{ role: 'user', content: fanoutPrompt(c.q) }], { model: MODELS.fanout, temperature: 0.3, maxTokens: 200 });
    let arr = parseJSON(raw, []); if (!Array.isArray(arr)) arr = [];
    fo.push(arr.filter(s => typeof s === 'string' && s.length > 1).slice(0, 6).map(s => s.trim()));
  }
  return fo;
}

const isMain = process.argv[1] && process.argv[1].endsWith('rb4_s3.mjs');
if (isMain) {
  const out = {};
  for (const v of VERTICALS) {
    const S = await loadVertical(v, { probes: false });
    const cands = S.gen.candidates, tau = CONFIG.tau;
    const cents = Object.fromEntries(S.tree.clusters.map(c => [c.cid, normalize(c.centroid)]));
    const cids = S.tree.clusters.map(c => c.cid);
    const fo = await cachedFanouts(S);
    const flat = fo.flat(), flatEmb = await embedN(flat);
    let p = 0; const subEmb = fo.map(f => { const e = flatEmb.slice(p, p + f.length); p += f.length; return e; });
    const cyc = (i, cid) => subEmb[i].length ? Math.max(...subEmb[i].map(e => dot(e, cents[cid]))) : -1;
    const own = cands.map((c, i) => cyc(i, c.cid));
    const maxDev = Math.max(...own.map((s, i) => Math.abs(s - S.filt.consistencyScores[i])));
    const direct = cands.map((c, i) => dot(S.candEmbN[i], cents[c.cid]));
    const dropped = cands.map((c, i) => i).filter(i => own[i] < tau);
    const r = { n: cands.length, nDropped: dropped.length, recomputeMaxDev: maxDev };

    // 1. bank impact
    const woS3 = lazyGreedy(S.allItems, S.tree.clusters, CONFIG.B, CONFIG.lambda).selectedIds;
    const fin = new Set(S.finalIds);
    r.bank = { sharedWithWoS3: woS3.filter(id => fin.has(id)).length, droppedCandidatesInWoS3Bank: woS3.filter(id => !S.keptSet.has(id)).length };
    // what share of the high-realism candidates (the ones selection prefers) does S3 remove?
    const rTop = quantile(S.cal.rq, 0.9);
    const top = cands.map((c, i) => i).filter(i => S.cal.rq[i] >= rTop);
    r.dropRateAmongTopRealism = top.filter(i => own[i] < tau).length / top.length;
    r.meanRealism = { kept: mean(cands.map((c, i) => i).filter(i => own[i] >= tau).map(i => S.cal.rq[i])), dropped: dropped.length ? mean(dropped.map(i => S.cal.rq[i])) : null };

    // 2. vs plain-similarity filter at a matched drop rate
    const tauD = quantile(direct, dropped.length / cands.length);
    const dDrop = new Set(cands.map((c, i) => i).filter(i => direct[i] < tauD));
    const inter = dropped.filter(i => dDrop.has(i)).length;
    r.vsDirect = { rhoCycleDirect: spearman(own, direct), matchedTau: tauD, overlapOfDropSets: inter / dropped.length, jaccard: inter / (dropped.length + dDrop.size - inter) };

    // 3. controlled drift: pair every candidate with a wrong cluster
    const sims = cids.map(a => cids.map(b => dot(cents[a], cents[b])));
    let rejRand = 0, rejNear = 0, dRejRand = 0, dRejNear = 0, nPairs = 0, nNear = 0;
    cands.forEach((c, i) => {
      const a = cids.indexOf(c.cid);
      let near = -1, ns = -2;
      cids.forEach((cid, b) => {
        if (b === a) return;
        nPairs++;
        if (cyc(i, cid) < tau) rejRand++;
        if (dot(S.candEmbN[i], cents[cid]) < tauD) dRejRand++;
        if (sims[a][b] > ns) { ns = sims[a][b]; near = b; }
      });
      nNear++;
      if (cyc(i, cids[near]) < tau) rejNear++;
      if (dot(S.candEmbN[i], cents[cids[near]]) < tauD) dRejNear++;
    });
    r.controlledDrift = {
      fanout: { rejectWrongCluster: rejRand / nPairs, rejectNearestWrongCluster: rejNear / nNear, falseRejectOwn: dropped.length / cands.length },
      plainSim: { rejectWrongCluster: dRejRand / nPairs, rejectNearestWrongCluster: dRejNear / nNear, falseRejectOwn: dDrop.size / cands.length },
    };
    r.examplesDropped = dropped.sort((x, y) => own[x] - own[y]).slice(0, 6).map(i => ({ q: cands[i].q, cluster: S.tree.clusters.find(c => c.cid === cands[i].cid).label, score: +own[i].toFixed(3), fanout: fo[i] }));
    out[v] = r;
    console.error(v, JSON.stringify({ ...r, examplesDropped: undefined }));
  }
  save('e4_s3_free.json', out);
  for (const v of VERTICALS) { console.log('\n' + v); for (const e of out[v].examplesDropped) console.log(' ', e.score, '|', e.cluster, '|', e.q, '=>', e.fanout.join(' ; ')); }
}
