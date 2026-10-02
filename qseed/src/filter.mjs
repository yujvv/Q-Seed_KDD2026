// S3 - Reverse fan-out cycle-consistency filter.
// A DIFFERENT model (gemini flash-lite) simulates the engine's query fan-out for
// each candidate q. The candidate survives iff the source keyword's semantic
// space is recovered among the sub-queries: max_k' sim(k', k_source) >= tau.
import { chat, embed, parseJSON, pool } from './llm.mjs';
import { cosine, normalize } from './util.mjs';
import { CONFIG, MODELS } from './config.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

export async function fanoutFilter(bank, gen, tree) {
  const ns = 'filter';
  const ckey = bank.vertical + ':B' + CONFIG.B + ':tau' + CONFIG.tau + ':v2';
  const cached = cacheGet(ns, ckey);
  if (cached) return cached;

  const cands = gen.candidates;
  // Simulate fan-out for each candidate
  const fanouts = await pool(cands, async (cand) => {
    const prompt = `A generative search engine answers a user query by decomposing it into 3-5 focused retrieval sub-queries (query fan-out), then retrieving documents for each.
User query: "${cand.q}"
Output ONLY a JSON array of 3-5 short retrieval sub-queries (keyword-style search strings) the engine would issue.`;
    const raw = await chat([{ role: 'user', content: prompt }], { model: MODELS.fanout, temperature: 0.3, maxTokens: 200 });
    let arr = parseJSON(raw, []);
    if (!Array.isArray(arr)) arr = [];
    return arr.filter(s => typeof s === 'string' && s.length > 1).slice(0, 6).map(s => s.trim());
  }, 20);

  // Embed all sub-queries + all source-cluster member keywords for sim check
  const allSub = [];
  const subIndex = fanouts.map(fo => {
    const start = allSub.length;
    (fo || []).forEach(s => allSub.push(s));
    return [start, allSub.length];
  });
  const subEmb = allSub.length ? (await embed(allSub)).map(normalize) : [];

  // cluster centroid embeddings already available in tree; also embed source keywords
  const centById = {};
  for (const cl of tree.clusters) centById[cl.cid] = normalize(cl.centroid);

  const kept = [];
  const drops = [];
  const consistencyScores = [];
  cands.forEach((cand, i) => {
    const [a, b] = subIndex[i];
    let bestSim = -1;
    const cent = centById[cand.cid];
    for (let j = a; j < b; j++) {
      const s = cosine(subEmb[j], cent);
      if (s > bestSim) bestSim = s;
    }
    consistencyScores.push(bestSim);
    if (bestSim >= CONFIG.tau) { cand.cycleSim = bestSim; kept.push(cand); }
    else drops.push({ id: cand.id, sim: bestSim });
  });

  const result = {
    vertical: bank.vertical,
    kept,
    nIn: cands.length,
    nKept: kept.length,
    dropRate: 1 - kept.length / cands.length,
    consistencyScores,
  };
  return cacheSet(ns, ckey, result);
}

// Forward-only fan-out baseline: generate via fan-out but WITHOUT the cycle
// -consistency test (keep all). Used as an ablation/baseline.
export async function forwardFanoutOnly(gen) {
  return gen.candidates.slice();
}
