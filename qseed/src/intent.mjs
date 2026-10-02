// S1 - Intent decomposition & hierarchy.
// Embed keywords -> spherical k-means -> LLM-label each cluster (type + facet).
// Cluster weight = sum of log(1+volume) (log-volume damps head skew).
import { embed, chat, parseJSON } from './llm.mjs';
import { sphericalKMeans, meanCentroidSim } from './linalg.mjs';
import { normalize } from './util.mjs';
import { CONFIG, MODELS } from './config.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

export async function buildIntentTree(bank, trainKeywords) {
  const ns = 'intent';
  const ckey = bank.vertical + ':' + trainKeywords.length + ':v2';
  const cached = cacheGet(ns, ckey);
  if (cached) {
    // rehydrate embeddings (not cached in the small record) lazily by caller if needed
    return cached;
  }
  const kws = trainKeywords.map(k => k.kw);
  const emb = await embed(kws);
  const embN = emb.map(normalize);
  const N = kws.length;

  // pick k by coarse scan of mean-centroid-sim (elbow-ish), bounded by config
  const kBase = CONFIG.nClustersHeuristic(N);
  let best = null;
  for (const k of [kBase - 4, kBase, kBase + 4].filter(k => k >= 8 && k < N / 3)) {
    const res = sphericalKMeans(embN, k, { seed: CONFIG.seed, restarts: 3 });
    const q = meanCentroidSim(embN, res);
    // penalize larger k slightly to avoid trivial singletons
    const score = q - 0.004 * k;
    if (!best || score > best.score) best = { res, k, score };
  }
  const { res } = best;

  // group keywords per cluster
  const clusters = [];
  for (let c = 0; c < res.k; c++) {
    const members = [];
    for (let i = 0; i < N; i++) if (res.assign[i] === c) members.push(trainKeywords[i]);
    if (!members.length) continue;
    const weight = members.reduce((s, m) => s + m.logvol, 0);
    clusters.push({ cid: clusters.length, centroid: res.centroids[c], members, weight });
  }

  // LLM-label each cluster
  for (const cl of clusters) {
    const examples = cl.members.slice(0, 12).map(m => m.kw).join(', ');
    const prompt = `These SEO keywords form one intent cluster in the domain of ${bank.label}:
${examples}
Return ONLY JSON: {"type":"<informational|navigational|transactional|commercial-investigation>","facet":"<one of: comparison, how-to, troubleshooting, recommendation, fact, other>","label":"<3-5 word cluster name>"}`;
    const raw = await chat([{ role: 'user', content: prompt }], { model: MODELS.generator, temperature: 0.2, maxTokens: 120 });
    const j = parseJSON(raw, {}) || {};
    cl.type = j.type || 'informational';
    cl.facet = j.facet || 'other';
    cl.label = j.label || ('cluster ' + cl.cid);
  }

  const tree = {
    vertical: bank.vertical,
    k: res.k,
    clusters: clusters.map(c => ({ cid: c.cid, centroid: c.centroid, weight: c.weight, type: c.type, facet: c.facet, label: c.label, memberIds: c.members.map(m => m.id), memberKw: c.members.map(m => m.kw) })),
  };
  return cacheSet(ns, ckey, tree);
}
