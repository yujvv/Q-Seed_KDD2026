// S2 - Contextualized candidate generation.
// For each (cluster, facet) drive the generator with a persona x scenario x
// funnel-stage matrix. Every candidate keeps a provenance tuple for auditing.
import { chat, parseJSON, pool } from './llm.mjs';
import { rng, sample } from './util.mjs';
import { CONFIG, MODELS } from './config.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

export async function generateCandidates(bank, tree) {
  const ns = 'generate';
  const ckey = bank.vertical + ':B' + CONFIG.B + ':v2';
  const cached = cacheGet(ns, ckey);
  if (cached) return cached;

  const targetPool = CONFIG.candidatesPerB * CONFIG.B;
  const perCluster = Math.max(6, Math.ceil(targetPool / tree.clusters.length));
  const rand = rng(CONFIG.seed + 5);

  const jobs = [];
  for (const cl of tree.clusters) {
    // choose a small conditioning matrix per cluster to hit perCluster candidates
    const nCombos = Math.max(3, Math.ceil(perCluster / 5));
    for (let c = 0; c < nCombos; c++) {
      jobs.push({
        cl,
        persona: sample(CONFIG.personas, 1, rand)[0],
        scenario: sample(CONFIG.scenarios, 1, rand)[0],
        funnel: CONFIG.funnels[Math.min(2, Math.floor(rand() * 3))],
        n: 5,
      });
    }
  }

  const outputs = await pool(jobs, async (job) => {
    const seedKws = job.cl.memberKw.slice(0, 8).join(', ');
    const prompt = `You are simulating how real people phrase questions to an AI search engine (like Perplexity or ChatGPT Search) in the domain of ${bank.label}.
Intent cluster: "${job.cl.label}" (type: ${job.cl.type}, facet: ${job.cl.facet}).
Representative source keywords: ${seedKws}
Write ${job.n} DISTINCT natural-language conversational queries that a "${job.persona}" would ask, in the scenario of "${job.scenario}", at the "${job.funnel}" stage of their journey.
Rules: full natural questions or requests (not keywords); vary phrasing and length; include realistic constraints/context; stay on the intent cluster's topic.
Return ONLY a JSON array of ${job.n} strings.`;
    const raw = await chat([{ role: 'user', content: prompt }], { model: MODELS.generator, temperature: 0.95, maxTokens: 500 });
    let arr = parseJSON(raw, []);
    if (!Array.isArray(arr)) arr = [];
    return arr.filter(s => typeof s === 'string' && s.length > 8 && s.length < 300)
      .map(q => ({ q: q.trim(), cid: job.cl.cid, facet: job.cl.facet, persona: job.persona, scenario: job.scenario, funnel: job.funnel, sourceKw: job.cl.memberKw[0] }));
  }, 14);

  // flatten + dedup (exact + near-exact lowercased)
  const seen = new Set();
  const candidates = [];
  for (const arr of outputs) {
    if (!Array.isArray(arr)) continue;
    for (const cand of arr) {
      const key = cand.q.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
      if (key.length < 6 || seen.has(key)) continue;
      seen.add(key);
      cand.id = candidates.length;
      candidates.push(cand);
    }
  }
  const result = { vertical: bank.vertical, candidates };
  return cacheSet(ns, ckey, result);
}
