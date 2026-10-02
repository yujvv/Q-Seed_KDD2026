// Baseline query-bank constructors. Each returns a list of query strings of
// size B (or as close as the method allows), from the SAME keyword bank.
import { chat, parseJSON, pool } from './llm.mjs';
import { rng, sample } from './util.mjs';
import { CONFIG, MODELS } from './config.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

// 1. KW-as-Query: submit top-B keywords by volume verbatim (industry lower bound).
export function kwAsQuery(bank, B = CONFIG.B) {
  return bank.keywords.slice().sort((a, b) => b.volume - a.volume).slice(0, B).map(k => k.kw);
}

// 2. Naive Paraphrase: one-prompt LLM rewrite of a keyword into a question.
export async function naiveParaphrase(bank, B = CONFIG.B) {
  const ns = 'bl_naive';
  const ck = bank.vertical + ':B' + B + ':v2';
  const hit = cacheGet(ns, ck); if (hit) return hit;
  const rand = rng(CONFIG.seed + 21);
  const picks = sample(bank.keywords, Math.min(B, bank.keywords.length), rand);
  const out = await pool(picks, async (k) => {
    const raw = await chat([{ role: 'user', content: `Rewrite this search keyword as a single natural-language question a person would ask a chatbot. Keyword: "${k.kw}". Return only the question.` }], { model: MODELS.generator, temperature: 0.7, maxTokens: 60 });
    return String(raw).replace(/^["']|["']$/g, '').trim().split('\n')[0];
  }, 16);
  const res = out.filter(q => typeof q === 'string' && q.length > 6);
  return cacheSet(ns, ck, res);
}

// 3. PAA-Template: People-Also-Ask style template expansion (deterministic).
export function paaTemplate(bank, B = CONFIG.B) {
  const templates = [
    k => `What is the best ${k}?`,
    k => `How do I choose a ${k}?`,
    k => `Is ${k} worth it?`,
    k => `What should I look for in a ${k}?`,
    k => `${k}: which one is right for me?`,
    k => `How much does ${k} cost?`,
  ];
  const rand = rng(CONFIG.seed + 22);
  const picks = sample(bank.keywords, Math.min(B, bank.keywords.length), rand);
  return picks.map((k, i) => templates[i % templates.length](k.kw));
}

// 4. Forward Fan-out: generate conversational queries by forward fan-out of
//    keywords WITHOUT cycle-consistency or submodular selection.
export async function forwardFanout(bank, B = CONFIG.B) {
  const ns = 'bl_fwd';
  const ck = bank.vertical + ':B' + B + ':v2';
  const hit = cacheGet(ns, ck); if (hit) return hit;
  const rand = rng(CONFIG.seed + 23);
  const picks = sample(bank.keywords, Math.min(B, bank.keywords.length), rand);
  const out = await pool(picks, async (k) => {
    const raw = await chat([{ role: 'user', content: `A user is interested in "${k.kw}". Write one realistic conversational question they might ask an AI search engine about it. Return only the question.` }], { model: MODELS.generator, temperature: 0.85, maxTokens: 60 });
    return String(raw).replace(/^["']|["']$/g, '').trim().split('\n')[0];
  }, 16);
  const res = out.filter(q => typeof q === 'string' && q.length > 6);
  return cacheSet(ns, ck, res);
}

// 5. GEO-bench-like: keyword-agnostic generic questions for the vertical
//    (mimics using an off-the-shelf general QA benchmark subset).
export async function geoBenchLike(bank, B = CONFIG.B) {
  const ns = 'bl_geobench';
  const ck = bank.vertical + ':B' + B + ':v2';
  const hit = cacheGet(ns, ck); if (hit) return hit;
  const raw = await chat([{ role: 'user', content: `List ${B + 10} diverse general questions people ask AI assistants about the broad topic of ${bank.label}. Do NOT tailor them to any specific product list; keep them generic and encyclopedic. Return ONLY a JSON array of strings.` }], { model: MODELS.generator, temperature: 0.9, maxTokens: 1600 });
  let arr = parseJSON(raw, []);
  if (!Array.isArray(arr)) arr = [];
  const res = arr.filter(q => typeof q === 'string' && q.length > 6).slice(0, B);
  return cacheSet(ns, ck, res);
}
