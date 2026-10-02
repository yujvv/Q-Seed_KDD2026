// Build semi-synthetic keyword banks (curated head terms -> LLM long-tail
// expansion + Zipf-modelled volumes) and load the real WildChat reference corpus.
import { chat, embed, parseJSON, pool } from './llm.mjs';
import { rng, cosine, normalize } from './util.mjs';
import { CONFIG, MODELS } from './config.mjs';
import { SEEDS, VERTICAL_LABEL } from './seeds.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

const INTENTS = ['I', 'N', 'T', 'C']; // informational, navigational, transactional, commercial-investigation

// Expand one seed into long-tail keyword-tool-style variants with intent tags.
async function expandSeed(seed, verticalLabel) {
  const prompt = `You are a keyword research tool for the domain of ${verticalLabel}.
Given the head keyword "${seed}", output 5 realistic long-tail keyword variations that a real
SEO keyword database would contain for this head term (modifiers like brands, use-cases,
price qualifiers, comparisons, "best", "vs", "near me", years, etc.). Keep them lowercase,
2-7 words, the way they appear in a keyword tool (NOT full questions).
For each, tag the dominant search intent: I=informational, N=navigational, T=transactional, C=commercial-investigation.
Return ONLY a JSON array like: [{"kw":"...","intent":"C"}, ...] with exactly 5 items.`;
  const raw = await chat([{ role: 'user', content: prompt }], { model: MODELS.generator, temperature: 0.8, maxTokens: 300, json: false });
  let arr = parseJSON(raw, []);
  if (!Array.isArray(arr)) arr = [];
  return arr.filter(x => x && x.kw).slice(0, 5).map(x => ({ kw: String(x.kw).toLowerCase().trim(), intent: INTENTS.includes(x.intent) ? x.intent : 'I' }));
}

// Zipf-ish monthly search volume: head terms high, tail lower, log-normal noise.
function assignVolume(headRank, isHead, rand) {
  const base = 60000 / Math.pow(headRank + 1, 0.75); // head decays with rank
  const factor = isHead ? 1 : (0.05 + 0.35 * rand());
  const noise = Math.exp((rand() - 0.5) * 0.8);
  return Math.max(30, Math.round(base * factor * noise));
}

export async function buildKeywordBank(vertical) {
  const ns = 'kwbank';
  const cached = cacheGet(ns, vertical + ':v2');
  if (cached) return cached;
  const label = VERTICAL_LABEL[vertical];
  const seeds = SEEDS[vertical];
  const rand = rng(CONFIG.seed ^ hashStr(vertical));
  const expansions = await pool(seeds, s => expandSeed(s, label), 12);
  const bank = [];
  const seen = new Set();
  seeds.forEach((seed, r) => {
    if (!seen.has(seed)) { seen.add(seed); bank.push({ kw: seed, intent: 'C', volume: assignVolume(r, true, rand), seed }); }
    const exps = Array.isArray(expansions[r]) ? expansions[r] : [];
    for (const e of exps) {
      if (e.kw && !seen.has(e.kw) && e.kw.length <= 60) {
        seen.add(e.kw);
        bank.push({ kw: e.kw, intent: e.intent, volume: assignVolume(r, false, rand), seed });
      }
    }
  });
  // index + logvolume weight
  bank.forEach((b, i) => { b.id = i; b.logvol = Math.log1p(b.volume); });
  const result = { vertical, label, keywords: bank };
  return cacheSet(ns, vertical + ':v2', result);
}

function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

// Holdout split for Intent-Recall (queries in holdout are "unseen real intents").
export function splitHoldout(bank) {
  const rand = rng(CONFIG.seed + 1);
  const idx = bank.keywords.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const nHold = Math.floor(bank.keywords.length * CONFIG.holdoutFrac);
  const holdSet = new Set(idx.slice(0, nHold));
  return {
    train: bank.keywords.filter(k => !holdSet.has(k.id)),
    holdout: bank.keywords.filter(k => holdSet.has(k.id)),
  };
}

// ---- Real reference corpus: WildChat-1M first user turns via HF datasets-server ----
async function fetchWildChatPool(target = 2400) {
  const ns = 'wildchat';
  const cached = cacheGet(ns, 'pool:' + target);
  if (cached) return cached;
  const out = [];
  let offset = 0;
  const length = 100;
  while (out.length < target && offset < 12000) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=allenai/WildChat-1M&config=default&split=train&offset=${offset}&length=${length}`;
    let json;
    try {
      const res = await fetch(url);
      json = await res.json();
    } catch { break; }
    const rows = json.rows || [];
    if (!rows.length) break;
    for (const r of rows) {
      const row = r.row || {};
      if (row.language && row.language !== 'English') continue;
      const conv = row.conversation;
      if (!Array.isArray(conv) || !conv.length) continue;
      const first = conv.find(m => m.role === 'user');
      if (!first || !first.content) continue;
      const text = String(first.content).replace(/\s+/g, ' ').trim();
      // keep short, single-turn, search-like prompts
      if (text.length < 12 || text.length > 220) continue;
      { let ok=true; for(let ci=0;ci<text.length;ci++){ if(text.charCodeAt(ci)>126){ ok=false; break; } } if(!ok) continue; }
      out.push(text);
    }
    offset += length;
  }
  return cacheSet(ns, 'pool:' + target, out);
}

// Lightweight search-intent heuristic (info-seeking, not roleplay/coding/creative).
function looksSearchy(t) {
  const s = t.toLowerCase();
  if (/(write|code|python|javascript|essay|poem|story|roleplay|pretend|as an? |translate|def |function |import |\bhtml\b|resume|cover letter|rewrite|paraphrase|correct my|fix this)/.test(s)) return false;
  const q = /(what|which|how|why|when|where|best|vs|compare|should i|is it|are there|recommend|suggest|looking for|i need|difference|top \d|good |better )/.test(s);
  const wc = s.split(' ').length;
  return q && wc >= 3 && wc <= 36;
}

// Independently-constructed reference bank for external validity: convert the
// vertical's top head keywords into natural questions using a DIFFERENT model
// (the fan-out model) with a plain one-shot prompt -- no personas, no
// cycle-consistency, no submodular selection. Its domain-visibility ranking is
// an independent construct-validity signal for the Q-SEED ranking.
async function buildIndependentRefBank(bank) {
  const heads = bank.keywords.slice().sort((a, b) => b.volume - a.volume).slice(0, CONFIG.refBankSize);
  const out = await pool(heads, async (k) => {
    const raw = await chat([{ role: 'user', content: `Turn this ${bank.label} search topic into one natural question a person would type into an AI search engine. Topic: "${k.kw}". Return only the question.` }], { model: MODELS.fanout, temperature: 0.6, maxTokens: 60 });
    return String(raw).replace(/^["']|["']$/g, '').trim().split('\n')[0];
  }, 12);
  return out.filter(q => typeof q === 'string' && q.length > 8);
}

// Build reference sets: a broad real WildChat style pool (for S4 realism
// calibration) + a per-vertical independently-constructed reference bank.
export async function buildReferenceSets(banks) {
  const ns = 'refsets';
  const cached = cacheGet(ns, 'v4:' + banks.map(b => b.vertical).join(','));
  if (cached) return cached;
  const pool = (await fetchWildChatPool(14000)).filter(looksSearchy);
  const globalRealPool = [...new Set(pool)];
  const perVertical = {};
  for (const b of banks) perVertical[b.vertical] = await buildIndependentRefBank(b);
  const result = { perVertical, globalRealPool };
  return cacheSet(ns, 'v4:' + banks.map(b => b.vertical).join(','), result);
}
