// OpenRouter client: chat, embeddings, and generative-engine probing.
// Bounded-concurrency async pool + retry with backoff + on-disk cache.
import { API_KEY, BASE_URL, MODELS } from './config.mjs';
import { cacheGet, cacheSet } from './cache.mjs';

let LIVE_CALLS = 0;
let TOTAL_COST = 0;
export function stats() { return { liveCalls: LIVE_CALLS, cost: TOTAL_COST }; }

// Offline mode (QSEED_OFFLINE=1): every request must be served from the
// content-addressed cache; a miss is a hard error instead of a paid call. This
// is what makes the released artifact's "reproduce every number for $0" claim
// checkable -- a reviewer can re-run the pipeline and any cache gap surfaces as
// a crash rather than a silent charge.
export const OFFLINE = process.env.QSEED_OFFLINE === '1';

async function post(pathname, body, { retries = 5, timeoutMs = 120000 } = {}) {
  if (OFFLINE) {
    throw new Error(
      `QSEED_OFFLINE=1: cache miss on ${pathname} for model=${body?.model}. ` +
      `Refusing to issue a live (billable) call.`);
  }
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(BASE_URL + pathname, {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer ' + API_KEY,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://github.com/qseed/qseed',
          'X-Title': 'Q-SEED',
        },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      clearTimeout(t);
      if (res.status === 429 || res.status >= 500) {
        lastErr = new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
        await sleep(1500 * Math.pow(2, attempt) + Math.floor(seededJitter() * 1000));
        continue;
      }
      const json = await res.json();
      if (json.error) {
        lastErr = new Error(`API error: ${JSON.stringify(json.error).slice(0, 200)}`);
        // Non-retryable content errors: bail early.
        if (String(json.error.code) === '400') throw lastErr;
        await sleep(1500 * Math.pow(2, attempt));
        continue;
      }
      LIVE_CALLS++;
      if (json.usage?.cost) TOTAL_COST += json.usage.cost;
      return json;
    } catch (e) {
      clearTimeout(t);
      lastErr = e;
      await sleep(1500 * Math.pow(2, attempt));
    }
  }
  throw lastErr;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let _jit = 12345;
function seededJitter() { _jit = (_jit * 1103515245 + 12345) & 0x7fffffff; return _jit / 0x7fffffff; }

// ---- bounded-concurrency map ----
export async function pool(items, worker, concurrency = 16, onProgress) {
  const results = new Array(items.length);
  let idx = 0, done = 0;
  async function run() {
    while (idx < items.length) {
      const i = idx++;
      try { results[i] = await worker(items[i], i); }
      catch (e) { results[i] = { __error: String(e).slice(0, 300) }; }
      done++;
      if (onProgress && done % Math.max(1, Math.floor(items.length / 20)) === 0) onProgress(done, items.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
  return results;
}

// ---- chat ----
export async function chat(messages, { model = MODELS.generator, temperature = 0.7, maxTokens = 900, json = false, seed } = {}) {
  const body = { model, messages, temperature, max_tokens: maxTokens };
  if (json) body.response_format = { type: 'json_object' };
  if (seed !== undefined) body.seed = seed;
  const ck = JSON.stringify(body);
  const hit = cacheGet('chat', ck);
  if (hit !== undefined) return hit;
  const res = await post('/chat/completions', body);
  const content = res.choices?.[0]?.message?.content ?? '';
  return cacheSet('chat', ck, content);
}

// ---- embeddings (batched) ----
export async function embed(texts, { model = MODELS.embed, batch = 96 } = {}) {
  const out = new Array(texts.length);
  const missing = [];
  for (let i = 0; i < texts.length; i++) {
    const hit = cacheGet('embed', model + '\n' + texts[i]);
    if (hit !== undefined) out[i] = hit; else missing.push(i);
  }
  for (let b = 0; b < missing.length; b += batch) {
    const chunkIdx = missing.slice(b, b + batch);
    const input = chunkIdx.map(i => texts[i]);
    const res = await post('/embeddings', { model, input });
    for (let k = 0; k < chunkIdx.length; k++) {
      const v = res.data[k].embedding;
      out[chunkIdx[k]] = cacheSet('embed', model + '\n' + texts[chunkIdx[k]], v);
    }
  }
  return out;
}

// ---- generative-engine probe: returns cited eTLD+1 domains ----
import { parse as tldParse } from 'tldts';
export async function probe(query, engineModel, rep) {
  const body = {
    model: engineModel,
    messages: [{ role: 'user', content: query }],
    temperature: 0.4,
    max_tokens: 600,
    // rep index folded into cache key so repeats are distinct cached calls
  };
  const ck = JSON.stringify(body) + '::rep' + rep;
  const hit = cacheGet('probe', ck);
  if (hit !== undefined) return hit;
  const res = await post('/chat/completions', body, { timeoutMs: 120000 });
  const msg = res.choices?.[0]?.message ?? {};
  const domains = new Set();
  for (const a of (msg.annotations || [])) {
    const u = a?.url_citation?.url;
    if (u) { const d = tldParse(u).domain; if (d) domains.add(d.toLowerCase()); }
  }
  const value = { domains: [...domains], nCitations: (msg.annotations || []).length };
  return cacheSet('probe', ck, value);
}

// Parse a JSON array/object out of a possibly-fenced model response.
export function parseJSON(text, fallback = null) {
  if (text == null) return fallback;
  let s = String(text).trim();
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) s = fence[1].trim();
  // find first [ or { and matching last ] or }
  const first = Math.min(...['[', '{'].map(c => { const i = s.indexOf(c); return i < 0 ? Infinity : i; }));
  if (first !== Infinity) {
    const lastArr = s.lastIndexOf(']'), lastObj = s.lastIndexOf('}');
    const last = Math.max(lastArr, lastObj);
    if (last > first) s = s.slice(first, last + 1);
  }
  try { return JSON.parse(s); } catch { return fallback; }
}
