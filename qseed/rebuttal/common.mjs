// Shared loader for the rebuttal experiments. Rebuilds the exact per-vertical
// state of run_pipeline.mjs from the content-addressed cache (no live calls when
// QSEED_OFFLINE=1), and provides a budget-capped client for the few NEW calls
// the rebuttal experiments issue (cached under their own `rb_*` namespaces so
// the released cache and every number in the submitted paper stay untouched).
import fs from 'node:fs';
import path from 'node:path';
import { parse as tldParse } from 'tldts';
import { CONFIG, VERTICALS, ENGINES, API_KEY, BASE_URL, ROOT } from '../src/config.mjs';
import { embed } from '../src/llm.mjs';
import { normalize } from '../src/util.mjs';
import { buildKeywordBank, splitHoldout, buildReferenceSets } from '../src/data.mjs';
import { buildIntentTree } from '../src/intent.mjs';
import { generateCandidates } from '../src/generate.mjs';
import { fanoutFilter } from '../src/filter.mjs';
import { calibrateRealism } from '../src/calibrate.mjs';
import { lazyGreedy } from '../src/select.mjs';
import { probeQueries, buildDomainUniverse } from '../src/probe.mjs';
import { cacheGet, cacheSet } from '../src/cache.mjs';

export { CONFIG, VERTICALS, ENGINES };
export const OUT = path.join(ROOT, 'results', 'rebuttal');
fs.mkdirSync(OUT, { recursive: true });
export const save = (name, obj) => fs.writeFileSync(path.join(OUT, name), JSON.stringify(obj, null, 1));
export const load = (name) => JSON.parse(fs.readFileSync(path.join(OUT, name), 'utf8'));
export const VLABEL = { consumer_electronics: 'CE', personal_finance: 'PF', health_wellness: 'HW' };

export const embedN = async (texts, opts) => (await embed(texts, opts)).map(normalize);

let BANKS = null;
export async function loadVertical(vertical, { probes = true } = {}) {
  if (!BANKS) { BANKS = []; for (const v of VERTICALS) BANKS.push(await buildKeywordBank(v)); }
  const bank = BANKS.find(b => b.vertical === vertical);
  const { train, holdout } = splitHoldout(bank);
  const tree = await buildIntentTree(bank, train);
  const gen = await generateCandidates(bank, tree);
  const candQueries = gen.candidates.map(c => c.q);
  const candEmbN = await embedN(candQueries);
  const refsets = await buildReferenceSets(BANKS);
  const realPool = refsets.globalRealPool;
  const refBank = refsets.perVertical[vertical];
  const cal = await calibrateRealism(candQueries, realPool);
  const filt = await fanoutFilter(bank, gen, tree);
  const keptSet = new Set(filt.kept.map(c => c.id));
  const mk = (cands) => cands.map(c => ({ id: c.id, emb: candEmbN[c.id], r: cal.rq[c.id], cid: c.cid }));
  const keptItems = mk(filt.kept), allItems = mk(gen.candidates);
  const g0 = lazyGreedy(keptItems, tree.clusters, CONFIG.Q0factor * CONFIG.B, CONFIG.lambda);
  const Q0ids = g0.selectedIds;
  const qById = (id) => gen.candidates[id].q;
  const Q0queries = Q0ids.map(qById);
  const finalIds = Q0ids.slice(0, CONFIG.B);
  const out = {
    vertical, bank, train, holdout, tree, gen, candQueries, candEmbN, realPool, refBank, cal, filt,
    keptSet, keptItems, allItems, Q0ids, Q0queries, finalIds, finalQueries: finalIds.map(qById), qById,
  };
  if (probes) {
    out.recsQ0 = await probeQueries(Q0queries, { label: vertical + '/Q0', repeats: CONFIG.probeRepeats });
    out.recsQ0r2 = await probeQueries(Q0queries, { label: vertical + '/Q0r2', repeats: CONFIG.probeRepeats, repOffset: 100 });
    out.recsRef = await probeQueries(refBank, { label: vertical + '/ref', repeats: CONFIG.refBankRepeats });
    out.domains = buildDomainUniverse(out.recsQ0, ENGINES, { minQueries: 3, maxD: 60 });
    out.recsFinal = out.recsQ0.slice(0, CONFIG.B);
    out.recsFinalR2 = out.recsQ0r2.slice(0, CONFIG.B);
  }
  return out;
}

// ---------------- budget-capped live client for NEW rebuttal calls ----------------
const LEDGER = path.join(OUT, '_ledger.json');
// Read-modify-write ledger so several scripts can run without clobbering each other.
const readLedger = () => fs.existsSync(LEDGER) ? JSON.parse(fs.readFileSync(LEDGER, 'utf8')) : { cost: 0, calls: 0, byTag: {} };
const base = readLedger().cost;
const delta = { cost: 0, calls: 0, byTag: {} };
let sessionCost = 0;
export const BUDGET_USD = Number(process.env.RB_BUDGET || 12);
export const spent = () => base + sessionCost;
function charge(tag, cost, calls = 1) {
  sessionCost += cost; delta.cost += cost; delta.calls += calls;
  const t = (delta.byTag[tag] ||= { cost: 0, calls: 0 }); t.cost += cost; t.calls += calls;
  if (delta.calls >= 25) flushLedger();
}
export function flushLedger() {
  if (!delta.calls && !delta.cost) return;
  const l = readLedger(); l.cost += delta.cost; l.calls += delta.calls;
  for (const [k, v] of Object.entries(delta.byTag)) { const t = (l.byTag[k] ||= { cost: 0, calls: 0 }); t.cost += v.cost; t.calls += v.calls; }
  fs.writeFileSync(LEDGER, JSON.stringify(l, null, 1));
  delta.cost = 0; delta.calls = 0; delta.byTag = {};
}
process.on('exit', flushLedger);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function post(pathname, body, tag, { retries = 4, timeoutMs = 150000 } = {}) {
  if (process.env.QSEED_OFFLINE === '1') throw new Error('offline: cache miss for ' + tag);
  if (spent() >= BUDGET_USD) throw new Error(`rebuttal budget cap $${BUDGET_USD} reached`);
  let lastErr;
  for (let a = 0; a <= retries; a++) {
    const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(BASE_URL + pathname, {
        method: 'POST', signal: ctrl.signal,
        headers: { Authorization: 'Bearer ' + API_KEY, 'Content-Type': 'application/json', 'X-Title': 'Q-SEED' },
        body: JSON.stringify({ ...body, usage: { include: true } }),
      });
      clearTimeout(t);
      if (res.status === 429 || res.status >= 500) { lastErr = new Error('HTTP ' + res.status + ' ' + (await res.text()).slice(0, 200)); await sleep(1500 * 2 ** a); continue; }
      const json = await res.json();
      if (json.error) { lastErr = new Error('API error: ' + JSON.stringify(json.error).slice(0, 300)); if (res.status === 400 || res.status === 404 || res.status === 402) throw lastErr; await sleep(1500 * 2 ** a); continue; }
      charge(tag, json.usage?.cost || 0);
      return json;
    } catch (e) { clearTimeout(t); lastErr = e; if (/API error|budget/.test(String(e))) throw e; await sleep(1500 * 2 ** a); }
  }
  throw lastErr;
}

// Chat call cached under its own namespace (keeps the released `chat` cache clean).
export async function rbChat(prompt, { model, temperature = 0, maxTokens = 200, tag = 'chat' } = {}) {
  const body = { model, messages: [{ role: 'user', content: prompt }], temperature, max_tokens: maxTokens };
  const ck = JSON.stringify(body);
  const hit = cacheGet('rb_chat', ck); if (hit !== undefined) return hit;
  const res = await post('/chat/completions', body, tag);
  return cacheSet('rb_chat', ck, res.choices?.[0]?.message?.content ?? '');
}

// Engine probe that keeps the full citation URLs (the released probe cache keeps
// only eTLD+1 domains). `extra` carries e.g. plugins:[{id:'web',engine:'exa'}].
export async function rbProbe(query, model, rep, { extra = {}, tag = 'probe' } = {}) {
  const body = { model, messages: [{ role: 'user', content: query }], temperature: 0.4, max_tokens: 600, ...extra };
  const ck = JSON.stringify(body) + '::rep' + rep;
  const hit = cacheGet('rb_probe', ck); if (hit !== undefined) return hit;
  const res = await post('/chat/completions', body, tag);
  const msg = res.choices?.[0]?.message ?? {};
  const urls = [];
  for (const a of (msg.annotations || [])) { const u = a?.url_citation?.url; if (u) urls.push(u); }
  for (const u of (res.citations || [])) if (typeof u === 'string' && !urls.includes(u)) urls.push(u);
  const domains = [...new Set(urls.map(u => tldParse(u).domain).filter(Boolean).map(d => d.toLowerCase()))];
  return cacheSet('rb_probe', ck, { domains, urls, nCitations: (msg.annotations || []).length, cost: res.usage?.cost || 0 });
}

// Probe queries on an engine list [{id, model, extra}] -> recs in the pipeline's shape.
export async function rbProbeQueries(queries, engines, { repeats = 2, concurrency = 12, tag = 'probe', repOffset = 0 } = {}) {
  const tasks = [];
  queries.forEach((q, qi) => { for (const e of engines) for (let r = 0; r < repeats; r++) tasks.push({ q, qi, e, r: r + repOffset }); });
  const recs = queries.map(() => Object.fromEntries(engines.map(e => [e.id, { domainCounts: {}, trials: 0, urls: [], nCit: [] }])));
  let i = 0, fail = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, async () => {
    while (i < tasks.length) {
      const t = tasks[i++];
      try {
        const out = await rbProbe(t.q, t.e.model, t.r, { extra: t.e.extra || {}, tag });
        const cell = recs[t.qi][t.e.id];
        cell.trials += 1; cell.urls.push(out.urls); cell.nCit.push(out.domains.length);
        for (const d of out.domains) cell.domainCounts[d] = (cell.domainCounts[d] || 0) + 1;
      } catch (e) { fail++; if (/budget/.test(String(e))) throw e; if (fail <= 3) console.error('probe fail:', String(e).slice(0, 160)); }
    }
  }));
  flushLedger();
  if (fail) console.error(`[${tag}] ${fail}/${tasks.length} probes failed`);
  return recs;
}

export async function pmap(items, fn, concurrency = 12) {
  const out = new Array(items.length); let i = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (i < items.length) { const k = i++; try { out[k] = await fn(items[k], k); } catch (e) { if (/budget/.test(String(e))) throw e; out[k] = null; } }
  }));
  flushLedger();
  return out;
}

// Record spend made through the pipeline's own client (src/llm.mjs stats()).
export function chargeExternal(tag, cost, calls) { charge(tag, cost, calls); flushLedger(); }

// ---------------- free-tier client (":free" models; 20 req/min, daily cap) ----------------
// Used for the rebuttal analyses that need an LLM or encoder from a family that
// took no part in the pipeline. Throttled, cached, and never billed.
let lastFree = 0;
async function freePost(pathname, body, { retries = 6 } = {}) {
  if (!/:free$/.test(body.model)) throw new Error('freePost called with a paid model: ' + body.model);
  let lastErr;
  for (let a = 0; a <= retries; a++) {
    const wait = lastFree + 3300 - Date.now(); if (wait > 0) await sleep(wait); lastFree = Date.now();
    try {
      const res = await fetch(BASE_URL + pathname, { method: 'POST', headers: { Authorization: 'Bearer ' + API_KEY, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const json = await res.json();
      if (json.error || !res.ok) { lastErr = new Error('free API error ' + res.status + ': ' + JSON.stringify(json.error || json).slice(0, 200)); if (/per-day|daily/i.test(String(lastErr))) throw lastErr; await sleep(8000 * (a + 1)); continue; }
      charge('free:' + body.model, 0);
      return json;
    } catch (e) { lastErr = e; if (/per-day|daily/i.test(String(e))) throw e; await sleep(8000 * (a + 1)); }
  }
  throw lastErr;
}
export async function freeChat(prompt, { model, temperature = 0, maxTokens = 400 } = {}) {
  const body = { model, messages: [{ role: 'user', content: prompt }], temperature, max_tokens: maxTokens };
  const ck = JSON.stringify(body);
  const hit = cacheGet('rb_chat', ck); if (hit !== undefined) return hit;
  const res = await freePost('/chat/completions', body);
  const content = res.choices?.[0]?.message?.content;
  if (!content) throw new Error('empty completion from ' + model);
  return cacheSet('rb_chat', ck, content);
}
export async function freeEmbed(texts, model, batch = 64) {
  const out = new Array(texts.length), missing = [];
  texts.forEach((t, i) => { const hit = cacheGet('embed', model + '\n' + t); if (hit !== undefined) out[i] = hit; else missing.push(i); });
  for (let b = 0; b < missing.length; b += batch) {
    const idx = missing.slice(b, b + batch);
    const res = await freePost('/embeddings', { model, input: idx.map(i => texts[i]) });
    idx.forEach((i, k) => { out[i] = cacheSet('embed', model + '\n' + texts[i], res.data[k].embedding); });
  }
  return out.map(normalize);
}
