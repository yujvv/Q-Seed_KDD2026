// Which retrieval backend did `openai/gpt-4o-mini:online` use in the submitted
// run? Re-issue a few bank queries with the web plugin pinned to `exa` and to
// `native`, and compare the cited domains with the cached (submitted) probes.
// Also dump one raw response per candidate third engine to see cost and whether
// the engine exposes its actual search queries.
import fs from 'node:fs';
import path from 'node:path';
import { loadVertical, save, OUT, flushLedger, spent } from './common.mjs';
import { probe } from '../src/llm.mjs';
import { API_KEY, BASE_URL } from '../src/config.mjs';
import { parse as tldParse } from 'tldts';
import { mean } from '../src/util.mjs';

async function raw(body) {
  const res = await fetch(BASE_URL + '/chat/completions', { method: 'POST', headers: { Authorization: 'Bearer ' + API_KEY, 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, usage: { include: true } }) });
  return res.json();
}
const doms = (j) => [...new Set((j.choices?.[0]?.message?.annotations || []).map(a => tldParse(a?.url_citation?.url || '').domain).filter(Boolean))];
const jac = (a, b) => { const A = new Set(a), B = new Set(b); let k = 0; for (const x of A) if (B.has(x)) k++; return (A.size + B.size - k) ? k / (A.size + B.size - k) : null; };

const S = await loadVertical('consumer_electronics', { probes: false });
const qs = S.finalQueries.slice(0, 6);
const variants = {
  default: { model: 'openai/gpt-4o-mini:online' },
  exa: { model: 'openai/gpt-4o-mini', plugins: [{ id: 'web', engine: 'exa' }] },
  native: { model: 'openai/gpt-4o-mini', plugins: [{ id: 'web', engine: 'native' }] },
};
const res = {}; let total = 0;
for (const [name, v] of Object.entries(variants)) {
  const js = [], costs = [], n = [];
  for (const q of qs) {
    const cached = new Set(); for (const rep of [0, 1, 2]) for (const d of (await probe(q, 'openai/gpt-4o-mini:online', rep)).domains) cached.add(d);
    const j = await raw({ ...v, messages: [{ role: 'user', content: q }], temperature: 0.4, max_tokens: 600 });
    if (j.error) { console.log(name, 'ERROR', JSON.stringify(j.error).slice(0, 300)); break; }
    const d = doms(j); js.push(jac(d, [...cached])); costs.push(j.usage?.cost || 0); n.push(d.length); total += j.usage?.cost || 0;
    if (q === qs[0]) fs.writeFileSync(path.join(OUT, 'raw_' + name + '.json'), JSON.stringify(j, null, 1));
  }
  res[name] = { jaccardWithCachedRun: mean(js.filter(x => x !== null)), meanCost: mean(costs), domainsPerAnswer: mean(n), provider: undefined };
  console.log(name, JSON.stringify(res[name]));
}
// candidate third engines: one raw call each
for (const m of (process.env.RB_CANDS || 'google/gemini-2.5-flash:online,anthropic/claude-haiku-4.5:online,x-ai/grok-4-fast:online').split(',')) {
  const j = await raw({ model: m, messages: [{ role: 'user', content: qs[0] }], temperature: 0.4, max_tokens: 600 });
  if (j.error) { console.log(m, 'ERROR', JSON.stringify(j.error).slice(0, 300)); continue; }
  fs.writeFileSync(path.join(OUT, 'raw_' + m.replace(/[\/:]/g, '_') + '.json'), JSON.stringify(j, null, 1));
  total += j.usage?.cost || 0;
  console.log(m, 'provider', j.provider, 'cost', j.usage?.cost, 'domains', doms(j).length, doms(j).join(','), '| msg keys', Object.keys(j.choices?.[0]?.message || {}).join(','));
}
save('e7_engine_check.json', { res, total });
console.log('total cost of this check: $' + total.toFixed(4));
