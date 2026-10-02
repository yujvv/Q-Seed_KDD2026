import fs from 'node:fs'; import path from 'node:path';
import { API_KEY, BASE_URL, ROOT } from '../src/config.mjs';
import { parse as tldParse } from 'tldts';
const q = "What are the best noise cancelling headphones for long flights under $300?";
const cands = [
  { model: 'openai/gpt-4o-mini-search-preview' },
  { model: 'openai/gpt-4.1-mini', plugins: [{ id: 'web', engine: 'native' }] },
  { model: 'openai/gpt-5-mini', plugins: [{ id: 'web', engine: 'native' }] },
  { model: 'google/gemini-2.5-flash', plugins: [{ id: 'web', engine: 'native' }] },
  { model: 'google/gemini-2.5-flash-lite', plugins: [{ id: 'web', engine: 'native' }] },
  { model: 'anthropic/claude-haiku-4.5', plugins: [{ id: 'web', engine: 'native' }] },
];
let total = 0;
for (const c of cands) {
  const body = { ...c, messages: [{ role: 'user', content: q }], max_tokens: 600, usage: { include: true } };
  if (!/search-preview|gpt-5/.test(c.model)) body.temperature = 0.4;
  const t0 = Date.now();
  const r = await fetch(BASE_URL + '/chat/completions', { method: 'POST', headers: { Authorization: 'Bearer ' + API_KEY, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  const tag = c.model + (c.plugins ? '+native' : '');
  if (j.error) { console.log(tag, 'ERROR', JSON.stringify(j.error).slice(0, 220)); continue; }
  fs.writeFileSync(path.join(ROOT, 'results', 'rebuttal', 'rawn_' + c.model.replace(/[\/:]/g, '_') + '.json'), JSON.stringify(j, null, 1));
  const ann = j.choices?.[0]?.message?.annotations || [];
  const d = [...new Set(ann.map(a => tldParse(a?.url_citation?.url || '').domain).filter(Boolean))];
  total += j.usage?.cost || 0;
  console.log(tag, '| provider', j.provider, '| cost', j.usage?.cost, '| sec', ((Date.now() - t0) / 1000).toFixed(1), '| nAnn', ann.length, '| domains', d.length, d.join(','), '| usage', JSON.stringify(j.usage?.server_tool_use || j.usage?.completion_tokens_details || {}));
}
console.log('total $' + total.toFixed(4));
