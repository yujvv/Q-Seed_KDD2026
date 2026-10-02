// Stream the public Bing keyword inventory (HF: vtempest/seo-search-keywords-100M,
// "SEO keywords from Bing collected in 2020", keyphrase + 0-10 search-popularity
// index) and keep every keyphrase that phrase-matches one of our curated head
// terms -- the way a keyword tool builds a bank from a seed list. Nothing is
// generated: keyphrases and popularity values are taken verbatim.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { Readable } from 'node:stream';
import readline from 'node:readline';
import { SEEDS } from '../src/seeds.mjs';
import { ROOT } from '../src/config.mjs';

const STOP = new Set(['for', 'to', 'vs', 'how', 'the', 'a', 'of', 'in']);
const strip = (s) => s.split(' ').filter(w => !STOP.has(w)).join(' ');
const pats = [];
for (const [v, seeds] of Object.entries(SEEDS)) for (const s of seeds) {
  for (const form of new Set([s, strip(s), s.replace(/-/g, ' ')])) pats.push({ v, seed: s, form });
}
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const big = new RegExp('(?:^|[ "])(' + [...new Set(pats.map(p => esc(p.form)))].sort((a, b) => b.length - a.length).join('|') + ')s?(?:$|[ "])');
const byForm = {}; for (const p of pats) (byForm[p.form] ||= []).push(p);

const url = 'https://huggingface.co/datasets/vtempest/seo-search-keywords-100M/resolve/main/keywords.tar.gz';
const res = await fetch(url);
if (!res.ok) throw new Error('HTTP ' + res.status);
const rl = readline.createInterface({ input: Readable.fromWeb(res.body).pipe(zlib.createGunzip()), crlfDelay: Infinity });
const out = Object.fromEntries(Object.keys(SEEDS).map(v => [v, []]));
let n = 0, hits = 0;
const line = /^\["(.*)",(\d+)\],?\s*$/;
for await (const l of rl) {
  n++;
  if (n % 20000000 === 0) console.error(`${(n / 1e6).toFixed(0)}M lines, ${hits} hits`);
  const m = big.exec(l);
  if (!m) continue;
  const lm = line.exec(l);
  if (!lm) continue;
  for (const p of byForm[m[1]] || []) { out[p.v].push({ kw: lm[1], pop: +lm[2], seed: p.seed }); hits++; }
}
const dir = path.join(ROOT, 'data', 'bing_real'); fs.mkdirSync(dir, { recursive: true });
for (const [v, rows] of Object.entries(out)) fs.writeFileSync(path.join(dir, v + '.json'), JSON.stringify(rows));
console.error('done', n, 'lines;', Object.entries(out).map(([v, r]) => v + '=' + r.length).join(' '));
