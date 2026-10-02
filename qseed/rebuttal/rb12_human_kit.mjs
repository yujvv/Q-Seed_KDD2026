// Builds a blinded human-evaluation kit for query realism (no API calls).
//   Part A  "real or model-written?"   120 single queries, shuffled
//           40 real WildChat search prompts / 40 Q-SEED / 20 Naive Paraphrase / 20 Forward Fan-out
//   Part B  "which was typed by a real person?"  60 pairs, Q-SEED vs a baseline, side randomised
// Annotators get only items_A.csv / pairs_B.csv; key.json stays with the authors.
// After collecting answers:  node rebuttal/rb12_human_kit.mjs analyze <answers_dir>
import fs from 'node:fs';
import path from 'node:path';
import { loadVertical, VERTICALS, CONFIG } from './common.mjs';
import { ROOT } from '../src/config.mjs';
import * as BL from '../src/baselines.mjs';
import { rng, shuffle, mean } from '../src/util.mjs';

const DIR = path.resolve(ROOT, '..', 'rebuttal', 'human_study');
const csv = (rows) => rows.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(',')).join('\r\n') + '\r\n';
const parseCsv = (txt) => txt.split(/\r?\n/).filter(Boolean).map(l => { const out = []; let cur = '', q = false; for (let i = 0; i < l.length; i++) { const c = l[i]; if (q) { if (c === '"' && l[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; } else if (c === '"') q = true; else if (c === ',') { out.push(cur); cur = ''; } else cur += c; } out.push(cur); return out; });

if (process.argv[2] === 'analyze') {
  // answers_dir (default: the kit folder) holds one pair of files per annotator: the filled-in
  // items_A*.csv (answer H|M in the last column) and pairs_B*.csv (answer A|B in the last column)
  const dir = process.argv[3] || DIR, key = JSON.parse(fs.readFileSync(path.join(DIR, 'key.json'), 'utf8'));
  const files = fs.readdirSync(dir);
  const A = {}, Bw = {}; const perAnn = [];
  const sum = (xs) => xs.reduce((s, x) => s + x, 0);
  for (const f of files.filter(f => /_A(_[\w-]+)?\.csv$/i.test(f))) {
    let ok = 0, n = 0;
    for (const row of parseCsv(fs.readFileSync(path.join(dir, f), 'utf8')).slice(1)) {
      const id = row[0], ans = row[row.length - 1].trim();
      const src = key.A[id]; if (!src || !/^[HM]$/i.test(ans)) continue;
      const saidHuman = /^H/i.test(ans); (A[src] ||= []).push(+saidHuman); n++; if (saidHuman === (src === 'real')) ok++;
    }
    if (n) perAnn.push({ file: f, accuracy: ok / n, n });
  }
  for (const f of files.filter(f => /_B(_[\w-]+)?\.csv$/i.test(f))) for (const row of parseCsv(fs.readFileSync(path.join(dir, f), 'utf8')).slice(1)) {
    const id = row[0], ans = row[row.length - 1].trim();
    const k = key.B[id]; if (!k || !/^[AB]$/i.test(ans)) continue; (Bw[k.baseline] ||= []).push(+(ans.toUpperCase() === k.qseedSide));
  }
  console.log('Part A - judged "typed by a real person", by true source:'); for (const [s, xs] of Object.entries(A)) console.log('  ', s, sum(xs) + '/' + xs.length, '=', (100 * mean(xs)).toFixed(1) + '%');
  console.log('  annotator accuracy:', perAnn.map(p => p.file + ' ' + (100 * p.accuracy).toFixed(1) + '% (n=' + p.n + ')').join(', '));
  console.log('Part B - Q-SEED chosen as the real one over:'); for (const [b, xs] of Object.entries(Bw)) console.log('  ', b, sum(xs) + '/' + xs.length, '=', (100 * mean(xs)).toFixed(1) + '%');
  const all = Object.values(Bw).flat(); console.log('   all baselines', sum(all) + '/' + all.length, '=', (100 * mean(all)).toFixed(1) + '%');
  process.exit(0);
}

fs.mkdirSync(DIR, { recursive: true });
const rand = rng(CONFIG.seed + 2027);
const A = [], B = [];
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const take = (xs, n) => shuffle(xs, rand).slice(0, n);
  const bl = { 'PAA-Template': BL.paaTemplate(S.bank), 'Naive Paraphrase': await BL.naiveParaphrase(S.bank), 'Forward Fan-out': await BL.forwardFanout(S.bank), 'GEO-bench-like': await BL.geoBenchLike(S.bank) };
  const qs = shuffle(S.finalQueries, rand);
  for (const q of qs.slice(0, 14)) A.push({ src: 'Q-SEED', q });
  for (const q of take(bl['Naive Paraphrase'], 7)) A.push({ src: 'Naive Paraphrase', q });
  for (const q of take(bl['Forward Fan-out'], 7)) A.push({ src: 'Forward Fan-out', q });
  if (v === VERTICALS[0]) for (const q of take(S.realPool.filter(t => t.length <= 200), 40)) A.push({ src: 'real', q });
  let i = 14;
  for (const [name, list] of Object.entries(bl)) for (const b of take(list, 5)) { const side = rand() < 0.5 ? 'A' : 'B'; const q = qs[i++]; B.push({ baseline: name, qseedSide: side, A: side === 'A' ? q : b, B: side === 'A' ? b : q }); }
}
const As = shuffle(A, rand).map((x, i) => ({ id: 'A' + String(i + 1).padStart(3, '0'), ...x }));
const Bs = shuffle(B, rand).map((x, i) => ({ id: 'B' + String(i + 1).padStart(3, '0'), ...x }));
fs.writeFileSync(path.join(DIR, 'items_A.csv'), csv([['id', 'query', 'answer (H = typed by a real person, M = written by a language model)'], ...As.map(x => [x.id, x.q, ''])]));
fs.writeFileSync(path.join(DIR, 'pairs_B.csv'), csv([['id', 'query A', 'query B', 'answer (A or B: which was typed by a real person?)'], ...Bs.map(x => [x.id, x.A, x.B, ''])]));
fs.writeFileSync(path.join(DIR, 'key.json'), JSON.stringify({ A: Object.fromEntries(As.map(x => [x.id, x.src])), B: Object.fromEntries(Bs.map(x => [x.id, { baseline: x.baseline, qseedSide: x.qseedSide }])) }, null, 1));
console.log('wrote', As.length, 'single items and', Bs.length, 'pairs to', DIR, '| sources', JSON.stringify(As.reduce((h, x) => (h[x.src] = (h[x.src] || 0) + 1, h), {})));
