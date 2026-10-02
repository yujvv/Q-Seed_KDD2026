// Rebuttal E8 (all four reviewers on realism circularity): realism measured in
// representations the S4 selector never sees.
//  A. hand-built style features DISJOINT from the nine S4 features
//  B. function-word skeleton n-grams (content words masked -> topic removed,
//     features learned from data rather than hand-picked)
//  C. LLM judges from families that took no part in generation or S4
//     (absolute human-likelihood score -> AUC vs real WildChat prompts; and
//      blind pairwise preference Q-SEED vs each baseline)
// A and B are free. C issues ~3.5k short judge calls.
import { loadVertical, VERTICALS, CONFIG, save, rbChat, pmap, spent } from './common.mjs';
import { fitLogReg, calibrateRealism } from '../src/calibrate.mjs';
import { lazyGreedy } from '../src/select.mjs';
import * as BL from '../src/baselines.mjs';
import { rng, shuffle, auc, mean } from '../src/util.mjs';

// ---------- A: disjoint style features ----------
const FUNC = new Set('the a an of to in on at and or but that this these those it its be been being was were am have has had will would could may might must shall not no so if then as by from about into over after before up down out off again also very too more most some any each such only own same just than'.split(' '));
const POLITE = /\b(please|thanks|thank you|could you|can you|would you|help me)\b/;
const HEDGE = /\b(just|really|actually|kind of|sort of|maybe|probably|basically|honestly|anyone|any ideas?)\b/;
export function style2(q) {
  const s = q.toLowerCase(), words = s.split(/\s+/).filter(Boolean), wc = words.length || 1;
  const toks = words.map(w => w.replace(/[^a-z0-9']/g, ''));
  return [
    /^[a-z]/.test(q) ? 1 : 0,                                   // starts lowercase
    /[.?!]$/.test(q.trim()) ? 0 : 1,                             // no terminal punctuation
    toks.filter(w => FUNC.has(w)).length / wc,                   // function-word rate
    (q.match(/,/g) || []).length / wc,                           // comma density
    /\b\w+(n't|'s|'re|'ll|'d)\b/.test(s) ? 1 : 0,                // contraction (not i'm / i've)
    q.split(/[.?!]+/).filter(x => x.trim()).length,              // sentence count
    new Set(toks).size / wc,                                     // type-token ratio
    POLITE.test(s) ? 1 : 0,                                      // politeness / request marker
    HEDGE.test(s) ? 1 : 0,                                       // hedge / filler
    /[;:()"]/.test(q) ? 1 : 0,                                   // clause punctuation
    /\byou(r)?\b/.test(s) ? 1 : 0,                               // second person
    /\b(and|or|but|so|because)\b/.test(s) ? 1 : 0,               // clause conjunction
  ];
}
// ---------- B: function-word skeleton ----------
const KEEP = new Set([...FUNC, 'i', 'my', 'me', 'we', 'our', 'you', 'your', 'what', 'which', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'do', 'does', 'can', 'should', 'for', 'with', 'without', 'under', 'between', 'vs', 'there', 'here', 'need', 'want', 'looking', 'best', 'good', 'please', 'help']);
export function skeleton(q) {
  const out = ['<s>'];
  for (const raw of q.toLowerCase().split(/\s+/).filter(Boolean)) {
    const punct = raw.match(/[?.!,;:]+$/)?.[0];
    const w = raw.replace(/[^a-z0-9']/g, '');
    if (w) out.push(KEEP.has(w) ? w : /^\d/.test(w) ? 'NUM' : 'X');
    if (punct) out.push(punct[0]);
  }
  out.push('</s>');
  // collapse runs of masked content words: length of a noun phrase is topic-ish detail
  return out.filter((t, i) => !(t === 'X' && out[i - 1] === 'X'));
}
function ngramVocab(docs, max = 250) {
  const df = new Map();
  for (const d of docs) { const seen = new Set(); for (let i = 0; i < d.length; i++) { seen.add(d[i]); if (i + 1 < d.length) seen.add(d[i] + ' ' + d[i + 1]); } for (const g of seen) df.set(g, (df.get(g) || 0) + 1); }
  return [...df.entries()].filter(([, c]) => c >= 3).sort((a, b) => b[1] - a[1]).slice(0, max).map(([g]) => g);
}
function vec(d, vocab, index) {
  const v = new Array(vocab.length).fill(0);
  for (let i = 0; i < d.length; i++) { if (d[i] in index) v[index[d[i]]] += 1; if (i + 1 < d.length) { const g = d[i] + ' ' + d[i + 1]; if (g in index) v[index[g]] += 1; } }
  const n = d.length || 1; return v.map(x => x / n);
}
// held-out AUC of a real-vs-synthetic discriminator, averaged over repeated 70/30 splits
function heldOutAUC(realRows, synRows, featurize, seeds, opts) {
  const aucs = [];
  for (const seed of seeds) {
    const rand = rng(seed);
    const R = shuffle(realRows, rand), Sy = shuffle(synRows, rand);
    const cr = Math.floor(R.length * 0.7), cs = Math.floor(Sy.length * 0.7);
    const f = featurize(R.slice(0, cr).concat(Sy.slice(0, cs)));
    const clf = fitLogReg(R.slice(0, cr).map(f).concat(Sy.slice(0, cs).map(f)), R.slice(0, cr).map(() => 1).concat(Sy.slice(0, cs).map(() => 0)), opts);
    aucs.push(auc(R.slice(cr).map(x => clf.predict(f(x))).concat(Sy.slice(cs).map(x => clf.predict(f(x)))), R.slice(cr).map(() => 1).concat(Sy.slice(cs).map(() => 0))));
  }
  return mean(aucs);
}
const SEEDS = Array.from({ length: 20 }, (_, i) => 100 + i);
const featA = () => style2;
const featB = (train) => { const vocab = ngramVocab(train.map(skeleton)); const index = Object.fromEntries(vocab.map((g, i) => [g, i])); return (q) => vec(skeleton(q), vocab, index); };

// ---------- C: LLM judges ----------
const JUDGES = (process.env.RB_JUDGES || 'anthropic/claude-haiku-4.5,google/gemini-2.5-flash').split(',');
const absPrompt = (q) => `Below is a single message that was sent to an AI search assistant (such as Perplexity or ChatGPT Search).
Some messages in this study were typed by real people; others were written by a language model imitating a user.
Message: """${q}"""
How likely is it that this message was typed by a real person? Reply with ONLY an integer from 0 (certainly model-written) to 100 (certainly a real person).`;
const pairPrompt = (a, b) => `Two messages sent to an AI search assistant are shown. Decide which one is more likely to have been typed by a real person (as opposed to written by a language model or filled into a template).
A: """${a}"""
B: """${b}"""
Reply with ONLY the letter A or B.`;
const num = (s) => { const m = String(s ?? '').match(/\d{1,3}/); return m ? Math.min(100, +m[0]) : null; };

const out = {}; const DO_JUDGE = process.env.RB_NOJUDGE !== '1';
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const real = shuffle(S.realPool, rng(4)).slice(0, 1200);
  const banks = {
    'KW-as-Query': BL.kwAsQuery(S.bank), 'PAA-Template': BL.paaTemplate(S.bank), 'Naive Paraphrase': await BL.naiveParaphrase(S.bank),
    'Forward Fan-out': await BL.forwardFanout(S.bank), 'GEO-bench-like': await BL.geoBenchLike(S.bank), 'Q-SEED': S.finalQueries,
    'Q-SEED w/o S4': lazyGreedy(S.keptItems, S.tree.clusters, CONFIG.B, 0).selectedIds.map(S.qById),
    'S2 pool (unselected)': shuffle(S.candQueries, rng(11)).slice(0, 300),
  };
  const res = {};
  for (const [name, qs] of Object.entries(banks)) {
    res[name] = {
      n: qs.length,
      s4_auc: mean([4, 5, 6, 7, 8, 9, 10, 11, 12, 13].map(seed => calibrateRealism(qs, S.realPool, { seed }).discAUC)), // the paper's metric, 10 seeds
      disjoint_auc: heldOutAUC(real, qs, featA, SEEDS, { l2: 1e-2 }),
      skeleton_auc: heldOutAUC(real, qs, featB, SEEDS.slice(0, 8), { l2: 3e-2, iters: 250 }),
    };
  }
  if (DO_JUDGE) {
    const realJ = shuffle(S.realPool, rng(77)).slice(0, 150);
    for (const J of JUDGES) {
      const key = J.split('/')[1];
      const realScores = (await pmap(realJ, q => rbChat(absPrompt(q), { model: J, maxTokens: 8, tag: 'judge' }), 12)).map(num).filter(x => x !== null);
      for (const [name, qs] of Object.entries(banks)) {
        if (name === 'S2 pool (unselected)') continue;
        const sc = (await pmap(qs, q => rbChat(absPrompt(q), { model: J, maxTokens: 8, tag: 'judge' }), 12)).map(num).filter(x => x !== null);
        (res[name].judge ||= {})[key] = { meanScore: mean(sc), n: sc.length, auc: auc(realScores.concat(sc), realScores.map(() => 1).concat(sc.map(() => 0))) };
      }
      res.__real = { ...(res.__real || {}), [key]: { meanScore: mean(realScores), n: realScores.length } };
      // blind pairwise: Q-SEED vs each baseline, order randomised per pair
      for (const name of ['PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED w/o S4']) {
        const rand = rng(CONFIG.seed + name.length * 13);
        const A = shuffle(banks['Q-SEED'], rand), Bq = shuffle(banks[name], rand), n = Math.min(A.length, Bq.length);
        const pairs = Array.from({ length: n }, (_, i) => ({ q: A[i], b: Bq[i], first: rand() < 0.5 }));
        const ans = await pmap(pairs, p => rbChat(p.first ? pairPrompt(p.q, p.b) : pairPrompt(p.b, p.q), { model: J, maxTokens: 4, tag: 'judge' }), 12);
        let win = 0, tot = 0;
        ans.forEach((a, i) => { const m = String(a ?? '').trim().toUpperCase()[0]; if (m !== 'A' && m !== 'B') return; tot++; if ((m === 'A') === pairs[i].first) win++; });
        (res[name].pairwise ||= {})[key] = { qseedWins: win, n: tot };
      }
      console.error(v, J, 'done; spent $' + spent().toFixed(3));
    }
  }
  out[v] = res;
}
save('e8_realism.json', out);
const names = Object.keys(out[VERTICALS[0]]).filter(n => n !== '__real');
const a = (n, f) => { const xs = VERTICALS.map(v => f(out[v][n])).filter(x => x !== undefined); return xs.length ? mean(xs) : NaN; };
console.log('method | S4-space AUC (paper metric) | disjoint-feature AUC | skeleton n-gram AUC' + (DO_JUDGE ? ' | ' + JUDGES.map(j => j.split('/')[1] + ' AUC / mean score').join(' | ') : ''));
for (const n of names) console.log([n, a(n, x => x.s4_auc).toFixed(3), a(n, x => x.disjoint_auc).toFixed(3), a(n, x => x.skeleton_auc).toFixed(3), ...(DO_JUDGE ? JUDGES.map(j => { const k = j.split('/')[1]; return a(n, x => x.judge?.[k]?.auc).toFixed(3) + ' / ' + a(n, x => x.judge?.[k]?.meanScore).toFixed(1); }) : [])].join(' | '));
if (DO_JUDGE) {
  console.log('real WildChat mean score:', JUDGES.map(j => { const k = j.split('/')[1]; return k + '=' + mean(VERTICALS.map(v => out[v].__real[k].meanScore)).toFixed(1); }).join(' '));
  console.log('\npairwise: Q-SEED preferred over ...');
  for (const n of names) { if (!out[VERTICALS[0]][n].pairwise) continue; console.log(n, JUDGES.map(j => { const k = j.split('/')[1]; const w = VERTICALS.reduce((s, v) => s + out[v][n].pairwise[k].qseedWins, 0), t = VERTICALS.reduce((s, v) => s + out[v][n].pairwise[k].n, 0); return `${k}: ${w}/${t} = ${(100 * w / t).toFixed(0)}%`; }).join(' | ')); }
}
