// Rebuttal E8-C: LLM-as-judge realism, with a judge from a model family that took
// no part in generation (OpenAI), fan-out (Google) or S4 (no LLM).
// Free-tier judge, so items are scored in shuffled, source-mixed batches.
//  abs : P(typed by a real person) per query -> AUC vs real WildChat prompts
//  pair: blind preference, Q-SEED vs a baseline query, side randomised
import { loadVertical, VERTICALS, CONFIG, save, freeChat } from './common.mjs';
import { lazyGreedy } from '../src/select.mjs';
import * as BL from '../src/baselines.mjs';
import { rng, shuffle, auc, mean } from '../src/util.mjs';

const JUDGE = process.env.RB_JUDGE || 'qwen/qwen3.8-27b:free';
const KEY = JUDGE.split('/')[1].replace(':free', '');
const BATCH = 25, NPAIR = Number(process.env.RB_NPAIR || 30), NREAL = 50;
const MODE = new Set((process.env.RB_MODE || 'abs,pair').split(','));

const absPrompt = (qs) => 'Each message below was sent to an AI search assistant (such as Perplexity or ChatGPT Search). Some were typed by real people; others were written by a language model or filled into a template to imitate a user.\n' +
  'For each message, estimate the probability (0-100) that it was typed by a real person.\n\n' +
  qs.map((q, i) => (i + 1) + '. """' + q + '"""').join('\n') +
  '\n\nReply with ONLY a JSON array of ' + qs.length + ' integers, one per message, in order.';
const pairPrompt = (ps) => 'Each numbered item shows two messages (A and B) sent to an AI search assistant. For each item, decide which message is more likely to have been typed by a real person, as opposed to written by a language model or filled into a template.\n\n' +
  ps.map((p, i) => (i + 1) + '.\nA: """' + p.A + '"""\nB: """' + p.B + '"""').join('\n') +
  '\n\nReply with ONLY a JSON array of ' + ps.length + ' strings, each "A" or "B", in order.';
async function ask(prompt, n) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const raw = await freeChat(prompt + (attempt ? '\n' : ''), { model: JUDGE, maxTokens: 4000 });
      const m = raw.match(/\[[\s\S]*\]/); const arr = m ? JSON.parse(m[0]) : null;
      if (Array.isArray(arr) && arr.length === n) return arr;
    } catch (e) { console.error('judge fail', String(e).slice(0, 140)); }
  }
  return null;
}

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const banks = {
    'KW-as-Query': BL.kwAsQuery(S.bank), 'PAA-Template': BL.paaTemplate(S.bank), 'Naive Paraphrase': await BL.naiveParaphrase(S.bank),
    'Forward Fan-out': await BL.forwardFanout(S.bank), 'GEO-bench-like': await BL.geoBenchLike(S.bank), 'Q-SEED': S.finalQueries,
    'Q-SEED w/o S4': lazyGreedy(S.keptItems, S.tree.clusters, CONFIG.B, 0).selectedIds.map(S.qById),
  };
  const res = { judge: KEY };
  if (MODE.has('abs')) {
    const real = shuffle(S.realPool, rng(77 + VERTICALS.indexOf(v))).slice(0, NREAL);
    const items = shuffle([...real.map(q => ({ src: 'real', q })), ...Object.entries(banks).flatMap(([src, qs]) => qs.map(q => ({ src, q })))], rng(CONFIG.seed + 5150));
    const scores = {};
    for (let i = 0; i < items.length; i += BATCH) {
      const chunk = items.slice(i, i + BATCH);
      const arr = await ask(absPrompt(chunk.map(x => x.q)), chunk.length);
      if (arr) chunk.forEach((x, k) => { const s = Number(arr[k]); if (Number.isFinite(s)) (scores[x.src] ||= []).push(s); });
      console.error(v, 'abs', Math.min(i + BATCH, items.length) + '/' + items.length, arr ? 'ok' : 'FAILED');
    }
    res.abs = Object.fromEntries(Object.entries(scores).map(([src, xs]) => [src, { n: xs.length, meanScore: mean(xs), auc: src === 'real' ? null : auc((scores.real || []).concat(xs), (scores.real || []).map(() => 1).concat(xs.map(() => 0))) }]));
    console.error(v, JSON.stringify(res.abs));
  }
  if (MODE.has('pair')) {
    const pairs = [];
    for (const name of ['PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED w/o S4']) {
      const rand = rng(CONFIG.seed + name.length * 13);
      const A = shuffle(banks['Q-SEED'], rand), Bq = shuffle(banks[name], rand);
      for (let i = 0; i < Math.min(NPAIR, A.length, Bq.length); i++) { const first = rand() < 0.5; pairs.push({ name, first, A: first ? A[i] : Bq[i], B: first ? Bq[i] : A[i] }); }
    }
    const sh = shuffle(pairs, rng(CONFIG.seed + 99));
    const tally = {};
    for (let i = 0; i < sh.length; i += BATCH) {
      const chunk = sh.slice(i, i + BATCH);
      const arr = await ask(pairPrompt(chunk), chunk.length);
      if (arr) chunk.forEach((p, k) => { const a = String(arr[k]).trim().toUpperCase()[0]; if (a !== 'A' && a !== 'B') return; const t = (tally[p.name] ||= { qseedWins: 0, n: 0 }); t.n++; if ((a === 'A') === p.first) t.qseedWins++; });
      console.error(v, 'pair', Math.min(i + BATCH, sh.length) + '/' + sh.length, arr ? 'ok' : 'FAILED');
    }
    res.pair = tally; console.error(v, JSON.stringify(tally));
  }
  out[v] = res; save('e8c_judge_' + KEY + '.json', out);
}
const names = ['KW-as-Query', 'PAA-Template', 'Naive Paraphrase', 'Forward Fan-out', 'GEO-bench-like', 'Q-SEED', 'Q-SEED w/o S4'];
if (MODE.has('abs')) { console.log('judge', KEY, '| real mean score', mean(VERTICALS.map(v => out[v].abs.real.meanScore)).toFixed(1)); for (const n of names) console.log(n, '| mean score', mean(VERTICALS.map(v => out[v].abs[n].meanScore)).toFixed(1), '| AUC vs real', mean(VERTICALS.map(v => out[v].abs[n].auc)).toFixed(3), '| n', VERTICALS.map(v => out[v].abs[n].n).join('/')); }
if (MODE.has('pair')) for (const n of names) { const t = VERTICALS.map(v => out[v].pair[n]).filter(Boolean); if (!t.length) continue; const w = t.reduce((s, x) => s + x.qseedWins, 0), tot = t.reduce((s, x) => s + x.n, 0); console.log('Q-SEED preferred over', n, w + '/' + tot, '=', (100 * w / tot).toFixed(0) + '%'); }
