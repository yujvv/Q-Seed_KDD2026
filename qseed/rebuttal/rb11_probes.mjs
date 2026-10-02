// Rebuttal E11: all NEW engine probes, issued in one session so that every
// comparison is same-time (the submitted probes are ~11 weeks old).
//  A. the two submitted engines re-probed on the delivered banks (1 repeat)
//     -> 11-week test-retest; same-time reference for B and C
//  A'. three further engines on the first 20 queries of each bank:
//     OpenAI native search, Anthropic native search (two more retrieval stacks),
//     and Exa retrieval with a DIFFERENT generator (retrieval held fixed)
//  B. the banks built from the real Bing keyword inventory (rb6)
//  C. real user queries (MS MARCO, Bing logs) on the same head terms
import { loadVertical, VERTICALS, CONFIG, save, load, rbProbeQueries, spent, flushLedger } from './common.mjs';
import { domainVisibility, buildDomainUniverse } from '../src/probe.mjs';
import { cacheGet, cacheSet } from '../src/cache.mjs';
import { SEEDS } from '../src/seeds.mjs';
import { spearman, kendallTau, rng, shuffle, mean } from '../src/util.mjs';

const SONAR = { id: 'sonar', model: 'perplexity/sonar' };
const EXA4O = { id: 'gpt4o-online', model: 'openai/gpt-4o-mini:online' };           // = Exa retrieval + GPT-4o-mini (submitted engine)
const EXAGEM = { id: 'exa-gemini', model: 'google/gemini-2.5-flash-lite:online' };   // = Exa retrieval + a different generator
const OAI = { id: 'openai-native', model: 'openai/gpt-4.1-mini', extra: { plugins: [{ id: 'web', engine: 'native' }] } };
const ANT = { id: 'anthropic-native', model: 'anthropic/claude-haiku-4.5', extra: { plugins: [{ id: 'web', engine: 'native' }] } };
const TWO = [SONAR, EXA4O];
const PART = new Set((process.env.RB_PARTS || 'A,A2,B,C').split(','));
const NSUB = 20;

const vis = (recs, D, engs) => domainVisibility(recs, D, engs).map(x => x.rate);
const setOf = (cell) => new Set(Object.keys(cell.domainCounts));
const jac = (A, B) => { if (!A.size && !B.size) return null; let k = 0; for (const x of A) if (B.has(x)) k++; return k / (A.size + B.size - k); };
const mj = (xs) => { const y = xs.filter(x => x !== null); return y.length ? mean(y) : null; };
const universe = (recSets, engs, minQ = 3, maxD = 60) => buildDomainUniverse(recSets.flat(), engs, { minQueries: minQ, maxD });
function splitHalfSB(recs, D, engs, n = 200, seed = 9100) {
  const xs = [];
  for (let k = 0; k < n; k++) { const o = shuffle(recs.map((_, i) => i), rng(seed + k)), h = Math.floor(o.length / 2); const c = spearman(vis(o.slice(0, h).map(i => recs[i]), D, engs), vis(o.slice(h, 2 * h).map(i => recs[i]), D, engs)); xs.push(2 * c / (1 + c)); }
  return mean(xs);
}
// real user queries for a vertical: MS MARCO (Bing log) queries retrieved by head term
async function msmarcoQueries(v, target = 30) {
  const hit = cacheGet('rb_msmarco', v + ':' + target + ':v1'); if (hit) return hit;
  const picked = [], seen = new Set();
  for (const head of SEEDS[v]) {
    if (picked.length >= target) break;
    const words = head.split(' ').filter(w => !['for', 'to', 'vs', 'how', 'the', 'a'].includes(w)).map(w => w.replace(/s$/, ''));
    let rows = [];
    for (let attempt = 0; attempt < 3 && !rows.length; attempt++) {
      try {
        const r = await fetch(`https://datasets-server.huggingface.co/search?dataset=BeIR/msmarco&config=queries&split=queries&query=${encodeURIComponent(head)}&offset=0&length=40`, { signal: AbortSignal.timeout(25000) });
        const j = await r.json(); if (j.rows) { rows = j.rows; break; }
      } catch { }
      await new Promise(r => setTimeout(r, 4000));
    }
    const cand = rows.map(r => String(r.row.text).trim()).filter(t => { const s = t.toLowerCase(); return words.every(w => s.includes(w)) && t.split(' ').length >= 4 && t.length <= 160 && !seen.has(s); });
    if (cand.length) { picked.push({ head, q: cand[0] }); seen.add(cand[0].toLowerCase()); }
  }
  if (picked.length < 15) throw new Error('MS MARCO search returned only ' + picked.length + ' queries for ' + v + ' (index not ready?); not caching');
  return cacheSet('rb_msmarco', v + ':' + target + ':v1', picked);
}

const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v);
  const r = {};
  // ---------------- A: submitted engines, now ----------------
  const now = await rbProbeQueries(S.finalQueries, TWO, { repeats: 1, tag: 'A_now' });
  const D = S.domains; // the submitted 60-domain universe
  r.retest11wk = {};
  for (const e of TWO) {
    const a = vis(S.recsFinal, D, [e]), b = vis(now, D, [e]);
    r.retest11wk[e.id] = { spearman: spearman(a, b), kendall: kendallTau(a, b), perQueryJaccard: mj(S.recsFinal.map((rec, i) => jac(setOf(rec[e.id]), setOf(now[i][e.id])))) };
  }
  { const a = vis(S.recsFinal, D, TWO), b = vis(now, D, TWO); r.retest11wk.pooled = { spearman: spearman(a, b), kendall: kendallTau(a, b) }; }
  r.crossEngineNow = spearman(vis(now, D, [SONAR]), vis(now, D, [EXA4O]));
  console.error(v, 'A', JSON.stringify(r.retest11wk), 'cross now', r.crossEngineNow.toFixed(3), '$' + spent().toFixed(2));

  // ---------------- A': more engines on the first 20 queries ----------------
  if (PART.has('A2')) {
    const sub = S.finalQueries.slice(0, NSUB);
    const oai = await rbProbeQueries(sub, [OAI], { repeats: 2, tag: 'A2_openai_native', concurrency: 8 });
    const ant = await rbProbeQueries(sub, [ANT], { repeats: 1, tag: 'A2_anthropic_native', concurrency: 6 });
    const exg = await rbProbeQueries(sub, [EXAGEM], { repeats: 1, tag: 'A2_exa_gemini' });
    const E5 = [SONAR, EXA4O, EXAGEM, OAI, ANT];
    const merged = sub.map((_, i) => ({ ...now[i], ...oai[i], ...ant[i], ...exg[i] }));
    const U = buildDomainUniverse(merged, E5, { minQueries: 2, maxD: 80 });
    const pair = {};
    for (let a = 0; a < E5.length; a++) for (let b = a + 1; b < E5.length; b++) {
      pair[E5[a].id + ' ~ ' + E5[b].id] = { rho: spearman(vis(merged, U, [E5[a]]), vis(merged, U, [E5[b]])), perQueryJaccard: mj(merged.map(m => jac(setOf(m[E5[a].id]), setOf(m[E5[b].id])))) };
    }
    // within-engine reference: OpenAI native repeat 0 vs repeat 1 (URL lists kept per repeat)
    const dom = (u) => { try { return new URL(u).hostname.replace(/^www\./, '').split('.').slice(-2).join('.'); } catch { return null; } };
    const withinOAI = mj(oai.map(m => { const u = m[OAI.id].urls; return u.length === 2 ? jac(new Set(u[0].map(dom)), new Set(u[1].map(dom))) : null; }));
    r.moreEngines = { nQueries: NSUB, universe: U.length, pair, withinOpenAINativeJaccard: withinOAI, domainsPerAnswer: Object.fromEntries(E5.map(e => [e.id, mean(merged.map(m => mean(m[e.id].nCit.length ? m[e.id].nCit : [0])))])), failed: Object.fromEntries(E5.map(e => [e.id, merged.filter(m => !m[e.id].trials).length])) };
    console.error(v, "A'", JSON.stringify(r.moreEngines), '$' + spent().toFixed(2));
  }

  // ---------------- B: real-keyword-bank ----------------
  if (PART.has('B')) {
    const real = load('e6_realbank_' + v + '.json');
    const rb = await rbProbeQueries(real.finalQueries, TWO, { repeats: 1, tag: 'B_realbank' });
    const U = universe([rb, now], TWO);
    const Ur = universe([rb], TWO);
    r.realBank = {
      universe: U.length,
      splitHalfSB: splitHalfSB(rb, Ur, TWO), splitHalfSB_syntheticNow: splitHalfSB(now, universe([now], TWO), TWO),
      crossEngine: spearman(vis(rb, Ur, [SONAR]), vis(rb, Ur, [EXA4O])),
      agreementWithSyntheticBank: { pooled: spearman(vis(rb, U, TWO), vis(now, U, TWO)), sonar: spearman(vis(rb, U, [SONAR]), vis(now, U, [SONAR])), exa: spearman(vis(rb, U, [EXA4O]), vis(now, U, [EXA4O])) },
      topDomains: Object.fromEntries(TWO.map(e => [e.id, domainVisibility(rb, Ur, [e]).sort((a, b) => b.rate - a.rate).slice(0, 5).map(x => x.domain + ':' + x.rate.toFixed(2))])),
    };
    console.error(v, 'B', JSON.stringify(r.realBank), '$' + spent().toFixed(2));
  }

  // ---------------- C: real user queries (MS MARCO) ----------------
  if (PART.has('C')) {
    const hq = await msmarcoQueries(v, 30);
    const hr = await rbProbeQueries(hq.map(x => x.q), TWO, { repeats: 1, tag: 'C_human' });
    const U = universe([now], TWO);
    const conv = spearman(vis(now, U, TWO), vis(hr, U, TWO));
    const relH = splitHalfSB(hr, U, TWO), relQ = splitHalfSB(now, U, TWO);
    r.humanCriterion = { n: hq.length, examples: hq.slice(0, 6).map(x => x.q), convergent: conv, perEngine: Object.fromEntries(TWO.map(e => [e.id, spearman(vis(now, U, [e]), vis(hr, U, [e]))])), splitHalfHuman: relH, splitHalfBank: relQ, disattenuated: conv / Math.sqrt(Math.max(1e-6, relH * relQ)) };
    console.error(v, 'C', JSON.stringify(r.humanCriterion), '$' + spent().toFixed(2));
    save('e11_human_queries_' + v + '.json', hq);
  }
  out[v] = r; save(process.env.RB_OUT || 'e11_probes.json', out);
}
flushLedger();
const avg = (f) => mean(VERTICALS.map(v => f(out[v])).filter(x => x !== null && x !== undefined && !Number.isNaN(x)));
console.log('11-week retest rho:', TWO.map(e => e.id + '=' + avg(x => x.retest11wk[e.id].spearman).toFixed(2) + ' (tau ' + avg(x => x.retest11wk[e.id].kendall).toFixed(2) + ', per-query J ' + avg(x => x.retest11wk[e.id].perQueryJaccard).toFixed(2) + ')').join('  '), '| pooled', avg(x => x.retest11wk.pooled.spearman).toFixed(2), 'tau', avg(x => x.retest11wk.pooled.kendall).toFixed(2), '| cross-engine now', avg(x => x.crossEngineNow).toFixed(2));
if (out[VERTICALS[0]].moreEngines) { console.log('pairwise engines (rho / per-query Jaccard):'); for (const k of Object.keys(out[VERTICALS[0]].moreEngines.pair)) console.log('  ', k, avg(x => x.moreEngines.pair[k].rho).toFixed(2), '/', avg(x => x.moreEngines.pair[k].perQueryJaccard).toFixed(2)); console.log('   within OpenAI-native J', avg(x => x.moreEngines.withinOpenAINativeJaccard).toFixed(2)); }
if (out[VERTICALS[0]].realBank) console.log('real bank: split-half', avg(x => x.realBank.splitHalfSB).toFixed(2), '(synthetic now', avg(x => x.realBank.splitHalfSB_syntheticNow).toFixed(2) + ') cross-engine', avg(x => x.realBank.crossEngine).toFixed(2), 'agreement with synthetic bank pooled/sonar/exa', avg(x => x.realBank.agreementWithSyntheticBank.pooled).toFixed(2), avg(x => x.realBank.agreementWithSyntheticBank.sonar).toFixed(2), avg(x => x.realBank.agreementWithSyntheticBank.exa).toFixed(2));
if (out[VERTICALS[0]].humanCriterion) console.log('human-query criterion: rho', avg(x => x.humanCriterion.convergent).toFixed(2), 'disattenuated', avg(x => x.humanCriterion.disattenuated).toFixed(2), 'n', VERTICALS.map(v => out[v].humanCriterion.n).join('/'));
console.log('total rebuttal spend so far: $' + spent().toFixed(2));
