// How different is the convergent-validity reference bank from the Q-SEED bank? (free)
import { loadVertical, VERTICALS, save } from './common.mjs';
import { mean } from '../src/util.mjs';
const STOP = new Set('the a an of to in on at and or for with is are what which how do does can i my me should best good that this it be you your'.split(' '));
const toks = (q) => new Set(q.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter(w => w && !STOP.has(w)));
const jac = (A, B) => { let k = 0; for (const x of A) if (B.has(x)) k++; return k / (A.size + B.size - k || 1); };
const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v, { probes: false });
  const R = S.refBank.map(toks), Q = S.finalQueries.map(toks);
  const best = Q.map(q => Math.max(...R.map(r => jac(q, r))));
  const top60 = new Set(S.bank.keywords.slice().sort((a, b) => b.volume - a.volume).slice(0, 60).map(k => k.kw));
  const heads = new Set(S.bank.keywords.filter(k => k.kw === k.seed).map(k => k.kw));
  out[v] = { nRef: S.refBank.length, wordsRef: mean(S.refBank.map(q => q.split(/\s+/).length)), wordsBank: mean(S.finalQueries.map(q => q.split(/\s+/).length)), meanBestTokenJaccard: mean(best), nearDuplicates: best.filter(x => x >= 0.6).length, refKeywordsThatAreHeadTerms: [...top60].filter(k => heads.has(k)).length, shareOfBankVolumeInRefKeywords: S.bank.keywords.filter(k => top60.has(k.kw)).reduce((s, k) => s + k.logvol, 0) / S.bank.keywords.reduce((s, k) => s + k.logvol, 0) };
  console.log(v, JSON.stringify(out[v]));
}
save('e13_refbank_overlap.json', out);
