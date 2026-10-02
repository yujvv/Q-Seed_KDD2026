// What changed between the submitted probes (July) and the re-probe (October)?
// Per-engine top domains and source mix, then vs now, on the delivered banks. Cached.
import { loadVertical, VERTICALS, save, rbProbeQueries } from './common.mjs';
import { domainVisibility } from '../src/probe.mjs';
import { spearman, mean } from '../src/util.mjs';
const TWO = [{ id: 'sonar', model: 'perplexity/sonar' }, { id: 'gpt4o-online', model: 'openai/gpt-4o-mini:online' }];
const UGC = new Set(['youtube.com', 'reddit.com', 'quora.com', 'facebook.com', 'medium.com', 'tiktok.com', 'instagram.com', 'x.com', 'twitter.com']);
const out = {};
for (const v of VERTICALS) {
  const S = await loadVertical(v);
  const now = await rbProbeQueries(S.finalQueries, TWO, { repeats: 1, tag: 'A_now' });
  const r = {};
  for (const e of TWO) {
    const stat = (recs) => {
      const cnt = {}; let tot = 0, answers = 0;
      for (const rec of recs) { answers += rec[e.id].trials; for (const [d, c] of Object.entries(rec[e.id].domainCounts)) { cnt[d] = (cnt[d] || 0) + c; tot += c; } }
      const top = Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([d, c]) => d + ' ' + Math.round(100 * c / answers) + '%');
      const ugc = Object.entries(cnt).filter(([d]) => UGC.has(d)).reduce((s, [, c]) => s + c, 0) / tot;
      return { top, ugcShare: ugc, domainsPerAnswer: tot / answers, distinct: Object.keys(cnt).length };
    };
    r[e.id] = { july: stat(S.recsFinal), october: stat(now) };
  }
  const D = S.domains, vis = (recs, e) => domainVisibility(recs, D, [e]).map(x => x.rate);
  r.cross = { july: spearman(vis(S.recsFinal, TWO[0]), vis(S.recsFinal, TWO[1])), october: spearman(vis(now, TWO[0]), vis(now, TWO[1])) };
  // same comparison on the July data restricted to ONE repeat is not available (counts are pooled), so report as is
  out[v] = r; console.log(v, JSON.stringify(r, null, 0));
}
save('e14_drift.json', out);
console.log('cross-engine rho July', mean(VERTICALS.map(v => out[v].cross.july)).toFixed(2), 'October', mean(VERTICALS.map(v => out[v].cross.october)).toFixed(2));
for (const e of TWO) console.log(e.id, 'UGC share July', mean(VERTICALS.map(v => out[v][e.id].july.ugcShare)).toFixed(3), 'October', mean(VERTICALS.map(v => out[v][e.id].october.ugcShare)).toFixed(3), '| domains/answer', mean(VERTICALS.map(v => out[v][e.id].july.domainsPerAnswer)).toFixed(1), '->', mean(VERTICALS.map(v => out[v][e.id].october.domainsPerAnswer)).toFixed(1));
