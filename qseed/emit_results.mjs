// Read results/*.json and emit LaTeX tables, pgfplots data files, and a macro
// file of inline numbers into the paper's gen/ directory.
//
// Every number that appears in the paper is produced here: the body cites
// \qs<name> macros only, so a re-run of the pipeline propagates to the PDF with
// no hand-editing. Nothing is transcribed by hand.
import fs from 'node:fs';
import path from 'node:path';
import { VERTICALS } from './src/config.mjs';
import { loadJSON } from './src/cache.mjs';
// Imported purely so the paper's stated style-feature count is derived from the
// implementation rather than typed by hand (it had drifted: the text said six).
import { styleFeatures } from './src/calibrate.mjs';
import {
  mean, std, quantile, rng, spearman, kendallTau,
  bootstrapCI, pairedBootstrapDiff, holm,
} from './src/util.mjs';

const OUT = path.resolve('..', 'Q-SEED_KDD2027_2026-07-12', 'gen');
fs.mkdirSync(OUT, { recursive: true });
const w = (name, content) => { fs.writeFileSync(path.join(OUT, name), content); };
// Table bodies get a trailing \hline INSIDE the file: a row-ending \\ at an
// \input boundary breaks TeX's \\ optional-arg lookahead, so the bottom rule
// must live in the body file (see paper build notes).
const wtable = (name, body) => { fs.writeFileSync(path.join(OUT, name), body.replace(/\s*$/, '') + '\n\\hline\n'); };

const VLABEL = { consumer_electronics: 'Cons.\\ Electronics', personal_finance: 'Pers.\\ Finance', health_wellness: 'Health' };
const VSHORT = { consumer_electronics: 'CE', personal_finance: 'PF', health_wellness: 'HW' };
// Display names for every method key. An explicit map, not a regex over
// camelCase: the old auto-speller mangled acronym/hyphen names ("KW-as-Query"
// -> "K W-as- Query") and forced hand-editing of the emitted table.
const MLABEL = {
  'KW-as-Query': 'KW-as-Query',
  'PAA-Template': 'PAA-Template',
  'NaiveParaphrase': 'Naive Paraphrase',
  'ForwardFanout': 'Forward Fan-out',
  'GEO-bench-like': 'GEO-bench-like',
  'Q-SEED': '\\textbf{Q-SEED (ours)}',
  'Q-SEED wo/S3': '\\;-- w/o S3 (fan-out)',
  'Q-SEED wo/S4': '\\;-- w/o S4 (realism)',
  'Q-SEED random-sel': '\\;-- random selection',
  'Q-SEED +IRTdistill': '\\;$+$ IRT distillation',
};
const ENGLABEL = { 'sonar': 'Sonar', 'gpt4o-online': 'GPT-4o' };

const R = {};
for (const v of VERTICALS) { const r = loadJSON(v + '.json'); if (r) R[v] = r; }
const verts = Object.keys(R);
if (!verts.length) { console.error('no results found'); process.exit(1); }

const fmt = (x, d = 3) => (x == null || Number.isNaN(x)) ? '--' : x.toFixed(d);
const pct = (x, d = 1) => (x == null || Number.isNaN(x)) ? '--' : (100 * x).toFixed(d);
// LaTeX-escape free text (query strings, domain names) before it enters a table.
// LLM-written queries carry smart quotes/dashes/ellipses that pdflatex would
// either drop or mis-set under the default OT1 encoding, so fold them to ASCII
// first, then escape the TeX specials.
const tex = (s) => String(s)
  .replace(/[‘’‛]/g, "'")
  .replace(/[“”]/g, '"')
  .replace(/—/g, '---').replace(/–/g, '--')
  .replace(/…/g, '...')
  .replace(/ /g, ' ')
  .replace(/\\/g, '\\textbackslash{}')
  .replace(/([&%$#_{}])/g, '\\$1')
  .replace(/~/g, '\\textasciitilde{}').replace(/\^/g, '\\textasciicircum{}');

// ---------- aggregate helpers ----------
function methodAgg(metric) {
  // returns {method: {mean, values[]}}
  const methods = Object.keys(R[verts[0]].methods);
  const out = {};
  for (const m of methods) {
    const vals = verts.map(v => R[v].methods[m]?.[metric]).filter(x => x != null && !Number.isNaN(x));
    out[m] = { mean: mean(vals), values: vals };
  }
  return out;
}
const mm = (f) => mean(verts.map(f));

// Bootstrap a rank-correlation that is computed over DOMAINS, resampling domain
// indices with replacement independently within each vertical and averaging the
// per-vertical coefficient. The resulting interval reflects domain-sampling
// noise in the reported cross-vertical mean. It deliberately does NOT model
// vertical-to-vertical heterogeneity: with 3 verticals that component is not
// estimable, so per-vertical values are reported separately in the tables.
function bootstrapDomainCorr(pick, corr = spearman, { nboot = 2000, seed = 17 } = {}) {
  const rand = rng(seed);
  const perV = verts.map(v => pick(R[v]));      // [{x:[], y:[]}]
  const est = mean(perV.map(({ x, y }) => corr(x, y)));
  const reps = [];
  for (let b = 0; b < nboot; b++) {
    const vals = perV.map(({ x, y }) => {
      const n = x.length, xs = new Array(n), ys = new Array(n);
      for (let i = 0; i < n; i++) { const j = Math.floor(rand() * n); xs[i] = x[j]; ys[i] = y[j]; }
      return corr(xs, ys);
    });
    reps.push(mean(vals));
  }
  reps.sort((a, b) => a - b);
  return { est, lo: quantile(reps, 0.025), hi: quantile(reps, 0.975) };
}
// "0.68 [0.55, 0.78]" -- the form used inline in the paper.
const ciText = (c, d = 2) => `${fmt(c.est, d)}\\,[${fmt(c.lo, d)},\\,${fmt(c.hi, d)}]`;

// ---------- Table: dataset / bank stats ----------
{
  let s = '';
  for (const v of verts) {
    const r = R[v];
    s += `${VLABEL[v]} & ${r.bankSize} & ${r.nClusters} & ${r.nCandidates} & ${r.S3.kept} & ${pct(r.S3.dropRate)}\\% & ${r.nDomains} \\\\\n`;
  }
  wtable('tab_dataset.tex', s);
}

// ---------- Table: main quality (RQ1) averaged across verticals ----------
{
  const order = ['KW-as-Query', 'PAA-Template', 'NaiveParaphrase', 'ForwardFanout', 'GEO-bench-like', 'Q-SEED'];
  const metrics = ['coverage', 'intentRecall', 'vendi', 'mauve', 'discAUC'];
  const aggs = Object.fromEntries(metrics.map(m => [m, methodAgg(m)]));
  // best per metric (discAUC best = closest to 0.5)
  const bestOf = (m) => {
    const entries = order.map(mm2 => [mm2, aggs[m][mm2].mean]);
    if (m === 'discAUC') return entries.reduce((a, b) => Math.abs(b[1] - 0.5) < Math.abs(a[1] - 0.5) ? b : a)[0];
    return entries.reduce((a, b) => b[1] > a[1] ? b : a)[0];
  };
  const best = Object.fromEntries(metrics.map(m => [m, bestOf(m)]));
  let s = '';
  for (const mm2 of order) {
    const cells = metrics.map(m => {
      let val = m === 'discAUC' ? fmt(aggs[m][mm2].mean, 3) : (m === 'vendi' ? fmt(aggs[m][mm2].mean, 1) : pct(aggs[m][mm2].mean));
      if (best[m] === mm2) val = `\\textbf{${val}}`;
      return val;
    });
    s += `${MLABEL[mm2]} & ${cells.join(' & ')} \\\\\n`;
    if (mm2 === 'GEO-bench-like') s += '\\midrule\n';
  }
  wtable('tab_quality.tex', s);
}

// ---------- Table: ablations ----------
{
  const order = ['Q-SEED', 'Q-SEED wo/S3', 'Q-SEED wo/S4', 'Q-SEED random-sel', 'Q-SEED +IRTdistill'];
  const metrics = ['coverage', 'intentRecall', 'vendi', 'mauve', 'discAUC'];
  const aggs = Object.fromEntries(metrics.map(m => [m, methodAgg(m)]));
  let s = '';
  for (const mm2 of order) {
    const cells = metrics.map(m => m === 'discAUC' ? fmt(aggs[m][mm2].mean, 3) : (m === 'vendi' ? fmt(aggs[m][mm2].mean, 1) : pct(aggs[m][mm2].mean)));
    const label = mm2 === 'Q-SEED' ? 'Full Q-SEED (greedy $B$)' : MLABEL[mm2];
    s += `${label} & ${cells.join(' & ')} \\\\\n`;
    if (mm2 === 'Q-SEED') s += '\\midrule\n';
  }
  wtable('tab_ablation.tex', s);
}

// ---------- Table: reliability + external validity + cross-engine ----------
{
  let s = '';
  for (const v of verts) {
    const r = R[v];
    s += `${VLABEL[v]} & ${fmt(r.reliability.splitHalf.spearmanBrown)} & ${fmt(r.reliability.retest.kendall)} & ${fmt(r.external.main.spearman)} & ${fmt(r.crossEngine.visibility.spearman)} \\\\\n`;
  }
  wtable('tab_reliability.tex', s);
}

// ---------- Table: cross-engine IRT item-parameter agreement (RQ4) ----------
// Visibility disagreement could be a pure level effect; if the ENGINES ranked
// the same queries as easy/discriminating, the instrument would still transfer.
// These columns test that directly.
{
  let s = '';
  for (const v of verts) {
    const ce = R[v].crossEngine;
    s += `${VLABEL[v]} & ${fmt(ce.visibility.spearman)} & ${fmt(ce.b.spearman)} & ${fmt(ce.a.spearman)} & ` +
      `${ce.engCited[ce.engIds[0]]} / ${ce.engCited[ce.engIds[1]]} \\\\\n`;
  }
  wtable('tab_crossengine.tex', s);
}

// ---------- Table: per-engine top domains (qualitative engine specificity) ----------
{
  let s = '';
  for (const v of verts) {
    const r = R[v], ce = r.crossEngine, [e1, e2] = ce.engIds;
    const topK = (eid, k = 3) => r.domains
      .map((d, i) => ({ d, rate: ce.perEngVis[eid][i] }))
      .sort((a, b) => b.rate - a.rate).slice(0, k)
      .map(o => `\\texttt{${tex(o.d)}} {\\scriptsize(${pct(o.rate, 0)})}`).join(', ');
    s += `${VLABEL[v]} & ${ENGLABEL[e1] ?? e1} & ${topK(e1)} \\\\\n`;
    s += ` & ${ENGLABEL[e2] ?? e2} & ${topK(e2)} \\\\\n`;
    if (v !== verts[verts.length - 1]) s += '\\midrule\n';
  }
  wtable('tab_topdomains.tex', s);
}

// ---------- Table: example Q-SEED queries ----------
{
  let s = '';
  for (const v of verts) {
    const qs = R[v].finalQueries.slice(0, 2);
    qs.forEach((q, i) => {
      s += `${i === 0 ? VLABEL[v] : ''} & \`\`${tex(q)}'' \\\\\n`;
    });
    if (v !== verts[verts.length - 1]) s += '\\midrule\n';
  }
  wtable('tab_examples.tex', s);
}

// ---------- Table: lambda sensitivity (coverage/realism trade-off in F) ----------
{
  const grid = R[verts[0]].lambdaSweep.map(s => s.lambda);
  const opLam = loadJSON('_index.json')?.config?.lambda ?? 0.35;
  let s = '';
  grid.forEach((lam, i) => {
    const g = (k) => mean(verts.map(v => R[v].lambdaSweep[i][k]));
    const isOp = Math.abs(lam - opLam) < 1e-9;
    const lab = isOp ? `\\textbf{${lam.toFixed(2)}}` : lam.toFixed(2)
      + (lam === 0 ? '\\rlap{$^{\\dagger}$}' : '');
    s += `${lab} & ${pct(g('coverage'))} & ${fmt(g('meanR'), 3)} & ${fmt(g('mauve'), 3)} & ${fmt(g('discAUC'), 3)} \\\\\n`;
    if (isOp) s += '\\midrule\n';
  });
  wtable('tab_lambda.tex', s);
}

// ---------- Table: tau sensitivity of the S3 filter ----------
// The cycle-consistency scores of every candidate are stored, so the drop rate
// at any threshold is recomputable with no further API calls. This shows the
// operating point tau=0.62 is not knife-edge tuned.
const TAU_GRID = [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75];
{
  let s = '';
  for (const t of TAU_GRID) {
    const rates = verts.map(v => {
      const cs = R[v].consistencyScores;
      return cs.filter(x => x < t).length / cs.length;
    });
    const isOp = Math.abs(t - R[verts[0]].S3.tau) < 1e-9;
    const cells = rates.map(x => pct(x, 1) + '\\%').join(' & ');
    const lab = isOp ? `\\textbf{${t.toFixed(2)}}` : t.toFixed(2);
    s += `${lab} & ${cells} & ${pct(mean(rates), 1)}\\% \\\\\n`;
    if (isOp) s += '\\midrule\n';
  }
  wtable('tab_tausens.tex', s);
}

// ---------- pgfplots data: coverage-budget (mean across verticals) ----------
{
  const steps = R[verts[0]].coverageBudget.steps;
  let s = ''; // headerless: index columns 0=n 1=greedy 2=random 3=kw
  steps.forEach((n, i) => {
    const g = mean(verts.map(v => R[v].coverageBudget.greedy[i]));
    const rr = mean(verts.map(v => R[v].coverageBudget.random[i]));
    const kw = mean(verts.map(v => R[v].coverageBudget.kw[i]));
    s += `${n} ${fmt(g, 4)} ${fmt(rr, 4)} ${fmt(kw, 4)}\n`;
  });
  w('dat_coverage.dat', s);
}

// ---------- pgfplots data: efficiency curve (mean Kendall tau) ----------
{
  const steps = R[verts[0]].efficiency.steps;
  let s = ''; // 0=n 1=irt 2=random
  steps.forEach((n, i) => {
    const it = mean(verts.map(v => R[v].efficiency.irt[i]?.kendall).filter(x => x != null));
    const rr = mean(verts.map(v => R[v].efficiency.random[i]?.kendall).filter(x => x != null));
    s += `${n} ${fmt(it, 4)} ${fmt(rr, 4)}\n`;
  });
  w('dat_efficiency.dat', s);
}

// ---------- pgfplots data: test information (mean) ----------
{
  const grid = R[verts[0]].testInfo.thetaGrid;
  let s = ''; // 0=theta 1=final 2=random
  grid.forEach((t, i) => {
    const f = mean(verts.map(v => R[v].testInfo.final[i]));
    const rr = mean(verts.map(v => R[v].testInfo.random[i]));
    s += `${t} ${fmt(f, 4)} ${fmt(rr, 4)}\n`;
  });
  w('dat_testinfo.dat', s);
}

// ---------- pgfplots data: cross-engine per-domain visibility scatter ----------
{
  let s = ''; // 0=engine1_vis 1=engine2_vis
  for (const v of verts) {
    const ce = R[v].crossEngine;
    const [i1, i2] = ce.engIds;
    const a = ce.perEngVis[i1], b = ce.perEngVis[i2];
    for (let i = 0; i < a.length; i++) s += `${fmt(a[i], 4)} ${fmt(b[i], 4)}\n`;
  }
  w('dat_crossengine.dat', s);
}

// ---------- pgfplots data: discrimination histogram (pooled a_q) ----------
{
  const allA = [];
  for (const v of verts) allA.push(...R[v].irt.pooled.a);
  const bins = 16, lo = 0, hi = Math.max(...allA);
  const counts = new Array(bins).fill(0);
  for (const a of allA) { const bi = Math.min(bins - 1, Math.floor((a - lo) / (hi - lo) * bins)); counts[bi]++; }
  let s = ''; // 0=a 1=count
  counts.forEach((c, i) => { s += `${fmt(lo + (i + 0.5) * (hi - lo) / bins, 3)} ${c}\n`; });
  w('dat_adist.dat', s);
}

// ---------- pgfplots data: S3 cycle-consistency score histogram ----------
// Shows tau sits in the low-density valley between the grounded and drifted
// modes -- the visual companion to tab_tausens.
{
  const all = [];
  for (const v of verts) all.push(...R[v].consistencyScores);
  const bins = 30, lo = Math.min(...all), hi = Math.max(...all);
  const counts = new Array(bins).fill(0);
  for (const x of all) { const bi = Math.min(bins - 1, Math.floor((x - lo) / (hi - lo) * bins)); counts[bi]++; }
  let s = '';
  counts.forEach((c, i) => { s += `${fmt(lo + (i + 0.5) * (hi - lo) / bins, 4)} ${c}\n`; });
  w('dat_cyclehist.dat', s);
}

// ---------- macros: inline numbers ----------
{
  const cov = methodAgg('coverage'), rec = methodAgg('intentRecall'), mv = methodAgg('mauve'), au = methodAgg('discAUC'), vd = methodAgg('vendi');
  // Intent-Recall gain of Q-SEED over best baseline
  const baselines = ['KW-as-Query', 'PAA-Template', 'NaiveParaphrase', 'ForwardFanout', 'GEO-bench-like'];
  const bestBaseRec = Math.max(...baselines.map(b => rec[b].mean));
  const recGain = 100 * (rec['Q-SEED'].mean - bestBaseRec);
  const bestBaseCov = Math.max(...baselines.map(b => cov[b].mean));
  const covGain = 100 * (cov['Q-SEED'].mean - bestBaseCov);
  // paired bootstrap Q-SEED vs best baseline on coverage (per-vertical values)
  const bestBaseCovName = baselines.reduce((a, b) => cov[b].mean > cov[a].mean ? b : a);
  const covTest = pairedBootstrapDiff(cov['Q-SEED'].values, cov[bestBaseCovName].values);
  // efficiency: mean ranking fidelity in the tight-budget regime (n<=25),
  // discrimination-prioritized vs random ordering (vs independent round-2 GT).
  const smallSteps = [10, 15, 20, 25];
  const smallMean = (which) => mean(verts.map(v => {
    const st = R[v].efficiency.steps, cur = R[v].efficiency[which];
    const vals = smallSteps.map(n => { const i = st.indexOf(n); return i >= 0 ? cur[i].kendall : null; }).filter(x => x != null);
    return mean(vals);
  }));
  const effIRTsmall = smallMean('irt'), effRandsmall = smallMean('random');
  const totalProbes = verts.reduce((s, v) => s + (R[v].nProbeCalls || 0), 0);
  const groupThousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, '{,}');

  // ---- coverage at a tight budget (the submodular payoff, cited in RQ1) ----
  const covSteps = R[verts[0]].coverageBudget.steps;
  const TIGHT = 15, ti = covSteps.indexOf(TIGHT);
  const covTightGreedy = mean(verts.map(v => R[v].coverageBudget.greedy[ti]));
  const covTightRand = mean(verts.map(v => R[v].coverageBudget.random[ti]));
  const covTightKw = mean(verts.map(v => R[v].coverageBudget.kw[ti]));

  // ---- bootstrap CIs over domains for the headline measurement claims ----
  const ciRetest = bootstrapDomainCorr(r => ({ x: r.visVectors.r1, y: r.visVectors.r2 }), kendallTau);
  const ciExt = bootstrapDomainCorr(r => ({ x: r.visVectors.r1, y: r.visVectors.ref }), spearman);
  const ciCross = bootstrapDomainCorr(r => ({ x: r.crossEngine.perEngVis[r.crossEngine.engIds[0]], y: r.crossEngine.perEngVis[r.crossEngine.engIds[1]] }), spearman);

  // ---- IRT discrimination spread (the diagnostic payload) ----
  const allA = []; for (const v of verts) allA.push(...R[v].irt.pooled.a);
  const aQ1 = quantile(allA, 0.25), aQ3 = quantile(allA, 0.75);

  // ---- Concentration of the measured trait (why the IRT null happens) ----
  // The paper's explanation for the efficiency null is that domain visibility is
  // head-dominated, so any covering subset estimates the head well. That is a
  // measurable property of the citation mass, not an intuition -- quantify it.
  const giniOf = (xs) => {
    const a = xs.slice().sort((x, y) => x - y), n = a.length;
    const tot = a.reduce((s, x) => s + x, 0);
    if (!tot) return 0;
    let cum = 0;
    for (let i = 0; i < n; i++) cum += (2 * (i + 1) - n - 1) * a[i];
    return cum / (n * tot);
  };
  // share of total citation mass held by the top-5 domains
  const top5Share = mm(v => {
    const r = R[v].visVectors.r1.slice().sort((a, b) => b - a);
    const tot = r.reduce((s, x) => s + x, 0);
    return tot ? r.slice(0, 5).reduce((s, x) => s + x, 0) / tot : 0;
  });
  const visGini = mm(v => giniOf(R[v].visVectors.r1));
  // how many domains account for half the citation mass
  const halfMass = mm(v => {
    const r = R[v].visVectors.r1.slice().sort((a, b) => b - a);
    const tot = r.reduce((s, x) => s + x, 0);
    let cum = 0, k = 0;
    while (k < r.length && cum < tot / 2) { cum += r[k]; k++; }
    return k;
  });

  // ---- S3 tau sensitivity summary: drop rate at the extremes of the grid, and
  // the share of candidates sitting within +-0.05 of the operating threshold
  // (a thin boundary zone => the keep/drop partition is stable).
  const dropAt = (t) => mean(verts.map(v => { const cs = R[v].consistencyScores; return cs.filter(x => x < t).length / cs.length; }));
  const tauOp = R[verts[0]].S3.tau;
  const bandFrac = mean(verts.map(v => {
    const cs = R[v].consistencyScores;
    return cs.filter(x => x >= tauOp - 0.05 && x <= tauOp + 0.05).length / cs.length;
  }));

  // keys map to \qs<key> in main.tex (toMacro prepends 'qs' verbatim).
  const macros = {
    nVerticals: verts.length,
    nEngines: Object.keys(R[verts[0]].irt.perEngine).length,
    avgBankSizeText: Math.round(mm(v => R[v].bankSize)),
    avgClustersText: fmt(mm(v => R[v].nClusters), 1),
    avgCandidatesText: Math.round(mm(v => R[v].nCandidates)),
    avgDropRateText: pct(mm(v => R[v].S3.dropRate)),
    avgDomainsText: Math.round(mm(v => R[v].nDomains)),
    BbudgetText: R[verts[0]].methods['Q-SEED'].n,
    candPerBtext: Math.round(mm(v => R[v].nCandidates) / R[verts[0]].methods['Q-SEED'].n),
    tauText: fmt(R[verts[0]].S3.tau, 2),
    probeRepText: R[verts[0]].probeRepeats ?? 3,
    qseedCoverageText: pct(cov['Q-SEED'].mean),
    qseedRecallText: pct(rec['Q-SEED'].mean),
    recallGain: recGain.toFixed(1),
    recallGainText: recGain.toFixed(1),
    covGainText: covGain.toFixed(1),
    qseedMauveText: fmt(mv['Q-SEED'].mean, 3),
    qseedDiscAUCText: fmt(au['Q-SEED'].mean, 3),
    qseedVendiText: fmt(vd['Q-SEED'].mean, 1),
    covPvalText: covTest.p < 0.001 ? '<0.001' : ('=' + covTest.p.toFixed(3)),
    effIRTsmallText: fmt(effIRTsmall, 2),
    effRandsmallText: fmt(effRandsmall, 2),
    effGainText: ((effIRTsmall - effRandsmall) * 100).toFixed(0),
    splitHalfText: fmt(mm(v => R[v].reliability.splitHalf.spearmanBrown), 2),
    retestText: fmt(mm(v => R[v].reliability.retest.kendall), 2),
    extMainText: fmt(mm(v => R[v].external.main.spearman), 2),
    extWoSThreeText: fmt(mm(v => R[v].external.woS3.spearman), 2),
    crossVisText: fmt(mm(v => R[v].crossEngine.visibility.spearman), 2),
    crossBText: fmt(mm(v => R[v].crossEngine.b.spearman), 2),
    crossAText: fmt(mm(v => R[v].crossEngine.a.spearman), 2),
    // Per-vertical spread of cross-engine item-parameter agreement. The mean
    // hides a split: one vertical shows no transfer at all, the others moderate
    // transfer. The prose cites these extremes, so they are emitted, not typed.
    crossBminText: fmt(Math.min(...verts.map(v => R[v].crossEngine.b.spearman)), 2),
    crossBmaxText: fmt(Math.max(...verts.map(v => R[v].crossEngine.b.spearman)), 2),
    crossAminText: fmt(Math.min(...verts.map(v => R[v].crossEngine.a.spearman)), 2),
    crossAmaxText: fmt(Math.max(...verts.map(v => R[v].crossEngine.a.spearman)), 2),
    // vertical whose item parameters transfer worst (by difficulty agreement)
    crossWorstVertText: (() => {
      const v = verts.reduce((a, b) => R[b].crossEngine.b.spearman < R[a].crossEngine.b.spearman ? b : a);
      return VLABEL[v].replace(/\\ /g, ' ').replace('Cons. Electronics', 'consumer electronics')
        .replace('Pers. Finance', 'personal finance').replace('Health', 'health');
    })(),
    apiCostText: (loadJSON('_index.json')?.apiStats?.cost ?? 0).toFixed(2),
    totalProbesText: groupThousands(totalProbes),

    // ---------------- added: numbers the prose previously hard-coded ----------------
    nStyleFeatText: styleFeatures('a probe query for counting features?').length,
    lambdaText: fmt(loadJSON('_index.json')?.config?.lambda ?? 0.35, 2),
    refBankSizeText: R[verts[0]].refBank?.length ?? 60,
    nQzeroText: R[verts[0]].Q0queries.length,
    // per-vertical Q-SEED discriminator AUC, e.g. "0.83/0.83/0.81"
    qseedAUCperVertText: verts.map(v => fmt(R[v].methods['Q-SEED'].discAUC, 2)).join('/'),
    // coverage at the tight budget B=15 (submodular payoff)
    covTightBtext: TIGHT,
    covTightGreedyText: pct(covTightGreedy, 0),
    covTightRandText: pct(covTightRand, 0),
    covTightKwText: pct(covTightKw, 0),
    // IRT-distillation coverage (the cost of the cautionary null)
    distillCoverageText: pct(cov['Q-SEED +IRTdistill'].mean, 1),

    // ---------------- added: S3 grounding gap (was an unquantified claim) ----------------
    sThreeKeptSimText: fmt(mm(v => R[v].S3.keptMeanSim), 2),
    sThreeDropSimText: fmt(mm(v => R[v].S3.dropMeanSim), 2),
    sThreeBandFracText: pct(bandFrac, 1),
    sThreeDropLoTauText: pct(dropAt(0.50), 1),
    sThreeDropHiTauText: pct(dropAt(0.75), 1),

    // ---------------- added: candidate pool realism before S4-weighted selection ----------------
    poolDiscAUCText: fmt(mm(v => R[v].S4.candPoolDiscAUC), 3),
    poolMauveText: fmt(mm(v => R[v].S4.candPoolMauve), 3),
    // how much realism-weighted selection improves on the pool it draws from
    mauveGainText: fmt(mv['Q-SEED'].mean / mm(v => R[v].S4.candPoolMauve), 1),
    ablWoSFourAUCText: fmt(au['Q-SEED wo/S4'].mean, 3),
    ablWoSFourMauveText: fmt(mv['Q-SEED wo/S4'].mean, 3),

    // ---------------- added: reliability detail + bootstrap CIs over domains ----------------
    retestSpearmanText: fmt(mm(v => R[v].reliability.retest.spearman), 2),
    retestCIText: ciText(ciRetest),
    extMainCIText: ciText(ciExt),
    crossVisCIText: ciText(ciCross),

    // ---------------- added: IRT discrimination spread ----------------
    meanAtext: fmt(mm(v => R[v].irt.meanA), 2),
    aIQRtext: `${fmt(aQ1, 2)}--${fmt(aQ3, 2)}`,
    aMaxText: fmt(Math.max(...allA), 2),

    // ---------------- added: lambda sensitivity ----------------
    // Range of lambda>0 probed, and the spread of realism AUC over it: the
    // headline is that everything above zero behaves alike.
    lambdaGridLoText: (() => { const g = R[verts[0]].lambdaSweep.map(s => s.lambda).filter(x => x > 0); return fmt(Math.min(...g), 2); })(),
    lambdaGridHiText: (() => { const g = R[verts[0]].lambdaSweep.map(s => s.lambda); return fmt(Math.max(...g), 1); })(),
    lambdaPosAUCloText: (() => {
      const idx = R[verts[0]].lambdaSweep.map((s, i) => [s.lambda, i]).filter(([l]) => l > 0).map(([, i]) => i);
      return fmt(Math.min(...idx.map(i => mean(verts.map(v => R[v].lambdaSweep[i].discAUC)))), 3);
    })(),
    lambdaPosAUChiText: (() => {
      const idx = R[verts[0]].lambdaSweep.map((s, i) => [s.lambda, i]).filter(([l]) => l > 0).map(([, i]) => i);
      return fmt(Math.max(...idx.map(i => mean(verts.map(v => R[v].lambdaSweep[i].discAUC)))), 3);
    })(),
    // worst coverage seen anywhere on the lambda grid
    lambdaMinCovText: (() => {
      const n = R[verts[0]].lambdaSweep.length;
      return pct(Math.min(...Array.from({ length: n }, (_, i) => mean(verts.map(v => R[v].lambdaSweep[i].coverage)))));
    })(),
    // mean realism r(q) of the bank at lambda=0 vs at the operating point
    lambdaZeroMeanRtext: fmt(mean(verts.map(v => R[v].lambdaSweep[0].meanR)), 3),
    lambdaOpMeanRtext: (() => {
      const opLam = loadJSON('_index.json')?.config?.lambda ?? 0.35;
      const i = R[verts[0]].lambdaSweep.findIndex(s => Math.abs(s.lambda - opLam) < 1e-9);
      return fmt(mean(verts.map(v => R[v].lambdaSweep[i].meanR)), 3);
    })(),

    // ---------------- added: head-dominance of the measured trait (RQ2) ----------------
    visTopFiveShareText: pct(top5Share, 0),
    visGiniText: fmt(visGini, 2),
    visHalfMassText: fmt(halfMass, 1),
  };
  let s = '% auto-generated numeric macros -- do not edit by hand.\n';
  s += '% Regenerate with:  node emit_results.mjs   (after run_pipeline.mjs)\n';
  for (const [k, val] of Object.entries(macros)) s += `\\newcommand{\\${toMacro(k)}}{${val}}\n`;
  w('macros.tex', s);
  console.error('macros:', JSON.stringify(macros, null, 1));
}

function toMacro(k) { return 'qs' + k; }
console.error('emit done ->', OUT);
