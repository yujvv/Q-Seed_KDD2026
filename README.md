# Q-SEED: From SEO Keyword Banks to Measurable Generative-Engine Query Banks

Code, query banks, cached engine probes, results and rebuttal experiments for
KDD 2027 Research Track submission 645.

## Layout

```
qseed/
  src/                 pipeline modules (S1-S5, baselines, metrics, IRT, probing)
  run_pipeline.mjs     end-to-end driver            -> results/<vertical>.json
  emit_results.mjs     results -> tables / plot data / numeric macros of the paper
  results/             results of the submitted paper (one JSON per vertical)
  rebuttal/            scripts of the rebuttal experiments (rb0 ... rb14)
  results/rebuttal/    their outputs (JSON), incl. the spend ledger
  cache/               content-addressed cache of every LLM call and engine probe
rebuttal/
  new_results.md       evidence log of the rebuttal experiments (R1-R8)
  human_study/         blinded human realism study: items, answers, key
```

`qseed/README.md` documents the pipeline itself.

## Where each rebuttal result comes from

| Section of `rebuttal/new_results.md` | Script(s) in `qseed/rebuttal/` | Output in `qseed/results/rebuttal/` |
|---|---|---|
| R1 real keyword banks | `rb0_fetch_real_keywords`, `rb6_realbank`, `rb9b_coverage_indep_real`, `rb11_probes` (part B) | `e6_realbank*.json`, `e9b_*.json`, `e11_probes_full.json` |
| R2 realism outside the S4 features | `rb8_realism`, `rb8b_judge`, `rb12_human_kit` | `e8_realism.json`, `e8c_*.json`, `rebuttal/human_study/` |
| R3 what S3 does | `rb4_s3`, `rb10_fanout_models` | `e4_s3_free.json`, `e10_fanout_models.json` |
| R4 coverage in independent encoders | `rb9_coverage_indep`, `rb9c`-`rb9f` | `e9*.json` |
| R5 selection baselines | `rb1_selection` | `e1_selection.json` |
| R6 IRT stability | `rb2_irt` | `e2_irt_stability.json` |
| R7 engines, drift | `rb3_engines`, `rb7_engine_check`, `rb11_probes` (parts A, A'), `rb14_drift` | `e3_*.json`, `e7_*.json`, `e11_probes_full.json`, `e14_drift.json` |
| R8 style bias, reference bank, real-user criterion | `rb5_style_bias`, `rb13_refbank_overlap`, `rb11_probes` (part C) | `e5_*.json`, `e13_*.json`, `e11_probes.json`, `e11_human_queries_*.json` |

## Running

```bash
cd qseed
npm install
QSEED_OFFLINE=1 node run_pipeline.mjs      # paper results, served from the cache
node emit_results.mjs
QSEED_OFFLINE=1 node rebuttal/rb1_selection.mjs    # any rebuttal script
node rebuttal/rb12_human_kit.mjs analyze           # score the human study
```

`QSEED_OFFLINE=1` turns every cache miss into an error instead of a paid API call.

**Cache contents.** The repository holds the cache of every engine probe
(`cache/probe`, `cache/rb_probe`), every LLM call (`cache/chat`, `cache/rb_chat`)
and every pipeline intermediate (keyword banks, intent trees, candidate pools,
filter decisions, baselines, reference sets). The embedding cache (`cache/embed`,
2.6 GB) is too large for git and is not included. Scripts that need embeddings
therefore recompute them through the API (set `OPENROUTER_API_KEY` in `../.env`
and drop `QSEED_OFFLINE`); for the paper pipeline this costs a few cents. All
reported numbers are in `results/` regardless.

**Real keyword banks.** `rb0_fetch_real_keywords.mjs` streams the public Bing
keyword inventory (Hugging Face `vtempest/seo-search-keywords-100M`) and keeps the
keyphrases that match the head terms; the banks actually used are stored in
`results/rebuttal/e6_realbank_<vertical>.json`.

Requires Node.js 20 or later.
