# Q-SEED: Keyword-Grounded GEO Query Bank Construction

Reference implementation for the paper *"Q-SEED: From SEO Keyword Banks to
Measurable Generative-Engine Query Banks."* The pipeline turns an SEO keyword
bank into a size-bounded, measurement-grade query bank for probing generative
engines (GEs), with provable coverage guarantees and psychometric validity.

Everything runs on the [OpenRouter](https://openrouter.ai) API in Node.js — no
GPU, no training. Every LLM/engine call is content-addressed and cached, so
re-runs and ablations are free and results are exactly reproducible.

## Layout

```
src/
  config.mjs     # models, engines, budgets, thresholds
  llm.mjs        # OpenRouter client: chat / embeddings / GE probe, async pool, cache
  cache.mjs      # content-addressed on-disk cache (sha256 of model+payload)
  util.mjs       # RNG, correlations, bootstrap, Holm, AUC, Krippendorff alpha
  linalg.mjs     # spherical / Euclidean k-means, PCA (Gram trick), Vendi score
  seeds.mjs      # curated real head keywords per vertical
  data.mjs       # keyword-bank build + WildChat-1M reference corpus (HF)
  intent.mjs     # S1  intent decomposition (embed -> cluster -> LLM label)
  generate.mjs   # S2  contextualized generation (persona x scenario x funnel)
  filter.mjs     # S3  reverse fan-out cycle-consistency filter
  calibrate.mjs  # S4  stylistic realism (discriminator, MAUVE)
  select.mjs     # S5  submodular lazy-greedy + IRT distillation
  irt.mjs        # 2PL IRT joint MLE (Adam), Fisher information
  probe.mjs      # multi-engine probe harness -> item-response matrices
  baselines.mjs  # KW-as-Query, PAA-Template, Naive Paraphrase, Forward Fan-out, GEO-bench-like
  metrics.mjs    # coverage, intent-recall, diversity, reliability, efficiency, external validity
run_pipeline.mjs # end-to-end driver -> results/<vertical>.json
emit_results.mjs # results -> LaTeX tables + pgfplots data + numeric macros
```

## Run

```bash
# put OPENROUTER_API_KEY in ../.env
npm install
node run_pipeline.mjs            # all verticals (uses cache; safe to re-run)
node run_pipeline.mjs consumer_electronics   # one vertical
node emit_results.mjs            # regenerate paper tables/figures/macros
```

Outputs land in `results/*.json`; the paper's `gen/` directory receives
`tab_*.tex`, `dat_*.dat`, and `macros.tex`.

## Engines probed

- `perplexity/sonar`
- `openai/gpt-4o-mini:online`

Both expose OpenAI-style `message.annotations[].url_citation`, parsed to
eTLD+1 domains via `tldts`.

## Notes / limitations

- Keyword banks are semi-synthetic: curated real head terms expanded into
  long-tail variants with Zipf-modeled volumes.
- Clustering uses spherical k-means (HDBSCAN needs a native build blocked on the
  target machine's application-control policy).
- Test-retest uses independently issued probe rounds within the study window.
