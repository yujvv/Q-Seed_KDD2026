# Q-SEED rebuttal: new experiments (evidence log)

Every number below is produced by a script in `qseed/rebuttal/` and stored in
`qseed/results/rebuttal/`. Nothing in the submitted paper, its cache or its
results was changed. Unless stated, values are means over the three verticals
(CE = consumer electronics, PF = personal finance, HW = health and wellness).

Dates: the submitted probes are from 6 July 2026; all new engine probes are from
2 October 2026.

| ID | Question | Script | Status |
|---|---|---|---|
| R1 | Does the pipeline work on real keyword banks? | `rb0`, `rb6`, `rb9b`, `rb11` | done, 3 verticals, incl. engine probing |
| R2 | Is the realism gain visible outside the S4 feature space? | `rb8`, `rb8b` | done: 2 feature controls, 3 LLM judges, human study (1 annotator) |
| R3 | What does S3 do? How sensitive is it to the fan-out model? | `rb4`, `rb10` | done |
| R4 | Is coverage visible outside the S1 embedding space? | `rb9`, `rb9c`-`rb9f` | done |
| R5 | Stronger selection baselines on the same pool | `rb1` | done |
| R6 | Stability of the IRT item parameters | `rb2` | done |
| R7 | What the cross-engine disagreement consists of; five engines; 12-week drift | `rb3`, `rb7`, `rb11`, `rb14` | done |
| R8 | Style bias; reference-bank independence; real-user-query criterion | `rb5`, `rb13`, `rb11` | done; criterion bank is small (67 queries) |

---

## R1. Real keyword banks

**Data.** Public Bing keyword inventory (Hugging Face `vtempest/seo-search-keywords-100M`,
"SEO keywords from Bing collected in 2020", 112M keyphrases, each with a 0-10
search-popularity index). For each of our 40 curated head terms per vertical we
keep the keyphrases that phrase-match it, as a keyword tool does, and take the 50
most popular per head term. Keyphrases and popularity values are used verbatim.
No keyword is generated and no volume is modelled. The popularity index is
log-scaled, so it replaces `log(1+volume)` in the cluster weights.

| | N keywords | matched in inventory | M clusters | cluster size | cohesion | weight Gini | S3 drop |
|---|---|---|---|---|---|---|---|
| CE real | 1,854 | 87,061 | 24 | 37-124 | 0.79 | 0.21 | 2.5% |
| PF real | 1,536 | 56,332 | 24 | 10-117 | 0.81 | 0.32 | 4.1% |
| HW real | 1,798 | 42,082 | 28 | 12-154 | 0.80 | 0.33 | 2.3% |
| CE semi-synthetic (paper) | 240 | | 18 | 6-18 | 0.75 | 0.23 | 6.3% |
| PF semi-synthetic (paper) | 240 | | 18 | 3-26 | 0.77 | 0.24 | 0.8% |
| HW semi-synthetic (paper) | 239 | | 18 | 3-23 | 0.75 | 0.30 | 5.0% |

Cohesion = mean cosine of a keyword to its cluster centroid. The real banks
contain the noise one expects of an export: `webcams monterey`, `b12 vitamin
tablets` (matched through "tablet"), `banner life insurance agent login`,
`collagen dressing`, brand and location modifiers. The pipeline uses no
per-keyword intent label (S1 infers intent per cluster), so the absence of intent
labels in the inventory does not matter. Banks built by phrase match on head terms
are cohesive by construction, which is why cohesion is not lower than in the
semi-synthetic banks.

**Bank quality on real banks** (3 verticals, unchanged code and CONFIG, B = 50).
"Cov (S1)" is the paper's metric. "Cov (indep.)" re-clusters the full real bank
with two encoders the pipeline never used (R4). AUC is the paper's discriminator,
averaged over 10 seeds.

| Method | Cov (S1) | Recall (held-out) | Cov (indep.) | Vendi | MAUVE | AUC |
|---|---|---|---|---|---|---|
| KW-as-Query | 46.4 | 45.5 | 45.4 | 10.5 | 1.0 | 0.987 |
| PAA-Template | 89.5 | 89.1 | 85.8 | 20.3 | 9.6 | 0.947 |
| Naive Paraphrase | 81.8 | 81.6 | 81.1 | 22.4 | 9.8 | 0.924 |
| Forward Fan-out | 88.7 | 88.6 | 83.8 | 22.6 | 8.6 | 0.907 |
| GEO-bench-like | 72.1 | 74.6 | 72.7 | 23.7 | 8.5 | 0.938 |
| **Q-SEED** | **97.9** | **97.6** | 83.3 | 19.7 | **12.4** | **0.872** |
| Q-SEED w/o S3 | 97.9 | 97.6 | | 19.7 | 12.4 | 0.872 |
| Q-SEED w/o S4 | 97.9 | 97.6 | | 18.3 | 4.3 | 0.961 |

Per vertical, Q-SEED AUC is 0.857 (CE), 0.884 (PF), 0.875 (HW); the closest
baseline is 0.896, 0.896 and 0.917.

**Coverage against budget on real banks** (greedy order vs random draws from the
same pool vs keywords by popularity):

| B | 10 | 15 | 20 | 25 | 50 |
|---|---|---|---|---|---|
| S1 space: greedy / random / keyword order | 49 / 33 / 19 | 67 / 45 / 22 | 88 / 55 / 29 | 96 / 64 / 33 | 98 / 86 / 46 |
| independent encoders: greedy / random | 49 / 37 | 63 / 47 | 76 / 55 | 80 / 61 | 83 / 77 |

**Engine probing of the real-bank query banks** (`rb11` part B; both submitted
engines, one repeat; the semi-synthetic banks were re-probed in the same session
for a same-time comparison):

| Mean over verticals | Real-bank Q-SEED bank | Semi-synthetic Q-SEED bank, same day |
|---|---|---|
| Split-half reliability of the domain ranking (Spearman-Brown, 200 splits) | 0.65 | 0.54 |
| Cross-engine rho | 0.39 | 0.50 |
| Agreement between the two banks' domain rankings (pooled / Sonar / Exa) | 0.45 / 0.48 / 0.48 | |

The two banks come from different keyword sources (a real inventory against an
LLM expansion) for the same head terms. Their rankings agree at rho 0.45, which is
0.76 after correcting for the two split-half reliabilities.

**What replicates.** The Table 2 realism ordering in the S4 space (Q-SEED lowest
in all three verticals); the collapse of that realism without S4; the tight-budget
advantage of greedy selection, in the S1 space and in independent encoders (+12
to +21 points in independent encoders at B = 10-25, more in the S1 space); a measurement reliability at least as high as on the
semi-synthetic banks. The gap between Q-SEED and the keyword-sampling baselines in
S1 coverage widens, because 50 randomly drawn keywords no longer span a
1,500-1,900-keyword bank.

**What does not.** In independent encoders Q-SEED's full-budget coverage (83.3)
is level with Forward Fan-out (83.8) and below PAA-Template (85.8). See R4.

**Limits.** The popularity value is a binned index, not a monthly volume. The
bank is a public inventory filtered by our head terms, not one brand's private
export.

---

## R2. Realism outside the S4 feature space

Clarification of the submitted protocol: the AUC in Table 2 is not computed with
the S4 scorer. A new discriminator is fitted for each bank (70/30 split) and its
held-out AUC is reported. It does share the nine-feature representation with S4,
which is the circularity the reviewers describe.

The paper reported one discriminator seed (Q-SEED 0.825). Over 10 seeds the value
is 0.860; baselines move by at most 0.02 and the ordering is unchanged. The table
uses the 10-seed protocol throughout.

Two further representations that S4 never sees:

- **Disjoint style features (12).** Lower-case start, no terminal punctuation,
  function-word rate, comma density, contractions, sentence count, type-token
  ratio, politeness markers, hedges, clause punctuation, second person, clause
  conjunctions. None is among the nine S4 features.
- **Function-word skeleton n-grams.** Every content word is masked (`X`), numbers
  become `NUM`, function words and punctuation are kept; unigrams and bigrams of
  that skeleton (top 250 by document frequency) feed a logistic discriminator.
  Topic is removed by construction and the features are learned, not hand-picked.

Held-out AUC of real WildChat prompts vs each bank (lower = harder to tell apart;
20 and 8 repeated splits respectively):

| Bank | S4 space (paper metric) | Disjoint features | Skeleton n-grams |
|---|---|---|---|
| KW-as-Query | 0.978 | 0.945 | 0.986 |
| PAA-Template | 0.960 | 0.963 | 0.999 |
| Naive Paraphrase | 0.934 | 0.936 | 0.966 |
| Forward Fan-out | 0.941 | 0.899 | 0.974 |
| GEO-bench-like | 0.938 | 0.935 | 0.929 |
| **Q-SEED** | **0.860** | **0.891** | **0.924** |
| Q-SEED w/o S4 | 0.967 | 0.913 | 0.965 |
| S2 pool, unselected | 0.975 | 0.906 | 0.971 |

Per vertical (S4 / disjoint / skeleton), Q-SEED: CE 0.888 / 0.861 / 0.941, PF
0.856 / 0.891 / 0.962, HW 0.836 / 0.921 / 0.869. Best baseline: CE 0.928 / 0.919 /
0.936, PF 0.920 / 0.872 / 0.917, HW 0.932 / 0.907 / 0.933.

**Reading.** In the S4 space Q-SEED leads every baseline in every vertical. Outside
it, Q-SEED has the lowest mean AUC in both representations, but by 0.005-0.008
over the best baseline, and it is the best method in only one vertical of three in
each. The S4 gain (w/o S4 to full: 0.107 in the S4 space) transfers at about a
fifth (0.022) to the disjoint features and two fifths (0.041) to the skeleton. All
banks, Q-SEED included, remain clearly separable from real prompts (AUC about
0.9). The surface noise of real users (lower-case starts, missing punctuation) is
absent from the LLM-generated pool, so selection cannot recover it.

**LLM judges.** Three judges from families that took no part in S4.

*Blind pairwise preference* ("which message was more likely typed by a real
person?", Q-SEED against one baseline query from the same vertical, side
randomised, 150 pairs per cell):

| Q-SEED preferred over | Claude Haiku 4.5 | Gemini 2.5 Flash |
|---|---|---|
| PAA-Template | 39% | 64% |
| Naive Paraphrase | 55% | 77% |
| Forward Fan-out | 66% | 82% |
| GEO-bench-like | 69% | 92% |
| Q-SEED w/o S4 | 49% | 50% |

*Absolute score* (probability 0-100 that a real person typed the message):

| Source | Haiku 4.5 | Gemini 2.5 Flash | Qwen 3.8 (CE only, batched) |
|---|---|---|---|
| real WildChat prompts | 77.4 | 83.4 | 77.3 |
| KW-as-Query | 89.0 | 89.8 | 83.8 |
| PAA-Template | 79.7 | 85.3 | 57.3 |
| Naive Paraphrase | 89.1 | 88.9 | 64.0 |
| Forward Fan-out | 85.0 | 89.5 | 52.3 |
| GEO-bench-like | 85.2 | 90.3 | 48.4 |
| Q-SEED | 80.7 | 88.4 | 49.2 |
| Q-SEED w/o S4 | 79.1 | 88.1 | 50.8 |

Reading. (1) All three judges are indifferent to S4: 49-50% in the pairwise test,
and absolute scores within 2 points with and without it. (2) In the pairwise test
both judges prefer Q-SEED queries to Forward Fan-out and GEO-bench-like queries
(66-92%); against Naive Paraphrase and the templates the two judges disagree. What
the judges reward is therefore the contextualized generation of S2 (persona,
scenario, constraint), not the S4 selection. (3) Absolute scores do not separate
the methods consistently: Haiku and Gemini score almost every synthetic set as at
least as human as the real prompts, and Qwen penalises long queries. LLM judges
are not a substitute for people here.

**Human study** (`rb12`; materials, answers and key in `rebuttal/human_study/`).
One annotator, blind to source, labelled 124 single queries as typed by a person
or written by a model, and chose the human-written query in 60 side-randomised
pairs (Q-SEED against one baseline query from the same vertical).

| Single items, true source | Judged "typed by a person" |
|---|---|
| real WildChat prompts | 40 / 40 |
| Q-SEED | 3 / 42 (7%) |
| Naive Paraphrase | 0 / 21 |
| Forward Fan-out | 0 / 21 |

| Pairs: Q-SEED chosen as the human-written one, against | |
|---|---|
| GEO-bench-like | 9 / 15 |
| PAA-Template | 9 / 15 |
| Forward Fan-out | 7 / 15 |
| Naive Paraphrase | 3 / 15 |
| all | 28 / 60 (47%) |

Reading. The annotator separates synthetic from real queries with 97.6% accuracy.
Q-SEED is the only synthetic source with any query taken for human (3 of 42
against 0 of 42), a difference that is not significant at this sample size. In
pairs there is no preference for Q-SEED overall (28 of 60), and Naive Paraphrase
is preferred to it (12 of 15, two-sided binomial p = 0.035, uncorrected). Limits:
one annotator; in the single-item part the real prompts cover arbitrary topics and
the synthetic ones three verticals, so topic is a cue; the pairs are within
vertical and free of it.

**Consequence for the paper.** "Most human-realistic among all baselines" is
withdrawn. The supported statements are: S4 calibrates the bank on the features it
scores; a fifth to two fifths of that gain transfers to other style features; LLM
judges and a human annotator do not see it; S2's contextualized queries are preferred to plain forward
generation and to generic questions in blind pairwise judgments. The Discussion
paragraph "Realism requires selection" is reversed: selection is not enough, the
generator has to produce the missing surface forms.

---

## R3. What S3 does

| | CE | PF | HW |
|---|---|---|---|
| candidates / dropped by S3 | 1075 / 68 | 1075 / 9 | 1080 / 54 |
| delivered-bank queries unchanged without S3 (of 50) | 47 | 50 | 49 |
| Spearman, fan-out score vs plain sim(q, k_src) | 0.79 | 0.59 | 0.79 |
| S3 drops also dropped by a plain-similarity filter at the same drop rate | 53% | 11% | 50% |
| controlled drift, wrong cluster at random: rejected by S3 / by plain similarity | 98.1% / 98.3% | 97.1% / 97.4% | 98.4% / 99.0% |
| controlled drift, nearest wrong cluster: rejected by S3 / by plain similarity | 84.7% / 86.5% | 69.3% / 70.7% | 85.6% / 90.6% |

"Controlled drift" pairs every candidate with a source cluster it was not
generated for and asks whether the filter rejects it; the plain-similarity
threshold is set to the same false-rejection rate as S3.

On the real banks (R1) the bank with and without S3 is identical (50 of 50).

**Reading.** S3 is a working drift detector: it rejects 97-98% of candidates
attached to a wrong cluster at a 1-6% false-rejection rate. It is not a better
detector than cosine similarity between the query and its source cluster, which
needs no LLM call. Its drops differ from the plain filter's in about half the
cases, but nothing we measured shows those drops to be the better ones, and
several are artefacts of heterogeneous clusters (for example "OLED vs QLED"
dropped from a cluster labelled "2023 consumer electronics reviews"). With a
generator that drifts in 1-6% of candidates, the delivered bank changes by 0-3
queries and no downstream number moves.

**Consequence for the paper.** S3 is kept as an optional grounding audit with
recorded scores, described as an LLM simulation, and removed from the list of
contributions. The phrases "the engine's internal retrieval path" and "the
essential difference from naive paraphrasing" are deleted. No claim in RQ1-RQ4
depends on S3.

**Sensitivity to the fan-out model** (`rb10`). The identical fan-out prompt was
re-run on all 3,230 candidates with two other model families and compared with
the submitted run (Gemini 2.5 Flash-Lite), at the same threshold 0.62:

| Fan-out model | Drop rate | Kept / dropped mean score | rho of scores with submitted | Same keep/drop decision | Cohen's kappa | Delivered-bank queries unchanged (of 50) |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash-Lite (submitted) | 4.1% | 0.78 / 0.57 | | | | |
| GPT-4o-mini | 2.6% | 0.78 / 0.57 | 0.81 | 97.0% | 0.45 | 49.3 |
| Llama 3.3 70B | 3.0% | 0.78 / 0.57 | 0.82 | 96.6% | 0.42 | 49.0 |

The score is fairly stable across fan-out models (rho 0.8) and the delivered bank
is almost identical (49 of 50). Which individual candidates fall below the
threshold is only moderately reproducible (kappa 0.4-0.5), as expected when 3-4%
of candidates sit near a cut-off. The kept/dropped means are identical for every
model because they are a property of thresholding one score at 0.62, not evidence
of validity.

---

## R4. Coverage outside the S1 embedding space

Two encoders that the pipeline never used (NVIDIA Nemotron-3 Embed 1B, Liquid LFM
2.5 Embedding). The full keyword bank, including the held-out 20%, is re-clustered
in each encoder at k = 12, 18, 24, 30 (three seeds each), and each bank is scored
by nearest-cluster assignment with log-volume weights.

Semi-synthetic banks of the paper, B = 50:

| Method | Cov (S1, paper) | Nemotron | LFM |
|---|---|---|---|
| KW-as-Query | 94.9 | 92.7 | 93.5 |
| PAA-Template | 96.7 | 94.1 | 94.3 |
| Naive Paraphrase | 98.4 | 92.2 | 93.2 |
| Forward Fan-out | 97.4 | 94.6 | 92.6 |
| GEO-bench-like | 79.6 | 78.7 | 79.2 |
| Q-SEED | 100.0 | 84.8 | 83.7 |

By granularity of the independent clustering (both encoders pooled):

| | k=8 | k=12 | k=18 | k=24 | k=30 | k=40 |
|---|---|---|---|---|---|---|
| Naive Paraphrase | 99.9 | 97.8 | 95.0 | 90.8 | 87.1 | 77.4 |
| Forward Fan-out | 99.9 | 99.4 | 95.8 | 91.3 | 87.9 | 79.2 |
| GEO-bench-like | 89.2 | 88.1 | 81.2 | 77.0 | 69.4 | 64.4 |
| Q-SEED | 98.8 | 95.9 | 87.9 | 81.6 | 71.7 | 60.2 |

Tight budgets, greedy prefix vs random draws from the same pool (independent
encoders): B=10: 53 vs 44; B=15: 68 vs 56; B=20: 75 vs 63; B=25: 76 vs 69.

**Reading.** The reviewer's concern is correct. The 100% in Table 2 is coverage of
the 18 clusters the method itself formed. In an independent space Q-SEED covers
coarse intents (99% at k=8, 96% at k=12) and loses ground as the partition gets
finer than its own M. The keyword-sampling baselines draw 50 different keywords
and so spread more evenly at fine granularity. What survives independent
evaluation is (a) the advantage over a keyword-agnostic set and (b) the advantage
of greedy over random selection at tight budgets.

**Cause and remedy.** The cause is the granularity M of the objective, not the
objective. Re-selecting from the same candidate pool with the same F, with the
intent space cut finer (selection is local, so this costs nothing):

| Intent units in F | lambda | Cov (indep., k=12-40) | AUC (S4 space) | queries shared with paper bank |
|---|---|---|---|---|
| M = 18 clusters (paper) | 0.35 | 79.5 | 0.867 | 50 |
| M = 40 | 0.35 | 87.1 | 0.950 | 15 |
| M = N, one unit per keyword | 0.35 | 89.5 | 0.959 | 17 |
| M = N | 2 | 90.0 | 0.925 | |
| **M = N** | **5** | **88.1** | **0.874** | |
| M = N | 15 | 85.5 | 0.833 | |
| Naive Paraphrase (reference) | | 89.6 | 0.932 | |
| Forward Fan-out (reference) | | 90.7 | 0.940 | |

With one facility per keyword and lambda = 5, the same pool yields a bank whose
independent coverage (88.1) is within 2-3 points of the keyword-sampling
baselines while keeping the paper's realism level (0.874 against 0.867). With
M = 18 and B = 50, the picks made after the 18 clusters are covered are driven
mostly by the realism term; this is why coverage was coarse and realism high. lambda = 5 was chosen after
seeing this table and is reported as such.

**Consequence for the paper.** Coverage is reported in an independent encoder
alongside the S1 metric; "complete intent coverage" becomes "complete coverage of
the S1 intent clusters"; the default objective uses per-keyword facilities, which
is the textbook facility-location form and leaves Theorem 1 unchanged.

---

## R5. Selection baselines on the same candidate pool

All selectors draw B = 50 from the identical post-S3 pool. "+ realism" variants
receive the same r(q) score Q-SEED uses. Coverage is the paper's S1 metric; AUC is
the 10-seed S4-space value.

| Selector | Cov@10 | Cov@15 | Cov@20 | Cov@50 | F / F_greedy | mean r | AUC | Vendi |
|---|---|---|---|---|---|---|---|---|
| random | 42.3 | 54.7 | 65.8 | 92.9 | 0.893 | 0.047 | 0.969 | 17.9 |
| k-means medoids | 52.0 | 75.5 | 89.9 | 97.7 | 0.955 | 0.052 | 0.972 | 18.6 |
| k-center (max-min) | 54.2 | 75.5 | 97.7 | 100.0 | 0.905 | 0.084 | 0.958 | 23.7 |
| MMR (alpha 0.5) | 48.7 | 77.0 | 95.6 | 100.0 | 0.982 | 0.074 | 0.960 | 21.5 |
| MMR (alpha 0.7) | 41.6 | 77.2 | 85.9 | 97.9 | 0.985 | 0.082 | 0.957 | 17.3 |
| MMR (alpha 0.5) + realism | 47.3 | 76.5 | 90.1 | 100.0 | 0.975 | 0.323 | 0.831 | 21.3 |
| k-means + realism | 53.9 | 79.3 | 93.6 | 100.0 | 0.932 | 0.310 | 0.848 | 20.8 |
| S1-stratified + realism | 73.2 | 92.7 | 100.0 | 100.0 | 0.940 | 0.357 | 0.817 | 19.5 |
| **Q-SEED greedy** | 62.9 | 86.3 | 97.2 | 100.0 | 1.000 | 0.323 | 0.860 | 16.7 |

**Reading.** (1) Without a realism score no selector moves AUC (0.96-0.97): the
realism gain in the S4 space comes from r(q), not from the optimizer. (2) Among
generic selectors (MMR, k-center, k-means) greedy facility location has the best
coverage at B = 10 and 15, by 7-21 points. (3) A round-robin over the S1 clusters
that takes the most realistic candidate generated for each cluster beats greedy
on S1 coverage at tight budgets (73 vs 63 at B = 10) and matches it on realism.
That heuristic is Q-SEED's own S1 and S4 with a simpler S5; it optimizes the
assignment metric directly, whereas F optimizes similarity mass. The gains
therefore come from the intent decomposition and the realism score. Greedy on F
adds the guarantee and the best objective value, not a coverage margin over an
intent-aware heuristic.

---

## R6. Stability of the 2PL item parameters

| | CE | PF | HW | mean |
|---|---|---|---|---|
| Refit on the independent retest round: rho(a_q) | 0.94 | 0.98 | 0.98 | 0.97 |
| ... rho(b_q) | 0.95 | 0.99 | 0.98 | 0.97 |
| ... rho(theta_d) | 0.98 | 0.99 | 0.98 | 0.98 |
| ... top-quartile-a_q overlap (Jaccard) | 0.79 | 0.82 | 0.79 | 0.80 |
| Bootstrap over domains (200): rho(a_q) with full fit [95% CI] | 0.61 [0.36, 0.78] | 0.79 [0.52, 0.92] | 0.74 [0.50, 0.90] | 0.71 |
| ... rho(b_q) | 0.61 [0.29, 0.81] | 0.84 [0.60, 0.95] | 0.74 [0.48, 0.91] | 0.73 |
| ... top-quartile overlap | 0.39 | 0.52 | 0.48 | 0.46 |
| Disjoint halves of the domains (50 splits): rho(a_q) | -0.07 | 0.36 | 0.24 | 0.18 |
| ... rho(b_q) | -0.09 | 0.51 | 0.23 | 0.22 |
| Top-quartile / bottom-quartile mean a_q across bootstrap [95% CI] | 8.0 [3.6, 15.7] | 11.7 [6.1, 18.8] | 12.5 [5.5, 23.0] | 10.7 |

**Reading.** Item parameters are stable under re-probing (rho 0.97): engine
stochasticity is not the problem. They are only moderately stable when the domain
panel is resampled (rho about 0.7) and unstable across disjoint domain halves (rho
about 0.2, and zero in CE). An item's discrimination is therefore a property of
the item together with the domain panel. Two statements in the paper survive:
discrimination is heterogeneous (the top quartile is 8 to 12 times the bottom
quartile, with a lower 95% bound of 3.6 or more in every vertical), and selecting by it does not help. A third
is now explained: compression by a_q fails because a_q does not transfer to
domains the fit did not see.

---

## R7. The cross-engine disagreement

**What engine B is.** Both engines are called through the OpenRouter API, not
through a web interface. `perplexity/sonar` is Perplexity's own Sonar API, with
Perplexity's retrieval. `openai/gpt-4o-mini:online` is OpenRouter's web plugin:
for a model without provider-native search it retrieves the top five results from
the Exa search API for the user query, injects them into the prompt, and returns
them as `url_citation` annotations. Verified on 1 Oct 2026: pinning
`engine: "exa"` reproduces the default behaviour and cost; `engine: "native"` is
rejected for this model. Engine B is therefore "Exa retrieval + GPT-4o-mini
generation". It is a retrieval-augmented cited-answer engine, not the consumer
ChatGPT Search product. In a spot check, the same query sent through the same
plugin with three different generators (GPT-4o-mini, Gemini 2.5 Flash, Claude
Haiku 4.5) returned the same five URLs: for this engine the cited set is fixed at
retrieval.

**Decomposition of the disagreement** (cached probes of the submitted run):

| | CE | PF | HW | mean |
|---|---|---|---|---|
| Test-retest of the domain ranking, Sonar (rho) | 0.98 | 0.99 | 0.98 | 0.98 |
| Test-retest, Exa + GPT-4o-mini (rho) | 0.96 | 0.98 | 0.97 | 0.97 |
| Cross-engine (rho) | 0.15 | 0.22 | 0.13 | 0.17 |
| Cross-engine, corrected for the two reliabilities | 0.16 | 0.23 | 0.13 | 0.17 |
| Per-query cited-domain Jaccard, same engine, two rounds (Sonar / Exa) | 0.88 / 0.84 | 0.84 / 0.85 | 0.84 / 0.82 | 0.85 / 0.84 |
| Per-query cited-domain Jaccard, across engines | 0.12 | 0.14 | 0.13 | 0.13 |
| Distinct domains cited per answer (Sonar / Exa) | 11.9 / 4.7 | 12.5 / 4.8 | 13.1 / 4.6 | 12.5 / 4.7 |
| Distinct domains cited over the whole bank (Sonar / Exa / both) | 524 / 305 / 131 | 603 / 278 / 122 | 739 / 335 / 151 | 622 / 306 / 135 |
| Share of citation mass on domains both engines cite (Sonar / Exa) | 0.59 / 0.66 | 0.47 / 0.68 | 0.48 / 0.67 | 0.52 / 0.67 |
| rho of citation counts on the domains both cite | 0.60 | 0.59 | 0.45 | 0.55 |

Across-engine Jaccard does not depend on the query's intent facet (comparison
0.11-0.14, recommendation 0.11-0.14, how-to 0.12-0.14).

**Reading.** (1) The disagreement is not measurement noise: each engine reproduces
its own ranking at rho 0.97-0.98, so the reliability-corrected cross-engine
correlation equals the raw one. (2) It is mostly a difference in which sources
enter the answer at all: the two engines share 135 of about 790 distinct domains
(Jaccard 0.17), and about half of Sonar's citation mass goes to domains the other
engine never cites. (3) On the domains both cite, agreement is moderate (rho
0.55). (4) The engines have different citation budgets (12.5 vs at most 5 domains
per answer). The evidence points to the retrieval stage (index and result-set
size) more than to answer generation, and for engine B the generator has no
influence on the cited set.

**Five engines** (`rb11` part A'; first 20 queries of each delivered bank, 60
queries in all; three engines added). OpenAI native = GPT-4.1-mini with OpenAI's
own web search; Anthropic native = Claude Haiku 4.5 with Anthropic's own web
search; Exa + Gemini = the same Exa retrieval as the submitted engine B with a
different generator (Gemini 2.5 Flash-Lite). Entries: Spearman rho of domain
visibility over the pooled domain universe / mean per-query Jaccard of cited
domains.

| | Exa + GPT-4o-mini | Exa + Gemini | OpenAI native | Anthropic native |
|---|---|---|---|---|
| Sonar | 0.35 / 0.13 | 0.30 / 0.12 | -0.05 / 0.03 | 0.26 / 0.06 |
| Exa + GPT-4o-mini | | **0.88 / 0.77** | 0.02 / 0.02 | 0.13 / 0.04 |
| Exa + Gemini | | | 0.01 / 0.02 | 0.06 / 0.03 |
| OpenAI native | | | | 0.11 / 0.02 |

Within-engine reference: two repeats of OpenAI native agree at per-query Jaccard
0.53; Sonar and Exa at 0.85 and 0.84 (table above).

Holding retrieval fixed and changing the generator leaves the citations almost
unchanged (rho 0.88, Jaccard 0.77). Every pair of engines with different retrieval
stacks disagrees (Jaccard 0.02-0.13, rho -0.05 to 0.35). Four retrieval stacks,
six pairs, one pattern: the cited set is decided by retrieval.

**Twelve-week drift** (`rb11` part A, `rb14`). The delivered banks were probed in
July (three repeats) and again in October (one repeat), same queries, same domain
universe.

| Mean over verticals | Sonar | Exa + GPT-4o-mini |
|---|---|---|
| Domain-ranking agreement, July vs October (rho / Kendall tau) | 0.57 / 0.45 | 0.63 / 0.53 |
| Per-query cited-domain Jaccard, July vs October | 0.26 | 0.28 |
| Distinct domains per answer, July -> October | 12.3 -> 16.4 | 4.7 -> 4.7 |
| Share of citations to video/community sites, July -> October | 9.6% -> 0.9% | 0% -> 0.1% |

Cross-engine rho on the same queries: **0.17 in July, 0.50 in October** (CE 0.18
-> 0.47, PF 0.21 -> 0.54, HW 0.13 -> 0.50).

In July Sonar cited `youtube.com` in 57-78% of answers and `reddit.com` in about
half of the consumer-electronics and finance answers. In October neither is in
its top five in any vertical; its top sources are editorial and reference sites
(`nytimes.com`, `techradar.com`, `forbes.com`, `nerdwallet.com`,
`healthline.com`), the same kind the Exa-backed engine cites. The Exa-backed
engine's source mix did not change. One engine changed its source policy, and
cross-engine agreement tripled.

**Reading of the drift.** (1) The July figure of 0.17 was a snapshot. The level of
cross-engine agreement depends on the engines' current source policies and moved
from weak to moderate in twelve weeks. (2) The July disagreement was largely a
source-type effect: one engine cited video and community sites and the other did
not. (3) An engine agrees with its own ranking of twelve weeks earlier (0.57-0.63)
about as much as two engines agree with each other on the same day (0.50).
Visibility is specific to the engine and to the date. Short-horizon reliability
(rho 0.97-0.98 between rounds issued within the study window) is unaffected.

**Consequence for the paper.** "Structural" is replaced by measured statements:
the disagreement is not noise, it is decided at retrieval, and its size depends on
the engines' source policies at the time of measurement. The headline becomes
"0.17 in July and 0.50 in October 2026", the retest section gains the 12-week
figures, the table of top domains is dated, and the recommendation becomes "report
per engine and per date". Engine B is named for what it is (Exa retrieval +
GPT-4o-mini) in the setup, tables and abstract; the tables currently abbreviate it
as "GPT-4o".

---

## R8. Style selection and the cited domains; the reference bank

**Does selecting for style change which domains are cited?** Among the 100 probed
queries per vertical, the 50 with the highest realism score (mean r = 0.43) and
the 50 with the lowest (mean r = 0.13) give domain-visibility rankings that
correlate at rho = 0.585. Random 50/50 splits of the same queries give 0.563 (95%
range 0.42-0.69). The style split sits at the 34th-78th percentile of the random
splits in the three verticals. Realism score and number of domains cited are
uncorrelated (rho 0.09, -0.15, 0.01). No bias is detectable.

**Diversity.** The low Vendi score does not come from the style term: removing S4
lowers Vendi further (15.3 against 16.7, ablation table of the paper). It comes from
facility-location selection at coarse granularity, the same cause as R4; the
per-keyword objective (lambda = 5) raises it to 19.0, in every vertical
(`rb9f`).

**How independent is the convergent-validity reference bank?**

| | CE | PF | HW |
|---|---|---|---|
| Words per query: reference / Q-SEED | 7.5 / 17.2 | 9.2 / 18.8 | 8.6 / 16.9 |
| Mean best content-token Jaccard of a Q-SEED query to any reference query | 0.21 | 0.22 | 0.23 |
| Q-SEED queries with a near-duplicate in the reference bank (Jaccard >= 0.6) | 1 | 0 | 1 |
| Share of bank log-volume covered by the 60 keywords the reference uses | 30% | 29% | 30% |
| Convergent rho (paper) | 0.82 | 0.69 | 0.54 |
| Split-half reliability (Spearman-Brown, 200 splits): reference / Q-SEED bank | 0.87 / 0.67 | 0.90 / 0.66 | 0.84 / 0.52 |
| Convergent rho corrected for both reliabilities | 1.07 | 0.90 | 0.82 |

**Reading.** The two banks differ in generator, prompt, length and wording, and
the reference is built from the 60 highest-volume keywords only. They share the
keyword bank and the fact that an LLM wrote them. The coefficient is therefore
alternate-form agreement between two keyword-grounded instruments, and after
correcting for the finite length of both banks the two forms agree almost
completely (0.8-1.0). It is not criterion validity against real user behaviour,
and the paper will say so. 
**Real-user-query criterion** (`rb11` part C). Real Bing user queries from MS MARCO
that contain a head term (for example "what is the best laptop to use for gaming",
"can you have a 401k and a roth ira", "what can lower blood pressure") were probed
on both engines the same day as the Q-SEED bank.

| | CE | PF | HW | mean |
|---|---|---|---|---|
| Real user queries found | 17 | 21 | 29 | |
| rho of domain ranking, criterion bank vs Q-SEED bank | 0.61 | 0.65 | 0.43 | 0.56 |
| ... Sonar / Exa | 0.62 / 0.46 | 0.59 / 0.38 | 0.41 / 0.49 | |
| Split-half reliability, criterion bank / Q-SEED bank (one repeat) | 0.71 / 0.72 | 0.81 / 0.60 | 0.77 / 0.31 | |
| rho corrected for both reliabilities | 0.85 | 0.94 | 0.88 | 0.89 |

A bank written by no LLM ranks domains much as the Q-SEED bank does (rho 0.56,
about 0.9 once the shortness of both banks is accounted for). The criterion bank
is small and MS MARCO queries are search-engine questions, not chat prompts, so
this supports the alternate-form reading without settling criterion validity.

---

## Cost, reproducibility, and what is still open

Total API spend of the rebuttal experiments: $7.93 (OpenRouter key usage; itemised
in `qseed/results/rebuttal/_ledger.json`). No existing cache entry was modified: rebuttal-only calls go to
separate namespaces (`rb_chat`, `rb_probe`), and the real-bank runs add entries
under new keys (`*_real`). With the cache in place every script re-runs offline
(`QSEED_OFFLINE=1`).

Still open:
- more than one human annotator;
- a brand's private keyword export with monthly volumes (ours is a public
  inventory with a popularity index);
- more than one repeat and more than 20 queries per vertical for the three added
  engines.
