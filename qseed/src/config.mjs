// Central configuration for the Q-SEED pipeline.
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
export const ROOT = path.resolve(__dirname, '..');

// Load OPENROUTER_API_KEY from ../.env (project root, one level above qseed/).
function loadEnv() {
  const envPath = path.resolve(ROOT, '..', '.env');
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
      if (m) process.env[m[1]] ??= m[2];
    }
  }
}
loadEnv();

export const API_KEY = process.env.OPENROUTER_API_KEY;
export const BASE_URL = 'https://openrouter.ai/api/v1';

export const MODELS = {
  embed: 'openai/text-embedding-3-small', // 1536-dim
  generator: 'openai/gpt-4o-mini',        // S1 label, S2 generate, keyword expansion
  fanout: 'google/gemini-2.5-flash-lite', // S3 reverse fan-out (DIFFERENT family, avoids self-verification)
  judge: 'openai/gpt-4o-mini',            // realism human-proxy / misc
};

// The generative engines probed for citations. Both expose OpenAI-style
// message.annotations[].url_citation, so domain extraction is uniform.
export const ENGINES = [
  { id: 'sonar', model: 'perplexity/sonar' },
  { id: 'gpt4o-online', model: 'openai/gpt-4o-mini:online' },
];

export const CONFIG = {
  B: 50,                 // query-bank budget per vertical
  candidatesPerB: 20,    // target candidate-pool size = candidatesPerB * B
  nClustersHeuristic: (N) => Math.max(10, Math.min(24, Math.round(Math.sqrt(N)))),
  keywordsPerVertical: 240,
  holdoutFrac: 0.20,     // fraction of keywords held out for Intent-Recall
  tau: 0.62,             // cycle-consistency / coverage cosine threshold
  lambda: 0.35,          // realism weight in submodular objective F(Q)
  probeRepeats: 3,       // r: repeats per (query, engine)
  Q0factor: 2,           // |Q0| = Q0factor * B (probed pool before IRT distillation)
  refBankSize: 60,       // independent real-query reference bank size per vertical (external validity)
  refBankRepeats: 2,
  facets: ['comparison', 'how-to / tutorial', 'troubleshooting', 'recommendation / best-of', 'fact / definition'],
  personas: ['a novice first-time buyer', 'a domain expert', 'a budget-conscious shopper', 'a time-pressured professional'],
  scenarios: ['an urgent problem', 'long-term research', 'comparing options before buying', 'a repeat/return decision'],
  funnels: ['awareness', 'consideration', 'decision'],
  seed: 20260706,
};

export const VERTICALS = ['consumer_electronics', 'personal_finance', 'health_wellness'];

export const DIRS = {
  cache: path.join(ROOT, 'cache'),
  data: path.join(ROOT, 'data'),
  results: path.join(ROOT, 'results'),
  figures: path.join(ROOT, '..', 'IEEE_Conference_Template__3_', 'figs'),
};
for (const d of Object.values(DIRS)) fs.mkdirSync(d, { recursive: true });

// In offline mode every call is served from the cache, so no key is required --
// which is exactly the path a reviewer reproducing the released artifact takes.
if (!API_KEY && process.env.QSEED_OFFLINE !== '1') {
  console.error('FATAL: OPENROUTER_API_KEY not found in environment or ../.env');
  console.error('       (set QSEED_OFFLINE=1 to reproduce entirely from the cache)');
  process.exit(1);
}
