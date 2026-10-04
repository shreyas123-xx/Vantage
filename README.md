# Drift Impact Estimator

Detects dataset drift between a reference (training/baseline) dataset and a
current (production) dataset, **predicts how much downstream model
performance will degrade because of it**, and recommends the smallest
retraining schedule that keeps performance above a threshold. Validated with
real, production-like classifiers and a multi-week longitudinal simulation.

Most drift-monitoring tools stop at "here's how much the data moved." This
project's core novelty is the second step: an **impact model** trained on
hundreds of simulated production scenarios (real classifiers trained,
drifted, and re-evaluated) that maps a drift signature onto predicted
performance loss, with confidence bands — plus a scheduler that turns that
into a minimal, cost-optimal retraining cadence instead of "just retrain
weekly and hope."

## Stack

- **Backend**: FastAPI + scikit-learn/scipy/pandas/numpy
- **Frontend**: React (Vite) + Recharts, plain CSS (no framework)

## Architecture

```
backend/
  app/
    core/
      drift_metrics.py     PSI, KS test, Jensen-Shannon divergence, Wasserstein
                            distance per feature; classifier two-sample test
                            (C2ST) for a multivariate drift summary.
      synthetic_data.py     Production-like labelled datasets with controllable
                            covariate/concept drift, for demos and simulation.
      impact_model.py       The novel piece: simulates hundreds of scenarios
                            (train a real classifier, measure its true
                            performance drop under drift, record the drift
                            signature) and fits gradient-boosted regressors
                            (+ quantile bands) mapping drift -> degradation.
      scheduler.py          Turns a drift trend + the impact model into a
                            minimal-cost retraining schedule (never / fixed
                            periodic / adaptive threshold).
      longitudinal.py       Multi-week deployment simulation: real classifier,
                            real retraining, real measured performance vs. the
                            impact model's blind predictions, side by side.
    api/                     FastAPI routers (datasets, drift, impact, schedule,
                             experiments)
    models/schemas.py        Pydantic request/response models
  train_impact_model.py      Resumable CLI to (re)train the impact model
  artifacts/                 Trained model + the simulated training scenarios

frontend/
  src/
    components/              DriftPanel, ImpactPanel, SchedulePanel,
                              LongitudinalPanel
    api/client.js             Thin axios wrapper around the backend
```

## Setup

### Backend

```bash
cd  D:\drift-impact-estimatorr\project\backend
python3 -m venv .venv 
# Activate the virtual environment:
source .venv/bin/activate          # Mac / Linux
# OR for Windows PowerShell:
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Train the impact model (one-time; ~5-6 minutes on a laptop).
# If you're on a machine that kills long-running processes, run it in chunks
.\.venv\Scripts\python.exe train_impact_model.py # (single-shot, ~450 scenarios; the
# shipped artifacts/ was built from 1050 via the chunked commands above)

.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

The API is now at `http://localhost:8000` (interactive docs at `/docs`).

### Frontend

```bash
cd D:\drift-impact-estimatorr\project\frontend
npm install
npm run dev
```

Open the printed localhost URL. It talks to the backend at
`http://localhost:8000` by default; override with a `VITE_API_URL` env var
if you're running the backend elsewhere.

## Using it

1. **Drift tab** — choose how to provide your data using the toggle:
   - **Synthetic generator**: Generate a simulated baseline/production pair with configurable magnitude, drift mechanism (covariate, concept, mixed), row counts, and seed[cite: 1].
   - **Upload custom CSVs**: Upload your own two CSV files directly in the UI — one for your **Reference** (baseline/training) data and one for your **Current** (production) data, with an optional target column name.
   Once generated or uploaded, running the analysis displays per-feature PSI, KS test results, Jensen-Shannon divergence, and a multivariate classifier two-sample test (C2ST) AUC, along with an overall drift score and severity badge.

2. **Impact tab** — reads that drift signature and predicts the expected drop
   in accuracy / F1 / ROC AUC, with a confidence interval and feature
   importances, plus the impact model's own validation metrics (R², MAE,
   broken out by drift type — see "Honest limitations" below).

3. **Schedule tab** — enter an observed drift history (or use the defaults),
   retraining cost, and a per-day cost of running a degraded model, and get a
   minimal-cost recommended retraining cadence versus a fixed-periodic
   baseline and a do-nothing baseline.

4. **Longitudinal tab** — the validation experiment: deploys a real
   RandomForest over many simulated weeks of drifting data, retrains it
   according to a chosen strategy, and plots the impact model's *predicted*
   degradation against the model's *true*, measured degradation each week —
   something only possible to check honestly in a simulation with ground
   truth, which is exactly why this exists.


### Custom CSV Upload Requirements

When uploading your own 2 files in the Drift tab:
- **Two separate files**: Provide one CSV as your baseline (Reference) and one CSV as your production batch (Current).
- **Matching headers**: Both files must share identical feature column names.
- **Numeric columns**: Drift statistics are calculated over numeric features.
- **Target column**: Optional; if your files contain a label/target, specify its column name so it is excluded from input feature drift calculations.


## Honest limitations (read this before trusting the numbers)

**Pure concept drift is structurally invisible to every metric in the drift
tab.** PSI, KS, Jensen-Shannon divergence, Wasserstein distance, and the
classifier two-sample test all operate on the *input* feature distributions.
If the input distribution P(X) doesn't move but the relationship P(y|X) does
(e.g. fraud patterns evolve, user preferences shift), none of these metrics
will show anything unusual — and this is true of PSI/KS/MMD/C2ST-based
drift detection in general, not a bug specific to this implementation.
Detecting it requires ground-truth or proxy labels, which is exactly why
production systems pair drift monitoring with a delayed-label feedback loop
(the Longitudinal tab's "true" performance line is standing in for that).

This shows up concretely in the trained impact model's validation (visible on
the Impact tab and in `artifacts/impact_models.joblib`'s stored metrics):
MAE is meaningfully lower for covariate/mixed drift scenarios (where the
input-distribution signal is real) than for concept-drift or no-drift
scenarios, because concept-drift and no-drift scenarios can produce nearly
identical drift feature vectors while having very different true performance
outcomes — the regressor can only split the difference between them. On the
1050 simulated scenarios shipped in this repo, accuracy-drop MAE is 0.086
(covariate) and 0.068 (mixed) versus 0.100 (concept) and 0.074 (none) — the
same pattern holds for f1 and roc_auc. Overall R² is roughly 0.08-0.20
depending on metric (see your own `/api/model-status` for current numbers);
that's a realistic, un-inflated reflection of the limit above, not a
modeling failure to be tuned away. As direct evidence of that: doubling the
training scenarios from ~450 to 1050 did *not* improve R² (it went from
~0.27 to ~0.20 for accuracy, with a larger, more reliable held-out test set) —
if this were a data-volume problem, more scenarios would have helped. They
didn't, because the ceiling is an identifiability problem (two different
ground-truth outcomes sharing the same visible features), not a sample-size
problem. More scenarios chiefly tightened the per-drift-type MAE estimates'
reliability, which is still useful.

**The synthetic data is a benchmark, not a substitute for your production
data.** It's built so reference and current batches are independent draws
from the same generative process at magnitude 0 (an earlier version of this
generator had a bug where they weren't, which manufactured fake drift at
"zero drift" — worth knowing if you extend it) and diverge in a controlled,
known way as magnitude increases. That makes it good for validating the
*mechanism*, not a claim that the numbers transfer directly to your models.

**The in-memory dataset/analysis store is deliberately simple** (see
`app/storage.py`) and won't survive a process restart or scale across workers
— swap in Redis/Postgres for anything beyond a demo or single-user tool. It
is now bounded (200 entries each, LRU eviction) so a long demo session
doesn't grow memory without limit; older entries simply stop resolving.

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

29 tests cover the drift metrics (including the severity-threshold and
all-NaN-column edge cases below), the synthetic data generator (the
zero-drift bug below is now a regression test), the scheduler (both bugs
below are regression tests), the bounded store's eviction, and an API-level
smoke test through the full dataset → drift → impact → schedule pipeline
plus the longitudinal endpoint.

## Changelog: bugs found and fixed after the first pass

Keeping this honest, in the same spirit as the limitations above — these
were found by writing tests and re-reading the code, not reported by a user:

- **Inconsistent severity logic.** `analyze_drift`'s severity used one rule
  (thresholds on mean PSI) below an overall-score cutoff and a different rule
  (thresholds on the overall score itself) above it — two branches that could
  disagree near the boundary. Replaced with a single monotonic function of
  the overall score (`_overall_severity`).
- **Scheduler: immediate threshold crossing was read as "never".** If the
  impact model's predicted degradation right after a hypothetical retrain
  already met the threshold, the code looked for the first *later* day it
  crossed, found none, and concluded the threshold was never reached —
  recommending "never retrain" when it should have recommended retraining
  as often as daily. Fixed, and it now also surfaces a `floor_warning`
  explaining that the threshold is likely set below the model's irreducible
  prediction floor (see "Honest limitations" above) rather than silently
  doing the wrong thing.
- **Scheduler: short horizons silently dropped the periodic strategy.** The
  candidate retraining intervals were a fixed list starting at 7 days, so
  `horizon_days=7` (the API's own minimum) excluded every candidate and
  `best_fixed_periodic` just disappeared from the response with no error.
  Candidates are now generated relative to `horizon_days`.
- **Scheduler: a strategy with zero retrains could cost more than literally doing nothing.** `cost_for_strategy` always measured time using the since-retrain curve (reset to ~zero drift), even *before* a strategy's first retrain had happened. So a periodic or adaptive strategy that ended up triggering zero retrains computed a different cost than the literal "never retrain" baseline, even though both describe the exact same outcome. Found via a test that compared the two costs directly. Fixed by tracking whether a retrain has happened yet: before the first one, cost uses the never-retrain curve (today's actual accumulated drift); only the time *after* an actual retrain uses the since-retrain curve. The adaptive strategy's timing is now explicitly two-phase too — `trigger_after_days` (first retrain, from today's real trajectory) and `repeat_every_days` (the cadence after each reset) are reported separately instead of conflated into one number.
- **Default thresholds were below the model's own floor.** The impact
  model's predicted degradation at zero drift is ~0.12 for all three metrics
  (a direct consequence of the concept-drift blind spot above), but the
  default `degradation_threshold` was 0.03-0.05 — meaning a first-time user
  would hit the floor_warning almost immediately regardless of their actual
  data. Raised the default to 0.15 in the API schema, the scheduler, and the
  frontend sliders.
- **Longitudinal experiment simulated the requested strategy twice.** The
  detailed week-by-week view and the three-way strategy comparison table
  each re-ran the full simulation independently, so whichever strategy the
  caller asked about was computed twice. Refactored into a single
  `_simulate_strategy` helper called once per strategy (three calls total
  instead of up to four), removing the duplication and the performance cost.
- **Unbounded in-memory store.** Every generated dataset and drift analysis
  stayed in memory forever with no eviction — fine for a quick demo, a slow
  leak for a long-running one. `storage.py` is now a bounded LRU store
  (200 entries each).
- **CORS allowed credentialed wildcard origins.** `allow_credentials=True`
  with `allow_origins=["*"]` made the server reflect *any* request origin
  with credentials allowed; harmless here since nothing uses cookies, but
  unnecessary. Set to `allow_credentials=False`.
- **Defensive gap in the numeric drift path.** A numeric column that's
  entirely NaN on one side (e.g. a field a new data source hasn't populated
  yet) would call `.min()`/`.max()` on an empty array and crash. Now returns
  a "no signal" result instead.
- **Custom CSV upload via UI.** Added direct frontend support in the Drift tab for uploading custom reference and current CSV datasets via a segmented toggle, with client-side schema alignment validation, hooking into `/api/datasets/upload` and `/api/drift/analyze`.