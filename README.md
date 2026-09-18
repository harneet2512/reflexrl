# ReflexRL

Can reinforcement learning teach a multimodal foundation model **when to
think** — allocating its own inference compute while acting in a
first-person world?

This repo tests one falsifiable hypothesis on ViZDoom `defend_the_center`:

> Under an explicit compute cost, an RL-learned depth router achieves a
> better task-performance/compute tradeoff than fixed-depth, random-depth,
> and confidence-based inference.

The agent is Qwen3-VL-2B (frozen) with three intermediate exits —
REFLEX (layer 9), FAST (19), DEEP (28) — plus a trained router, action
heads, and value head. Reward = task reward − λ·C(d)/C(DEEP). PPO trains
the heads and router on cached hidden states; the backbone never sees
gradients.

**Budget: $20 of Modal credits.** This is a mechanism demonstration, not
a benchmark claim — the full PRD matrix (multiple scenarios, λ sweep,
3+ seeds, continual learning) is cut. What $20 buys: one ReflexRL run,
three trained baselines, matched-compute evals, a post-hoc oracle
frontier, automaticity-over-training analysis, and a small LevDoom
novelty probe — all reported as preliminary and underpowered.

## Layout

```
reflexrl/
  envs/vizdoom_env.py    pixels-only DTC wrapper; debug_state() is eval-only
  models/qwen_vl.py      dual-mode backbone: features_collect / features_to
  models/heads.py        ExitPolicy: norms, action heads, router, value
  rl/                    rollout, GAE buffer, PPO on cached features, trainer
  baselines/cnn_policy.py  CNN specialist (trains locally, $0)
  eval/                  evaluate (oracle/matched/confidence), automaticity,
                         novelty (LevDoom), latency, compute FLOP model
  analysis/              pareto frontier, depth-over-training curves
  demo/compose.py        offline mp4: gameplay + compute-bar overlay
training/modal_app.py    Modal entrypoints: run_tests, profile, train, evaluate
configs/                 phase budgets and run matrix
tests/                   gates: env, model, compute, eval
scripts/                 smoke_e2e.py, train.py
```

## Ground rules (enforced by tests)

- Policy input is stacked RGB frames **only**. `debug_state()` exists for
  eval logging and is prohibited from the policy path.
- Exits are real: `features_to(d)` truncates the text stack; a gate
  asserts truncated-vs-collected hidden states match exactly.
- FLOPs are analytic (HF config), latency is measured; the two are kept
  separate so the Pareto claim doesn't depend on one machine.
- `λ=0` exists in configs: if the router doesn't collapse to DEEP with
  free compute, the penalty mechanism is broken, not interesting.

## Run it

Local (free): CPU smoke of the full loop with a stub backbone —

```bash
pip install -e .[dev]
pytest tests                      # all gates, CPU
python scripts/smoke_e2e.py       # 512-decision train+ckpt+metrics
python scripts/train.py --agent cnn --decisions 300000 --workdir runs/cnn0
```

Paid (Modal — CPU gates first, GPU after payment method is attached):

```bash
modal run training/modal_cpu.py::run_tests         # 25 gates, ~$0.003
modal run training/modal_cpu.py::budget_status     # spend ledger
modal run training/modal_app.py::profile           # latency/FLOPs on L4
modal run training/modal_app.py::train --tag pilot --total-decisions 150000
modal run training/modal_app.py::evaluate --run-dir /runs/pilot_s0 --mode router
```

Spend controls (four layers):

1. `max_containers=4` on every function — hard concurrency ceiling.
2. **Pre-launch gate**: persistent ledger on the runs volume
   (`reflexrl.cloud_budget`) refuses any run whose estimate pushes
   recorded spend past the **$20 hard cap**.
3. **In-run wind-down**: a `BudgetWatchdog` is checked every training
   iteration and every eval episode. It re-reads the ledger each check
   (so concurrent runs' spend counts) and raises `BudgetExhausted` at
   the cap — the run saves `ckpt_final.pt` + `done.json` with
   `stopped_reason` and exits instead of burning past $20 on a bad
   estimate.
4. **Police + billing kill switch**: the deployed `reflexrl-cpu` app
   runs `budget_police` every 5 min — when the ledger hits the cap it
   calls `AppStop` on every reflexrl app (including itself last). And
   the workspace spend limit in Modal dashboard should be set to $20
   as the billing-level backstop none of the code layers can exceed.

Every run writes `config.json`, `metrics.jsonl`, and `ckpt_*.pt` to a
Modal volume; nothing lives only in a notebook.

## Failure criteria (pre-registered)

The hypothesis is reported as **failed** if: confidence routing matches
ReflexRL; depth doesn't correlate with state difficulty; the local CNN
dominates on every axis; or the policy can't learn in-budget. A negative
result ships as a negative result.

## Status

Phase-0 gates: ✅ 25/25 pass on Modal CPU image ($0.003 spent of $20
ledger). ⛔ GPU functions blocked until a payment method is attached to
the Modal workspace — credits apply first once one is. Next after that:
`profile` → pilot → matrix. See `configs/phases.json`.
