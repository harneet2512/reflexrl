# Project status

Last updated: 2026-09-20. Generated metrics: [`results/METRICS.md`](results/METRICS.md)
(rebuild with `python scripts/metrics_report.py`).

## The goal, restated

A vision-language model knows what is in a game frame but cannot play the game.
**Move that knowledge into a small network that can, and measure the move.** At
deployment the VLM and the decision model are both gone; a 0.75M-parameter CNN plays
alone, with zero model calls.

Latency is a *consequence* of that design, not the claim. "A VLM is too slow to play a
shooter" is not a finding; everybody knows it. The findings are:

1. **Knowledge transfers, and it is worth 2.95x in sample efficiency.** More robustly, it
   is worth an 8x tighter seed-to-seed spread. A pre-registered control shows the gain is
   the *knowledge* and not the guidance machinery: cut the link between the teacher's
   advice and the frame, keep everything else, and 2.95x becomes 1.05x.
2. **What you distil matters more than how well you distil it.** Two students, same
   3,662 labels, same teacher, near-identical fidelity: copying the teacher's *actions*
   gives 0.88 kills, copying its *percept* gives 2.53.
3. **A 4.40-scoring teacher produces 7.34-scoring students.** The teacher is a
   curriculum, not a ceiling, and it is switched off by 6-10% of training.
4. **You can tell in advance whether a VLM can teach a task**, with a 20-minute
   perception probe, before spending a GPU-day on it.

## Done

| area | state | evidence |
|---|---|---|
| Teacher: Qwen3-VL-8B (NF4 weights, fp32 compute) + TypeSafe Jev 1.13 | done | 4.40 kills vs 1.67 for the same model choosing actions |
| Pre-registered teacher gate on 4 scenarios | done | `experiments/configs/phase0_gate.json`; 1 pass, 3 below bar |
| Perception probes: Doom x2, real Valorant footage x2 | done | 0.733 / 0.740 / 0.747 / 0.307 vs majority baselines |
| Perception distillation (least-squares inversion of Jev's table) | done | 2.53 kills vs 0.88 for action cloning |
| Guided PPO + adaptive handover, 3 seeds x 1.5M steps | done | **X = 2.95x**, final 7.34 |
| **Knowledge ablation** (shuffled teacher, 2 seeds x 1.5M) | done | **X = 1.05x**: the gain is the VLM's sight, not the mechanism |
| Controls: PPO from scratch, BC->PPO, fixed anneal, live DAgger | done | all in `results/METRICS.md` §1 |
| Leakage-free final evaluation, 50 unseen episodes | done | `results/final_eval.json`, seeds 7,000,000+ |
| Held-out map transfer (`defend_the_line`) | done | 3/3 seeds reach 21.5 vs 0/3 for PPO |
| Deployment benchmark (params, FLOPs, latency, cost, VRAM) | done | 751,526 params, 2.57 ms, 173x |
| Real-time evaluation (latency converted to dropped tics) | done | reflex 7.25 vs teacher -0.38 |
| 45-second demo video | done | `results/demo/reflexrl_demo.mp4` |
| Three README GIFs (teacher vs student, learning race, gameplay) | done | `scripts/make_gifs.py`, ~4 MB each, autoplay inline |
| Positioning against 2025-2026 literature | done | LVLM2P, GameSense, Playing DOOM with 1.3M Parameters |
| Test suite (pixels-only, IS correction, resume, seed hygiene, probe alignment) | done | 28 passing |
| Full local archive of every Kaggle job | done | `archive/kaggle/`, ~1.4 GB mirror on `D:\reflexrl_backup` |
| Public repo | done | https://github.com/harneet2512/reflexrl |
| Complete metrics document with sources | done | `results/METRICS.md` |

**Two measurement bugs were found and fixed during this work**, both of which had made
the teacher look *worse* than it is: Health-Gathering distance labels came from a
different rollout than the frames (now guarded by `tests/test_probe_alignment.py`), and
90 of 93 Valorant frames labelled "no enemy" actually contained a teammate, so the model
was being penalised for correctly seeing a person.

## In flight

Nothing. The knowledge ablation landed on 2026-09-20 and was the last open question.

| result | X | final |
|---|---|---|
| PPO from scratch | 1.00x | 6.69 |
| teacher with the frame link cut (control) | **1.05x** | 6.09 |
| the real Qwen3-VL + Jev teacher | **2.95x** | 7.23 |

Pre-registered prediction, written before the run: "if the VLM's knowledge is doing the
work, X collapses towards 1.0." It collapsed to 1.05. This converts "guidance helps" into
"**the VLM's knowledge** is what helps", which is the claim the whole project rests on.

## Left

1. Refresh `results/SCOREBOARD.md`, sync the archive, push. Small and mechanical.

The README rewrite, the metrics document, the GIFs, the literature positioning and the
ablation are all done.

## Known neighbours in the literature

Checked 2026-09-20, because citing only 2018-2023 work made this look like it was
engaging with a stale field.

| work | overlap | why this is still distinct |
|---|---|---|
| **LVLM2P** (Lee et al., May 2025) | closest: distils a large VLM into an RL agent for sample efficiency, teacher active only during training | it has the VLM supply **actions**; finding 2 here is that action-teaching is the part that fails (0.88 vs 2.53 from the same labels) |
| **GameSense** (Lu et al., Mar 2025) | same conclusion that VLMs should not drive the game directly | there the VLM *writes* the execution module; here RL trains it and a rule switches the teacher off |
| **Playing DOOM with 1.3M Parameters** (Golchinfar et al., Apr 2026) | small specialised model beats LLMs at real-time ViZDoom | this is the *premise*, not a competing result. It is published confirmation that the latency framing was not worth leading with |

Nothing found so far does perception-only VLM teaching with a typed decision layer and an
adaptive handover, which is the combination this project tests.

## Optional, if there is appetite

- A second task family beyond ViZDoom. The gate makes this cheap to *screen*: a
  20-minute probe says yes/no before any training. The Valorant localisation result
  (0.747 vs 0.493 baseline) says the perception half already transfers to real footage.
- More seeds. 3 is enough to show a spread, not enough for tight confidence intervals;
  this is the weakest part of the evidence and the cheapest to strengthen.
- Write-up of the fp16-corruption finding (fp16 silently destroys Qwen3-VL on pre-Ampere
  GPUs: 25% vs 100% on a synthetic control, with 0.03% of probability mass on the answer
  letters). This is a genuinely useful public-service bug report.
- X/Twitter thread built around the demo video.

## Budget

Paid compute: **$0.00** (free Kaggle T4s throughout). Paid API: **~$0.05** of Jev calls.
Kaggle GPU quota is the binding constraint, not money.
