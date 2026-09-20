# ReflexRL: complete metrics

Every measurement in the project, with the data behind it and the file it came from. Regenerate with `python scripts/metrics_report.py`; nothing here is typed by hand.

**The question this project answers:** a vision-language model knows what a game scene contains but cannot play. Can that knowledge be *moved* into a small network that can, and can the move be measured?

---

## The result in four numbers

| | ReflexRL | PPO from scratch | |
|---|---|---|---|
| environment steps to reach the target score | **304K** | 900K | **2.95x fewer** |
| score on 50 episodes nothing ever saw | **7.34** | 6.95 | **+6%** |
| spread across seeds (lower = more reliable) | **0.22** | 1.88 | **9x tighter** |
| seeds mastering a map never trained on | **3 of 3** | 0 of 3 | |

The teacher that produced this scores **4.40**. Its students finish above **7.2**. The teacher is a curriculum, not a ceiling. Both the vision model and the decision model are **deleted after training**: what ships is 751,526 parameters making **0 model calls** per action, on a CPU core.

---

## 1. Sample efficiency: the headline number

Target R\* = **5.46** kills, fixed before these runs as 80% of the way from random (0.57) to PPO's own final score (6.69); pre-registered in `experiments/configs/metrics_prereg.json`. X = median steps PPO needs / median steps the method needs. The 3,662 environment steps spent collecting teacher labels are added to every teacher-using method before the comparison.

| method | steps to R\*, per seed | median | **X** | last in-training eval, per seed | mean |
|---|---|---|---|---|---|
| PPO from scratch | 900K, 1400K, 600K | 900K | **1.00x** | 7.09, 5.47, 7.50 | 6.69 |
| BC -> PPO (same teacher, imitate then RL) | 604K, 704K, never | never | - | 7.31, 8.12, 4.53 | 6.66 |
| **ReflexRL (guided, adaptive handover)** | 304K, 304K, 304K | 304K | **2.95x** | 7.34, 7.03, 7.31 | 7.23 |
| ReflexRL (live Qwen+Jev DAgger rounds) | 353K, 353K | 353K | **2.54x** | 7.12, 7.28 | 7.20 |
| ReflexRL (fixed anneal, no adaptivity) | 453K | 453K | **1.98x** | 7.38 | 7.38 |

<sub>source: `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s0/done.json`, `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s1/done.json`, `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s2/done.json` and 7 more</sub>

The last two columns are the *in-training* evaluation (16 episodes, validation seeds). They are not the reported result: section 2 re-runs every finished checkpoint on 50 episodes from a seed stream nothing ever touched, and those are the numbers that count.

**BC -> PPO is the control that matters.** Identical teacher, identical labels, identical budget: imitate the teacher first, then run the same PPO. It does not reliably speed anything up and one seed never reaches the target at all. What produces X is *guiding exploration and then handing control back*, not having the teacher's answers in the weights.

### Learning curves (evaluation return, 16 episodes per point)

| env steps | PPO | BC->PPO | **ReflexRL** | ReflexRL live | ReflexRL fixed |
|---|---|---|---|---|---|
| 0 | 0.69 | 0.92 | 0.69 | 0.50 | 0.44 |
| 100,000 | 3.08 | 2.94 | 2.52 | 1.19 | 3.31 |
| 200,000 | 3.46 | 3.02 | 4.73 | 2.81 | 4.06 |
| 300,000 | 3.17 | 3.25 | 5.48 | 4.81 | 4.62 |
| 400,000 | 2.81 | 3.29 | 6.06 | 6.19 | 5.06 |
| 500,000 | 3.79 | 4.00 | 6.25 | 6.38 | 6.25 |
| 750,000 | 5.21 | 5.31 | 6.44 | 6.84 | 6.50 |
| 1,000,000 | 5.92 | 6.17 | 6.71 | 6.81 | 7.19 |
| 1,500,000 | 6.67 | 6.46 | 7.33 | 7.09 | 7.62 |

Mean over seeds of the last evaluation at or before each step.

### How little the teacher was actually used

| seed | env steps under teacher control | share of 1.5M | teacher gone by | final |
|---|---|---|---|---|
| reflexrl_s0 | 93,438 | 6.2% | 201K | 7.34 |
| reflexrl_s1 | 143,616 | 9.6% | 250K | 7.03 |
| reflexrl_s2 | 143,630 | 9.6% | 250K | 7.31 |

The teacher scores **4.40**. The students it guided finish above **7.2** and stop listening to it before 10% of training has elapsed. The teacher is not a performance ceiling; it is a curriculum.

<sub>source: `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s0/done.json`, `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s1/done.json`, `archive/kaggle/reflexrl-train-lane0/runs/train/defend_the_center/reflexrl_s2/done.json`</sub>

### Wall-clock cost of one 1.5M-step run (free Kaggle T4)

| method | minutes, per seed |
|---|---|
| PPO from scratch | 26, 26, 26 |
| BC -> PPO (same teacher, imitate then RL) | 28, 28, 28 |
| **ReflexRL (guided, adaptive handover)** | 27, 27, 27 |
| ReflexRL (live Qwen+Jev DAgger rounds) | 64, 63 |
| ReflexRL (fixed anneal, no adaptivity) | 29 |

---

## 2. Final scores on episodes nothing ever saw

50 episodes from seeds 7,000,000+, a stream disjoint from training (0-2011), from in-training evaluation (900,000+) and from teacher labelling (50,000+). No checkpoint, no handover decision and no hyper-parameter was chosen using these episodes.

| policy | kills | std err |
|---|---|---|
| bc_ppo_s1 | 8.14 | ±0.20 |
| ppo_s2 | 7.84 | ±0.25 |
| bc_ppo_s0 | 7.70 | ±0.17 |
| reflexrl_s0 | 7.42 | ±0.15 |
| reflexrl_s1 | 7.40 | ±0.16 |
| reflexrl_live_s0 | 7.26 | ±0.14 |
| reflexrl_live_s1 | 7.22 | ±0.14 |
| reflexrl_s2 | 7.20 | ±0.13 |
| ppo_s0 | 7.06 | ±0.20 |
| ppo_s1 | 5.96 | ±0.17 |
| bc_ppo_s2 | 4.56 | ±0.17 |
| distilled_perception_jev_teacher | 3.10 | ±0.25 |

### Grouped by method, where the spread is the point

| method | mean | worst seed | best seed | spread |
|---|---|---|---|---|
| PPO from scratch | 6.95 | 5.96 | 7.84 | 1.88 |
| BC -> PPO | 6.80 | 4.56 | 8.14 | 3.58 |
| **ReflexRL** | 7.34 | 7.20 | 7.42 | 0.22 |
| ReflexRL (live teacher) | 7.24 | 7.22 | 7.26 | 0.04 |

PPO has a 5.96 seed. BC -> PPO has a 4.56 seed. ReflexRL's three seeds land within 0.22 of each other. **Guidance buys reliability, not just speed**, and on a 3-seed budget that is the more honest claim.

<sub>source: `results/final_eval.json`</sub>

---

## 3. Transfer to a map the policy has never seen (`defend_the_line`)

Different layout and enemy placement, same controls, 750K steps, 3 seeds. The teacher is not involved except in the last row.

| condition | steps to 19.9 kills | seeds reaching 21.5 | final, mean |
|---|---|---|---|
| PPO from scratch | 350K | 0/3 | 21.54 |
| PPO policy fine-tuned | 350K | 1/3 | 21.17 |
| **ReflexRL policy fine-tuned** | 250K | 3/3 | 23.14 |
| ReflexRL fine-tuned *with* the teacher | 201K | 3/3 | 23.32 |

<sub>source: `archive/kaggle/reflexrl-heldout-a/runs/train/defend_the_line/ppo_ft_s0/done.json`, `archive/kaggle/reflexrl-heldout-a/runs/train/defend_the_line/ppo_ft_s1/done.json`, `archive/kaggle/reflexrl-heldout-a/runs/train/defend_the_line/ppo_s0/done.json` and 9 more</sub>

Zero-shot transfer is near random for every policy; what transfers is how fast the new map is re-learned. The last row is where the *adaptive* handover earns its keep: the arriving policy already outscores the 4.40 teacher, so the rule cuts teacher influence to zero at the first evaluation and guidance costs nothing. An earlier version that stepped down on a fixed schedule let that teacher override an 18-scoring student and made transfer **0.86x**, worse than no teacher. A fixed anneal cannot detect that.

---

## 4. The finding: *what* you distil beats *how well* you distil it

Both students are the same small CNN, trained on the same 3,662 frames labelled by the same Qwen3-VL-8B + Jev teacher, and both reproduce the teacher on held-out labels about equally well. The only difference is **what the labels are**.

| student | trained to copy the teacher's... | agreement with teacher | kills | std err |
|---|---|---|---|---|
| action-cloned | chosen **action** | 0.655 | 0.88 | ±0.21 |
| **perception-distilled** (+ Jev still deciding) | **percept** (monster left/centre/right/none) | 0.629 | **2.53** | ±0.39 |

Fidelity is *lower* for the student that scores 2.9x higher. Copying the teacher's behaviour faithfully copies its mistakes and throws away the structure that made it good; copying what it *saw* and rebuilding the decision on top keeps the useful part. An action-cloned teacher at 0.88 kills is barely above random. A VLM's action choice is not worth learning, and its percept is.

The perception student is recovered by least squares: the teacher's action distribution is pi_T = q . J for a known decision table J, so the percept q is inverted out of it, then fitted with mirror augmentation and inverse-frequency class weights. Training-set percept prior: left 15%, center 54%, right 11%, none 20%.

<sub>source: `archive/kaggle/reflexrl-train-lane0/runs/teachers/defend_the_center/perception_teacher.json`, `archive/kaggle/reflexrl-train-lane1/runs/teachers/defend_the_center/teacher.json`</sub>

This is the teacher that then guides RL, and note that it scores 2.53, well below the 4.40 of the live Qwen+Jev teacher it was distilled from. **Everything downstream is taught by a degraded copy, and the students still finish above 7.2.**

---

## 5. Why this teacher is worth learning from: the decision layer

No policy is learned here. Each controller is dropped into `defend_the_center` and the engine waits for it, so a 6-second decision costs nothing. This isolates *decision quality* from speed.

| controller | kills, mean | std err | episodes | vs random |
|---|---|---|---|---|
| random actions | 0.57 | 0.22 | 30 | +0.00 |
| Qwen3-VL-2B, picks actions (calibrated) | 1.73 | 0.27 | 30 | +1.17 |
| Qwen3-VL-2B, picks actions (uncalibrated) | 0.03 | 0.21 | 30 | -0.53 |
| Qwen3-VL-8B, picks actions (calibrated) | 1.67 | 0.22 | 30 | +1.10 |
| **Qwen3-VL-8B sees -> Jev 1.13 decides** | 4.40 | 0.36 | 30 | +3.83 |

Four times the parameters bought nothing (1.73 -> 1.67). Taking the *action choice away* from the same 8B model and handing it to a typed decision model multiplied the score by 2.6x.

<sub>source: `results/teacher_gate/qwen2b_fp32_cal_lane0.json`, `results/teacher_gate/qwen2b_fp32_cal_lane1.json`, `results/teacher_gate/qwen2b_gate.json` and 5 more</sub>

---

## 6. What the vision model can actually see

The same four-way question in four settings, scored against ground truth (ViZDoom object labels; the Valorant dataset's bounding boxes). 150 frames each, 2 option-order permutations averaged.

| setting | question | accuracy | majority-class baseline | margin |
|---|---|---|---|---|
| **Doom `defend_the_center`, 8B** | which way is the monster | **0.733** | 0.313 | +0.420 |
| Doom `defend_the_center`, 2B | which way is the monster | 0.687 | 0.313 | +0.373 |
| Doom `health_gathering` | is a medkit visible, and where | 0.740 | 0.640 | +0.100 |
| Doom `health_gathering` | how far away is it | 0.627 | 0.640 | -0.013 |
| **Valorant, real gameplay frames** | which way is the nearest player | **0.747** | 0.493 | +0.253 |
| Valorant, real gameplay frames | which of them is an *enemy* | 0.307 | 0.620 | -0.313 |

<sub>source: `archive/kaggle/reflexrl-hg-probe/hg_probe/hg_probe.json`, `archive/kaggle/reflexrl-real-frames/real_frames/real_frames_enemy_w640.json`, `archive/kaggle/reflexrl-real-frames/real_frames/real_frames_player_w640.json` and 1 more</sub>

**The model is good at *where* and bad at *whether* and *which kind*.** That single sentence predicts every result in this project: `defend_the_center` works because a monster is nearly always on screen so only direction matters; `health_gathering` fails because 96 of 150 frames contain no medkit and the model cannot report absence; the Valorant localisation transfers, the friend/foe judgement does not (that needs team colours nobody told it about).

### Confusion matrix, Valorant `player` (rows = truth, columns = answer)

| truth | left | center | right | none | n |
|---|---|---|---|---|---|
| left | **22** | 17 | 3 | 0 | 42 |
| center | 1 | **64** | 9 | 0 | 74 |
| right | 0 | 5 | **26** | 0 | 31 |
| none | 0 | 1 | 2 | **0** | 3 |

Errors are almost entirely left/right -> centre: the model rarely mistakes left for right, it hedges towards the middle.

<sub>source: `archive/kaggle/reflexrl-real-frames/real_frames/real_frames_player_w640.json`</sub>

### Contextual calibration helps the *action* teacher and hurts perception

| probe | raw | after blank-frame calibration | delta |
|---|---|---|---|
| Doom health_gathering (position) | 0.740 | 0.253 | -0.487 |
| Doom health_gathering (distance) | 0.627 | 0.133 | -0.493 |
| Valorant (player) | 0.747 | 0.727 | -0.020 |
| Valorant (enemy) | 0.307 | 0.253 | -0.053 |

Dividing out the answer prior measured on a blank frame is what rescued the action teacher on `defend_the_center`. On perception it *removes* real class skew: when 64% of frames genuinely contain no medkit, flattening the prior is destructive. Reported both ways rather than picking the flattering one.

### Input resolution matters (Valorant `enemy` probe)

| frame width fed to the model | accuracy | excluding teammate-only frames |
|---|---|---|
| 320 px | 0.313 | 0.550 |
| 640 px | 0.353 | 0.733 |

Downscaling to 320 px costs real accuracy. This is a cost of the free-tier GPU, not of the method: image tokens grow with the square of the width and the 8B model at 640 px only fits one frame per batch on a T4.

<sub>source: `archive/kaggle/reflexrl-real-frames/real_frames/real_frames_w320.json`, `archive/kaggle/reflexrl-real-frames/real_frames/real_frames_w640.json`</sub>

---

## 7. Deployment cost

| | reflex policy | its teacher | ratio |
|---|---|---|---|
| parameters | 751,526 | 2,127,532,032 | 2,831x |
| FLOPs per action | 28.9M | 1.29T | 44,613x |
| latency, same T4 | 2.57 ms | 445 ms | **173x** |
| latency, one CPU core | 2.77 ms | cannot run | - |
| USD per 10K decisions | $0.0004 (CPU) | $0.7299 | **173x** |
| peak VRAM | 12 MB | 8,346 MB | - |
| model calls at deployment | **0** | one per action | - |

<sub>source: `archive/kaggle/reflexrl-demo/results/benchmark/defend_the_center.json`</sub>

### The same controllers when the game does *not* wait

Decision latency is converted to dropped engine tics (1 tic = 28.57 ms), so a slow controller literally stands still while monsters close in.

| controller | kills | std err | ms per decision | tics skipped per decision |
|---|---|---|---|---|
| reflex | 7.25 | ±0.45 | 1.7 | 0 |
| qwen8b_jev_teacher | -0.38 | ±0.26 | 6190.9 | 217 |

The teacher scores **below random** when it has to play in real time. This is the least interesting number in the document, since everyone already knows a 6-second-per-frame model cannot play a shooter. It is included because it is the reason the knowledge has to be *moved* rather than queried.

<sub>source: `archive/kaggle/reflexrl-demo/results/demo/realtime.json`</sub>

---

## 8. Knowing in advance where this will work

A teacher is only worth using where it beats random, so every candidate scenario was put through the same cheap test *before* any training run used it. The rule was written down in `experiments/configs/phase0_gate.json` first: teacher mean > random mean, one-sided Welch p < 0.05, **and** Cohen's d >= 0.5.

| scenario | random | Qwen3-VL-2B | Welch p | Cohen's d | verdict |
|---|---|---|---|---|---|
| `defend_the_center` (kills) | 0.6 | 1.7 | 4.0e-04 | +0.87 | **PASS** |
| `health_gathering` (survival tics) | 499.5 | 474.9 | 6.5e-01 | -0.10 | below bar, not used |
| `health_gathering_supreme` | 370.8 | 360.2 | 6.6e-01 | -0.11 | below bar, not used |
| `deadly_corridor` | -82.0 | -70.4 | 2.6e-01 | +0.17 | below bar, not used |

One scenario cleared the bar, and that scenario carries the main result. The others are not a mystery: all three require reporting **absence**: is there a medkit, is the corridor clear. The perception probes in section 6 show precisely that this is what the model cannot do. The two agree, which is the useful part: **the probe predicts the gate**. A 20-minute perception probe tells you whether a VLM can teach a given task before you spend a GPU-day finding out.

Nothing in the pipeline is scenario-specific. Supply a teacher that clears the gate and the remaining machinery (perception distillation, guided PPO, adaptive handover) is unchanged.

<sub>source: `results/teacher_gate/qwen2b_fp32_cal_lane0.json`, `results/teacher_gate/qwen2b_fp32_cal_lane1.json`, `results/teacher_gate/random_gate.json`</sub>

---

## 9. Bugs and dead ends, measured

### fp16 silently corrupts Qwen3-VL on pre-Ampere GPUs

| precision | accuracy on a synthetic control | probability mass on *any* answer letter |
|---|---|---|
| fp16 | 0.25 | 0.0002 |
| fp32 | 1.00 | 1.0000 |

The control is a red circle drawn on the left, centre or right of a blank image, a question no vision model should miss. In fp16 the model answered 'A' every time with 0.02% of its probability on the letters at all; renormalising over A/B/C/D hid the corruption and made it look like multiple-choice position bias. Every teacher number measured before this was discarded. The code now runs fp32 (4-bit weights with fp32 compute for the 8B) and refuses to emit a label when letter mass drops below 0.5.

<sub>source: `results/teacher_diagnosis/diagnose_fp16_vs_fp32.json`</sub>

### Everything else that did not work

| attempt | result | why it is in the repo |
|---|---|---|
| Qwen3-VL-4B as teacher | failed the perception probe outright | size is not the axis |
| action-cloning the teacher into a CNN | 0.88 kills | cloning *decisions* destroys the teacher; cloning *perception* and keeping Jev preserves it (3.10) |
| `health_gathering`, `health_gathering_supreme`, `deadly_corridor` | teacher at or below random | three of four scenarios dropped, reported as negatives |
| contextual calibration on perception probes | 0.74 -> 0.25 | the fix for one problem is the bug for another |
| fixed teacher anneal on the held-out map | 0.86x, worse than no teacher | motivated the adaptive handover rule |

Pre-registrations for every gate and metric live in `experiments/configs/*.json`, written before the corresponding run.

---

## 10. What this cost

| resource | amount |
|---|---|
| Kaggle jobs archived | 23 |
| GPU wall-clock in training runs alone | 8.2 h |
| paid compute | **$0.00** (free Kaggle T4s) |
| paid API | **~$0.05** of TypeSafe Jev calls |
| model weights downloaded | Qwen3-VL 2B / 4B / 8B, open weights |

Jobs: `reflexrl-bench-cpu`, `reflexrl-bench-gpu`, `reflexrl-debias-probe`, `reflexrl-demo`, `reflexrl-diagnose`, `reflexrl-final-eval`, `reflexrl-gate-8b`, `reflexrl-gate-pjev`, `reflexrl-heldout-a`, `reflexrl-heldout-b`, `reflexrl-heldout-guided`, `reflexrl-hg-probe`, `reflexrl-live-teacher`, `reflexrl-perception-teacher`, `reflexrl-pilot-pjev`, `reflexrl-qwen-gate`, `reflexrl-real-frames`, `reflexrl-size-check`, `reflexrl-teachers-pjev`, `reflexrl-train-lane0`, `reflexrl-train-lane1`, `reflexrl-train-lane1-v1`, `reflexrl-train-ppo`

## 11. Leakage and seed hygiene

| stream | seed range | used for |
|---|---|---|
| training | 0 - 2,011 | environment resets during PPO |
| in-training evaluation | 900,000+ | the curves above, handover decisions |
| teacher labelling | 50,000+ | frames shown to Qwen |
| **final reported scores** | **7,000,000+** | section 4 only |

`tests/test_seed_hygiene.py` fails the build if any two of these overlap. `tests/test_core.py` asserts the policy receives pixels only: no depth buffer, no object labels, no game variables. `tests/test_probe_alignment.py` asserts probe frames and their oracle labels come from the same rollout (it exists because they once did not).

