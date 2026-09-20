# Scoreboard

Every number below is read from a results file in this repo; `python scripts/scoreboard.py` rebuilds it.

## Controllers (game paused; 30-episode gates, seeds 50,000+)

| controller | kills | episodes |
|---|---|---|
| random | 0.57 | 30 |
| Qwen3-VL-2B alone (calibrated) | 1.73 | 30 |
| Qwen3-VL-8B alone (calibrated) | 1.67 | 30 |
| **Qwen3-VL-8B sees + Jev decides** | 4.40 | 30 |

## Learned policies (1.5M steps; target R* = 5.46)

Teacher-label collection (3,662 env steps) is charged to every teacher-using method.

| method | final (per seed) | steps to R* | X |
|---|---|---|---|
| PPO from scratch | 6.69 (7.1, 5.5, 7.5) | 900K, 1400K, 600K | 1.00x |
| BC -> PPO | 6.66 (7.3, 8.1, 4.5) | 604K, 704K, never | - |
| ReflexRL (offline teacher) | 7.23 (7.3, 7.0, 7.3) | 304K, 304K, 304K | 2.95x |
| ReflexRL (fixed schedule) | 7.38 (7.4) | 453K | 1.98x |
| ReflexRL (live Qwen+Jev rounds) | 7.20 (7.1, 7.3) | 353K, 353K | 2.54x |

## Held-out map (`defend_the_line`, 750K steps)

| condition | steps to 19.9 kills | reaches 21.5 | final |
|---|---|---|---|
| PPO from scratch | 350K | 0/3 seeds | 21.54 |
| PPO policy fine-tuned | 350K | 1/3 seeds | 21.17 |
| ReflexRL policy fine-tuned | 250K | 3/3 seeds | 23.14 |

## Deployment (same T4)

| | reflex policy | teacher model |
|---|---|---|
| parameters | 751,526 | 2,127,532,032 |
| FLOPs per action | 28.9M | 1.29T |
| latency (GPU) | 2.57 ms | 445 ms |
| ratio | **173x faster**, **173x cheaper** | 1x |

## Real time (the game does not wait for a slow controller)

| controller | kills | ms per decision |
|---|---|---|
| reflex | 7.25 ± 0.45 | 1.7 |
| qwen8b_jev_teacher | -0.38 ± 0.26 | 6190.9 |
