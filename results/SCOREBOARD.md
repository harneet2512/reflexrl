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

## Deployment (same T4)

| | reflex policy | teacher model |
|---|---|---|
| parameters | 751,526 | 2,127,532,032 |
| FLOPs per action | 28.9M | 1.29T |
| latency (GPU) | 2.85 ms | 468 ms |
| ratio | **164x faster**, **164x cheaper** | 1x |

## Real time (the game does not wait for a slow controller)

| controller | kills | ms per decision |
|---|---|---|
| reflex | 7.12 ± 0.55 | 1.9 |
| qwen2b_calibrated | 2.88 ± 0.74 | 2247.3 |
