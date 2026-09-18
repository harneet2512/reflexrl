# Related work & prior art

What exists, and what ReflexRL claims that's different. Deliberately
short — this is a $20 mechanism study, not a survey.

## Conditional computation (deciding *where* to spend FLOPs)

- **Mixture-of-Depths** (Raposo et al., 2024, arXiv:2404.02258): routers
  allocate FLOPs across transformer depth per-token, trained on the LM
  objective. Ours differs: the router is trained by *RL reward*, and the
  choice changes behavior in a world, not next-token loss.
- **Early-exit / anytime inference** (BranchyNet, DeeBERT, EPNet, LEDE,
  BlockDrop lineage): cascade decisions under supervised or decoding
  objectives, often confidence-thresholded. Confidence-based routing is
  explicitly one of our *baselines* — if it matches the learned router,
  our hypothesis fails.
- **Speculative / cascade inference**: small model drafts, big model
  verifies. Same "cheap-unless-needed" idea, but a *decoding* trick; no
  learned state-dependent allocation tied to task reward.

## Multimodal models with exits / multi-level features

- **DeepStack** (Qwen3-VL, arXiv:2406.04334): visual features injected at
  multiple decoder depths. Relevant engineering fact — it means our
  exits at layers 9/19/28 are all *post-fusion*, so early exits are real
  representations of the fused input, not pre-vision shortcuts.
- Prior "VLM plays game" work exists, but the artifact here is the
  compute policy, not gameplay competence — see failure criteria.

## Embodied RL on ViZDoom

- ViZDoom (vizdoom.farama.org): pixel-based FPS RL, Gymnasium API, very
  fast sim. `defend_the_center` chosen for per-state difficulty variance
  (enemies from multiple directions) — gives the router something real
  to exploit.
- **LevDoom** (github.com/TTomilin/LevDoom): systematic visual/gameplay
  variation of ViZDoom for generalization — our novelty probe's held-out
  environment.
- **COOM** (github.com/hyintell/COOM): continual-learning sequences over
  3D tasks. Cut at $20; the natural next step if the mechanism shows.

## Compute-budget / adaptive compute in RL

- Adaptive computation time / pondering (ACT and successors): learned
  per-input compute under supervised sequence models. The idea of
  *cost-penalized* compute choice is borrowed; doing it inside closed-
  loop embodied RL, with the cost in the reward, is the contribution.
- Title collision: "Learning When to Think" is taken (arXiv:2505.10832,
  a different problem). Write-up uses a different title.

## Honest positioning

Learned early-exit routing is *not* novel in supervised/decoding
settings. The defensible claim is narrow: **a compute-allocation policy
learned by RL reward, inside a closed-loop embodied task, where compute
is an explicit action with an explicit cost.** Everything else —
multiple exits, confidence baselines, Pareto analysis — is standard and
exists to make that one claim testable.
