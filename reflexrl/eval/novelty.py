"""Novelty response probe: held-out visual conditions should push depth up.

Compares the router's depth distribution on the training environment vs a
held-out LevDoom variant of the same scenario. Prediction: d_t rises at
the transition and falls again with adaptation (adaptation itself is a
training run, outside this eval).

LevDoom varies visuals by swapping the map .wad under a shared conf.cfg
(levdoom/envs/<scenario>/{conf.cfg, maps/<name>.wad}). We reuse our own
wrapper with an absolute config + scenario_wad so obs shape, actions, and
frame stacking are identical — only the visual world changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reflexrl.eval.evaluate import evaluate

# LevDoom defend_the_center difficulty levels -> map wads. Level 0 is
# default (≈ training condition); higher levels add visual shifts.
DTC_LEVEL_WADS = {
    0: "default",
    1: "gore",
    2: "stone_wall_flying_enemies",
    3: "resized_flying_enemies_mossy_bricks",
    4: "complete",
}


def levdoom_env_kwargs(scenario: str = "defend_the_center",
                       level: int = 2) -> dict:
    """LevDoom (scenario, level) -> env kwargs for VizdoomPixelsEnv.

    Resolves the shared conf.cfg + level wad inside the installed
    ``levdoom`` package and returns absolute paths.
    """
    import levdoom

    pkg = Path(levdoom.__file__).parent
    scenario_dir = pkg / "envs" / scenario
    cfg = scenario_dir / "conf.cfg"
    if not cfg.is_file():
        raise FileNotFoundError(f"LevDoom conf.cfg not found at {cfg}")
    wad_name = DTC_LEVEL_WADS.get(level)
    wad = scenario_dir / "maps" / f"{wad_name}.wad"
    if not wad.is_file():
        cands = sorted((scenario_dir / "maps").glob("*.wad"))
        if not cands:
            raise FileNotFoundError(f"no LevDoom wads under {scenario_dir}/maps")
        wad = cands[min(level, len(cands) - 1)]
    return {"config": str(cfg), "scenario_wad": str(wad)}


def novelty_probe(agent, n_episodes: int = 5, level: int = 2,
                  base_kwargs: dict | None = None, seed: int = 30_000) -> dict:
    """Depth distribution on trained vs held-out visuals for one checkpoint."""
    base = evaluate(agent, n_episodes=n_episodes, mode="router", seed=seed,
                    env_kwargs=base_kwargs)
    novel = evaluate(agent, n_episodes=n_episodes, mode="router", seed=seed,
                     env_kwargs=levdoom_env_kwargs(level=level))
    return {
        "base": {
            "mean_depth": base.mean_depth, "depth_hist": base.depth_hist,
            "mean_return": float(np.mean(base.episode_returns)),
        },
        "novel": {
            "mean_depth": novel.mean_depth, "depth_hist": novel.depth_hist,
            "mean_return": float(np.mean(novel.episode_returns)),
        },
        "level": level,
        "depth_shift": novel.mean_depth - base.mean_depth,
    }
