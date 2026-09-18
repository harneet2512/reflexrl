"""Constrained-choice prompt: frames + lettered action list -> one letter.

The teacher never writes prose. Its policy is read off the next-token logits
over the answer letters, which yields a full distribution pi_T(a|o) from a
single forward pass.
"""

from __future__ import annotations

import string

from reflexrl.env.scenarios import Scenario

LETTERS = tuple(string.ascii_uppercase)


def action_letters(scenario: Scenario) -> tuple[str, ...]:
    return LETTERS[: len(scenario.actions)]


def build_prompt_text(scenario: Scenario) -> str:
    options = "\n".join(
        f"{letter}. {name}"
        for letter, name in zip(action_letters(scenario), scenario.action_names, strict=True))
    return (
        "You are playing the first-person shooter Doom. "
        f"{scenario.briefing}\n"
        "The images are consecutive frames from your point of view, oldest first; "
        "the last image is the current moment.\n"
        f"Choose the best action right now:\n{options}\n"
        "Answer with a single letter."
    )


def build_messages(scenario: Scenario, n_frames: int) -> list[dict]:
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": build_prompt_text(scenario)})
    return [{"role": "user", "content": content}]
