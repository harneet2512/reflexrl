"""Scenario registry: config file, named action vocabulary, teacher briefing.

Each action is a named button combo over the scenario's own button list, so
the teacher (Qwen) and the student act over the identical discrete set. The
legal action set is fixed by the environment's ``available_buttons``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    cfg: str
    buttons: tuple[str, ...]
    actions: tuple[tuple[str, tuple[str, ...]], ...]  # (name, pressed buttons)
    briefing: str  # one-paragraph task description for the teacher prompt
    split: str  # "train" or "heldout"
    episode_timeout: int | None = None  # tics; None keeps the .cfg value

    @property
    def action_names(self) -> list[str]:
        return [name for name, _ in self.actions]

    def button_vectors(self) -> list[list[int]]:
        return [[int(b in pressed) for b in self.buttons] for _, pressed in self.actions]


DEFEND_THE_CENTER = Scenario(
    name="defend_the_center",
    cfg="defend_the_center.cfg",
    buttons=("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    actions=(
        ("TURN_LEFT", ("TURN_LEFT",)),
        ("TURN_RIGHT", ("TURN_RIGHT",)),
        ("FIRE", ("ATTACK",)),
        ("TURN_LEFT_FIRE", ("TURN_LEFT", "ATTACK")),
        ("TURN_RIGHT_FIRE", ("TURN_RIGHT", "ATTACK")),
    ),
    briefing=(
        "You stand in the center of a circular room and cannot move, only turn. "
        "Monsters approach from all sides. Turn to face a monster and shoot it "
        "before it reaches you. Ammo is limited, so only shoot when a monster is "
        "in front of you."
    ),
    split="train",
)

HEALTH_GATHERING = Scenario(
    name="health_gathering",
    cfg="health_gathering.cfg",
    buttons=("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
    actions=(
        ("FORWARD", ("MOVE_FORWARD",)),
        ("TURN_LEFT", ("TURN_LEFT",)),
        ("TURN_RIGHT", ("TURN_RIGHT",)),
        ("FORWARD_TURN_LEFT", ("MOVE_FORWARD", "TURN_LEFT")),
        ("FORWARD_TURN_RIGHT", ("MOVE_FORWARD", "TURN_RIGHT")),
    ),
    briefing=(
        "The floor is acid and slowly drains your health. Medkits lie on the "
        "floor. Walk over medkits to heal and stay alive as long as possible. "
        "Turn toward the nearest medkit and move forward onto it."
    ),
    split="train",
)

DEADLY_CORRIDOR = Scenario(
    name="deadly_corridor",
    cfg="deadly_corridor.cfg",
    buttons=("MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD",
             "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"),
    actions=(
        ("FORWARD", ("MOVE_FORWARD",)),
        ("BACK", ("MOVE_BACKWARD",)),
        ("TURN_LEFT", ("TURN_LEFT",)),
        ("TURN_RIGHT", ("TURN_RIGHT",)),
        ("STRAFE_LEFT", ("MOVE_LEFT",)),
        ("STRAFE_RIGHT", ("MOVE_RIGHT",)),
        ("FIRE", ("ATTACK",)),
        ("TURN_LEFT_FIRE", ("TURN_LEFT", "ATTACK")),
        ("TURN_RIGHT_FIRE", ("TURN_RIGHT", "ATTACK")),
        ("STRAFE_LEFT_FIRE", ("MOVE_LEFT", "ATTACK")),
        ("STRAFE_RIGHT_FIRE", ("MOVE_RIGHT", "ATTACK")),
    ),
    briefing=(
        "You are at one end of a corridor. A green armor vest lies at the far "
        "end. Monsters stand on both sides of the corridor and shoot at you. "
        "Kill the monsters and advance forward down the corridor to reach the "
        "vest without dying."
    ),
    split="train",
)

# Held-out world: same buttons and goal as health_gathering, but a different
# map layout with poison vials and sparser medkits. Never trained on before
# the adaptation experiment.
HEALTH_GATHERING_SUPREME = Scenario(
    name="health_gathering_supreme",
    cfg="health_gathering_supreme.cfg",
    buttons=HEALTH_GATHERING.buttons,
    actions=HEALTH_GATHERING.actions,
    briefing=HEALTH_GATHERING.briefing + " Avoid poison vials.",
    split="heldout",
)

# Held-out world for the Defend-the-Center policy: same three buttons, different
# map and enemy layout (a line of enemies ahead). Its .cfg has no timeout, so it
# gets DTC's 2100-tic cap to keep returns comparable.
DEFEND_THE_LINE = Scenario(
    name="defend_the_line",
    cfg="defend_the_line.cfg",
    buttons=DEFEND_THE_CENTER.buttons,
    actions=DEFEND_THE_CENTER.actions,
    briefing=(
        "You stand at one end of a room and cannot move, only turn. A line of "
        "monsters faces you from the other side and they keep respawning. Turn to "
        "face a monster and shoot it. Ammo is limited, so only shoot when a "
        "monster is in front of you."
    ),
    split="heldout",
    episode_timeout=2100,
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (DEFEND_THE_CENTER, HEALTH_GATHERING, DEADLY_CORRIDOR,
                        HEALTH_GATHERING_SUPREME, DEFEND_THE_LINE)
}

# Short aliases for CLIs and run tags.
ALIASES = {"dtc": "defend_the_center", "hg": "health_gathering",
           "dc": "deadly_corridor", "hgs": "health_gathering_supreme",
           "dtl": "defend_the_line"}


def get_scenario(name: str) -> Scenario:
    return SCENARIOS[ALIASES.get(name, name)]
