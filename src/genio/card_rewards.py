from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from genio.card import Card

Rarity = Literal["common", "uncommon", "rare"]

MIN_RARE_OFFSET = -5
MAX_RARE_OFFSET = 40


@dataclass(frozen=True)
class RewardCardDef:
    title: str
    rarity: Rarity


@dataclass(frozen=True)
class CardRewardResult:
    cards: list[Card]
    rare_offset: int


CARD_REWARD_POOL: tuple[RewardCardDef, ...] = (
    RewardCardDef("*___ Attack", "common"),
    RewardCardDef("*___ Block", "common"),
    RewardCardDef("*___ Trick", "common"),
    RewardCardDef("Re____", "uncommon"),
    RewardCardDef("^*____", "uncommon"),
    RewardCardDef("*___ism", "uncommon"),
    RewardCardDef("____", "rare"),
)


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _template_and_role(defn: RewardCardDef) -> tuple[str, str | None]:
    parts = defn.title.split(maxsplit=1)
    role = parts[1] if len(parts) > 1 else None
    return parts[0], role


def _preview_description(role: str | None) -> str:
    match role:
        case "Attack":
            return "Complete the word. Becomes an Attack this turn."
        case "Block":
            return "Complete the word. Becomes a Block card this turn."
        case "Trick":
            return "Complete the word. Becomes a utility card this turn."
        case _:
            return "Complete the word. Becomes a card this turn."


def card_from_reward_def(
    defn: RewardCardDef, rng: np.random.Generator | None = None
) -> Card:
    template, role = _template_and_role(defn)
    card = Card(
        name=defn.title,
        description=_preview_description(role),
        prefix_template=template,
        prefix_role=role,
        energy_cost=1,
        rarity=defn.rarity,
    )
    card.prepare_prefix_reward(_rng(rng))
    return card


def roll_card_rarity(
    rare_offset: int, rng: np.random.Generator | None = None
) -> Rarity:
    rng = _rng(rng)
    common_chance = 60 - rare_offset
    uncommon_chance = 37
    rare_chance = 3 + rare_offset

    if rare_chance < 0:
        uncommon_chance += rare_chance
        rare_chance = 0

    roll = float(rng.random() * 100)
    if roll < rare_chance:
        return "rare"
    if roll < rare_chance + uncommon_chance:
        return "uncommon"
    return "common"


def update_rare_offset(rare_offset: int, rarity: Rarity) -> int:
    match rarity:
        case "common":
            return min(MAX_RARE_OFFSET, rare_offset + 1)
        case "rare":
            return MIN_RARE_OFFSET
        case "uncommon":
            return rare_offset


def generate_typist_card_rewards(
    rare_offset: int = MIN_RARE_OFFSET,
    rng: np.random.Generator | None = None,
    count: int = 3,
) -> CardRewardResult:
    rng = _rng(rng)
    current_offset = rare_offset
    offered: list[RewardCardDef] = []

    for _ in range(count):
        rarity = roll_card_rarity(current_offset, rng)
        candidates = [
            card
            for card in CARD_REWARD_POOL
            if card.rarity == rarity and card not in offered
        ]
        if not candidates:
            candidates = [card for card in CARD_REWARD_POOL if card not in offered]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        offered.append(chosen)
        current_offset = update_rare_offset(current_offset, chosen.rarity)

    cards = [card_from_reward_def(card, rng) for card in offered]
    return CardRewardResult(cards, current_offset)
