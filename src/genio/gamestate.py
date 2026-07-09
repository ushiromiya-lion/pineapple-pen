from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain

from genio.battle import (
    BattleBundle,
    BattlePrelude,
    CardBundle,
    EnemyBattler,
    EnemyProfile,
    PlayerBattler,
    setup_battle_bundle,
)
from genio.card import Card


@dataclass
class World:
    name: str = "Mystic Wilds"


@dataclass
class GameConfig:
    larger_font: bool = False
    music_volume: int = 3
    sfx_volume: int = 3

    def reset(self) -> None:
        self.larger_font = False
        self.music_volume = 3
        self.sfx_volume = 3


@dataclass(frozen=True, eq=True)
class StageDescription:
    name: str
    subtitle: str
    lore: str
    danger_level: int
    enemies: list[EnemyProfile] = field(default_factory=list)

    @staticmethod
    def default() -> StageDescription:
        return StageDescription(
            "1-1",
            "Beneath the Soil",
            "Beneath the sturdy bamboo, even sturdier roots spread out. Only foolish humans and youkai can see nothing but the surface.",
            1,
        )

    def generate_base_money(self) -> int:
        return 10 + 5 * self.danger_level


class GameState:
    battle_bundle: BattleBundle
    gold: float

    def __init__(self) -> None:
        self.stage = StageDescription.default()
        self.battle_bundle = setup_battle_bundle(
            "initial_deck", "players.starter", ["enemies.sneaky_gremlin"]
        )
        self.gold = 10
        self.battle_bundle.battle_logs = []
        self.world = World()
        self.config = GameConfig()
        self.card_reward_rare_offset = -5

    def gain_gold(self, amount: float) -> None:
        self.gold += amount

    def lose_gold(self, amount: float) -> None:
        self.gold -= amount
        self.gold = max(0, self.gold)

    def should_use_large_font(self) -> bool:
        return self.config.larger_font

    def current_deck_cards(self) -> list[Card]:
        card_bundle = self.battle_bundle.card_bundle
        card_bundle.revert_temporary_transforms()
        cards = chain(
            card_bundle.deck,
            card_bundle.hand,
            card_bundle.graveyard,
            card_bundle.resolving,
        )
        unique_cards = []
        seen = set()
        for card in cards:
            if card.id in seen:
                continue
            unique_cards.append(card)
            seen.add(card.id)
        return unique_cards

    def start_test_battle(self, reward_card: Card | None = None) -> None:
        deck = self.current_deck_cards()
        if reward_card is not None:
            deck.append(reward_card)

        card_bundle = CardBundle(deck)
        card_bundle.draw_to_hand()
        self.battle_bundle = BattleBundle(
            PlayerBattler.from_predef("players.starter"),
            [EnemyBattler.from_predef("enemies.sneaky_gremlin")],
            BattlePrelude.default(),
            card_bundle,
        )
        self.battle_bundle.battle_logs = []


game_state = GameState()
"""Singleton instance of the game state."""
