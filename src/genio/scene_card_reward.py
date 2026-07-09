from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pyxel
from pyxelxl import layout

from genio.base import WINDOW_HEIGHT, WINDOW_WIDTH, load_image
from genio.card import Card
from genio.card_rewards import generate_typist_card_rewards
from genio.components import cute_text, draw_rounded_rectangle, retro_text
from genio.constants import CARD_HEIGHT, CARD_WIDTH
from genio.gamestate import game_state
from genio.gears.card_printer import CardPrinter
from genio.scene import Scene
from genio.scene_stages import draw_lush_background


@dataclass
class RewardCardSprite:
    x: int
    y: int
    card: Card
    printer: CardPrinter

    def __post_init__(self) -> None:
        self.image = self.printer.print_card(self.card)
        self.hovering = False

    def update(self) -> None:
        self.hovering = self.is_mouse_over()

    def is_mouse_over(self) -> bool:
        return (
            self.x <= pyxel.mouse_x <= self.x + CARD_WIDTH
            and self.y <= pyxel.mouse_y <= self.y + CARD_HEIGHT
        )

    def draw(self) -> None:
        if self.hovering:
            draw_rounded_rectangle(
                self.x - 3, self.y - 3, CARD_WIDTH + 6, CARD_HEIGHT + 6, 3, 7
            )
            draw_rounded_rectangle(
                self.x - 2, self.y - 2, CARD_WIDTH + 4, CARD_HEIGHT + 4, 2, 1
            )
        pyxel.blt(
            self.x,
            self.y - (4 if self.hovering else 0),
            self.image,
            0,
            0,
            CARD_WIDTH,
            CARD_HEIGHT,
            colkey=254,
        )


class CardRewardScene(Scene):
    def __init__(self) -> None:
        super().__init__()
        self.card_printer = CardPrinter()
        reward = generate_typist_card_rewards(game_state.card_reward_rare_offset)
        game_state.card_reward_rare_offset = reward.rare_offset
        spacing = 66
        start_x = WINDOW_WIDTH // 2 - spacing
        self.cards = [
            RewardCardSprite(
                start_x + i * spacing - CARD_WIDTH // 2,
                WINDOW_HEIGHT // 2 - CARD_HEIGHT // 2 - 4,
                card,
                self.card_printer,
            )
            for i, card in enumerate(reward.cards)
        ]
        self.next_scene: str | None = None
        self.next_scene_requested = False

    def update(self) -> None:
        for card in self.cards:
            card.update()
        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            for card in self.cards:
                if card.hovering:
                    self.choose_card(card.card)
                    return
        for ix, key in enumerate((pyxel.KEY_1, pyxel.KEY_2, pyxel.KEY_3)):
            if ix < len(self.cards) and pyxel.btnp(key):
                self.choose_card(self.cards[ix].card)
                return
        if pyxel.btnp(pyxel.KEY_ESCAPE):
            game_state.start_test_battle()
            self.next_scene = "genio.gui"

    def choose_card(self, card: Card) -> None:
        game_state.start_test_battle(card)
        self.next_scene = "genio.gui"

    def draw(self) -> None:
        draw_lush_background()
        cute_text(
            0,
            24,
            "Choose a card",
            7,
            layout=layout(w=WINDOW_WIDTH, h=12, ha="center"),
        )
        for card in self.cards:
            card.draw()
        hovered = next((card.card for card in self.cards if card.hovering), None)
        if hovered:
            self.draw_hover_text(hovered)
        self.draw_mouse_cursor(pyxel.mouse_x, pyxel.mouse_y)

    def draw_hover_text(self, card: Card) -> None:
        has_flavor = bool(card.flavor_text)
        description_width = 28 if has_flavor else 46
        flavor_width = 18
        description_lines = textwrap.wrap(card.description or "", description_width)
        flavor_lines = (
            textwrap.wrap(card.flavor_text or "", flavor_width) if has_flavor else []
        )
        line_count = max(len(description_lines), len(flavor_lines))
        box_height = max(24, 14 + line_count * 7)
        y = min(178, WINDOW_HEIGHT - box_height - 4)
        draw_rounded_rectangle(96, y, 236, box_height, 3, 0)
        retro_text(
            102,
            y + 3,
            card.name,
            7,
            layout=layout(w=224, h=8, ha="center"),
        )
        text_y = y + 15
        for i, line in enumerate(description_lines):
            pyxel.text(102, text_y + i * 7, line, 7)
        if has_flavor:
            pyxel.line(248, y + 15, 248, y + box_height - 5, 5)
            for i, line in enumerate(flavor_lines):
                pyxel.text(254, text_y + i * 7, line, 5)

    def request_next_scene(self) -> str | None:
        if self.next_scene_requested or self.next_scene is None:
            return None
        self.next_scene_requested = True
        return self.next_scene

    def draw_mouse_cursor(self, x: int, y: int) -> None:
        cursor = load_image("cursor.png")
        pyxel.blt(x, y, cursor, 0, 0, 16, 16, colkey=254)


def gen_scene() -> Scene:
    return CardRewardScene()
