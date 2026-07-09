from __future__ import annotations

import re
import uuid
from base64 import b32encode
from dataclasses import dataclass, field
from functools import lru_cache
from string import ascii_lowercase
from typing import Any

from parse import parse

keywords = re.compile(
    r"\b(?:noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection|article|determiner|auxiliary verb|modal verb|particle|gerund|infinitive|participle)\b"
)

DEFAULT_TEMPLATE_MIN_BLANKS = 3
DEFAULT_TEMPLATE_MAX_BLANKS = 7


@lru_cache(16)
def judge_is_flashcard_like(card_description: str | None) -> bool:
    if not card_description:
        return False
    return re.search(keywords, card_description) is not None


@dataclass
class Card:
    name: str = ""
    description: str | None = None
    flavor_text: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    card_art_name: str | None = None
    prefix: str | None = None
    prefix_template: str | None = None
    prefix_draw_template: str | None = None
    prefix_role: str | None = None
    prefix_min_length: int = DEFAULT_TEMPLATE_MIN_BLANKS
    prefix_max_length: int = DEFAULT_TEMPLATE_MAX_BLANKS
    prefix_length: int | None = None
    prefix_typed: str = ""
    prefix_pending: bool = False
    energy_cost: int = 1
    rarity: str = "common"
    temporary_original_name: str | None = None
    temporary_original_description: str | None = None
    temporary_original_flavor_text: str | None = None
    temporary_original_card_art_name: str | None = None
    temporary_original_energy_cost: int | None = None

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "flavor_text": self.flavor_text,
            "energy_cost": self.energy_cost,
        }

    def to_plaintext(self) -> str:
        if self.description:
            return f"<{self.name}: {self.description}>"
        return f"<{self.name}>"

    def display_description(self) -> str:
        parts = []
        if self.description:
            parts.append(self.description)
        if self.flavor_text:
            parts.append(f'"{self.flavor_text}"')
        return "\n".join(parts)

    def short_id(self) -> str:
        return b32encode(bytes.fromhex(self.id[:8])).decode().lower()[:4]

    @staticmethod
    def parse(s: str) -> Card:
        if result := parse("<{}: {}>", s):
            name, description = result.fixed
            return Card(name=name, description=description)
        if result := parse("<{}>", s):
            return Card(name=result.fixed[0], description=None)
        raise ValueError("Invalid card format")

    def duplicate(self) -> Card:
        """Copy, but with a new ID."""
        return Card(
            name=self.name,
            description=self.description,
            flavor_text=self.flavor_text,
            card_art_name=self.card_art_name,
            prefix=self.prefix,
            prefix_template=self.prefix_template,
            prefix_draw_template=self.prefix_draw_template,
            prefix_role=self.prefix_role,
            prefix_min_length=self.prefix_min_length,
            prefix_max_length=self.prefix_max_length,
            prefix_length=self.prefix_length,
            prefix_typed=self.prefix_typed,
            prefix_pending=self.prefix_pending,
            energy_cost=self.energy_cost,
            rarity=self.rarity,
        )

    def begin_temporary_transform(
        self,
        name: str,
        description: str | None,
        flavor_text: str | None = None,
        card_art_name: str | None = None,
        energy_cost: int | None = None,
    ) -> None:
        if self.temporary_original_name is None:
            self.temporary_original_name = self.name
            self.temporary_original_description = self.description
            self.temporary_original_flavor_text = self.flavor_text
            self.temporary_original_card_art_name = self.card_art_name
            self.temporary_original_energy_cost = self.energy_cost
        self.name = name
        self.description = description
        self.flavor_text = flavor_text
        self.card_art_name = card_art_name
        if energy_cost is not None:
            self.energy_cost = energy_cost

    def revert_temporary_transform(self) -> None:
        if self.temporary_original_name is None:
            return
        self.name = self.temporary_original_name
        self.description = self.temporary_original_description
        self.flavor_text = self.temporary_original_flavor_text
        self.card_art_name = self.temporary_original_card_art_name
        if self.temporary_original_energy_cost is not None:
            self.energy_cost = self.temporary_original_energy_cost
        self.temporary_original_name = None
        self.temporary_original_description = None
        self.temporary_original_flavor_text = None
        self.temporary_original_card_art_name = None
        self.temporary_original_energy_cost = None

    def has_temporary_transform(self) -> bool:
        return self.temporary_original_name is not None

    def is_prefix_card(self) -> bool:
        return self.prefix is not None or self.prefix_template is not None

    def prefix_suffix_length(self) -> int:
        if not self.is_prefix_card() or self.prefix_length is None:
            if self.prefix_template is None:
                return 0
            return self.prefix_template.count("_")
        if self.prefix_template is not None:
            template = self.prefix_draw_template or self.prefix_template
            return template.count("_")
        return max(self.prefix_length - len(self.prefix or ""), 0)

    def is_prefix_filled(self) -> bool:
        return (
            self.is_prefix_card()
            and len(self.prefix_typed) == self.prefix_suffix_length()
        )

    def is_playable_prefix_state(self) -> bool:
        return not self.is_prefix_card() or (
            self.is_prefix_filled()
            and bool(self.description)
            and not self.prefix_pending
            and "_" not in self.name
        )

    def _random_template_letter(self, rng: Any | None = None) -> str:
        if rng is None:
            import random

            return random.choice(ascii_lowercase)
        return ascii_lowercase[int(rng.integers(0, len(ascii_lowercase)))]

    def _with_permanent_template_letters(self, rng: Any | None = None) -> str:
        template = self.prefix_template or ""
        if "^" not in template:
            return template
        template = "".join(
            self._random_template_letter(rng) if ch == "^" else ch
            for ch in template
        )
        self.prefix_template = template
        return template

    def _format_template_name(self, word_template: str) -> str:
        if self.prefix_role and "_" in word_template:
            return f"{word_template} {self.prefix_role}"
        return word_template

    def _with_randomized_blanks(self, template: str, blank_count: int) -> str:
        blank_count = max(0, blank_count)
        return re.sub(r"_+", "_" * blank_count, template)

    def _render_template_word(self) -> str:
        template = self.prefix_draw_template or self.prefix_template or ""
        typed = iter(self.prefix_typed)
        rendered = []
        for ch in template:
            if ch == "_":
                rendered.append(next(typed, "_"))
            else:
                rendered.append(ch)
        return "".join(rendered)

    def prepare_prefix_reward(self, rng: Any | None = None) -> None:
        if self.prefix_template is None:
            return
        self._with_permanent_template_letters(rng)
        self.name = self._format_template_name(self.prefix_template)

    def prepare_prefix_draw(self, length: int, rng: Any | None = None) -> None:
        if not self.is_prefix_card():
            return
        if self.prefix_template is not None:
            template = self._with_permanent_template_letters(rng)
            template = self._with_randomized_blanks(template, length)
            self.prefix_draw_template = "".join(
                self._random_template_letter(rng) if ch == "*" else ch
                for ch in template
            )
            self.prefix_length = len(self.prefix_draw_template)
            self.prefix_typed = ""
            self.prefix_pending = False
            self.description = None
            self.flavor_text = None
            self.name = self._format_template_name(self.prefix_draw_template)
            return
        prefix = self.prefix or ""
        self.prefix_length = max(length, len(prefix))
        self.prefix_typed = ""
        self.prefix_pending = False
        self.description = None
        self.flavor_text = None
        self.name = prefix + "_" * self.prefix_suffix_length()

    def set_prefix_suffix(self, suffix: str) -> None:
        if not self.is_prefix_card():
            return
        suffix = "".join(ch for ch in suffix if ch.isalpha()).lower()
        self.prefix_typed = suffix[: self.prefix_suffix_length()]
        if self.prefix_template is not None:
            self.name = self._format_template_name(self._render_template_word())
            self.description = None
            self.flavor_text = None
            self.prefix_pending = False
            return
        remaining = self.prefix_suffix_length() - len(self.prefix_typed)
        self.name = f"{self.prefix}{self.prefix_typed}{'_' * remaining}"
        self.description = None
        self.flavor_text = None
        self.prefix_pending = False

    def append_prefix_letter(self, letter: str) -> None:
        if len(letter) != 1 or not letter.isalpha():
            return
        self.set_prefix_suffix(self.prefix_typed + letter)

    def pop_prefix_letter(self) -> None:
        self.set_prefix_suffix(self.prefix_typed[:-1])

    def reset_prefix_entry(self) -> None:
        self.set_prefix_suffix("")

    def confirm_prefix_entry(self) -> None:
        if not self.is_prefix_filled():
            return
        if self.prefix_template is not None:
            word = self._render_template_word()
        else:
            word = f"{self.prefix}{self.prefix_typed}"
        self.name = word[:1].upper() + word[1:].lower()
        self.description = None
        self.flavor_text = None
        self.prefix_pending = True

    def finish_prefix_description(
        self,
        description: str,
        flavor_text: str | None = None,
        energy_cost: int | None = None,
    ) -> None:
        if not self.is_prefix_card():
            return
        self.description = description
        self.flavor_text = flavor_text
        if energy_cost is not None:
            self.energy_cost = energy_cost
        self.prefix_pending = False

    def __hash__(self) -> int:
        return hash(self.id)

    def is_flashcard_like(self) -> bool:
        return self.is_singleword_title() and judge_is_flashcard_like(self.description)

    def is_singleword_title(self) -> bool:
        return " " not in self.name
