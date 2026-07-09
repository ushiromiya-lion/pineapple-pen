from dataclasses import dataclass
from typing import Annotated, Protocol

from genio.card import Card
from genio.core.base import promptly


@dataclass
class BonusItems:
    """A list of objects with keys `title` and `delta`."""

    items: Annotated[
        list[dict],
        (
            "list of objects, of keys `title: string` and `delta: number`; "
            "where `title` can be things like 'Overkill', or 'Master of Poison', "
            "`delta` is the amount of bonus gold awarded, in pure numerical form."
        ),
    ]

    def to_individual_items(self) -> list["IndividualBonusItem"]:
        return [
            IndividualBonusItem(title=item["title"], delta=item["delta"])
            for item in self.items
        ]


@dataclass(frozen=True)
class IndividualBonusItem:
    title: str
    delta: float


@promptly()
def generate_bonus_items(base_money: float, battle_logs: list[str]) -> BonusItems:
    """\
    Act as an excellent GM. 
    The player has emerged victorious, and will be awarded base gold
    {{ base_money }}. However, based on what they performed in the battle,
    we can award them additional gold. Kind of like "mini achievements"
    in games that award interesting or positive behavior. For example,
    we might have something like this in some other games:

    "Overkill: +$3.00"
    "Triple Order: +$2.00"
    "Item Master: +$4.00"

    The bonus should be awarded roughly that `%15` of the base
    should be awarded for "normal" achievements, but if anything
    very interesting happens then the bonus can be higher, e.g., `+50%`.

    Your format should be roughly of the form of a list of objects.
    More smaller bonuses are preferred fewer larger bonuses.
    Consult the battle logs to judge the player's behaviors:

    {% for log in battle_logs %}
    "{{ log }}";
    {% endfor %}

    {{ formatting_instructions }}
    """


class CardsLike(Protocol):
    def to_cards(self) -> list[Card]:
        ...


@dataclass
class GenerateSATFlashCardResult(CardsLike):
    """A set of precisely 5 flashcards for SAT preparation."""

    flashcards: Annotated[
        list[dict],
        (
            "A list of flashcards, objects containing two keys: 'word' and 'definition', "
            "each corresponding to the word and its definition from dictionary. Definition "
            "should be in dictionary form, like (noun) a person who is very interested in [...]"
        ),
    ]

    def to_cards(self) -> list[Card]:
        return [Card(card["word"], card["definition"]) for card in self.flashcards]


@dataclass
class GenerateSTSCardResult(CardsLike):
    """A set of cards inspired by slay the spire."""

    cards: Annotated[
        list[dict],
        (
            "A list of cards, objects containing two keys: 'name' and 'description', "
            "inspired by Slay the Spire. Each card should have a unique name and a "
            "concise description of its effects or abilities without including costs.",
        ),
    ]

    def to_cards(self) -> list[Card]:
        return [Card(card["name"], card["description"]) for card in self.cards]


@dataclass
class GenerateNamedCardResult:
    """A generated rules text and optional flavor text for one named card."""

    energy_cost: Annotated[
        int,
        (
            "The explicit energy cost to print on the card and charge when played. "
            "Usually close to the suggested cost, but can be 0 or higher if the "
            "card name and effect call for it."
        ),
    ]
    description: Annotated[
        str,
        (
            "A concise Slay the Spire-style effect description for this card. "
            "Do not include an energy cost in this text; use the energy_cost field."
        ),
    ]
    flavor_text: Annotated[
        str | None,
        (
            "Optional MTG-style flavor text that is specific to this card name, "
            "the character, and the current situation. Use null if it does not fit."
        ),
    ] = None


@promptly()
def generate_named_card_description(
    card_name: str,
    card_cost: int,
    card_rarity: str,
    player_profile: str,
    combat_context: str,
    hand: list[str],
    card_role: str | None = None,
    previous_name: str | None = None,
    previous_description: str | None = None,
) -> GenerateNamedCardResult:
    """\
    Act as an imaginative card designer for a Slay-the-Spire-like roguelike.

    A card has just been named "{{ card_name }}" during combat.
    It starts from the neighborhood of a {{ card_cost }}-cost {{ card_rarity }}
    card, but you must choose and return its exact energy_cost. The name matters:
    if the name sounds strong, weak, clever, narrow, or situational, the card may
    cost more or less to match. Do not make it game-breaking.

    Character:
    {{ player_profile }}

    Current combat:
    {{ combat_context }}

    Current hand:
    {% for card in hand %}
    - {{ card }}
    {% endfor %}

    {% if previous_name %}
    This card used to be "{{ previous_name }}":
    {{ previous_description }}

    Let the new name take over, while keeping some memory of the old card's role
    or power level if it helps.
    {% endif %}

    Write the card rules text for "{{ card_name }}" and choose its exact cost.

    The battle is interpreted by an LLM game master, so the effect can be
    creative, contextual, referential, weird, or surprising. You are not limited
    to ordinary engine primitives.

    It is also fine, and often best, to make a straightforward old-fashioned
    Slay-the-Spire-style card. Not every card needs to be about typing, letters,
    the current enemy, or the current combat state. Treat the character profile
    and combat context as inspiration, not requirements.

    If "{{ card_name }}" is clearly a reference to another game, object, meme,
    myth, event, person, place, command, or real-world concept, you may use that
    reference's own logic. For example, a card named "Black Lotus" can say "Add
    three black mana," even if this game does not normally use mana. The battle
    GM will interpret what it means when played.

    Occasionally reward clever situational wordplay. Only make the card care
    about a specific enemy, status, intent, hand, or battle situation when the
    name naturally and specifically points there. For example, if the enemy is a
    chicken and the player names a card "Fry", the card can be devastating
    specifically against that chicken.

    {% if card_role %}
    It came from a visible "{{ card_role }}" template.
    {% endif %}

    Avoid overusing "if X, then Y" cards just to sound clever. Use conditions
    when they are natural, but many cards should be direct.

    Make the card specific to its name. If the name is nonsense, find a strange
    but playable association from its letters, sound, shape, typo-like quality,
    or context.

    {{ formatting_instructions }}
    """


@promptly()
def generate_sts_cards(avoid: list[str] | None = None) -> GenerateSTSCardResult:
    """\
    Act as an excellent game designer. Create a set of 5 cards inspired by the game Slay the Spire.
    Each card should have a unique name and a concise description of its effects or abilities without including costs.

    {% if avoid %}
    However, some words have already been generated and should be avoided. Their precise titles are below.
    {% for word in avoid %}
    - {{ word }}
    {% endfor %}
    {% endif %}

    {{ formatting_instructions }}
    """


@dataclass
class GenerateSpywareCardResult(CardsLike):
    """A set of cards inspired by spycraft, linguistic word-play, double-O-seven."""

    cards: Annotated[
        list[dict],
        (
            "A list of cards, objects containing two keys: 'name' and 'description', "
            "inspired by spyware. Each card should have a unique name and a "
            "concise description of its effects or abilities without including costs.",
        ),
    ]

    def to_cards(self) -> list[Card]:
        return [Card(card["name"], card["description"]) for card in self.cards]


@promptly()
def generate_spyware_cards() -> GenerateSpywareCardResult:
    """\
    Act as an excellent game designer. Create a set of 5 cards inspired by the world of espionage, linguistic word-play, and double-O-seven.

    Consult the following rulebook:

    {% include 'rulebook.md' %}

    {{ formatting_instructions }}
    """
    ...


@promptly
def generate_sat_flashcards(
    avoid: list[str] | None = None,
) -> GenerateSATFlashCardResult:
    """\
    Act as an excellent tutor and a test prep professional by designing
    a set of 5 flashcards for SAT preparation, as if pulled from a dictionary.

    {% if avoid %}
    However, some words have already been generated and should be avoided. Their precise titles are below.
    {% for word in avoid %}
    - {{ word }}
    {% endfor %}
    {% endif %}

    Write words in their entire form, and provide their definitions.
    Choose words that are no longer than 9 characters.

    {{ formatting_instructions }}
    """


@dataclass
class GenerateStageResult:
    subtitle: Annotated[
        str,
        (
            "A suitable short name for the stage, no longer than something like "
            "'Beneath the Soil' or 'Name of the Rose'. "
        ),
    ]
    lore: Annotated[
        str,
        (
            "A short twenty words of lore text, written as if by an excelllent game writer. The lore text style should be mystical, eerie, and contemplative."
        ),
    ]
    danger_level: Annotated[
        int, ("A number between 1 and 5, indicating the danger level of the stage. ")
    ]
    enemy_troop: Annotated[
        list[str],
        (
            "A list of enemies that the player will face in this stage. "
            "Jot down rough thoughts, the 'function' of this enemy within your grand design, and the name of the enemy. "
            "In the form of a list of strings, each string consisting of both the enemy's name and its function."
        ),
    ]


@promptly()
def generate_stage_description(
    stage_name: str,
    adventure_logs: list[str],
    inspiration: int,
) -> GenerateStageResult:
    """\
    Act as an excellent game writer. The player will arrive at a new stage named
    '{{ stage_name }}'. Your task is to provide a fitting description of this stage,
    considering the atmosphere for their next intriguing adventure.

    First, provide a subtitle for the stage. The subtitle should be short,
    capturing your main inspiration.

    Next, write a short lore text of no more than twenty words that captures the
    essence of the stage.
    Avoid stereotypical tropes such as "whispers", etc..

    Your writing style should take *strong* inspiration from the following:

    ```plaintext
    {% if inspiration == 0 %}
        {% include 'poems/our-land.md' %}
    {% elif inspiration == 1 %}
        {% include 'poems/the-snow-storm.md' %}
    {% elif inspiration == 2 %}
        {% include 'poems/the-waste-land.txt' %}
    {% elif inspiration == 3 %}
        {% include 'poems/wild-geese.md' %}
    {% elif inspiration == 4 %}
        {% include 'poems/our-land.md' %}
    {% elif inspiration == 5 %}
        {% include 'poems/our-land.md' %}
    {% endif %}
    ```

    Finally, assign a danger level to the stage, a number between 1 and 5.
    1 should indicate a beginner level stage, 3 is a somewhat challenging
    stage, and 5 is a stage that is extremely dangerous (e.g., optional level,
    for final boss). World 1 should only occasionally feature a danger level
    above 2, but otherwise the danger level should increase as the player
    progresses through the worlds.

    For your context, the player's adventure logs are as follows:

    {% for log in adventure_logs %}
    "{{ log }}";
    {% endfor %}

    {{ formatting_instructions }}
    """
