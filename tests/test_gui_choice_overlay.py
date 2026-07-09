from types import SimpleNamespace

from genio.battle_harness import ChooseCardsRequest
from genio.card import Card
from genio.gui import PlayerChoiceOverlay


def test_single_card_choice_does_not_deselect_on_second_click():
    first = Card("First")
    second = Card("Second")
    overlay = PlayerChoiceOverlay(
        SimpleNamespace(),
        ChooseCardsRequest(
            prompt="Choose one",
            reason="",
            cards=[first, second],
            min_count=1,
            max_count=1,
        ),
    )

    overlay._toggle(first)
    overlay._toggle(first)

    assert overlay.selected == {first.short_id()}

    overlay._toggle(second)

    assert overlay.selected == {second.short_id()}
