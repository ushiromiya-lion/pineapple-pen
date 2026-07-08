from genio.battle import (
    BattleBundle,
    BattlePrelude,
    CardBundle,
    EnemyBattler,
    PlayerBattler,
)
from genio.battle_commands import (
    ApplyStatus,
    CreateCard,
    DealDamage,
    DestroyCard,
    DestroyRule,
    DiscardCards,
    DrawCards,
    DuplicateCard,
    GainBlock,
    Heal,
    LoseBlock,
    RemoveStatus,
    TransformCard,
    command_from_effect,
)
from genio.card import Card
from genio.effect import (
    CreateCardEffect,
    DiscardCardsEffect,
    ModifyAmount,
    OnDamageTaken,
    SinglePointEffect,
    StatusDefinition,
)


def make_bundle() -> BattleBundle:
    player = PlayerBattler.from_predef("players.starter")
    enemy = EnemyBattler.from_predef("enemies.slime")
    card_bundle = CardBundle.from_predef("initial_deck")
    card_bundle.draw_to_hand()
    return BattleBundle(player, [enemy], BattlePrelude.default(), card_bundle)


def vulnerable_status() -> StatusDefinition:
    return StatusDefinition(
        name="vulnerable",
        trigger=OnDamageTaken(),
        reaction=ModifyAmount(expr="amount * 1.5"),
        counter_type="turns",
        description="Takes more damage.",
    )


def test_command_from_targeted_effect_damage():
    command = command_from_effect(("Slime A", SinglePointEffect.from_damage(4)))

    assert command == DealDamage(target="Slime A", amount=4)


def test_command_from_global_effect_create_and_discard():
    card = Card("shiv", "Deal 1 damage.")

    assert command_from_effect(CreateCardEffect(card=card, copies=2)) == CreateCard(
        name="shiv", description="Deal 1 damage.", copies=2
    )
    assert command_from_effect(DiscardCardsEffect(count=1)) == DiscardCards(count=1)


def test_apply_targeted_commands():
    bundle = make_bundle()
    player = bundle.player
    enemy = bundle.enemies[0]
    player.hp = 70
    enemy.shield_points = 3

    applied = bundle.apply_commands(
        [
            LoseBlock(target=enemy.name, amount=2),
            DealDamage(target=enemy.name, amount=4),
            Heal(target=player.name, amount=3),
            GainBlock(target=player.name, amount=5),
        ],
        caster=player,
    )

    assert enemy.hp == enemy.max_hp - 3
    assert player.hp == 73
    assert player.shield_points == 5
    assert enemy.shield_points == 0
    assert len(applied) == 4


def test_apply_status_and_remove_status_commands():
    bundle = make_bundle()
    enemy = bundle.enemies[0]

    bundle.apply_commands(
        [ApplyStatus(target=enemy.name, status=vulnerable_status(), duration=2)]
    )
    assert [status.name for status in enemy.status_effects] == ["vulnerable"]

    bundle.apply_commands([RemoveStatus(target=enemy.name, status_name="vulnerable")])
    assert enemy.status_effects == []


def test_apply_card_zone_commands():
    bundle = make_bundle()
    original_hand_size = len(bundle.card_bundle.hand)
    first_card = bundle.card_bundle.hand[0]
    first_card_id = first_card.short_id()

    bundle.apply_commands([DrawCards(count=1)])
    assert len(bundle.card_bundle.hand) == original_hand_size + 1

    bundle.apply_commands([DiscardCards(card_ids=(first_card_id,))])
    assert first_card not in bundle.card_bundle.hand
    assert first_card in bundle.card_bundle.graveyard

    bundle.apply_commands([CreateCard(name="shiv", description="Deal 1 damage.")])
    assert bundle.card_bundle.has_card("shiv") == "hand"

    shiv = bundle.card_bundle.seek_card("shiv")
    bundle.apply_commands([DuplicateCard(card_id=shiv.short_id(), copies=2)])
    assert bundle.card_bundle.count_cards("shiv") == 3

    bundle.apply_commands(
        [
            TransformCard(
                card_id=shiv.short_id(),
                name="knife",
                description="Deal 2 damage.",
            )
        ]
    )
    assert shiv.name == "knife"
    assert shiv.description == "Deal 2 damage."

    bundle.apply_commands([DestroyCard(card_ids=(shiv.short_id(),))])
    assert bundle.card_bundle.has_card("knife") is None


def test_apply_destroy_rule_command():
    bundle = make_bundle()
    assert bundle.rules[1] is not None

    bundle.apply_commands([DestroyRule(rule_id=1)])

    assert bundle.rules[1] is None


def test_process_effects_uses_command_shim_for_existing_dsl():
    bundle = make_bundle()
    enemy = bundle.enemies[0]

    flushed = bundle.process_and_flush_effects(f"[{enemy.name}: damaged 3]")

    assert flushed.total_damage() == 3
    assert enemy.hp == enemy.max_hp - 3
