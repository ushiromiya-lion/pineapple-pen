import pytest
from genio.battle import setup_battle_bundle
from genio.battle_commands import ApplyStatus, DealDamage
from genio.effect import (
    Advisory,
    ModifyAmount,
    OnDamageDealt,
    OnDamageTaken,
    StatusDefinition,
    UnsafeStatusExpression,
    evaluate_status_expr,
)


def test_typed_vulnerable_modifies_base_damage_once():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    vulnerable = StatusDefinition(
        name="vulnerable",
        trigger=OnDamageTaken(),
        reaction=ModifyAmount(expr="amount * 1.25"),
        counter_type="turns",
    )

    bundle.apply_commands([ApplyStatus(target=enemy.name, status=vulnerable, duration=2)])
    effects = bundle.apply_commands(
        [DealDamage(target=enemy.name, amount=4)], caster=bundle.player
    )

    assert effects.total_damage() == 5
    assert enemy.hp == enemy.max_hp - 5


def test_typed_strength_modifies_damage_dealt_once():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    strength = StatusDefinition(
        name="strength",
        trigger=OnDamageDealt(),
        reaction=ModifyAmount(expr="amount * 1.25"),
        counter_type="times",
    )

    bundle.apply_commands(
        [ApplyStatus(target=bundle.player.name, status=strength, duration=1)]
    )
    effects = bundle.apply_commands(
        [DealDamage(target=enemy.name, amount=4)], caster=bundle.player
    )

    assert effects.total_damage() == 5
    assert enemy.hp == enemy.max_hp - 5
    assert bundle.player.status_effects == []


def test_advisory_status_persists_ticks_and_has_no_engine_effect():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    drenched = StatusDefinition(
        name="drenched",
        trigger=None,
        reaction=Advisory("fire attacks against this target fizzle"),
        counter_type="turns",
    )

    bundle.apply_commands([ApplyStatus(target=enemy.name, status=drenched, duration=1)])
    assert "drenched" in bundle.status_snapshot_text()

    effects = bundle.apply_commands(
        [DealDamage(target=enemy.name, amount=4)], caster=bundle.player
    )
    assert effects.total_damage() == 4
    assert enemy.hp == enemy.max_hp - 4

    bundle.on_turn_end()
    assert enemy.status_effects == []


def test_unsafe_status_expression_is_rejected():
    with pytest.raises(UnsafeStatusExpression):
        evaluate_status_expr("__import__('os').system('echo nope')", amount=1, counter=1)
