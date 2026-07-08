import pytest
from genio.battle import setup_battle_bundle
from genio.battle_commands import ApplyStatus, DealDamage
from genio.effect import (
    Advisory,
    DealDamageToSelf,
    GainBlockReaction,
    HealSelf,
    ModifyAmount,
    OnBlockGained,
    OnDamageDealt,
    OnDamageTaken,
    OnTurnEnd,
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


def test_free_text_damage_source_does_not_crash_or_activate_status():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    strength = StatusDefinition(
        name="strength",
        trigger=OnDamageDealt(),
        reaction=ModifyAmount(expr="amount * 2"),
        counter_type="turns",
    )
    bundle.apply_commands([ApplyStatus(target=enemy.name, status=strength, duration=1)])

    effects = bundle.process_and_flush_effects(
        f"[{bundle.player.name}: damaged 4 by slime]"
    )

    assert effects.total_damage() == 4
    assert bundle.player.hp == bundle.player.max_hp - 4


def test_exact_damage_source_activates_status():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    strength = StatusDefinition(
        name="strength",
        trigger=OnDamageDealt(),
        reaction=ModifyAmount(expr="amount * 2"),
        counter_type="turns",
    )
    bundle.apply_commands([ApplyStatus(target=enemy.name, status=strength, duration=1)])

    effects = bundle.process_and_flush_effects(
        f"[{bundle.player.name}: damaged 4 by {enemy.name}]"
    )

    assert effects.total_damage() == 8
    assert bundle.player.hp == bundle.player.max_hp - 8


def test_invalid_legacy_status_expression_degrades_to_advisory():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]

    bundle.process_and_flush_effects(
        f"[{enemy.name}: +bad [1 turn] [ME: damaged {{:d}}] -> [ME: damaged {{{{m[0] * x}}}}];]"
    )
    assert isinstance(enemy.status_effects[0].defn.reaction, Advisory)

    effects = bundle.process_and_flush_effects(f"[{enemy.name}: damaged 3]")
    assert effects.total_damage() == 3


def test_invalid_end_of_turn_status_degrades_to_advisory():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]

    bundle.process_and_flush_effects(
        f"[{enemy.name}: +bad [1 turn] [ME: end of turn] -> [ME: damaged 3 points];]"
    )

    assert isinstance(enemy.status_effects[0].defn.reaction, Advisory)


def test_runtime_status_expression_errors_are_guarded_and_damage_clamped():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    unstable = StatusDefinition(
        name="unstable",
        trigger=OnDamageTaken(),
        reaction=ModifyAmount(expr="amount / (counter - 1)"),
        counter_type="turns",
    )
    inverted = StatusDefinition(
        name="inverted",
        trigger=OnDamageTaken(),
        reaction=ModifyAmount(expr="amount - 10"),
        counter_type="turns",
    )

    bundle.apply_commands([ApplyStatus(target=enemy.name, status=unstable, duration=1)])
    effects = bundle.apply_commands(
        [DealDamage(target=enemy.name, amount=3)], caster=bundle.player
    )
    assert effects.total_damage() == 3

    bundle.apply_commands([ApplyStatus(target=enemy.name, status=inverted, duration=1)])
    effects = bundle.apply_commands(
        [DealDamage(target=enemy.name, amount=3)], caster=bundle.player
    )
    assert effects.total_damage() == 0


def test_end_of_turn_reactions_evaluate_live_counter():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    scaling_poison = StatusDefinition(
        name="scaling poison",
        trigger=OnTurnEnd(),
        reaction=DealDamageToSelf("counter"),
        counter_type="turns",
    )

    bundle.apply_commands(
        [ApplyStatus(target=enemy.name, status=scaling_poison, duration=3)]
    )
    bundle.on_turn_end()

    assert enemy.hp == enemy.max_hp - 3
    assert enemy.status_effects[0].counter == 2


def test_legacy_regen_guard_and_frail_are_typed_reactions():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    enemy.hp = enemy.max_hp - 5

    bundle.process_and_flush_effects(
        f"[{enemy.name}: +regen [1 turn] [ME: end of turn] -> [ME: healed 3];]"
    )
    bundle.process_and_flush_effects(
        f"[{bundle.player.name}: +guard [1 turn] [ME: end of turn] -> [ME: shield 3];]"
    )
    bundle.process_and_flush_effects(
        f"[{bundle.player.name}: +frail [1 turn] [ME: block {{:b}}] -> [ME: block {{{{m[0] * 0.75}}}}];]"
    )

    assert isinstance(enemy.status_effects[0].defn.reaction, HealSelf)
    assert isinstance(bundle.player.status_effects[0].defn.reaction, GainBlockReaction)
    assert isinstance(bundle.player.status_effects[1].defn.trigger, OnBlockGained)

    bundle.process_and_flush_effects(f"[{bundle.player.name}: shield 8]")
    assert bundle.player.shield_points == 6

    bundle.on_turn_end()
    assert enemy.hp == enemy.max_hp - 2
    assert bundle.player.shield_points == 8.25
    assert enemy.status_effects == []
    assert bundle.player.status_effects == []


def test_advisory_status_is_in_prompt_injections():
    bundle = setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])
    enemy = bundle.enemies[0]
    drenched = StatusDefinition(
        name="drenched",
        trigger=None,
        reaction=Advisory("fire attacks against this target fizzle"),
        counter_type="turns",
    )

    bundle.apply_commands([ApplyStatus(target=enemy.name, status=drenched, duration=1)])

    assert "drenched" in "\n".join(bundle.prompt_injections())
