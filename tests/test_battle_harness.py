from genio.battle import setup_battle_bundle
from genio.battle_commands import ApplyStatus
from genio.battle_harness import BattleToolHarness
from genio.card import Card
from genio.effect import ModifyAmount, OnDamageDealt, StatusDefinition
from toolloop_fakes import fc_part, model_turn, scripted


def make_bundle():
    return setup_battle_bundle("initial_deck", "players.starter", ["enemies.slime"])


def response_payloads(content):
    return [part.function_response.response for part in content.parts]


def test_simple_attack_one_round_trip():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part("deal_damage", {"target": enemy.name, "amount": 6}),
                    fc_part("finish_resolution", {"reason": "hit", "significance": 1}),
                )
            ]
        ),
    )

    resolved, reason = harness.resolve("resolve", enemy_mode=False)

    assert enemy.hp == enemy.max_hp - 6
    assert resolved.rarity == 1
    assert reason == "hit"


def test_invalid_target_repair():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(fc_part("deal_damage", {"target": "Slim A", "amount": 6})),
                model_turn(
                    fc_part("deal_damage", {"target": enemy.name, "amount": 6}),
                    fc_part("finish_resolution", {"reason": "hit", "significance": 1}),
                ),
            ]
        ),
    )

    harness.resolve("resolve", enemy_mode=False)

    first_error = response_payloads(harness.session.history[2])[0]
    assert "Slime A" in first_error["error"]
    assert enemy.hp == enemy.max_hp - 6


def test_apply_at_finish_atomicity():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    hp_seen_after_staging = []
    responses = iter(
        [
            model_turn(fc_part("deal_damage", {"target": enemy.name, "amount": 6})),
            model_turn(fc_part("unknown", {})),
            model_turn(fc_part("unknown", {})),
            model_turn(fc_part("deal_damage", {"target": enemy.name, "amount": 6})),
        ]
    )

    def generate_fn(*, contents, config):
        if len(contents) >= 3 and not hp_seen_after_staging:
            hp_seen_after_staging.append(enemy.hp)
        return next(responses)

    harness = BattleToolHarness(bundle, generate_fn=generate_fn)

    resolved, reason = harness.resolve("resolve", enemy_mode=False)

    assert hp_seen_after_staging == [enemy.max_hp]
    assert enemy.hp == enemy.max_hp - 6
    assert resolved.rarity == 1
    assert reason == "(resolution aborted; applying staged commands)"


def test_staged_conflict_destroy_then_duplicate():
    bundle = make_bundle()
    card = bundle.card_bundle.hand[0]
    card_id = card.short_id()
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part("destroy_card", {"card_ids": [card_id]}),
                    fc_part("duplicate_card", {"card_id": card_id}),
                    fc_part("finish_resolution", {"reason": "burned", "significance": 1}),
                )
            ]
        ),
    )

    harness.resolve("resolve", enemy_mode=False)

    responses = [part.function_response.response for part in harness.session._pending_responses]
    assert responses[1]["error"] == f'card "{card_id}" was already consumed earlier in this resolution'
    assert all(card.short_id() != card_id for card in bundle.card_bundle.hand)
    assert all(card.short_id() != card_id for card in bundle.card_bundle.deck)
    assert all(card.short_id() != card_id for card in bundle.card_bundle.graveyard)


def test_staged_conflict_normalizes_card_name_before_consumed_check():
    bundle = make_bundle()
    card = bundle.card_bundle.hand[0]
    card_id = card.short_id()
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part("destroy_card", {"card_ids": [card_id.upper()]}),
                    fc_part("duplicate_card", {"card_id": card.name}),
                    fc_part("finish_resolution", {"reason": "burned", "significance": 1}),
                )
            ]
        ),
    )

    harness.resolve("resolve", enemy_mode=False)

    responses = [part.function_response.response for part in harness.session._pending_responses]
    assert responses[1]["error"] == f'card "{card_id}" was already consumed earlier in this resolution'


def test_apply_status_typed_reaction():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part(
                        "apply_status",
                        {
                            "target": enemy.name,
                            "name": "vulnerable",
                            "duration": 2,
                            "duration_type": "turns",
                            "trigger": "on_damage_taken",
                            "reaction": {
                                "kind": "modify_amount",
                                "expr": "amount * 1.5",
                            },
                            "description": "Takes more damage.",
                        },
                    ),
                    fc_part("finish_resolution", {"reason": "exposed", "significance": 1}),
                ),
                model_turn(
                    fc_part("deal_damage", {"target": enemy.name, "amount": 4}),
                    fc_part("finish_resolution", {"reason": "hit", "significance": 1}),
                ),
            ]
        ),
    )

    harness.resolve("apply status", enemy_mode=False)
    resolved, _ = harness.resolve("hit", enemy_mode=False)

    assert resolved.total_damage() == 6
    assert enemy.hp == enemy.max_hp - 6


def test_apply_status_invalid_expr_repair():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part(
                        "apply_status",
                        {
                            "target": enemy.name,
                            "name": "bad",
                            "duration": 2,
                            "duration_type": "turns",
                            "trigger": "on_damage_taken",
                            "reaction": {
                                "kind": "modify_amount",
                                "expr": "amount * x",
                            },
                            "description": "Bad math.",
                        },
                    ),
                    fc_part("finish_resolution", {"reason": "skip", "significance": 1}),
                )
            ]
        ),
    )

    harness.resolve("resolve", enemy_mode=False)

    responses = [part.function_response.response for part in harness.session._pending_responses]
    assert "invalid expression" in responses[0]["error"]
    assert enemy.status_effects == []


def test_advisory_status():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part(
                        "apply_status",
                        {
                            "target": enemy.name,
                            "name": "drenched",
                            "duration": 1,
                            "duration_type": "turns",
                            "reaction": {"kind": "advisory"},
                            "description": "fire attacks fizzle",
                        },
                    ),
                    fc_part("finish_resolution", {"reason": "wet", "significance": 1}),
                )
            ]
        ),
    )

    harness.resolve("resolve", enemy_mode=False)

    assert enemy.status_effects[0].name == "drenched"
    assert "advisory" in bundle.status_snapshot_text()
    hp = enemy.hp
    block = enemy.shield_points
    bundle.on_turn_end()
    assert enemy.hp == hp
    assert enemy.shield_points == block


def test_enemy_mode_requires_source():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    strength = StatusDefinition(
        name="strength",
        trigger=OnDamageDealt(),
        reaction=ModifyAmount(expr="amount * 2"),
        counter_type="turns",
    )
    bundle.apply_commands([ApplyStatus(target=enemy.name, status=strength, duration=1)])
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(fc_part("deal_damage", {"target": bundle.player.name, "amount": 2})),
                model_turn(
                    fc_part(
                        "deal_damage",
                        {"target": bundle.player.name, "amount": 2, "source": enemy.name},
                    ),
                    fc_part("finish_resolution", {"reason": "hit", "significance": 1}),
                ),
            ]
        ),
    )

    resolved, _ = harness.resolve("resolve", enemy_mode=True)

    assert "source is required" in response_payloads(harness.session.history[2])[0]["error"]
    assert resolved.total_damage() == 4
    assert bundle.player.hp == bundle.player.max_hp - 4


def test_engine_log_reaches_next_turn():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    captured_contents = []

    def generate_fn(*, contents, config):
        captured_contents.append(list(contents))
        if len(captured_contents) == 1:
            return model_turn(
                fc_part("deal_damage", {"target": enemy.name, "amount": 3}),
                fc_part("finish_resolution", {"reason": "hit", "significance": 1}),
            )
        return model_turn(
            fc_part("finish_resolution", {"reason": "next", "significance": 1})
        )

    harness = BattleToolHarness(bundle, generate_fn=generate_fn)
    harness.resolve("first", enemy_mode=False)
    harness.resolve("second", enemy_mode=False)

    opening = captured_contents[1][-1]
    assert opening.parts[0].function_response.name == "deal_damage"
    assert opening.parts[1].function_response.name == "finish_resolution"
    assert "engine_log" in opening.parts[1].function_response.response["output"]
    assert len(captured_contents[1]) == 1


def test_fallback_commit_when_model_never_finishes():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(fc_part("deal_damage", {"target": enemy.name, "amount": 5})),
                model_turn(fc_part("unknown", {})),
                model_turn(fc_part("unknown", {})),
                model_turn(fc_part("deal_damage", {"target": enemy.name, "amount": 5})),
            ]
        ),
    )

    resolved, reason = harness.resolve("resolve", enemy_mode=False)

    assert enemy.hp == enemy.max_hp - 5
    assert resolved.rarity == 1
    assert reason == "(resolution aborted; applying staged commands)"


def test_resolve_player_cards_uses_harness():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part("deal_damage", {"target": enemy.name, "amount": 7}),
                    fc_part("finish_resolution", {"reason": "fire", "significance": 2}),
                )
            ]
        ),
    )
    bundle._harness = harness
    card = Card("Fireball", "Deal 7 damage.", energy_cost=0)

    resolved = bundle.resolve_player_cards([card])

    assert enemy.hp == enemy.max_hp - 7
    assert resolved.rarity == 2


def test_resolve_player_cards_request_includes_played_card_short_ids():
    bundle = make_bundle()
    captured_messages = []

    def generate_fn(*, contents, config):
        captured_messages.append(contents[-1].parts[-1].text)
        return model_turn(
            fc_part("finish_resolution", {"reason": "nothing", "significance": 1})
        )

    harness = BattleToolHarness(bundle, generate_fn=generate_fn)
    bundle._harness = harness
    card = Card("Cinder Loop", "Destroy this card after use.", energy_cost=0)
    bundle.card_bundle.hand.append(card)
    bundle.card_bundle.hand_to_resolving([card])

    bundle.resolve_player_cards([card])

    assert f"- {card.short_id()}:" in captured_messages[0]
    assert "Resolving:" in captured_messages[0]


def test_delayed_commands_are_queued_not_reported_as_applied():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part(
                        "deal_damage",
                        {"target": enemy.name, "amount": 4, "delay": 1},
                    ),
                    fc_part("finish_resolution", {"reason": "later", "significance": 1}),
                )
            ]
        ),
    )

    resolved, _ = harness.resolve("resolve", enemy_mode=False)

    assert len(resolved) == 0
    assert enemy.hp == enemy.max_hp
    output = harness.session._pending_responses[-1].function_response.response["output"]
    assert output["applied"] == 0
    assert output["queued"] == ["DealDamage"]

    bundle.turn_counter = 1
    flushed = bundle.flush_expired_effects(bundle.rng)
    assert flushed.total_damage() == 4
    assert enemy.hp == enemy.max_hp - 4


def test_resolve_enemy_actions_uses_harness():
    bundle = make_bundle()
    enemy = bundle.enemies[0]
    enemy.current_intent = "curse the player"
    harness = BattleToolHarness(
        bundle,
        generate_fn=scripted(
            [
                model_turn(
                    fc_part(
                        "apply_status",
                        {
                            "target": bundle.player.name,
                            "name": "hexed",
                            "duration": 1,
                            "duration_type": "turns",
                            "reaction": {"kind": "advisory"},
                            "description": "ominous magic clings to them",
                        },
                    ),
                    fc_part("finish_resolution", {"reason": "curse", "significance": 1}),
                )
            ]
        ),
    )
    bundle._harness = harness

    resolved = bundle.resolve_enemy_actions()

    assert resolved.rarity == -1
    assert [status.name for status in bundle.player.status_effects] == ["hexed"]
