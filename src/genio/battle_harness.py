from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from google.genai import types
from structlog import get_logger

from genio.battle_commands import (
    ApplyStatus,
    BattleCommand,
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
)
from genio.card import Card
from genio.core.base import jinja_env
from genio.core.llm import GEMINI_MODEL, SAFETY_SETTINGS
from genio.core.toolloop import (
    LoopStatus,
    PendingInput,
    ToolError,
    ToolLoopSession,
    ToolSpec,
)
from genio.effect import (
    Advisory,
    DealDamageToSelf,
    GainBlockReaction,
    ModifyAmount,
    OnDamageDealt,
    OnDamageTaken,
    OnTurnEnd,
    OnTurnStart,
    StatusDefinition,
    StatusReaction,
    StatusTrigger,
    UnsafeStatusExpression,
    evaluate_status_condition,
    evaluate_status_expr,
)

if TYPE_CHECKING:
    from genio.battle import BattleBundle, Battler, ResolvedEffects

logger = get_logger()


@dataclass(frozen=True)
class ChooseCardsRequest:
    prompt: str
    reason: str
    cards: list[Card]
    min_count: int
    max_count: int


@dataclass(frozen=True)
class ChooseTargetsRequest:
    prompt: str
    reason: str
    candidates: list[Battler]
    min_count: int
    max_count: int


PlayerChoiceRequest = ChooseCardsRequest | ChooseTargetsRequest


def _s(schema_type: str, desc: str = "", **kwargs) -> types.Schema:
    return types.Schema(type=schema_type, description=desc, **kwargs)


def _obj(props: dict[str, types.Schema], required: list[str]) -> types.Schema:
    return types.Schema(type="OBJECT", properties=props, required=required)


def _array(items: types.Schema, desc: str = "") -> types.Schema:
    return types.Schema(type="ARRAY", items=items, description=desc)


def _decl(
    name: str,
    description: str,
    props: dict[str, types.Schema],
    required: list[str],
) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=_obj(props, required),
    )


TARGET = _s("STRING", "Exact target battler name from the state snapshot.")
AMOUNT_999 = _s("INTEGER", "Base amount.", minimum=0, maximum=999)
AMOUNT_99 = _s("INTEGER", "Base amount.", minimum=1, maximum=99)
DELAY = _s("INTEGER", "Delay in turns.", minimum=0, maximum=3)
WHERE = _s(
    "STRING",
    "Destination zone.",
    enum=["deck_top", "deck", "hand", "graveyard"],
)
CHOICE_ZONE = _s("STRING", "Choice source zone.", enum=["hand", "graveyard", "deck"])
DURATION_TYPE = _s("STRING", "Duration unit.", enum=["turns", "times"])
TRIGGER = _s(
    "STRING",
    "Typed status trigger.",
    enum=["on_damage_taken", "on_damage_dealt", "on_turn_start", "on_turn_end"],
)

REACTION = _obj(
    {
        "kind": _s(
            "STRING",
            (
                "modify_amount: rewrites the amount of the triggering event; "
                "expr is arithmetic over `amount` and `counter`, e.g. "
                "'amount * 1.5'; optional condition e.g. 'amount <= 2'. "
                "deal_damage_to_self / gain_block: amount required, trigger "
                "should be a turn event. advisory: no mechanical effect; the "
                "engine shows the description to you each turn and you honor "
                "it narratively."
            ),
            enum=["modify_amount", "deal_damage_to_self", "gain_block", "advisory"],
        ),
        "expr": _s("STRING", "Arithmetic expression over amount and counter."),
        "condition": _s("STRING", "Optional boolean expression."),
        "consumes_counter": _s("BOOLEAN", "Whether an interception consumes counter."),
        "amount": AMOUNT_99,
    },
    ["kind"],
)


def _tool_declarations() -> list[types.FunctionDeclaration]:
    return [
        _decl(
            "deal_damage",
            "Stage base damage against a target.",
            {
                "target": TARGET,
                "amount": AMOUNT_999,
                "source": _s(
                    "STRING",
                    "Exact acting battler name. Required during enemy resolution.",
                ),
                "pierce": _s("BOOLEAN", "Ignore block."),
                "drain": _s("BOOLEAN", "Heal source for damage dealt."),
                "delay": DELAY,
            },
            ["target", "amount"],
        ),
        _decl(
            "heal",
            "Stage healing.",
            {"target": TARGET, "amount": AMOUNT_999, "delay": DELAY},
            ["target", "amount"],
        ),
        _decl(
            "gain_block",
            "Stage block gain.",
            {
                "target": TARGET,
                "amount": AMOUNT_999,
                "source": _s("STRING", "Exact acting battler name."),
                "delay": DELAY,
            },
            ["target", "amount"],
        ),
        _decl(
            "lose_block",
            "Stage block loss.",
            {"target": TARGET, "amount": AMOUNT_999, "delay": DELAY},
            ["target", "amount"],
        ),
        _decl(
            "apply_status",
            "Stage a typed status application.",
            {
                "target": TARGET,
                "name": _s("STRING", "Status name."),
                "duration": _s("INTEGER", "Status duration.", minimum=1, maximum=9),
                "duration_type": DURATION_TYPE,
                "reaction": REACTION,
                "description": _s("STRING", "Status description for snapshots."),
                "trigger": TRIGGER,
            },
            ["target", "name", "duration", "duration_type", "reaction", "description"],
        ),
        _decl(
            "remove_status",
            "Stage status removal.",
            {"target": TARGET, "name": _s("STRING", "Status name.")},
            ["target", "name"],
        ),
        _decl(
            "draw_cards",
            "Stage card draw.",
            {"count": _s("INTEGER", "Number of cards.", minimum=1, maximum=5)},
            ["count"],
        ),
        _decl(
            "discard_cards",
            "Stage discarding cards by count or exact card ids.",
            {
                "count": _s("INTEGER", "Number of cards."),
                "card_ids": _array(_s("STRING"), "Exact card ids."),
            },
            [],
        ),
        _decl(
            "create_card",
            "Stage creating cards.",
            {
                "name": _s("STRING", "New card name."),
                "description": _s("STRING", "New card description."),
                "copies": _s("INTEGER", "Copies.", minimum=1, maximum=3),
                "where": WHERE,
            },
            ["name", "description"],
        ),
        _decl(
            "duplicate_card",
            "Stage duplicating an existing card.",
            {
                "card_id": _s("STRING", "Exact card id."),
                "copies": _s("INTEGER", "Copies.", minimum=1, maximum=3),
                "where": WHERE,
            },
            ["card_id"],
        ),
        _decl(
            "transform_card",
            "Stage transforming an existing card.",
            {
                "card_id": _s("STRING", "Exact card id."),
                "name": _s("STRING", "New card name."),
                "description": _s("STRING", "New card description."),
            },
            ["card_id", "name", "description"],
        ),
        _decl(
            "destroy_card",
            "Stage destroying cards.",
            {"card_ids": _array(_s("STRING"), "Exact card ids.")},
            ["card_ids"],
        ),
        _decl(
            "destroy_rule",
            "Stage destroying a rule.",
            {"rule_id": _s("INTEGER", "Rule id.")},
            ["rule_id"],
        ),
        _decl(
            "ask_player_choose_cards",
            "Ask the player to choose cards before continuing resolution.",
            {
                "prompt": _s("STRING", "Mechanical instruction shown to the player."),
                "reason": _s(
                    "STRING",
                    (
                        "One short sentence of in-world narration shown to the "
                        "player; match the battle's tone."
                    ),
                ),
                "zone": CHOICE_ZONE,
                "min_count": _s("INTEGER", "Minimum cards to choose.", minimum=0),
                "max_count": _s("INTEGER", "Maximum cards to choose.", minimum=1),
            },
            ["prompt", "min_count", "max_count"],
        ),
        _decl(
            "ask_player_choose_targets",
            "Ask the player to choose battler targets before continuing resolution.",
            {
                "prompt": _s("STRING", "Mechanical instruction shown to the player."),
                "reason": _s(
                    "STRING",
                    (
                        "One short sentence of in-world narration shown to the "
                        "player; match the battle's tone."
                    ),
                ),
                "candidates": _array(_s("STRING"), "Exact candidate battler names."),
                "min_count": _s("INTEGER", "Minimum targets to choose.", minimum=0),
                "max_count": _s("INTEGER", "Maximum targets to choose.", minimum=1),
            },
            ["prompt"],
        ),
        _decl(
            "finish_resolution",
            "Commit staged commands and finish this resolution.",
            {
                "reason": _s(
                    "STRING",
                    "One- to three-sentence narration of what happened and why.",
                ),
                "significance": _s(
                    "INTEGER", "1 routine, 2 strong, 3 spectacular.", minimum=1, maximum=3
                ),
            },
            ["reason", "significance"],
        ),
    ]


@dataclass
class BattleToolHarness:
    bundle: BattleBundle
    generate_fn: Any | None = None

    def __post_init__(self) -> None:
        self.session = ToolLoopSession(
            model=GEMINI_MODEL,
            tools=self._build_tools(),
            system_instruction=self._system_instruction(),
            safety_settings=SAFETY_SETTINGS,
            generate_fn=self.generate_fn,
        )
        self._staged: list[tuple[Battler | None, BattleCommand]] = []
        self._consumed_card_ids: set[str] = set()
        self._consumed_card_names: dict[str, str] = {}
        self._default_caster: Battler | None = None
        self._enemy_mode = False
        self._last_resolved: ResolvedEffects | None = None
        self._last_reason = ""
        self._queued_effects: list[str] = []
        self._resolve_lock = threading.RLock()

    def _build_tools(self) -> list[ToolSpec]:
        handlers = {
            "deal_damage": self._h_deal_damage,
            "heal": self._h_heal,
            "gain_block": self._h_gain_block,
            "lose_block": self._h_lose_block,
            "apply_status": self._h_apply_status,
            "remove_status": self._h_remove_status,
            "draw_cards": self._h_draw_cards,
            "discard_cards": self._h_discard_cards,
            "create_card": self._h_create_card,
            "duplicate_card": self._h_duplicate_card,
            "transform_card": self._h_transform_card,
            "destroy_card": self._h_destroy_card,
            "destroy_rule": self._h_destroy_rule,
            "ask_player_choose_cards": self._h_ask_choose_cards,
            "ask_player_choose_targets": self._h_ask_choose_targets,
            "finish_resolution": self._h_finish,
        }
        specs = []
        for declaration in _tool_declarations():
            assert declaration.name is not None
            specs.append(
                ToolSpec(
                    declaration=declaration,
                    handler=handlers[declaration.name],
                    terminal=declaration.name == "finish_resolution",
                )
            )
        return specs

    def _system_instruction(self) -> str:
        return jinja_env.get_template("battle_tools.md").render()

    def _valid_targets(self) -> str:
        return ", ".join(battler.name for battler in self.bundle.battlers())

    def _target(self, name: str) -> Battler:
        battler = self.bundle.search_exact(str(name))
        if battler is None:
            raise ToolError(
                f'target "{name}" does not exist. Valid targets: {self._valid_targets()}'
            )
        return battler

    def _source(self, name: str | None) -> Battler | None:
        if name is None or name == "":
            return self._default_caster
        battler = self.bundle.search_exact(str(name))
        if battler is None:
            raise ToolError(
                f'source "{name}" does not exist. Valid sources: {self._valid_targets()}'
            )
        return battler

    def _hand_ids(self) -> str:
        return ", ".join(card.short_id() for card in self.bundle.card_bundle.hand)

    def _card(self, card_id: str) -> Card:
        raw = str(card_id).lower()
        if raw in self._consumed_card_ids:
            raise ToolError(
                f'card "{raw}" was already consumed earlier in this resolution'
            )
        if consumed_id := self._consumed_card_names.get(raw):
            raise ToolError(
                f'card "{consumed_id}" was already consumed earlier in this resolution'
            )
        try:
            card = self.bundle.card_bundle.seek_card(card_id)
        except ValueError as exc:
            raise ToolError(
                f'card "{card_id}" does not exist. Valid hand ids: {self._hand_ids()}'
            ) from exc
        normalized_id = card.short_id()
        if normalized_id in self._consumed_card_ids:
            raise ToolError(
                f'card "{normalized_id}" was already consumed earlier in this resolution'
            )
        if consumed_id := self._consumed_card_names.get(card.name.lower()):
            raise ToolError(
                f'card "{consumed_id}" was already consumed earlier in this resolution'
            )
        return card

    def _req(self, args: dict, key: str) -> Any:
        if key not in args:
            raise ToolError(f'missing required argument "{key}"')
        return args[key]

    def _int(
        self,
        args: dict,
        key: str,
        lo: int,
        hi: int,
        default: int | None = None,
    ) -> int:
        if key not in args or args[key] is None:
            if default is not None:
                return default
            raise ToolError(f'missing required argument "{key}"')
        try:
            value = int(args[key])
        except (TypeError, ValueError) as exc:
            raise ToolError(f'"{key}" must be an integer from {lo} to {hi}') from exc
        if value < lo or value > hi:
            raise ToolError(f'"{key}" must be from {lo} to {hi}')
        return value

    def _duration_type(self, args: dict) -> str:
        duration_type = str(self._req(args, "duration_type"))
        if duration_type not in {"turns", "times"}:
            raise ToolError('"duration_type" must be "turns" or "times"')
        return duration_type

    def _where(self, args: dict) -> str:
        where = str(args.get("where", "hand"))
        if where not in {"deck_top", "deck", "hand", "graveyard"}:
            raise ToolError('"where" must be deck_top, deck, hand, or graveyard')
        return where

    def _choice_counts(self, args: dict) -> tuple[int, int]:
        min_count = self._int(args, "min_count", 0, 99, default=1)
        max_count = self._int(args, "max_count", 1, 99, default=1)
        if min_count > max_count:
            raise ToolError('"min_count" must be less than or equal to "max_count"')
        return min_count, max_count

    def _choice_text(self, args: dict) -> tuple[str, str]:
        prompt = " ".join(str(self._req(args, "prompt")).split())[:100]
        reason = " ".join(str(args.get("reason", "")).split())[:100]
        if not prompt:
            raise ToolError('"prompt" must not be empty')
        return prompt, reason

    def _trigger(self, trigger: str | None) -> StatusTrigger | None:
        if trigger is None:
            return None
        triggers = {
            "on_damage_taken": OnDamageTaken(),
            "on_damage_dealt": OnDamageDealt(),
            "on_turn_start": OnTurnStart(),
            "on_turn_end": OnTurnEnd(),
        }
        try:
            return triggers[str(trigger)]
        except KeyError as exc:
            raise ToolError(f'invalid trigger "{trigger}"') from exc

    def _reaction(
        self, args: dict, duration_type: str
    ) -> tuple[StatusTrigger | None, StatusReaction]:
        reaction_args = self._req(args, "reaction")
        if not isinstance(reaction_args, dict):
            raise ToolError('"reaction" must be an object')
        kind = str(self._req(reaction_args, "kind"))
        trigger = self._trigger(args.get("trigger"))
        if kind == "modify_amount":
            if not isinstance(trigger, OnDamageTaken | OnDamageDealt):
                raise ToolError(
                    'modify_amount requires trigger "on_damage_taken" or "on_damage_dealt"'
                )
            expr = str(self._req(reaction_args, "expr"))
            condition = reaction_args.get("condition")
            condition = str(condition) if condition is not None else None
            try:
                evaluate_status_expr(expr, amount=7, counter=1)
                evaluate_status_condition(condition, amount=7, counter=1)
            except (
                UnsafeStatusExpression,
                SyntaxError,
                ValueError,
                TypeError,
                ZeroDivisionError,
            ) as exc:
                raise ToolError(
                    f'invalid expression "{expr}": only arithmetic over amount and '
                    'counter is allowed, e.g. "amount * 1.5"'
                ) from exc
            consumes_counter = bool(
                reaction_args.get("consumes_counter", duration_type == "times")
            )
            return trigger, ModifyAmount(
                expr=expr,
                condition=condition,
                consumes_counter=consumes_counter,
            )
        if kind == "deal_damage_to_self":
            if not isinstance(trigger, OnTurnStart | OnTurnEnd):
                raise ToolError(
                    'deal_damage_to_self requires trigger "on_turn_start" or "on_turn_end"'
                )
            return trigger, DealDamageToSelf(
                self._int(reaction_args, "amount", 1, 99)
            )
        if kind == "gain_block":
            if not isinstance(trigger, OnTurnStart | OnTurnEnd):
                raise ToolError(
                    'gain_block requires trigger "on_turn_start" or "on_turn_end"'
                )
            return trigger, GainBlockReaction(
                self._int(reaction_args, "amount", 1, 99)
            )
        if kind == "advisory":
            return None, Advisory(str(self._req(args, "description")))
        raise ToolError(f'invalid reaction kind "{kind}"')

    def _stage(self, command: BattleCommand, caster: Battler | None) -> None:
        def remember(card_id: str) -> None:
            self._consumed_card_ids.add(card_id)
            try:
                name = self.bundle.card_bundle.seek_card(card_id).name.lower()
                self._consumed_card_names[name] = card_id
            except ValueError:
                pass

        self._staged.append((caster, command))
        match command:
            case DiscardCards(card_ids=card_ids):
                for card_id in card_ids:
                    remember(card_id)
            case DestroyCard(card_ids=card_ids):
                for card_id in card_ids:
                    remember(card_id)
            case TransformCard(card_id=card_id):
                remember(card_id)

    def _projection_note(self, command: BattleCommand, target: Battler | None) -> str:
        if not isinstance(command, DealDamage) or target is None:
            return ""
        amount = command.amount
        notes = []
        for status in target.status_effects:
            if status.is_expired() or not isinstance(status.defn.trigger, OnDamageTaken):
                continue
            reaction = status.defn.reaction
            if not isinstance(reaction, ModifyAmount):
                continue
            try:
                if not evaluate_status_condition(
                    reaction.condition, amount=amount, counter=status.counter
                ):
                    continue
                new_amount = evaluate_status_expr(
                    reaction.expr, amount=amount, counter=status.counter
                )
            except (UnsafeStatusExpression, ValueError, TypeError, ZeroDivisionError):
                continue
            notes.append(f"{status.name} {reaction.expr}")
            amount = max(0, new_amount)
        if not notes:
            return ""
        return (
            f"projected {command.amount:g} -> {amount:g} ({'; '.join(notes)}); "
            "final numbers computed on commit"
        )

    def _echo(self, command: BattleCommand, target: Battler | None = None) -> dict:
        return {
            "staged": True,
            "position": len(self._staged),
            "note": self._projection_note(command, target),
        }

    def _h_deal_damage(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        amount = self._int(args, "amount", 0, 999)
        caster = self._source(args.get("source"))
        if self._enemy_mode and caster is None:
            raise ToolError(
                "source is required when resolving enemy actions: name the acting enemy"
            )
        command = DealDamage(
            target=target.name,
            amount=amount,
            source=caster.name if caster else None,
            pierce=bool(args.get("pierce", False)),
            drain=bool(args.get("drain", False)),
            delay=self._int(args, "delay", 0, 3, default=0),
        )
        self._stage(command, caster)
        return self._echo(command, target)

    def _h_heal(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        command = Heal(
            target=target.name,
            amount=self._int(args, "amount", 0, 999),
            delay=self._int(args, "delay", 0, 3, default=0),
        )
        self._stage(command, None)
        return self._echo(command, target)

    def _h_gain_block(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        caster = self._source(args.get("source"))
        command = GainBlock(
            target=target.name,
            amount=self._int(args, "amount", 0, 999),
            delay=self._int(args, "delay", 0, 3, default=0),
        )
        self._stage(command, caster)
        return self._echo(command, target)

    def _h_lose_block(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        command = LoseBlock(
            target=target.name,
            amount=self._int(args, "amount", 0, 999),
            delay=self._int(args, "delay", 0, 3, default=0),
        )
        self._stage(command, None)
        return self._echo(command, target)

    def _h_apply_status(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        duration_type = self._duration_type(args)
        trigger, reaction = self._reaction(args, duration_type)
        status = StatusDefinition(
            name=str(self._req(args, "name")),
            trigger=trigger,
            reaction=reaction,
            counter_type=duration_type,
            description=str(self._req(args, "description")),
        )
        command = ApplyStatus(
            target=target.name,
            status=status,
            duration=self._int(args, "duration", 1, 9),
        )
        self._stage(command, None)
        return self._echo(command, target)

    def _h_remove_status(self, args: dict) -> dict:
        target = self._target(self._req(args, "target"))
        command = RemoveStatus(target=target.name, status_name=str(self._req(args, "name")))
        self._stage(command, None)
        return self._echo(command, target)

    def _h_draw_cards(self, args: dict) -> dict:
        command = DrawCards(count=self._int(args, "count", 1, 5))
        self._stage(command, None)
        return self._echo(command)

    def _h_discard_cards(self, args: dict) -> dict:
        has_count = args.get("count") is not None
        has_ids = bool(args.get("card_ids"))
        if has_count == has_ids:
            raise ToolError('discard_cards requires exactly one of "count" or "card_ids"')
        if has_count:
            command = DiscardCards(count=self._int(args, "count", 1, 99))
        else:
            card_ids = tuple(self._card(str(card_id)).short_id() for card_id in args["card_ids"])
            command = DiscardCards(card_ids=card_ids)
        self._stage(command, None)
        return self._echo(command)

    def _h_create_card(self, args: dict) -> dict:
        command = CreateCard(
            name=str(self._req(args, "name")),
            description=str(self._req(args, "description")),
            copies=self._int(args, "copies", 1, 3, default=1),
            where=self._where(args),
        )
        self._stage(command, None)
        return self._echo(command)

    def _h_duplicate_card(self, args: dict) -> dict:
        card = self._card(str(self._req(args, "card_id")))
        command = DuplicateCard(
            card_id=card.short_id(),
            copies=self._int(args, "copies", 1, 3, default=1),
            where=self._where(args),
        )
        self._stage(command, None)
        return self._echo(command)

    def _h_transform_card(self, args: dict) -> dict:
        card = self._card(str(self._req(args, "card_id")))
        command = TransformCard(
            card_id=card.short_id(),
            name=str(self._req(args, "name")),
            description=str(self._req(args, "description")),
        )
        self._stage(command, None)
        return self._echo(command)

    def _h_destroy_card(self, args: dict) -> dict:
        card_ids = tuple(
            self._card(str(card_id)).short_id() for card_id in self._req(args, "card_ids")
        )
        command = DestroyCard(card_ids=card_ids)
        self._stage(command, None)
        return self._echo(command)

    def _h_destroy_rule(self, args: dict) -> dict:
        rule_id = self._int(args, "rule_id", 1, len(self.bundle.rules) - 1)
        if self.bundle.rules[rule_id] is None:
            raise ToolError(f'rule "{rule_id}" does not exist')
        command = DestroyRule(rule_id=rule_id)
        self._stage(command, None)
        return self._echo(command)

    def _choice_cards(self, zone: str) -> list[Card]:
        match zone:
            case "hand":
                cards = self.bundle.card_bundle.hand
            case "graveyard":
                cards = self.bundle.card_bundle.graveyard
            case "deck":
                cards = self.bundle.card_bundle.deck
            case _:
                raise ToolError('"zone" must be hand, graveyard, or deck')
        return [
            card
            for card in cards
            if card.short_id() not in self._consumed_card_ids
            and card.name.lower() not in self._consumed_card_names
        ]

    def _h_ask_choose_cards(self, args: dict) -> dict | PendingInput:
        prompt, reason = self._choice_text(args)
        min_count, max_count = self._choice_counts(args)
        zone = str(args.get("zone", "hand"))
        options = self._choice_cards(zone)
        if not options:
            return {"card_ids": [], "cards": [], "note": "no cards available to choose"}
        max_count = min(max_count, len(options))
        min_count = min(min_count, max_count)

        def finalize(payload: dict) -> dict:
            if payload.get("cancelled"):
                raise ToolError("player cancelled the choice")
            card_ids = [str(card_id).lower() for card_id in payload.get("card_ids", [])]
            cards_by_id = {card.short_id(): card for card in options}
            return {
                "card_ids": card_ids,
                "cards": [
                    {"id": card_id, "name": cards_by_id[card_id].name}
                    for card_id in card_ids
                    if card_id in cards_by_id
                ],
            }

        return PendingInput(
            ChooseCardsRequest(
                prompt=prompt,
                reason=reason,
                cards=options,
                min_count=min_count,
                max_count=max_count,
            ),
            finalize,
        )

    def _h_ask_choose_targets(self, args: dict) -> dict | PendingInput:
        prompt, reason = self._choice_text(args)
        min_count, max_count = self._choice_counts(args)
        candidate_names = args.get("candidates")
        if candidate_names:
            candidates = [self._target(str(name)) for name in candidate_names]
        else:
            candidates = list(self.bundle.battlers())
        if not candidates:
            return {"targets": [], "note": "no targets available to choose"}
        max_count = min(max_count, len(candidates))
        min_count = min(min_count, max_count)

        def finalize(payload: dict) -> dict:
            if payload.get("cancelled"):
                raise ToolError("player cancelled the choice")
            return {"targets": [str(target) for target in payload.get("targets", [])]}

        return PendingInput(
            ChooseTargetsRequest(
                prompt=prompt,
                reason=reason,
                candidates=candidates,
                min_count=min_count,
                max_count=max_count,
            ),
            finalize,
        )

    def _commit(self, reason: str, significance: int):
        applied = []
        staged_count = len(self._staged)
        while self._staged:
            caster, command = self._staged.pop(0)
            if command.delay > 0:
                self._queued_effects.append(type(command).__name__)
            if result := self.bundle.apply_command(command, caster=caster):
                applied.append(result)
        self.bundle.clear_dead()
        flushed = self.bundle.flush_expired_effects(self.bundle.rng)
        applied.extend(flushed)
        from genio.battle import ResolvedEffects

        resolved = ResolvedEffects(applied)
        resolved.rarity = -1 if self._enemy_mode else significance
        self._last_resolved = resolved
        self._last_reason = reason
        logs = self.bundle._transform_to_battle_logs(resolved)
        return resolved, {
            "committed": staged_count,
            "applied": len(applied),
            "queued": list(self._queued_effects),
            "engine_log": logs,
        }

    def _h_finish(self, args: dict) -> dict:
        reason = str(self._req(args, "reason"))
        significance = self._int(args, "significance", 1, 3)
        _, output = self._commit(reason, significance)
        return output

    def _snapshot(self) -> str:
        lines = [
            f"Turn {self.bundle.turn_counter}.",
            (
                f"{self.bundle.player.name}: {self.bundle.player.hp}/"
                f"{self.bundle.player.max_hp} HP, "
                f"{self.bundle.player.shield_points} block, "
                f"{self.bundle.energy} energy"
            ),
            "Enemies:",
        ]
        for enemy in self.bundle.enemies:
            lines.append(
                f"- {enemy.name}: {enemy.hp}/{enemy.max_hp} HP, "
                f"{enemy.shield_points} block, intent: {enemy.current_intent}"
            )
        status_text = self.bundle.status_snapshot_text() or "none"
        lines.append("Active statuses:\n" + status_text)
        lines.append("Hand:")
        for card in self.bundle.card_bundle.hand:
            lines.append(f"- {card.short_id()}: {card.name} - {card.description or ''}")
        if self.bundle.card_bundle.resolving:
            lines.append("Resolving:")
            for card in self.bundle.card_bundle.resolving:
                lines.append(
                    f"- {card.short_id()}: {card.name} - {card.description or ''}"
                )
        rules = self.bundle.formatted_rules()
        if rules:
            lines.append("Rules:")
            lines.extend(f"- {rule}" for rule in rules)
        lines.append(
            "Amounts you send are BASE amounts. The engine applies all status "
            "modifiers; do not pre-multiply."
        )
        return "\n".join(lines)

    @property
    def pending_choice(self) -> PlayerChoiceRequest | None:
        if self.session.status != LoopStatus.AWAITING_INPUT:
            return None
        pending = self.session.pending_input
        if pending is None:
            return None
        request = pending.request
        if isinstance(request, ChooseCardsRequest | ChooseTargetsRequest):
            return request
        return None

    def submit_choice(self, payload: dict) -> None:
        pending = self.session.pending_input
        if pending is not None:
            pending.fulfill(payload)

    def abort_pending(self) -> None:
        pending = self.session.pending_input
        if pending is not None:
            pending.fulfill({"cancelled": True})

    def resolve(
        self,
        request: str,
        *,
        enemy_mode: bool,
        max_model_turns: int = 8,
        max_error_turns: int = 2,
    ) -> tuple[ResolvedEffects, str]:
        with self._resolve_lock:
            self.session.history = []
            self._staged.clear()
            self._consumed_card_ids.clear()
            self._consumed_card_names.clear()
            self._queued_effects.clear()
            self._enemy_mode = enemy_mode
            self._default_caster = None if enemy_mode else self.bundle.player
            self._last_resolved = None
            self._last_reason = ""
            message = self._snapshot() + "\n\n" + request
            self.session.run_turn(
                message,
                max_model_turns=max_model_turns,
                max_error_turns=max_error_turns,
            )
            if self._last_resolved is None:
                resolved, _ = self._commit(
                    "(resolution aborted; applying staged commands)",
                    1,
                )
                logger.warning("harness fallback commit", staged=len(self._staged))
                return resolved, self._last_reason
            return self._last_resolved, self._last_reason
