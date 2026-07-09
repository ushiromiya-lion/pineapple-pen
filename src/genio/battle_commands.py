from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from genio.card import Card
from genio.effect import (
    CreateCardEffect,
    DestroyCardEffect,
    DestroyRuleEffect,
    DiscardCardsEffect,
    DrawCardsEffect,
    DuplicateCardEffect,
    Effect,
    GlobalEffect,
    SinglePointEffect,
    StatusDefinition,
    TransformCardEffect,
)


@dataclass(frozen=True)
class CommandModifiers:
    delay: int = 0
    critical_chance: float = 0.0
    pierce: bool = False
    drain: bool = False
    accuracy: float = 1.0


@dataclass(frozen=True)
class DealDamage(CommandModifiers):
    target: str = ""
    amount: float = 0
    source: str | None = None


@dataclass(frozen=True)
class Heal(CommandModifiers):
    target: str = ""
    amount: float = 0


@dataclass(frozen=True)
class GainBlock(CommandModifiers):
    target: str = ""
    amount: float = 0


@dataclass(frozen=True)
class LoseBlock(CommandModifiers):
    target: str = ""
    amount: float = 0


@dataclass(frozen=True)
class ApplyStatus(CommandModifiers):
    target: str = ""
    status: StatusDefinition | None = None
    duration: int = 0


@dataclass(frozen=True)
class RemoveStatus(CommandModifiers):
    target: str = ""
    status_name: str = ""


@dataclass(frozen=True)
class DrawCards(CommandModifiers):
    count: int = 1


@dataclass(frozen=True)
class DiscardCards(CommandModifiers):
    count: int = 0
    card_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CreateCard(CommandModifiers):
    name: str = ""
    description: str | None = None
    energy_cost: int = 1
    copies: int = 1
    where: Literal["deck_top", "deck", "hand", "graveyard"] = "hand"


@dataclass(frozen=True)
class DuplicateCard(CommandModifiers):
    card_id: str = ""
    copies: int = 1
    where: Literal["deck_top", "deck", "hand", "graveyard"] = "hand"


@dataclass(frozen=True)
class TransformCard(CommandModifiers):
    card_id: str = ""
    name: str = ""
    description: str | None = None
    energy_cost: int | None = None


@dataclass(frozen=True)
class DestroyCard(CommandModifiers):
    card_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DestroyRule(CommandModifiers):
    rule_id: int = 0


BattleCommand: TypeAlias = (
    DealDamage
    | Heal
    | GainBlock
    | LoseBlock
    | ApplyStatus
    | RemoveStatus
    | DrawCards
    | DiscardCards
    | CreateCard
    | DuplicateCard
    | TransformCard
    | DestroyCard
    | DestroyRule
)


class CardContext:
    def seek_card(self, card_expr: str) -> Card:
        raise NotImplementedError


def _modifiers_from_effect(effect: SinglePointEffect | GlobalEffect) -> dict:
    return {
        "delay": effect.delay,
        "critical_chance": effect.critical_chance,
        "pierce": effect.pierce,
        "drain": effect.drain,
        "accuracy": effect.accuracy,
    }


def _modifiers_from_command(command: CommandModifiers) -> dict:
    return {
        "delay": command.delay,
        "critical_chance": command.critical_chance,
        "pierce": command.pierce,
        "drain": command.drain,
        "accuracy": command.accuracy,
    }


def command_from_effect(effect: Effect) -> BattleCommand | None:
    commands = commands_from_effect(effect)
    return commands[0] if commands else None


def commands_from_effect(effect: Effect) -> list[BattleCommand]:
    match effect:
        case (target, single_effect):
            return commands_from_targeted_effect(target, single_effect)
        case global_effect if isinstance(global_effect, GlobalEffect):
            return [command_from_global_effect(global_effect)]
    raise ValueError(f"Unsupported effect: {effect}")


def command_from_targeted_effect(
    target: str, effect: SinglePointEffect
) -> BattleCommand | None:
    commands = commands_from_targeted_effect(target, effect)
    return commands[0] if commands else None


def commands_from_targeted_effect(
    target: str, effect: SinglePointEffect
) -> list[BattleCommand]:
    modifiers = _modifiers_from_effect(effect)
    if effect.noop:
        return []
    commands: list[BattleCommand] = []
    if effect.delta_shield > 0:
        commands.append(GainBlock(target=target, amount=effect.delta_shield, **modifiers))
    if effect.delta_shield < 0:
        commands.append(LoseBlock(target=target, amount=-effect.delta_shield, **modifiers))
    if effect.delta_hp < 0:
        commands.append(
            DealDamage(
                target=target,
                amount=-effect.delta_hp,
                source=effect.source,
                **modifiers,
            )
        )
    if effect.delta_hp > 0:
        commands.append(Heal(target=target, amount=effect.delta_hp, **modifiers))
    if effect.remove_status:
        commands.append(
            RemoveStatus(target=target, status_name=effect.remove_status, **modifiers)
        )
    if effect.add_status:
        status, duration = effect.add_status
        commands.append(
            ApplyStatus(
                target=target,
                status=status,
                duration=duration,
                **modifiers,
            )
        )
    return commands


def command_from_global_effect(effect: GlobalEffect) -> BattleCommand:
    modifiers = _modifiers_from_effect(effect)
    match effect:
        case DrawCardsEffect(_):
            return DrawCards(count=effect.count, **modifiers)
        case DiscardCardsEffect(_) as discard:
            return DiscardCards(
                count=discard.count,
                card_ids=tuple(card.short_id() for card in discard.specifics),
                **modifiers,
            )
        case CreateCardEffect(_) as create:
            return CreateCard(
                name=create.card.name,
                description=create.card.description,
                energy_cost=create.card.energy_cost,
                copies=create.copies,
                where=create.where,
                **modifiers,
            )
        case DuplicateCardEffect(_) as duplicate:
            return DuplicateCard(
                card_id=duplicate.card.short_id(),
                copies=duplicate.copies,
                where=duplicate.where,
                **modifiers,
            )
        case TransformCardEffect(_) as transform:
            return TransformCard(
                card_id=transform.from_card.short_id(),
                name=transform.to_card.name,
                description=transform.to_card.description,
                energy_cost=transform.to_card.energy_cost,
                **modifiers,
            )
        case DestroyCardEffect(_) as destroy:
            return DestroyCard(
                card_ids=tuple(card.short_id() for card in destroy.cards),
                **modifiers,
            )
        case DestroyRuleEffect(_) as destroy_rule:
            return DestroyRule(rule_id=destroy_rule.rule_id, **modifiers)
    raise ValueError(f"Unsupported global effect: {effect}")


def effect_from_command(command: BattleCommand, context: CardContext) -> Effect:
    modifiers = _modifiers_from_command(command)
    match command:
        case DealDamage(target=target, amount=amount):
            return target, SinglePointEffect(
                delta_hp=-amount, source=command.source, **modifiers
            )
        case Heal(target=target, amount=amount):
            return target, SinglePointEffect(delta_hp=amount, **modifiers)
        case GainBlock(target=target, amount=amount):
            return target, SinglePointEffect(delta_shield=amount, **modifiers)
        case LoseBlock(target=target, amount=amount):
            return target, SinglePointEffect(delta_shield=-amount, **modifiers)
        case ApplyStatus(target=target, status=status, duration=duration):
            if status is None:
                raise ValueError("ApplyStatus requires a status definition")
            return target, SinglePointEffect(
                add_status=(status, duration), **modifiers
            )
        case RemoveStatus(target=target, status_name=status_name):
            return target, SinglePointEffect(remove_status=status_name, **modifiers)
        case DrawCards(count=count):
            return DrawCardsEffect(count=count, **modifiers)
        case DiscardCards(count=count, card_ids=card_ids):
            return DiscardCardsEffect(
                count=count,
                specifics=[context.seek_card(card_id) for card_id in card_ids],
                **modifiers,
            )
        case CreateCard(
            name=name,
            description=description,
            energy_cost=energy_cost,
            copies=copies,
            where=where,
        ):
            return CreateCardEffect(
                card=Card(
                    name=name,
                    description=description,
                    energy_cost=max(0, int(energy_cost)),
                ),
                copies=copies,
                where=where,
                **modifiers,
            )
        case DuplicateCard(card_id=card_id, copies=copies, where=where):
            return DuplicateCardEffect(
                card=context.seek_card(card_id),
                copies=copies,
                where=where,
                **modifiers,
            )
        case TransformCard(
            card_id=card_id,
            name=name,
            description=description,
            energy_cost=energy_cost,
        ):
            from_card = context.seek_card(card_id)
            return TransformCardEffect(
                from_card=from_card,
                to_card=Card(
                    name=name,
                    description=description,
                    energy_cost=(
                        from_card.energy_cost
                        if energy_cost is None
                        else max(0, int(energy_cost))
                    ),
                ),
                **modifiers,
            )
        case DestroyCard(card_ids=card_ids):
            return DestroyCardEffect(
                cards=[context.seek_card(card_id) for card_id in card_ids],
                **modifiers,
            )
        case DestroyRule(rule_id=rule_id):
            return DestroyRuleEffect(rule_id=rule_id, **modifiers)
    raise ValueError(f"Unsupported command: {command}")
