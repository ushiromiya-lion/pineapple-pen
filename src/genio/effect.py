from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

from parse import search
from structlog import get_logger

from genio.card import Card

logger = get_logger()


@dataclass(eq=True, frozen=True)
class BaseEffect:
    delay: int = 0
    critical_chance: float = 0.0
    pierce: bool = False
    drain: bool = False
    accuracy: float = 1.0
    noop: bool = False
    _uuid: str = field(default_factory=lambda: uuid4().hex)

    def equals_except_uuid(self, other: BaseEffect) -> bool:
        return self.__dict__ | {"_uuid": None} == other.__dict__ | {"_uuid": None}


@dataclass(eq=True, frozen=True)
class OnDamageTaken:
    pass


@dataclass(eq=True, frozen=True)
class OnDamageDealt:
    pass


@dataclass(eq=True, frozen=True)
class OnTurnStart:
    pass


@dataclass(eq=True, frozen=True)
class OnTurnEnd:
    pass


StatusTrigger: TypeAlias = OnDamageTaken | OnDamageDealt | OnTurnStart | OnTurnEnd


@dataclass(eq=True, frozen=True)
class ModifyAmount:
    expr: str
    condition: str | None = None
    consumes_counter: bool = False


@dataclass(eq=True, frozen=True)
class DealDamageToSelf:
    amount: float


@dataclass(eq=True, frozen=True)
class GainBlockReaction:
    amount: float


@dataclass(eq=True, frozen=True)
class Advisory:
    description: str


StatusReaction: TypeAlias = (
    ModifyAmount | DealDamageToSelf | GainBlockReaction | Advisory
)


@dataclass
class StatusDefinition:
    name: str
    trigger: StatusTrigger | None
    reaction: StatusReaction
    counter_type: Literal["turns", "times"]

    description: str = ""


class SinglePointEffectType(Enum):
    DAMAGE = 1
    HEAL = 2
    SHIELD_GAIN = 3
    SHIELD_LOSS = 4
    STATUS = 5
    OTHER = 6


@dataclass(eq=True, frozen=True)
class SinglePointEffect(BaseEffect):
    delta_shield: int = 0
    delta_hp: int = 0
    add_status: tuple[StatusDefinition, int] | None = None
    remove_status: str | None = None
    source: str | None = None

    @staticmethod
    def from_damage(damage: int, pierce: bool = False) -> SinglePointEffect:
        return SinglePointEffect(delta_hp=-damage, pierce=pierce)

    @staticmethod
    def from_heal(heal: int) -> SinglePointEffect:
        return SinglePointEffect(delta_hp=heal)

    @staticmethod
    def noop_effect() -> SinglePointEffect:
        return SinglePointEffect(noop=True)

    @property
    def damage(self) -> int:
        return max(-self.delta_hp, 0)

    @property
    def heal(self) -> int:
        return max(self.delta_hp, 0)

    @property
    def shield_gain(self) -> int:
        return max(self.delta_shield, 0)

    @property
    def shield_loss(self) -> int:
        return max(-self.delta_shield, 0)

    def classify_type(self) -> SinglePointEffectType:
        if self.delta_hp < 0:
            return SinglePointEffectType.DAMAGE
        if self.delta_hp > 0:
            return SinglePointEffectType.HEAL
        if self.delta_shield > 0:
            return SinglePointEffectType.SHIELD_GAIN
        if self.delta_shield < 0:
            return SinglePointEffectType.SHIELD_LOSS
        if self.add_status or self.remove_status:
            return SinglePointEffectType.STATUS
        return SinglePointEffectType.OTHER


@dataclass(eq=True, frozen=True)
class GlobalEffect(BaseEffect):
    pass


@dataclass(eq=True, frozen=True)
class DrawCardsEffect(GlobalEffect):
    count: int = 1


@dataclass(eq=True, frozen=True)
class DiscardCardsEffect(GlobalEffect):
    count: int = 0
    specifics: list[Card] = field(default_factory=list)


@dataclass(eq=True, frozen=True)
class CreateCardEffect(GlobalEffect):
    card: Card = field(default_factory=Card)
    where: Literal["deck_top", "deck", "hand", "graveyard"] = "hand"
    copies: int = 1


@dataclass(eq=True, frozen=True)
class DuplicateCardEffect(GlobalEffect):
    card: Card = field(default_factory=Card)
    copies: int = 1
    where: Literal["deck_top", "deck", "hand", "graveyard"] = "hand"


@dataclass(eq=True, frozen=True)
class TransformCardEffect(GlobalEffect):
    from_card: Card | None = None
    to_card: Card | None = None


@dataclass(eq=True, frozen=True)
class DestroyCardEffect(GlobalEffect):
    cards: list[Card] = field(default_factory=list)


@dataclass(eq=True, frozen=True)
class DestroyRuleEffect(GlobalEffect):
    rule_id: int = 0


TargetedEffect: TypeAlias = tuple[str, SinglePointEffect]
Effect: TypeAlias = GlobalEffect | TargetedEffect


class ParseEffectError(ValueError):
    ...


class UnsafeStatusExpression(ValueError):
    ...


_ALLOWED_EXPR_NAMES = {"amount", "counter"}
_ALLOWED_EXPR_CALLS = {"min": min, "max": max}


def _validate_status_expr_node(node: ast.AST, allow_compare: bool = False) -> None:
    match node:
        case ast.Expression(body=body):
            _validate_status_expr_node(body, allow_compare)
        case ast.Constant(value=value) if isinstance(value, int | float | bool):
            return
        case ast.Name(id=name) if name in _ALLOWED_EXPR_NAMES:
            return
        case ast.BinOp(left=left, op=op, right=right) if isinstance(
            op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.FloorDiv
        ):
            _validate_status_expr_node(left, allow_compare)
            _validate_status_expr_node(right, allow_compare)
        case ast.UnaryOp(op=op, operand=operand) if isinstance(op, ast.UAdd | ast.USub):
            _validate_status_expr_node(operand, allow_compare)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _ALLOWED_EXPR_CALLS:
            for arg in args:
                _validate_status_expr_node(arg, allow_compare)
        case ast.Compare(left=left, ops=ops, comparators=comparators) if allow_compare:
            if not all(
                isinstance(op, ast.Eq | ast.NotEq | ast.Lt | ast.LtE | ast.Gt | ast.GtE)
                for op in ops
            ):
                raise UnsafeStatusExpression(
                    f"Unsafe status comparison: {ast.dump(node)}"
                )
            _validate_status_expr_node(left, allow_compare)
            for comparator in comparators:
                _validate_status_expr_node(comparator, allow_compare)
        case ast.BoolOp(values=values) if allow_compare:
            for value in values:
                _validate_status_expr_node(value, allow_compare)
        case _:
            raise UnsafeStatusExpression(f"Unsafe status expression: {ast.dump(node)}")


def evaluate_status_expr(expr: str, *, amount: float, counter: int) -> float:
    tree = ast.parse(expr, mode="eval")
    _validate_status_expr_node(tree)
    return eval(
        compile(tree, "<status-expression>", "eval"),
        {"__builtins__": {}, **_ALLOWED_EXPR_CALLS},
        {"amount": amount, "counter": counter},
    )


def evaluate_status_condition(
    condition: str | None, *, amount: float, counter: int
) -> bool:
    if condition is None:
        return True
    tree = ast.parse(condition, mode="eval")
    _validate_status_expr_node(tree, allow_compare=True)
    return bool(
        eval(
            compile(tree, "<status-condition>", "eval"),
            {"__builtins__": {}, **_ALLOWED_EXPR_CALLS},
            {"amount": amount, "counter": counter},
        )
    )


def _legacy_expr_to_status_expr(expr: str, amount_match_index: int) -> str:
    expr = expr.strip()
    if expr.startswith("{{") and expr.endswith("}}"):
        expr = expr[2:-2].strip()
    expr = expr.replace(f"m[{amount_match_index}]", "amount")
    expr = expr.replace("m[0]", "amount" if amount_match_index == 0 else "0")
    expr = expr.replace("m[1]", "amount" if amount_match_index == 1 else "0")
    return expr


def _parse_legacy_damage_expr(replacement: str) -> str | None:
    match = re.search(r"damaged\s+(.+?)(?:\s+by\s+.*)?\]$", replacement)
    if not match:
        return None
    return match.group(1).strip()


def _parse_legacy_flat_status(
    name: str, counter_type: Literal["turns", "times"], rule: str
) -> StatusDefinition:
    return StatusDefinition(
        name=name,
        trigger=None,
        reaction=Advisory(rule),
        counter_type=counter_type,
        description=rule,
    )


def status_definition_from_legacy(
    name: str,
    counter_type: Literal["turns", "times"],
    rule: str,
) -> StatusDefinition:
    match = re.match(
        r"(?P<trigger>\[.*?\])\s*(?:if\s+(?P<condition>.*?)\s*)?->\s*(?P<replacement>\[.*\])$",
        rule.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        logger.warning("Unmapped legacy status rule", name=name, rule=rule)
        return _parse_legacy_flat_status(name, counter_type, rule)

    trigger_text = match.group("trigger").strip()
    condition = match.group("condition")
    replacement = match.group("replacement").strip()
    trigger_lower = trigger_text.lower()
    replacement_lower = replacement.lower()

    if "end of turn" in trigger_lower:
        if damage_expr := _parse_legacy_damage_expr(replacement_lower):
            try:
                amount = evaluate_status_expr(damage_expr, amount=0, counter=0)
            except UnsafeStatusExpression:
                logger.warning("Unmapped legacy end-of-turn status", name=name, rule=rule)
                return _parse_legacy_flat_status(name, counter_type, rule)
            return StatusDefinition(
                name=name,
                trigger=OnTurnEnd(),
                reaction=DealDamageToSelf(amount),
                counter_type=counter_type,
            )
        logger.warning("Unmapped legacy end-of-turn status", name=name, rule=rule)
        return _parse_legacy_flat_status(name, counter_type, rule)

    if "damaged {:d}" in trigger_lower:
        is_dealt = " by me" in trigger_lower
        amount_match_index = 1 if is_dealt else 0
        damage_expr = _parse_legacy_damage_expr(replacement_lower)
        if damage_expr is None:
            logger.warning("Unmapped legacy damage status", name=name, rule=rule)
            return _parse_legacy_flat_status(name, counter_type, rule)
        expr = _legacy_expr_to_status_expr(damage_expr, amount_match_index)
        condition = (
            _legacy_expr_to_status_expr(condition, amount_match_index)
            if condition
            else None
        )
        return StatusDefinition(
            name=name,
            trigger=OnDamageDealt() if is_dealt else OnDamageTaken(),
            reaction=ModifyAmount(
                expr=expr,
                condition=condition,
                consumes_counter=counter_type == "times",
            ),
            counter_type=counter_type,
        )

    logger.warning("Unmapped legacy status rule", name=name, rule=rule)
    return _parse_legacy_flat_status(name, counter_type, rule)


def extract_tokens(pattern: str, haystack: str) -> tuple[str, ...]:
    match = search(pattern, haystack)
    if not match:
        raise ParseEffectError(f"Invalid format: {haystack}")
    return match.fixed


def parse_global_effect(modifier: str, context: CardContext) -> GlobalEffect:
    match = re.match(r"\[(.*)\]", modifier)
    if not match:
        raise ValueError("Invalid format")
    effect = match.group(1).strip()

    tokens = effect.split("|")
    common_modifiers = parse_common_modifiers(tokens)

    if "draw" in effect:
        count = int(tokens[0].split(" ")[1])
        return DrawCardsEffect(count=count, **common_modifiers)
    elif "discard" in effect:
        to_discard = tokens[0].split(" ")[1:]
        parsed_tokens = []
        has_int = False
        for c in to_discard:
            if len(c) < 3:
                if c.startswith("#"):
                    parsed_tokens.append(c)
                else:
                    parsed_tokens.append(int(c))
                    has_int = True
            else:
                parsed_tokens.append(c)
        if has_int:
            count = parsed_tokens[0]
            return DiscardCardsEffect(count=count, **common_modifiers)
        return DiscardCardsEffect(
            specifics=[context.seek_card(expr) for expr in parsed_tokens],
            **common_modifiers,
        )
    elif "duplicate" in effect:
        card_specifier, where = search("[duplicate {}in {:w}", modifier).fixed
        if "*" in card_specifier:
            card_specifier, postfix = card_specifier.split()
        else:
            postfix = ""
        mult = 1
        if mult_expr := postfix.replace(" ", ""):
            mult = search("*{:d}", mult_expr).fixed[0]
        card = context.seek_card(card_specifier.strip())
        return DuplicateCardEffect(
            card=card, where=where, copies=mult, **common_modifiers
        )
    elif "transform" in effect[:20]:
        pat = "[transform {} to <{}>"
        from_card, to_card = extract_tokens(pat, modifier)
        to_card_desc = f"<{to_card}>"
        from_card = context.seek_card(from_card)
        to_card = Card.parse(to_card_desc)
        return TransformCardEffect(from_card=from_card, to_card=to_card)
    elif "create" in effect[:20]:
        card_desc, postfix, where = extract_tokens("[create <{}>{}in {:w}", modifier)
        card_desc = f"<{card_desc}>"
        mult = 1
        if mult_expr := postfix.replace(" ", ""):
            mult = search("*{:d}", mult_expr).fixed[0]
        card = Card.parse(card_desc)
        return CreateCardEffect(card=card, where=where, copies=mult, **common_modifiers)
    elif "destroy-rule" in effect[:20]:
        # "[destroy-rule R??] - Destroy a rule. E.g., `[destroy-rule R01]` (destroy rule `R01`).",
        rxx = extract_tokens("[destroy-rule R{:d}", modifier)[0]
        return DestroyRuleEffect(rule_id=int(rxx))
    elif "destroy" in effect[:20]:
        to_destroy = tokens[0].split(" ")[1:]
        return DestroyCardEffect(
            cards=[context.seek_card(expr) for expr in to_destroy], **common_modifiers
        )
    else:
        raise ValueError(f"Invalid format: {effect}")


def parse_targeted_effect(
    modifier: str, context: CardContext | None = None
) -> TargetedEffect:
    status_effect_pat = "[{}: +{} [{:d} {:w}] {};]"
    if match := search(status_effect_pat, modifier):
        entity, name, counter, counter_type, effects = match.fixed
        tokens = effects.split("|")
        common_modifiers = parse_common_modifiers(tokens[1:])
        status_def = status_definition_from_legacy(name, counter_type, tokens[0])
        return entity, SinglePointEffect(
            add_status=(status_def, counter), **common_modifiers
        )
    match = re.match(r"\[(.*): (.*)\]", modifier)
    if not match:
        raise ValueError("Invalid format")

    entity = match.group(1).strip()
    effects = match.group(2).split("|")

    if "end of turn" in effects:
        return entity, SinglePointEffect.noop_effect()

    delta_shield = 0
    delta_hp = 0
    source = None

    for effect in effects:
        effect = effect.strip()
        if "shield" in effect:
            delta_shield = float(effect.split(" ")[1])
        elif "damaged" in effect or "healed" in effect:
            effect_tokens = effect.split(" ")
            delta_hp = float(effect_tokens[1])
            if "damaged" in effect:
                delta_hp *= -1
            if " by " in effect:
                source = effect.split(" by ", 1)[1].strip()

    common_modifiers = parse_common_modifiers(effects)
    return entity, SinglePointEffect(
        delta_shield=delta_shield,
        delta_hp=delta_hp,
        source=source,
        **common_modifiers,
    )


def parse_common_modifiers(tokens: list[str]) -> dict:
    modifiers = {
        "delay": 0,
        "critical_chance": 0.0,
        "pierce": False,
        "drain": False,
        "accuracy": 1.0,
    }

    for token in tokens:
        token = token.strip()
        if "crit" in token:
            modifiers["critical_chance"] = float(token.split(" ")[1])
        elif "acc" in token:
            modifiers["accuracy"] = float(token.split(" ")[1])
        elif "delay" in token:
            modifiers["delay"] = int(token.split(" ")[1])
        elif "pierce" in token:
            modifiers["pierce"] = True
        elif "drain" in token:
            modifiers["drain"] = True

    return modifiers


def parse_effect(bracket_expr: str, context: CardContext) -> Effect:
    if re.match(r"^\[[\w\s,]*:", bracket_expr):
        return parse_targeted_effect(bracket_expr, context)
    return parse_global_effect(bracket_expr, context)


class CardContext(Protocol):
    def seek_card(self, card_expr: str) -> Card:
        ...
