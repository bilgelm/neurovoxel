"""Split fixed and random effects before constructing Formulaic matrices."""

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass

from formulaic.parser.algos.tokenize import (
    tokenize,  # pyright: ignore[reportUnknownVariableType]
)
from formulaic.parser.types import Token


@dataclass(frozen=True)
class ModelFormula:
    """A voxelwise formula with an optional correlated random-effects term."""

    lhs: str
    fixed: str
    random: str | None = None
    group: str | None = None


def split_formula(query: str) -> ModelFormula:  # noqa: C901, PLR0912
    """Parse ``y ~ fixed + (random | group)`` for one grouping column.

    Formulaic tokenization keeps quoted names and Python transforms intact.
    Multiple, nested, crossed, and uncorrelated random terms are rejected.
    """
    if query.count("~") != 1:
        msg = "Invalid formula syntax: formula should contain one '~'"
        raise ValueError(msg)
    lhs, rhs = (part.strip() for part in query.split("~"))
    tokens = list(tokenize(rhs))
    bars = [
        t for t in tokens if t.kind == Token.Kind.OPERATOR and "|" in t.token
    ]
    if not bars:
        return ModelFormula(lhs, rhs)
    msg = (
        "Use one additive random-effects term '(1 + age | id)' with a single "
        "grouping column; multiple, nested and '||' terms are not supported"
    )
    if len(bars) != 1 or bars[0] != "|":
        raise ValueError(msg)
    stack: list[Token] = []
    opening = closing = None
    for item in tokens:
        if item == "(":
            stack.append(item)
        elif item == ")":
            if not stack:
                raise ValueError(msg)
            candidate = stack.pop()
            if opening is candidate:
                closing = item
        elif item is bars[0]:
            if len(stack) != 1:
                raise ValueError(msg)
            opening = stack[0]
    if stack or opening is None or closing is None:
        raise ValueError(msg)
    assert opening.source_start is not None  # noqa: S101
    assert closing.source_end is not None  # noqa: S101
    assert bars[0].source_start is not None  # noqa: S101
    assert bars[0].source_end is not None  # noqa: S101
    before = rhs[: opening.source_start].strip()
    after = rhs[closing.source_end + 1 :].strip()
    if (before and not before.endswith("+")) or (
        after and not after.startswith("+")
    ):
        raise ValueError(msg)
    fixed = (before.removesuffix("+") + after).strip().removeprefix("+")
    random = rhs[opening.source_start + 1 : bars[0].source_start].strip()
    group = rhs[bars[0].source_end + 1 : closing.source_end].strip()
    if not random or not group.isidentifier():
        raise ValueError(msg)
    return ModelFormula(lhs, fixed.strip() or "1", random, group)
