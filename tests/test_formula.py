"""Formula handling for fixed and longitudinal random effects."""

# pyright: reportMissingTypeStubs=false

from typing import cast

import pandas as pd
import pytest
from formulaic import model_matrix  # pyright: ignore[reportUnknownVariableType]

from neurovoxel.utils.formula import ModelFormula, split_formula


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("y ~ age", ModelFormula("y", "age")),
        ("y ~ age + (1 + age | id)", ModelFormula("y", "age", "1 + age", "id")),
        ("y ~ (1|subject)", ModelFormula("y", "1", "1", "subject")),
        ("y ~ (1|id) + age", ModelFormula("y", "age", "1", "id")),
        ("y ~ age + (1|id) + sex", ModelFormula("y", "age + sex", "1", "id")),
        (
            "y ~ 0 + age + (0 + age|id)",
            ModelFormula("y", "0 + age", "0 + age", "id"),
        ),
        (
            "y ~ age + (1 + np.log(age)|id)",
            ModelFormula("y", "age", "1 + np.log(age)", "id"),
        ),
        (
            "y ~ C(sex, Treatment('a|b'))",
            ModelFormula("y", "C(sex, Treatment('a|b'))"),
        ),
        ("y ~ `a|b`", ModelFormula("y", "`a|b`")),
    ],
)
def test_split_formula(query: str, expected: ModelFormula) -> None:
    """Keep fixed-effect encoding and Python transforms intact."""
    assert split_formula(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "y age",
        "y ~ age ~ sex",
        "y ~ age | id",
        "y ~ age + (1||id)",
        "y ~ age + (1|id) + (0+age|id)",
        "y ~ age + (1|site/id)",
        "y ~ age * (1|id)",
        "y ~ (1|id) * age",
        "y ~ ((1|id))",
        "y ~ age + (1|id",
        "y ~ age + (|id)",
        "y ~ age + (1|id))",
    ],
)
def test_unsupported_formula(query: str) -> None:
    """Reject unsupported structures instead of silently changing the model."""
    with pytest.raises(ValueError, match="formula|random-effects"):
        split_formula(query)


@pytest.mark.parametrize(
    ("query", "fixed_intercept", "random_intercept"),
    [
        ("gm ~ age + (age | subject)", True, True),
        ("gm ~ 0 + age + (age | subject)", False, True),
        ("gm ~ age + (0 + age | subject)", True, False),
        ("gm ~ 0 + age + (0 + age | subject)", False, False),
        ("gm ~ age - 1 + (age - 1 | subject)", False, False),
    ],
)
def test_fixed_and_random_intercepts(
    query: str, fixed_intercept: bool, random_intercept: bool
) -> None:
    """Include intercepts by default and honor their independent removal."""
    data = pd.DataFrame(
        {"age": [50.0, 51.0, 60.0, 61.0], "subject": ["a", "a", "b", "b"]}
    )
    formula = split_formula(query)
    assert formula.random is not None
    for rhs, include_intercept in (
        (formula.fixed, fixed_intercept),
        (formula.random, random_intercept),
    ):
        actual = cast("pd.DataFrame", model_matrix(rhs, data))
        explicit_rhs = "1 + age" if include_intercept else "0 + age"
        explicit = cast("pd.DataFrame", model_matrix(explicit_rhs, data))
        pd.testing.assert_frame_equal(actual, explicit)
        assert list(actual.columns) == (
            ["Intercept", "age"] if include_intercept else ["age"]
        )
        if include_intercept:
            assert (actual["Intercept"] == 1).all()
