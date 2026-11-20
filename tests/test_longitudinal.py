"""Derived longitudinal covariates and formula previews."""

import pandas as pd
import pytest

from neurovoxel.utils.formula import split_formula
from neurovoxel.utils.load_parse import parse_query
from neurovoxel.utils.longitudinal import (
    add_longitudinal_terms,
    longitudinal_dependencies,
    uses_longitudinal_terms,
)


def test_baseline_is_minimum_age_within_subject() -> None:
    """Use age rather than row/session order and overwrite stale columns."""
    tbl = pd.DataFrame(
        {
            "subject": ["a", "b", "a", "b", "a"],
            "age": [65.0, 80.0, 60.0, 78.0, 60.0],
            "baseline_age": -1.0,
            "time": -1.0,
            "time_from_baseline": -1.0,
        }
    )
    original = tbl.copy(deep=True)
    result = add_longitudinal_terms(tbl)
    assert list(result["baseline_age"]) == [60.0, 78.0, 60.0, 78.0, 60.0]
    assert list(result["time"]) == [5.0, 2.0, 0.0, 0.0, 0.0]
    pd.testing.assert_series_equal(
        result["time"],
        result["time_from_baseline"],
        check_names=False,
    )
    pd.testing.assert_frame_equal(tbl, original)


@pytest.mark.parametrize("term", ["baseline_age", "time", "time_from_baseline"])
@pytest.mark.parametrize("missing", ["age", "subject"])
def test_derived_terms_require_source_columns(term: str, missing: str) -> None:
    """Explain missing prerequisites even if a stale derived column exists."""
    tbl = pd.DataFrame({"age": [60.0, 61.0], "subject": ["a", "a"], term: -1.0})
    tbl = tbl.drop(columns=[missing])
    with pytest.raises(ValueError, match=f"missing: {missing}"):
        parse_query(f"gm ~ {term} + (1 | subject)", ["gm"], tbl)


@pytest.mark.parametrize("age", [["60", "61"], [True, False]])
def test_derived_terms_require_numeric_age(age: list[object]) -> None:
    """Avoid string sorting or Boolean subtraction in age-based terms."""
    tbl = pd.DataFrame({"subject": ["a", "a"], "age": age})
    with pytest.raises(ValueError, match="numeric 'age'"):
        add_longitudinal_terms(tbl)


@pytest.mark.parametrize("term", ["time", "time_from_baseline"])
def test_query_preview_accepts_derived_terms_without_mutating_table(
    term: str,
) -> None:
    """Offer term names while leaving final baseline selection to fitting."""
    tbl = pd.DataFrame(
        {"subject": ["a", "a", "b", "b"], "age": [60.0, 62.0, 70.0, 73.0]}
    )
    original = tbl.copy(deep=True)
    lhs, columns = parse_query(
        f"gm ~ baseline_age + {term} + ({term} | subject)",
        ["gm"],
        tbl,
    )
    assert lhs == "gm"
    assert list(columns) == ["Intercept", "baseline_age", term]
    pd.testing.assert_frame_equal(tbl, original)


def test_random_only_derived_term_and_source_dependencies() -> None:
    """Compute aliases used only by random effects and keep source columns."""
    formula = split_formula("gm ~ age + (time | subject)")
    assert uses_longitudinal_terms(formula)
    tbl = pd.DataFrame({"subject": ["a", "a"], "age": [60.0, 61.0]})
    assert longitudinal_dependencies({"time", "sex"}, tbl) == {
        "age",
        "subject",
        "sex",
    }
    _, columns = parse_query("gm ~ age + (time | subject)", ["gm"], tbl)
    assert list(columns) == ["Intercept", "age"]


def test_ordinary_linear_models_keep_user_time_column() -> None:
    """Derived terms are specific to mixed models, not ordinary regressions."""
    tbl = pd.DataFrame({"time": [1.0, 2.0]})
    assert not uses_longitudinal_terms(split_formula("gm ~ time"))
    _, columns = parse_query("gm ~ time", ["gm"], tbl)
    assert list(columns) == ["Intercept", "time"]
