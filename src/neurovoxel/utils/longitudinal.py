"""Age-based covariates derived from the retained longitudinal visits."""

# pyright: reportMissingTypeStubs=false

from typing import Any, cast

import pandas as pd
from formulaic import Formula

from neurovoxel.utils.formula import ModelFormula

LONGITUDINAL_TERMS = frozenset({"baseline_age", "time", "time_from_baseline"})


def uses_longitudinal_terms(formula: ModelFormula) -> bool:
    """Identify computed terms in a mixed-effects formula."""
    if formula.random is None:
        return False
    required: set[str] = (
        cast("Any", Formula)(formula.fixed).required_variables
        | cast("Any", Formula)(formula.random).required_variables
        | {formula.group}
    )
    return bool(required & LONGITUDINAL_TERMS)


def _validate_sources(tbl: pd.DataFrame) -> None:
    missing = {"age", "subject"}.difference(tbl.columns)
    if missing:
        msg = (
            "Derived longitudinal terms require 'age' and 'subject' columns; "
            f"missing: {', '.join(sorted(missing))}"
        )
        raise ValueError(msg)
    if not pd.api.types.is_numeric_dtype(tbl["age"].dtype) or (
        pd.api.types.is_bool_dtype(tbl["age"].dtype)
    ):
        msg = "Derived longitudinal terms require a numeric 'age' column"
        raise ValueError(msg)


def longitudinal_dependencies(
    required: set[str], tbl: pd.DataFrame
) -> set[str]:
    """Replace computed names with the source columns needed to derive them."""
    if not required & LONGITUDINAL_TERMS:
        return set(required)
    _validate_sources(tbl)
    return (required - LONGITUDINAL_TERMS) | {"age", "subject"}


def add_longitudinal_terms(tbl: pd.DataFrame) -> pd.DataFrame:
    """Derive baseline and elapsed age for the supplied visits without mutation.

    Fitting callers must supply image-matched visits and recompute after any
    further exclusions. Query previews use these columns only to obtain term
    names; their provisional values are never used for fitting.
    """
    _validate_sources(tbl)
    result = tbl.copy()
    by_subject = result.groupby("subject", sort=False, observed=True)  # pyright: ignore[reportUnknownMemberType]
    result["baseline_age"] = by_subject["age"].transform("min")  # pyright: ignore[reportUnknownMemberType]
    elapsed = result["age"] - result["baseline_age"]
    result["time"] = elapsed
    result["time_from_baseline"] = elapsed
    return result
