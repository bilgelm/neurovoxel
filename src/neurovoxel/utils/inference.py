"""Multiple comparison correction shared by regression models."""

# pyright: reportMissingTypeStubs=false

from typing import Any, cast

import numpy as np
from scipy.stats import (
    false_discovery_control,  # pyright: ignore[reportUnknownVariableType]
)


def fdr_log_p_values(
    p_values: np.typing.NDArray[Any],
) -> np.typing.NDArray[np.float64]:
    """Return -log10 BH-adjusted p-values across voxels for each term.

    Input is a (terms, voxels) array of uncorrected p-values. Model-specific
    distributions and degrees of freedom belong to the model fitting code.
    """
    if p_values.ndim != 2 or not p_values.size:  # noqa: PLR2004
        msg = "FDR requires a nonempty (terms, voxels) p-value array"
        raise ValueError(msg)
    adjusted_p = cast(
        "np.typing.NDArray[np.float64]",
        false_discovery_control(p_values, axis=1, method="bh"),
    )
    return -np.log10(np.maximum(adjusted_p, np.finfo(float).tiny))
