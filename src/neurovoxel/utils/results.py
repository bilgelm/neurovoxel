"""Shared statistical result types."""

from typing import Any, NotRequired, TypedDict

import numpy as np

from neurovoxel.utils.mixed_diagnostics import MixedDiagnostics


class AnalysisResult(TypedDict):
    """Voxelwise fixed-effect results with FWE or FDR corrected p-values."""

    mixed_diagnostics: NotRequired[MixedDiagnostics]
    tested_var_names: list[str]
    beta: np.typing.NDArray[Any]
    t: NotRequired[np.typing.NDArray[Any]]
    z: NotRequired[np.typing.NDArray[Any]]
    converged: NotRequired[np.typing.NDArray[Any]]
    logp_max_t: NotRequired[np.typing.NDArray[Any]]
    h0_max_t: NotRequired[np.typing.NDArray[Any]]
    logp_fdr: NotRequired[np.typing.NDArray[Any]]
    tfce: NotRequired[np.typing.NDArray[Any]]
    logp_max_tfce: NotRequired[np.typing.NDArray[Any]]
    h0_max_tfce: NotRequired[np.typing.NDArray[Any]]
    size: NotRequired[np.typing.NDArray[Any]]
    logp_max_size: NotRequired[np.typing.NDArray[Any]]
    h0_max_size: NotRequired[np.typing.NDArray[Any]]
    mass: NotRequired[np.typing.NDArray[Any]]
    logp_max_mass: NotRequired[np.typing.NDArray[Any]]
    h0_max_mass: NotRequired[np.typing.NDArray[Any]]
    r_square: np.typing.NDArray[Any]
    r_square_conditional: NotRequired[np.typing.NDArray[Any]]
    f_stat: np.typing.NDArray[Any]
    f_pvalues: np.typing.NDArray[Any]


# Compatibility with the original result type name.
PermutedOLSResult = AnalysisResult
