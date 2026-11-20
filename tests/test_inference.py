"""Model-independent FDR correction."""

import numpy as np
import pytest

from neurovoxel.utils.inference import fdr_log_p_values


def test_fdr_adjusts_each_term_separately() -> None:
    """Check known BH values, ties, p=0, and p=1."""
    p = np.array([[0.01, 0.04, 0.03, 1], [0, 0.5, 0.5, 1]])
    expected = [[0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 1], [0, 2 / 3, 2 / 3, 1]]
    logp = fdr_log_p_values(p)
    assert np.isfinite(logp).all()
    np.testing.assert_allclose(10.0**-logp, expected, atol=1e-15)


@pytest.mark.parametrize("p", [[], [0.1], [[np.nan]], [[-0.1]], [[1.1]]])
def test_fdr_rejects_invalid_p_values(p: list[object]) -> None:
    """Require finite uncorrected p-values in a two-dimensional array."""
    with pytest.raises(ValueError, match="FDR|p-values|ps"):
        fdr_log_p_values(np.asarray(p))
