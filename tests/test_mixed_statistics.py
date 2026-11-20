"""Analytic checks of mixed-model variance partitions and joint Wald tests."""

# pyright: reportMissingTypeStubs=false

import numpy as np
import pytest
from scipy.stats import chi2, norm

from neurovoxel.utils.mixed_statistics import mixed_r_squared, mixed_wald_f


def test_r_squared_includes_random_slope_covariance() -> None:
    """Average visit-specific random variances, including the covariance."""
    prediction = np.array([1.0, 3.0, 5.0])
    z = np.column_stack((np.ones(3), [-1.0, 0.0, 2.0]))
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    # Sample fixed variance = 4; random variances = [2, 2, 8]; residual = 2.
    actual = mixed_r_squared(prediction, z, covariance, 2.0)
    np.testing.assert_allclose(actual, [0.4, 0.8])
    # Express time in decades with the equivalent random-effects covariance.
    transform = np.diag([1.0, 10.0])
    rescaled = mixed_r_squared(
        prediction,
        z @ np.linalg.inv(transform),
        transform @ covariance @ transform,
        2.0,
    )
    np.testing.assert_allclose(rescaled, actual)


@pytest.mark.parametrize("invalid", ["residual", "covariance", "total"])
def test_r_squared_unavailable_variances(invalid: str) -> None:
    """Do not manufacture R² when variance estimates are unavailable."""
    prediction = np.ones(4)
    z = np.ones((4, 1))
    covariance = np.array([[np.nan if invalid == "covariance" else 0.0]])
    residual = -1.0 if invalid == "residual" else 0.0
    assert all(
        np.isnan(value)
        for value in mixed_r_squared(prediction, z, covariance, residual)
    )


def test_joint_wald_uses_all_slopes_and_their_covariance() -> None:
    """Joint inference excludes the intercept and accounts for covariance."""
    beta = np.array([100.0, 2.0, -1.0])
    covariance = np.array([[9.0, 0, 0], [0, 1.0, 0.3], [0, 0.3, 2.0]])
    statistic, p = mixed_wald_f(beta, covariance, np.array([False, True, True]))
    wald = float(beta[1:] @ np.linalg.solve(covariance[1:, 1:], beta[1:]))
    np.testing.assert_allclose(statistic, wald / 2)
    np.testing.assert_allclose(p, chi2.sf(wald, 2))  # pyright: ignore[reportUnknownMemberType]


def test_one_effect_wald_agrees_with_two_sided_z_test() -> None:
    """For one fixed effect, F equals z² and the p-values agree."""
    statistic, p = mixed_wald_f(
        np.array([3.0]), np.array([[4.0]]), np.array([True])
    )
    np.testing.assert_allclose(statistic, 1.5**2)
    np.testing.assert_allclose(p, 2 * norm.sf(1.5))  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("covariance", "selected"),
    [
        (None, [True, True]),
        (np.eye(2), [False, False]),
        (np.zeros((2, 2)), [True, True]),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), [True, True]),
        (np.full((2, 2), np.nan), [True, True]),
    ],
)
def test_unavailable_wald_covariance_returns_nan(
    covariance: np.typing.NDArray[np.float64] | None,
    selected: list[bool],
) -> None:
    """Intercept-only, singular and indefinite tests remain unavailable."""
    assert all(
        np.isnan(value)
        for value in mixed_wald_f(np.ones(2), covariance, np.asarray(selected))
    )
