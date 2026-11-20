"""Variance-partition R² and asymptotic omnibus tests for Gaussian LME."""

# pyright: reportMissingTypeStubs=false

import numpy as np
from scipy.stats import chi2


def mixed_r_squared(
    fixed_prediction: np.typing.NDArray[np.float64],
    random_design: np.typing.NDArray[np.float64],
    random_covariance: np.typing.NDArray[np.float64],
    residual_variance: float,
) -> tuple[float, float]:
    """Return Nakagawa marginal and conditional R², including random slopes.

    Fixed variance is the sample variance of X beta. Random variance is
    mean(diag(Z G Z.T)), so intercept/slope covariance and unequal visit
    counts are included without constructing an observations-squared matrix.
    G must be the fitted covariance on the outcome scale (fit.cov_re).
    """
    if (
        not np.isfinite(fixed_prediction).all()
        or not np.isfinite(random_covariance).all()
        or not np.isfinite(residual_variance)
        or residual_variance < 0
    ):
        return np.nan, np.nan
    fixed_variance = float(np.var(fixed_prediction, ddof=1))
    visit_variances: np.typing.NDArray[np.float64] = np.sum(
        (random_design @ random_covariance) * random_design, axis=1
    )
    random_variance = float(np.mean(visit_variances))
    total = fixed_variance + random_variance + residual_variance
    if not np.isfinite(total) or total <= 0 or random_variance < 0:
        return np.nan, np.nan
    return fixed_variance / total, (fixed_variance + random_variance) / total


def mixed_wald_f(
    beta: np.typing.NDArray[np.float64],
    covariance: np.typing.NDArray[np.float64] | None,
    non_intercept: np.typing.NDArray[np.bool_],
) -> tuple[float, float]:
    """Test all non-intercept fixed effects using F = Wald chi² / q.

    This is an asymptotic F(q, infinity) test, equivalent to chi²(q), with
    uncorrected p-values. No finite-sample denominator df are estimated.
    Intercept-only models and invalid contrast covariances return NaN.
    """
    if covariance is None or not non_intercept.any():
        return np.nan, np.nan
    effects = beta[non_intercept]
    contrast_covariance = covariance[np.ix_(non_intercept, non_intercept)]
    if not np.isfinite(contrast_covariance).all():
        return np.nan, np.nan
    try:
        # Positive definiteness is required for usable joint Wald inference.
        standardized = np.linalg.solve(
            np.linalg.cholesky(contrast_covariance), effects
        )
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    wald = float(standardized @ standardized)
    if not np.isfinite(wald):
        return np.nan, np.nan
    q = int(non_intercept.sum())
    return wald / q, float(chi2.sf(wald, q))  # pyright: ignore[reportUnknownMemberType]
