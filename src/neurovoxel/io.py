"""Data I/O."""

from pathlib import Path

import numpy as np
from bids.layout import BIDSLayout  # pyright: ignore[reportMissingTypeStubs]
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import (  # pyright: ignore[reportMissingTypeStubs]
    MultiNiftiMasker,  # pyright: ignore[reportMissingTypeStubs]
)


def load_bids(bids_root: Path, config_fname: Path | None = None) -> BIDSLayout:
    """Load BIDS dataset."""
    layout = BIDSLayout(
        bids_root,
        validate=False,
        derivatives=False,
        config=["bids", "derivatives", config_fname] if config_fname else None,
    )

    layout.add_derivatives(  # pyright: ignore[reportUnknownMemberType]
        bids_root / "derivatives",
        config=["bids", "derivatives", config_fname] if config_fname else None,
    )
    return layout


def load_images(
    image_paths: str | Nifti1Image | list[str | Nifti1Image],
    mask: str | Nifti1Image,
    smoothing_fwhm: float,
    vox_size: float,
    n_jobs: int = -1,
) -> np.ndarray:
    """Load spatially-aligned images for voxelwise analysis.

    Only voxels within the binary mask will be loaded. Specified smoothing will
    be applied and images will be resampled to the specified voxel size.

    Args:
        image_paths: list of spatially-aligned image file names
        mask: binary mask specifying voxels to use in statistical analysis
        smoothing_fwhm: full width at half max for the Gaussian smoothing
        vox_size: voxel size for the statistical analysis space
        n_jobs: number of parallel jobs (set to -1 to maximize)
    """
    masker = MultiNiftiMasker(
        mask,
        smoothing_fwhm=smoothing_fwhm,
        target_affine=np.eye(3) * vox_size,
        n_jobs=n_jobs,
    )
    return np.vstack(masker.fit_transform(image_paths))  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]
