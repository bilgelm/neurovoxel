"""Data I/O."""

from pathlib import Path

from bids.layout import BIDSLayout  # pyright: ignore[reportMissingTypeStubs]


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
