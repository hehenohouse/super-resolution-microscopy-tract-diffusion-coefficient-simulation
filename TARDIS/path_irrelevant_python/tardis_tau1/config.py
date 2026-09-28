from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Tau1Config:
    """Configuration locked to the first distribution-only milestone."""

    tau_frames: int = 1
    max_distance_m: float = 3e-6
    n_bins: int = 300
    min_distance_background_fit: float = 0.7
    chunk_size: int = 256
    bootstrap_count: int = 2_000
    bootstrap_seed: int = 1_729
    save_plots: bool = True
    schema_version: str = "tau1-cell-v3"
    zero_distance_policy: str = "legacy_remove_all"

    def __post_init__(self) -> None:
        if self.tau_frames != 1:
            raise ValueError("This pipeline currently supports tau_frames=1 only")
        if self.max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if not 0 <= self.min_distance_background_fit < 1:
            raise ValueError("min_distance_background_fit must be in [0, 1)")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.bootstrap_count < 0:
            raise ValueError("bootstrap_count cannot be negative")
        if self.zero_distance_policy != "legacy_remove_all":
            raise ValueError("Only legacy_remove_all is supported in version 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
