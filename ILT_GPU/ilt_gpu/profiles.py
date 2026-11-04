from dataclasses import dataclass, replace

from .config import OpticsConfig, OptimizationConfig


@dataclass(frozen=True)
class RunProfile:
    name: str
    optics: OpticsConfig
    optimization: OptimizationConfig
    verification: OpticsConfig


def make_run_profile(name: str, iterations: int = 1000) -> RunProfile:
    """Return physically scaled demo or native ICCAD run settings."""
    if name == "demo":
        optics = OpticsConfig()
        optimization = replace(OptimizationConfig(), iterations=iterations)
        verification = replace(
            optics, source_rings=6, source_points_per_ring=24,
            source_batch_size=8,
        )
    elif name == "iccad-native":
        optics = replace(
            OpticsConfig(), grid_size=1024, pixel_size_nm=1.0,
            source_batch_size=1,
        )
        optimization = replace(
            OptimizationConfig(), iterations=iterations,
            log_every=10,
            mask_filter_sigma_px=5.0,       # 5 nm, same as demo profile
            border_margin_px=40,            # 40 nm, same as demo profile
            boundary_radius_px=8,           # 8 nm, same as demo profile
            tv_weight=2.0e-3,               # compensate for 4x finer sampling
        )
        # Independent denser radial sampling while keeping 8 GB verification safe.
        verification = replace(
            optics, source_rings=6, source_points_per_ring=12,
            source_batch_size=1,
        )
    else:
        raise ValueError(f"Unknown run profile: {name}")
    return RunProfile(name, optics, optimization, verification)
