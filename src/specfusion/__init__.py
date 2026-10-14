"""SpecFusion physics-guided hyperspectral representation toolkit."""

from .descriptors import compute_descriptors
from .fusion import fuse
from .harmonization import harmonize

__all__ = ["compute_descriptors", "fuse", "harmonize"]
__version__ = "0.1.0"
