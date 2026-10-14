"""Publication-oriented Indian Pines visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-cleantill", "Wheat",
    "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]


def _nearest(wavelengths: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(wavelengths - value)))


def _stretch(image: np.ndarray) -> np.ndarray:
    stretched = np.empty_like(image, dtype=float)
    for channel in range(image.shape[-1]):
        values = image[..., channel]
        low, high = np.percentile(values, [2, 98])
        stretched[..., channel] = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return stretched


def plot_indian_pines_overview(
    cube: np.ndarray,
    labels: np.ndarray,
    wavelengths_nm: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Render false color, ground truth, and class-mean spectra in one figure."""
    values = np.asarray(cube, dtype=float)
    target = np.asarray(labels)
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if values.ndim != 3 or target.shape != values.shape[:2]:
        raise ValueError("cube and labels must have matching spatial dimensions")
    if wavelengths.shape != (values.shape[-1],):
        raise ValueError("wavelengths must match the cube's spectral axis")

    rgb_indices = [_nearest(wavelengths, nm) for nm in (800.0, 650.0, 550.0)]
    false_color = _stretch(values[..., rgb_indices])
    tab20 = plt.get_cmap("tab20").colors
    colors = [(0.03, 0.03, 0.03, 1.0)] + [(*tab20[i], 1.0) for i in range(16)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 17.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].imshow(false_color)
    axes[0].set_title("(a) False-color composite")
    axes[0].axis("off")
    image = axes[1].imshow(target, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("(b) Ground-truth labels")
    axes[1].axis("off")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02, ticks=[0, 4, 8, 12, 16])
    colorbar.set_label("Class ID")

    flat = values.reshape(-1, values.shape[-1])
    flat_labels = target.reshape(-1)
    scale = 10000.0 if np.nanmax(np.abs(flat)) > 2 else 1.0
    for class_id, name in enumerate(CLASS_NAMES, start=1):
        class_values = flat[flat_labels == class_id]
        if class_values.size:
            axes[2].plot(
                wavelengths,
                class_values.mean(axis=0) / scale,
                color=colors[class_id],
                linewidth=1.0,
                alpha=0.9,
                label=f"{class_id}: {name}",
            )
    axes[2].set(
        title="(c) Class-mean spectra",
        xlabel="Wavelength (nm)",
        ylabel="Scaled reflectance",
        xlim=(wavelengths.min(), wavelengths.max()),
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=5.3, ncol=2, frameon=False, loc="upper right")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output
