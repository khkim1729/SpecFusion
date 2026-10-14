"""Command-line interface for SpecFusion."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import labeled_spectra, load_indian_pines
from .descriptors import compute_descriptors
from .evaluate import evaluate_representations
from .harmonization import harmonize
from .plotting import plot_indian_pines_overview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    cube, labels = load_indian_pines(config["data_path"], config["labels_path"])
    source_nm = np.linspace(float(config["wavelength_min_nm"]), float(config["wavelength_max_nm"]), cube.shape[-1])
    target_nm = np.linspace(float(config["wavelength_min_nm"]), float(config["wavelength_max_nm"]), int(config.get("target_bands", 220)))
    spectra, target = labeled_spectra(cube, labels)
    raw, valid = harmonize(spectra, source_nm, target_nm)
    physics, _ = compute_descriptors(raw, target_nm, mask=valid)
    results = evaluate_representations(
        raw,
        physics,
        target,
        seed=int(config.get("seed", 42)),
        n_estimators=int(config.get("n_estimators", 100)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "representation_comparison.csv", index=False)
    plot_indian_pines_overview(
        cube, labels, source_nm, args.output_dir / "indian_pines_overview.png"
    )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
