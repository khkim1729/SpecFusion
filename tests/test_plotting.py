import numpy as np

from specfusion.plotting import plot_indian_pines_overview


def test_plot_indian_pines_overview_writes_nonempty_png(tmp_path):
    rng = np.random.default_rng(2)
    cube = rng.uniform(0, 1, size=(5, 6, 12))
    labels = np.tile(np.arange(6), 5).reshape(5, 6)
    output = tmp_path / "overview.png"

    result = plot_indian_pines_overview(
        cube, labels, np.linspace(400, 2500, 12), output
    )

    assert result == output
    assert output.stat().st_size > 1000
