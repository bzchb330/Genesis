import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _analysis_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_correlation.py"
    spec = importlib.util.spec_from_file_location("phase2_analysis_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_wilson_interval_and_natural_occupied_categories():
    module = _analysis_module()
    low, high = module._wilson(5, 10, 0.05)
    assert 0 < low < 0.5 < high < 1
    frame = pd.DataFrame({
        "occupied_finger_count": [2, 2, 3, 3, 4, 4],
        "both_retained": [0, 1, 1, 1, 0, 0],
    })
    rows = module._binned(frame, "occupied_finger_count", 5, 0.05)
    assert [row["category"] for row in rows] == ["2", "3", "4"]
    assert all(0 <= row["wilson_low"] <= row["rate"] <= row["wilson_high"] <= 1 for row in rows)


def test_raw_standardized_and_grasp_clustered_logit_reporting():
    module = _analysis_module()
    rng = np.random.default_rng(4)
    rows = []
    for grasp in range(20):
        occupied = 2 + grasp % 3
        workspace = 1e-4 + rng.uniform(-2e-5, 2e-5)
        palm = 3.2e-3 + rng.uniform(-2e-4, 2e-4)
        for repeat in range(5):
            probability = 1 / (1 + np.exp(-(-1.0 + 0.35 * occupied + repeat * 0.08)))
            rows.append({
                "grasp_id": f"g{grasp}", "occupied_finger_count": occupied,
                "free_finger_workspace_vol_m3": workspace,
                "free_palm_volume_m3": palm,
                "both_retained": int(rng.random() < probability),
            })
    result = module._safe_regressions(pd.DataFrame(rows))
    assert set(result) == {
        "raw_physical_units", "standardized_predictors",
        "raw_clustered_robust", "standardized_clustered_robust",
    }
    assert result["raw_physical_units"]["N_valid_trials"] == 100
    assert result["raw_clustered_robust"]["clustered_by_grasp"] is True
