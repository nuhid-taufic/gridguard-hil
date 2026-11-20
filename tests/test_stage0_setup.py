from pathlib import Path
import yaml
import pandas as pd
import lightgbm as lgb


def test_config_loads():
    cfg_path = Path("configs/grid_code_ieee.yaml")
    assert cfg_path.exists(), "Missing configs/grid_code_ieee.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["grid_code"]["nominal_frequency_hz"] == 50.0
    assert "stab" not in cfg["dataset_uci"]["raw_input_features"]
    assert "p1" not in cfg["dataset_uci"]["raw_input_features"]


def test_uci_dataset_integrity():
    csv_path = Path("data/uci_raw/Data_for_UCI_named.csv")
    assert csv_path.exists(), "Download Data_for_UCI_named.csv into data/uci_raw/"
    df = pd.read_csv(csv_path)
    assert df.shape == (10000, 14), f"Expected (10000, 14), got {df.shape} (Avoid 60k leaky CSV!)"
    expected_cols = {
        "tau1", "tau2", "tau3", "tau4",
        "p1", "p2", "p3", "p4",
        "g1", "g2", "g3", "g4",
        "stab", "stabf",
    }
    assert set(df.columns) == expected_cols


def test_lightgbm_runtime():
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    assert model is not None
