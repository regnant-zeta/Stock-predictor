import numpy as np
import pandas as pd
import pytest
from model import compute_rsi, ModelConfig, build_features

def test_compute_rsi_bounds():
    """Validates that computed RSI values strictly conform to realistic scaling limits."""
    prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    rsi = compute_rsi(prices, length=14)
    
    assert isinstance(rsi, pd.Series)
    assert rsi.max() <= 100.0
    assert rsi.min() >= 0.0
    assert not rsi.isna().any()

def test_build_features_shapes():
    """Ensures that the output contains tracking columns and rows align with dropna parameters."""
    dates = pd.date_range(start="2026-01-01", periods=20, freq="D")
    mock_data = pd.DataFrame({
        "Open": np.random.uniform(100, 110, size=20),
        "High": np.random.uniform(111, 120, size=20),
        "Low": np.random.uniform(90, 99, size=20),
        "Close": np.random.uniform(100, 110, size=20),
        "Volume": np.random.randint(1000, 5000, size=20)
    }, index=dates)
    
    config = ModelConfig(RSI_LENGTH=5, SMA_FAST=2, SMA_SLOW=5, EMA_SPAN=5)
    features = build_features(mock_data, config)
    
    assert "label" in features.columns
    assert "next_day_ret" in features.columns
    assert len(features) < 20  # Rows dropped due to lookbacks/shifting
