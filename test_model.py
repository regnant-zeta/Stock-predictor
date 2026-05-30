
  import numpy as np
import pandas as pd
import pytest
from production_predictor import calculate_rsi_metrics, ModelConfig, generate_isolated_features, split_production_data

@pytest.fixture
def mock_market_data():
    """Generates a structured mock dataframe to simulate real stock history."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=50, freq="D")
    return pd.DataFrame({
        "Open": np.random.uniform(100, 110, size=50),
        "High": np.random.uniform(111, 120, size=50),
        "Low": np.random.uniform(90, 99, size=50),
        "Close": np.random.uniform(100, 110, size=50),
        "Volume": np.random.randint(1000, 5000, size=50)
    }, index=dates)


def test_rsi_evaluation_boundaries():
    """Validates that computed RSI values conform to technical boundaries [0, 100]."""
    prices = pd.Series([10 + i * 0.5 for i in range(20)])
    rsi = calculate_rsi_metrics(prices, length=14)
    
    assert rsi.max() <= 100.0
    assert rsi.min() >= 0.0
    assert not rsi.isna().any()


def test_production_feature_isolation_and_shapes(mock_market_data):
    """Verifies that data processing keeps live prediction features cleanly separated from labels."""
    config = ModelConfig(RSI_LENGTH=5, SMA_FAST=2, SMA_SLOW=5, EMA_SPAN=5)
    processed = generate_isolated_features(mock_market_data, config)
    
    X, y, returns, live_row = split_production_data(processed, config)
    
    # Ensure live inference row is present and matches feature column counts
    assert live_row.shape[0] == 1
    assert live_row.shape[1] == len(config.feature_columns)
    
    # Assert historical metrics are dropna trimmed for fitting while live_row remains separate
    assert len(X) == len(y) == len(returns)
    assert not X.isna().any().any()
