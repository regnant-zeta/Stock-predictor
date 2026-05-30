import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Enforce structured corporate logging output
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Immutable production environment configurations and internal engine hyperparameters."""
    RSI_LENGTH: int = 14
    SMA_FAST: int = 5
    SMA_SLOW: int = 10
    EMA_SPAN: int = 10
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 8
    RANDOM_STATE: int = 42
    WALK_FORWARD_SPLITS: int = 5
    RISK_FREE_RATE: float = 0.04  # Modernized benchmark risk-free metric

    @property
    def feature_columns(self) -> List[str]:
        return [
            "return_1d", "return_5d", "sma_5_ratio", 
            "sma_10_ratio", "ema_10_ratio", "vol_change_1d", 
            "rsi_14", "hl_range"
        ]


def download_market_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Downloads raw market ticks and extracts base parameters safely.
    Flattens modern structural MultiIndex formats natively.
    """
    logger.info(f"Initiating market stream download for security: {ticker} | Window: {period}")
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        raise RuntimeError(f"Critical failure communication with primary ingestion API: {e}")
        
    if df.empty:
        raise ValueError(f"Ingested dataset contains null dimensions for security token: {ticker}")
        
    # Safely flatten multi-level matrix structures if thrown by yfinance updates
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df


def calculate_rsi_metrics(series: pd.Series, length: int) -> pd.Series:
    """Calculates smoothed relative index distributions with boundary protection parameters."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def generate_isolated_features(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """
    Builds data vectors. Preserves structural records matching execution horizons
    by isolating target structures away from raw technical states.
    """
    logger.info("Parsing analytical telemetry matrix and building engine features.")
    working_df = pd.DataFrame(index=df.index)
    
    # Mathematical momentum vectors
    working_df["return_1d"] = df["Close"].pct_change()
    working_df["return_5d"] = df["Close"].pct_change(5)
    
    # Normalized technical trend ratios
    working_df["sma_5_ratio"] = df["Close"] / df["Close"].rolling(config.SMA_FAST).mean().replace(0, np.nan)
    working_df["sma_10_ratio"] = df["Close"] / df["Close"].rolling(config.SMA_SLOW).mean().replace(0, np.nan)
    working_df["ema_10_ratio"] = df["Close"] / df["Close"].ewm(span=config.EMA_SPAN, adjust=False).mean().replace(0, np.nan)
    
    # Volume dynamics & price variance bounds
    working_df["vol_change_1d"] = df["Volume"].pct_change().fillna(0)
    working_df["rsi_14"] = calculate_rsi_metrics(df["Close"], config.RSI_LENGTH)
    working_df["hl_range"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    
    # Define directional downstream target labels shifted backwards chronologically
    working_df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    working_df["next_day_ret"] = df["Close"].pct_change().shift(-1)
    
    return working_df


def split_production_data(
    features_df: pd.DataFrame, 
    config: ModelConfig
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Extracts high-fidelity inference rows before pruning training NaN layers.
    Guarantees no target variables cross over.
    """
    feature_cols = config.feature_columns
    
    # Extract the absolute latest index row for production execution
    live_inference_row = features_df[feature_cols].tail(1)
    
    # Drop rows that don't have historical lookback data or forward labels
    historical_matrix = features_df.dropna(subset=feature_cols + ["label", "next_day_ret"])
    
    X = historical_matrix[feature_cols]
    y = historical_matrix["label"]
    returns = historical_matrix["next_day_ret"]
    
    return X, y, returns, live_inference_row


def execute_walk_forward_validation(X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> None:
    """Executes robust time-series validation blocks across multiple historical splits."""
    logger.info(f"Initializing walk-forward evaluation protocol. Splits: {config.WALK_FORWARD_SPLITS}")
    tscv = TimeSeriesSplit(n_splits=config.WALK_FORWARD_SPLITS)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                max_depth=config.MAX_DEPTH,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_scores.append(acc)
        logger.info(f"Validation Block {fold + 1}/{config.WALK_FORWARD_SPLITS} Out-Of-Sample Accuracy: {acc:.4f}")


def evaluate_backtest_utility(y_test: pd.Series, y_pred: np.ndarray, next_day_returns: pd.Series, config: ModelConfig) -> None:
    """Calculates financial returns and risk-adjusted metrics for predictions."""
    # Simulation baseline: Long-only on directional asset matching upward calls, cash on negative signals
    strategy_returns = y_pred * next_day_returns
    
    cum_strategy_ret = (1 + strategy_returns).prod() - 1
    cum_bh_ret = (1 + next_day_returns).prod() - 1
    
    daily_rf = config.RISK_FREE_RATE / 252
    excess_returns = strategy_returns - daily_rf
    std_dev = excess_returns.std()
    
    sharpe_ratio = (excess_returns.mean() / std_dev) * np.sqrt(252) if std_dev > 1e-6 else 0.0
    
    print("\n📊 Out-Of-Sample Strategy Utility Report")
    print("=" * 45)
    print(f"Strategy Cumulative Growth : {cum_strategy_ret * 100:.2f}%")
    print(f"Passive Buy & Hold Growth  : {cum_bh_ret * 100:.2f}%")
    print(f"Annualized Sharpe Metric   : {sharpe_ratio:.4f}")
    print("=" * 45)


def main() -> None:
    parser = argparse.ArgumentParser(description="Production Grade Financial Predictor Pipeline")
    parser.add_argument("--ticker", required=True, type=str, help="Target market equity token ticker")
    parser.add_argument("--period", default="3y", type=str, help="Historical training data depth window")
    args = parser.parse_args()

    config = ModelConfig()

    try:
        raw_data = download_market_data(args.ticker, period=args.period)
        processed_features = generate_isolated_features(raw_data, config)
    except Exception as e:
        logger.error(f"Execution engine tracking termination due to pipeline data errors: {e}")
        return

    # Split arrays via strict target isolation bounds
    X, y, next_day_returns, live_inference_row = split_production_data(processed_features, config)

    if len(X) < 100:
        logger.error("Insufficient historical records remain after cleaning technical metrics.")
        return

    # Run structural walk-forward tests to ensure stability across regimes
    execute_walk_forward_validation(X, y, config)

    # Chronological holdout split for core system backtesting (Last 20% of temporal data)
    split_idx = int(len(X) * (1 - 0.2))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_returns = next_day_returns.iloc[split_idx:]

    # Build and optimize production pipeline
    production_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    logger.info("Fitting production model architecture on historical training set.")
    production_pipeline.fit(X_train, y_train)
    
    # Evaluate classification metrics
    y_pred = production_pipeline.predict(X_test)
    print("\n🔮 Evaluation Metrics (Holdout Evaluation Segment)")
    print(classification_report(y_test, y_pred, digits=4))

    # Evaluate algorithmic utility performance metrics
    evaluate_backtest_utility(y_test, y_pred, test_returns, config)

    # Live operational prediction for the next market session
    if not live_inference_row.empty:
        # Full end-to-end transformation inside production pipeline safely
        live_prediction = production_pipeline.predict(live_inference_row)[0]
        direction = "UP" if live_prediction == 1 else "DOWN"
        print(f"\n🚀 Production Live Prediction for {args.ticker} (Next Trading Session): {direction}\n")


if __name__ == "__main__":
    main()
