import argparse
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure standard structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
def ModelConfig:
    """Centralized configuration for model hyperparameters and settings."""
    RSI_LENGTH: int = 14
    SMA_FAST: int = 5
    SMA_SLOW: int = 10
    EMA_SPAN: int = 10
    N_ESTIMATORS: int = 150
    MAX_DEPTH: int = 10
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    RISK_FREE_RATE: float = 0.02  # Annualized risk-free rate for Sharpe Ratio


def download_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Downloads historical data and flattens potential multi-index headers."""
    logger.info(f"Downloading data for ticker: {ticker} (Period: {period})")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
        
    # Flatten MultiIndex columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Computes RSI using exponential moving average smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(df: pd.DataFrame, config: ModelConfig = ModelConfig()) -> pd.DataFrame:
    """Engineers relative features and targets while avoiding structural leakage."""
    features = pd.DataFrame(index=df.index)
    
    # Target calculations
    features["return_1d"] = df["Close"].pct_change()
    features["return_5d"] = df["Close"].pct_change(5)
    
    # Normalized technical trends
    features["sma_5_ratio"] = df["Close"] / df["Close"].rolling(config.SMA_FAST).mean().replace(0, np.nan)
    features["sma_10_ratio"] = df["Close"] / df["Close"].rolling(config.SMA_SLOW).mean().replace(0, np.nan)
    features["ema_10_ratio"] = df["Close"] / df["Close"].ewm(span=config.EMA_SPAN, adjust=False).mean().replace(0, np.nan)
    
    # Volatility & Momentum
    features["vol_change_1d"] = df["Volume"].pct_change().fillna(0)
    features["rsi_14"] = compute_rsi(df["Close"], length=config.RSI_LENGTH)
    features["hl_range"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    
    # Forward look-ahead label (Target)
    features["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    # Store future baseline returns explicitly for vectorized backtesting later
    features["next_day_ret"] = df["Close"].pct_change().shift(-1)
    
    return features.dropna()


def train_model(X: pd.DataFrame, y: pd.Series, config: ModelConfig = ModelConfig()) -> Pipeline:
    """Instantiates and fits the scaling and classification pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS, 
            max_depth=config.MAX_DEPTH, 
            random_state=config.RANDOM_STATE, 
            n_jobs=-1
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def evaluate_financial_performance(y_test: pd.Series, y_pred: np.ndarray, next_day_returns: pd.Series, config: ModelConfig = ModelConfig()) -> None:
    """Calculates economic utility metrics using a vectorized strategy baseline."""
    # Map predictions from [0, 1] to [-1, 1] for shorting capability, or use prediction directly for long-only
    # Here we simulate long-only when predicted UP (1), cash when predicted DOWN (0)
    strategy_returns = y_pred * next_day_returns
    
    cum_strategy_ret = (1 + strategy_returns).prod() - 1
    cum_bh_ret = (1 + next_day_returns).prod() - 1
    
    # Annualized Sharpe Ratio calculation (assuming 252 trading days)
    daily_rf = config.RISK_FREE_RATE / 252
    excess_returns = strategy_returns - daily_rf
    std_dev = excess_returns.std()
    
    sharpe_ratio = (excess_returns.mean() / std_dev) * np.sqrt(252) if std_dev != 0 else 0.0
    
    print("\n📈 Financial Strategy Metrics (Test Set)")
    print(f"Strategy Cumulative Return: {cum_strategy_ret * 100:.2f}%")
    print(f"Buy & Hold Cumulative Return: {cum_bh_ret * 100:.2f}%")
    print(f"Strategy Sharpe Ratio:       {sharpe_ratio:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock market up/down prediction prototype")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to download")
    parser.add_argument("--period", default="2y", help="Historical period to download (e.g. 1y, 2y, 5y)")
    args = parser.parse_args()

    config = ModelConfig()

    try:
        raw_data = download_data(args.ticker, period=args.period)
        features = build_features(raw_data, config)
    except Exception as e:
        logger.error(f"Execution failed during data preparation: {e}")
        return

    feature_cols = ["return_1d", "return_5d", "sma_5_ratio", "sma_10_ratio", "ema_10_ratio", "vol_change_1d", "rsi_14", "hl_range"]
    
    last_row = features[feature_cols].tail(1)
    
    X = features[feature_cols].iloc[:-1]
    y = features["label"].iloc[:-1]
    next_day_rets = features["next_day_ret"].iloc[:-1]

    if len(X) < 20:
        logger.error("Insufficient sample size remaining after technical indicators processing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, shuffle=False)
    _, _, _, test_rets = train_test_split(X, next_day_rets, test_size=config.TEST_SIZE, shuffle=False)
    
    model = train_model(X_train, y_train, config)
    y_pred = model.predict(X_test)

    print("\n🤖 Machine Learning Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    evaluate_financial_performance(y_test, y_pred, test_rets, config)

    if not last_row.empty:
        prediction = model.predict(last_row)[0]
        direction = "UP" if prediction == 1 else "DOWN"
        print(f"\n🔮 Next-day operational prediction for {args.ticker}: {direction}")


if __name__ == "__main__":
    main()
