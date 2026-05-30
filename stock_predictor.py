import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def download_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return df


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["return_1d"] = features["Close"].pct_change()
    features["return_5d"] = features["Close"].pct_change(5)
    features["sma_5"] = features["Close"].rolling(5).mean()
    features["sma_10"] = features["Close"].rolling(10).mean()
    features["ema_10"] = features["Close"].ewm(span=10, adjust=False).mean()
    features["vol_change_1d"] = features["Volume"].pct_change()
    features["rsi_14"] = compute_rsi(features["Close"], length=14)
    features["hl_range"] = (features["High"] - features["Low"]) / features["Open"].replace(0, np.nan)
    features["label"] = (features["Close"].shift(-1) > features["Close"]).astype(int)
    features = features.dropna()
    return features


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X, y)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock market up/down prediction prototype")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to download")
    parser.add_argument("--period", default="2y", help="Historical period to download (e.g. 1y, 2y, 5y)")
    args = parser.parse_args()

    data = download_data(args.ticker, period=args.period)
    features = build_features(data)

    X = features[["Open", "High", "Low", "Close", "Volume", "return_1d", "return_5d", "sma_5", "sma_10", "ema_10", "vol_change_1d", "rsi_14", "hl_range"]]
    y = features["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    last_row = X.tail(1)
    if not last_row.empty:
        prediction = model.predict(last_row)[0]
        direction = "UP" if prediction == 1 else "DOWN"
        print(f"\nNext-day prediction for {args.ticker}: {direction}")


if __name__ == "__main__":
    main()
