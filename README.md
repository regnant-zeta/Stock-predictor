# Stock-predictor
This is an AI created code constructed with prompts, is aimed to predict trajectories of stocks based on past patterns and parameters which affect path.
This is a simple Python prototype that downloads historical stock data, creates trajectory-based features, and trains a classifier to predict whether the next day closes higher or lower.

## Setup

1. Open PowerShell in `C:\Users\user\stock-predictor`
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Run

```powershell
python stock_predictor.py --ticker AAPL --period 2y
```

## Notes

- This is a prototype, not financial advice.
- It uses historical price and volume data only.
- You can extend features with sentiment, macro data, or additional technical indicators.

