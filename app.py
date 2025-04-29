import base64
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib
matplotlib.use('Agg')

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Path setup
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "crypto_lstm_final.keras"   # New model path
SCALER_PATH = BASE_DIR / "crypto_scaler.pkl"        # New scaler path
WEIGHTS_PATH = BASE_DIR / "crypto_lstm.weights.h5"  # Optional

# File existence checks
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CRYPTO dictionary
CRYPTOS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'SOL-USD': 'Solana',
    'XRP-USD': 'Ripple',
    'ADA-USD': 'Cardano',
    'DOGE-USD': 'Dogecoin',
    'DOT-USD': 'Polkadot',
    'SHIB-USD': 'Shiba Inu',
    'AVAX-USD': 'Avalanche'
}


def plot_to_html():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.getbuffer()).decode('ascii')
    buf.close()
    return f"data:image/png;base64,{img}"


def prepare_input_data(crypto, days=100):
    end = datetime.now()
    start = end - timedelta(days=days*2)  # Get extra data

    try:
        # Download data and ensure proper format
        data = yf.download(crypto, start=start, end=end)
        if data.empty:
            return None

        # Convert to plain DataFrame if MultiIndex exists
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Create a clean copy with only needed columns
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Calculate basic indicators manually (fallback if TA fails)
        df['SMA_7'] = df['Close'].rolling(7).mean()
        df['SMA_21'] = df['Close'].rolling(21).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df = df.dropna()

        # Try to add TA features if available
        try:
            from ta import add_all_ta_features
            df = add_all_ta_features(
                df,
                open="Open", high="High", low="Low",
                close="Close", volume="Volume",
                fillna=True
            )
            # Select specific features if TA worked
            features = df[['Close', 'volume_adi', 'volatility_bbm',
                          'trend_macd', 'momentum_rsi', 'trend_ema_fast']]
        except:
            # Fallback to basic features if TA fails
            features = df[['Close', 'SMA_7', 'SMA_21', 'Price_Change']]

        # Ensure we have exactly 100 days
        features = features.tail(100)

        # Scale and reshape for LSTM
        scaled = scaler.transform(features)
        return scaled.reshape(1, 100, -1)  # Shape: (1, 100, n_features)

    except Exception as e:
        print(f"Error preparing data for {crypto}: {str(e)}")
        return None


def predict_future(crypto, days=10):
    seq = prepare_input_data(crypto)
    if seq is None:
        return None

    predictions = []
    current_seq = seq.copy()

    for _ in range(days):
        pred = model.predict(current_seq, verbose=0)
        predictions.append(pred[0, 0])

        # Append prediction as next day's close
        new_row = np.zeros((1, 1, 6))
        new_row[0, 0, 0] = pred
        current_seq = np.append(current_seq[:, 1:, :], new_row, axis=1)

    # Inverse transform predictions
    dummy = np.zeros((len(predictions), 6))
    dummy[:, 0] = predictions
    return scaler.inverse_transform(dummy)[:, 0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['crypto'] = request.form.get('crypto')
        return redirect(url_for('predict_days'))
    return render_template('index.html', cryptos=CRYPTOS)


@app.route('/predict', methods=['GET', 'POST'])
def predict_days():
    if request.method == 'POST':
        days = int(request.form.get('days', 10))
        crypto = session.get('crypto')
        preds = predict_future(crypto, days)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, days+1), preds, marker='o', color='purple')
        plt.title(f"{CRYPTOS[crypto]} Price Prediction (Next {days} Days)")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price (USD)")
        plt.grid(alpha=0.3)
        for i, price in enumerate(preds):
            plt.text(i+1, price, f"${price:.2f}", ha='center', va='bottom')
        plot_url = plot_to_html()
        plt.close()

        return render_template('result.html',
                               plot_url=plot_url,
                               crypto_name=CRYPTOS[crypto],
                               predictions=zip(range(1, days+1), preds))

    return render_template('predict_days.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        selected = request.form.getlist('cryptos')
        days = int(request.form.get('days', 10))

        results = {}
        for crypto in selected:
            preds = predict_future(crypto, days)
            results[CRYPTOS[crypto]] = preds

        plt.figure(figsize=(14, 7))
        for name, prices in results.items():
            plt.plot(range(1, days+1), prices, marker='o', label=name)

        plt.title(f"Crypto Price Predictions (Next {days} Days)")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(alpha=0.3)

        plot_url = plot_to_html()
        plt.close()

        return render_template('compare_result.html',
                               plot_url=plot_url,
                               results=results)

    return render_template('compare.html', cryptos=CRYPTOS)


if __name__ == '__main__':
    app.run(debug=True)
