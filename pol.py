import tkinter as tk
import threading
import time
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pushbullet import Pushbullet
import ccxt

# Ortam değişkeninden token al
PB_TOKEN = os.environ.get("PB_TOKEN")
if not PB_TOKEN:
    raise ValueError("PB_TOKEN ortam değişkeni bulunamadı.")
pb = Pushbullet(PB_TOKEN)

# Coin listesi
COINS = ["storj", "xrp", "eigen", "eth", "btc", "rune", "avax", "aave", "ada", "dot",
         "rlc", "xlm", "amp", "etc", "fet", "enj", "ethw", "algo", "sol", "axs", "trx",
         "chz", "skl", "zrx", "omg", "xtz", "ape", "api3", "mana", "neo"]

exchange = ccxt.binance()

# GUI sınıfı
class PolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("pol")
        self.root.configure(bg="gray20")

        self.running = False
        self.start_time = None
        self.thread = None

        self.start_button = tk.Button(root, text="Start/Stop", command=self.toggle, bg="green", fg="black", font=("Comic Sans MS", 12))
        self.start_button.grid(row=0, column=0, padx=20, pady=20)

        self.time_label = tk.Label(root, text="Working Time:", bg="gray20", fg="white", font=("Comic Sans MS", 14))
        self.time_label.grid(row=0, column=1)

        self.time_display = tk.Label(root, text="0.0", bg="green", fg="black", font=("Comic Sans MS", 14))
        self.time_display.grid(row=0, column=2)

        self.coin_label = tk.Label(root, text="Analized Coin", bg="gray20", fg="white", font=("Comic Sans MS", 14))
        self.coin_label.grid(row=1, column=0)
        self.notification_label = tk.Label(root, text="Notification", bg="gray20", fg="white", font=("Comic Sans MS", 14))
        self.notification_label.grid(row=1, column=1)
        self.success_label = tk.Label(root, text="Success Rate", bg="gray20", fg="white", font=("Comic Sans MS", 14))
        self.success_label.grid(row=1, column=2)

        self.coin_box = tk.Text(root, height=20, width=30, bg="green")
        self.coin_box.grid(row=2, column=0)
        self.notification_box = tk.Text(root, height=20, width=60, bg="green")
        self.notification_box.grid(row=2, column=1)
        self.success_box = tk.Text(root, height=20, width=30, bg="green")
        self.success_box.grid(row=2, column=2)

        self.update_clock()

    def toggle(self):
        if self.running:
            self.running = False
        else:
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self.run_analysis)
            self.thread.start()

    def update_clock(self):
        if self.running and self.start_time:
            elapsed = time.time() - self.start_time
            self.time_display.config(text=f"{elapsed/60:.1f} min")
        self.root.after(1000, self.update_clock)

    def run_analysis(self):
        while self.running:
            results = []
            for coin in COINS:
                symbol = f"{coin.upper()}/USDT"
                try:
                    df = get_binance_ohlcv(symbol)
                    if df.empty:
                        continue

                    fisher = fisher_transform(df)
                    supertrend = supertrend_signal(df)
                    fractal = williams_fractal(df)
                    news_score = get_news_score(coin)

                    score = fisher + supertrend + fractal + news_score
                    results.append((coin.upper(), score))
                except Exception as e:
                    print(f"Veri çekme hatası {symbol}: {e}")
                    continue

            top_coins = sorted(results, key=lambda x: x[1], reverse=True)[:4]

            self.coin_box.delete(1.0, tk.END)
            self.notification_box.delete(1.0, tk.END)
            self.success_box.delete(1.0, tk.END)

            for coin, score in top_coins:
                price = get_price(f"{coin}/USDT")
                buy_price = round(price, 2)
                sell_price = round(price * 1.15, 2)
                notification = f"{coin} ${buy_price} al ${sell_price} sat emri ver"
                pb.push_note("pol", notification)

                self.coin_box.insert(tk.END, coin + "\n")
                self.notification_box.insert(tk.END, notification + "\n")
                self.success_box.insert(tk.END, f"%{int(score*10)} doğruluk" + "\n")

            time.sleep(3600)

# Binance'ten veri çek
def get_binance_ohlcv(symbol):
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=since)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def get_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker["last"]

# İndikatör fonksiyonları
def fisher_transform(df):
    high = df['High']
    low = df['Low']
    hl2 = (high + low) / 2
    n = 10
    min_val = hl2.rolling(n).min()
    max_val = hl2.rolling(n).max()
    value = 2 * ((hl2 - min_val) / (max_val - min_val) - 0.5)
    value = value.replace([np.inf, -np.inf], 0).fillna(0)
    fish = np.arctanh(value.clip(-0.999, 0.999))
    return fish.iloc[-1]

def supertrend_signal(df, atr_period=10, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2
    tr1 = abs(df['High'] - df['Low'])
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    close = df['Close']
    if close.index.equals(lowerband.index):
        return 1 if close.iloc[-1] > lowerband.iloc[-1] else -1
    else:
        return 0

def williams_fractal(df):
    high = df['High'].reset_index(drop=True)
    low = df['Low'].reset_index(drop=True)
    if len(high) < 5:
        return 0

    for i in range(2, len(high) - 2):
        if high[i] > high[i - 2] and high[i] > high[i - 1] and high[i] > high[i + 1] and high[i] > high[i + 2]:
            return -1
        if low[i] < low[i - 2] and low[i] < low[i - 1] and low[i] < low[i + 1] and low[i] < low[i + 2]:
            return 1
    return 0

def get_news_score(coin):
    if not coin:
        return 0
    url = f"https://www.cryptocraft.com/coins/{coin}"
    try:
        response = requests.get(url)
        if not response.ok:
            return 0
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.find_all("h3")[:5]
        positive_words = ["up", "gain", "rise", "positive", "bull"]
        negative_words = ["down", "fall", "drop", "negative", "bear"]
        score = 0
        for title in titles:
            text = title.text.lower()
            if any(word in text for word in positive_words):
                score += 1
            if any(word in text for word in negative_words):
                score -= 1
        return score / 5
    except Exception as e:
        print(f"News fetch error for {coin}: {e}")
        return 0

# GUI çalıştır
if __name__ == "__main__":
    root = tk.Tk()
    app = PolApp(root)
    root.mainloop()
