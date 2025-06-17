# hyperstrike.py

"""
ChatGPT HyperStrike 9000
Multi-Agent, Meta-Adaptive, Cross-Exchange Bitcoin Daytrade Bot
"""

import os
import sys
import asyncio
import logging
import argparse
import json
import random
from datetime import datetime, timezone, timedelta
from collections import deque

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import pywt
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
class Config:
    SYMBOLS                 = ["BTC/USDT"]
    TIMEFRAMES              = ["1m","5m","15m"]
    EXCHANGES               = ["binance","bybit","ftx"]
    LOOKBACK                = 500
    SIMULATE                = True
    INITIAL_CAPITAL         = 500.0
    RISK_PER_TRADE          = 0.02
    MAX_CONSECUTIVE_WINS    = 3
    OVERFIT_REDUCE_FACTOR   = 0.8
    TRAIL_EMA_SHORT         = 5
    TRAIL_EMA_LONG          = 20
    SAFE_DRAWDOWN           = 0.015
    SAFE_PAUSE_MIN          = 10
    GEN_PERIOD_HOURS        = 12
    N_GEN_STRATS            = 5
    AUTOENCODER_LATENT_DIM  = 4
    KS_ALPHA                = 0.05
    SLACK_WEBHOOK           = os.getenv("SLACK_WEBHOOK_URL","")
    WEB_PORT                = int(os.getenv("WEB_PORT","8500"))
    ASCII_REFRESH_SEC       = 10

# -------------------------------------------------------------------------
# 2. LOGGING
# -------------------------------------------------------------------------
logger = logging.getLogger("HyperStrike")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

# -------------------------------------------------------------------------
# 3. MULTI-AGENT SIGNALS
# -------------------------------------------------------------------------
class NeuralTrend:
    def __init__(self, seq_len=50, features=1):
        inp = Input((seq_len, features))
        x = LSTM(32, return_sequences=False)(inp)
        out = Dense(1, activation="sigmoid")(x)
        self.model = Model(inp, out)
        self.model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

    def predict(self, data):
        # data: np.array shape (seq_len, features)
        return float(self.model.predict(data.reshape(1,*data.shape))[0,0])

class QuantumBreak:
    @staticmethod
    def wavelet_energy(series, wavelet="db4", level=3):
        coeffs = pywt.wavedec(series, wavelet, level=level)
        energy = [np.sum(c**2) for c in coeffs]
        return energy[1]/(energy[0]+1e-6)

class SentimentScan:
    @staticmethod
    def score():
        # placeholder scraper; returns random sentiment [0,1]
        return random.random()

# -------------------------------------------------------------------------
# 4. META-ADAPTIVE HYPERPARAMETERS
# -------------------------------------------------------------------------
class MetaLearner:
    def __init__(self):
        self.hyper = {"threshold":0.55}

    def adapt(self, recent_pnl):
        # simple rule: if good pnl, raise threshold a bit
        if recent_pnl > 0:
            self.hyper["threshold"] = min(0.7, self.hyper["threshold"] + 0.01)
        else:
            self.hyper["threshold"] = max(0.4, self.hyper["threshold"] - 0.01)

# -------------------------------------------------------------------------
# 5. RISK MANAGEMENT
# -------------------------------------------------------------------------
class RiskManager:
    @staticmethod
    def kelly(p_win, win_loss_ratio):
        return max(0, p_win - (1 - p_win) / win_loss_ratio)

    @staticmethod
    def trailing_stop(price, short_ema, long_ema):
        # orbit stop-loss trigger
        if short_ema < long_ema:
            return price * 0.99
        return None

# -------------------------------------------------------------------------
# 6. CROSS-EXCHANGE DATA FETCH
# -------------------------------------------------------------------------
class MarketData:
    def __init__(self):
        self.exchanges = {name: getattr(ccxt, name)({"enableRateLimit":True}) for name in Config.EXCHANGES}
        for ex in self.exchanges.values():
            ex.set_sandbox_mode(Config.SIMULATE)

    async def fetch(self, symbol, timeframe, limit):
        frames = []
        for name, ex in self.exchanges.items():
            try:
                data = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(data, columns=["time","o","h","l","c","v"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df.set_index("time", inplace=True)
                df = df.add_suffix(f"_{name}")
                frames.append(df)
            except:
                continue
        if not frames: return None
        df = pd.concat(frames, axis=1).ffill().bfill()
        return df

# -------------------------------------------------------------------------
# 7. MICRO-STRATEGY GENETIC ALGO
# -------------------------------------------------------------------------
class StrategyGen:
    @staticmethod
    def random_strategy():
        return {
            "ema_short": random.randint(3,10),
            "ema_long": random.randint(11,30),
            "rsi_lb": random.randint(10,20),
            "rsi_ub": random.randint(60,80)
        }

    @staticmethod
    def backtest(df, strat):
        # simple EMA+RSI backtest
        df = df.copy()
        df["ema_s"] = df["c_binance"].ewm(span=strat["ema_short"]).mean()
        df["ema_l"] = df["c_binance"].ewm(span=strat["ema_long"]).mean()
        df["rsi"] = 100 - 100/(1+df["c_binance"].pct_change().rolling(strat["rsi_lb"]).apply(lambda x: (x[x>0].sum()/(-x[x<0].sum()+1e-6))))
        cash, pos = Config.INITIAL_CAPITAL, 0
        for i in range(1,len(df)):
            if df.ema_s.iloc[i]>df.ema_l.iloc[i] and df.rsi.iloc[i]<strat["rsi_ub]"]:
                if pos==0:
                    pos = cash/df.c_binance.iloc[i]
                    cash = 0
            elif pos>0 and df.ema_s.iloc[i]<df.ema_l.iloc[i]:
                cash = pos*df.c_binance.iloc[i]
                pos = 0
        final = cash + pos*df.c_binance.iloc[-1]
        return final - Config.INITIAL_CAPITAL

# -------------------------------------------------------------------------
# 8. DRIFT & ANOMALY DETECTION
# -------------------------------------------------------------------------
class DriftDetector:
    def __init__(self):
        self.ref = None
        self.autoencoder = self._build_autoencoder()

    def _build_autoencoder(self):
        inp = Input((len(Config.TIMEFRAMES)*len(Config.EXCHANGES),))
        x = Dense(8, activation="relu")(inp)
        z = Dense(Config.AUTOENCODER_LATENT_DIM, activation="relu")(x)
        x2 = Dense(8, activation="relu")(z)
        out = Dense(inp.shape[-1], activation="linear")(x2)
        model = Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def detect_ks(self, arr):
        if self.ref is None: self.ref = arr
        stat, p = K.mstats.kstest(arr, self.ref)
        return p < Config.KS_ALPHA

    def detect_autoencoder(self, arr):
        pred = self.autoencoder.predict(arr.reshape(1,-1))
        loss = np.mean((pred - arr)**2)
        return loss > np.percentile(self._reconstruct_losses, 95)

# -------------------------------------------------------------------------
# 9. DASHBOARD ASCII
# -------------------------------------------------------------------------
class AsciiDash:
    def __init__(self):
        pass

    def render(self, equity, pnl, wins, losses):
        bar = "#"*int((equity/ (Config.INITIAL_CAPITAL*2))*50)
        print(f"Equity: ${equity:.2f} | PnL: ${pnl:.2f}")
        print(f"[{bar:<50}]")
        print(f"W:{wins} L:{losses}")
        print(random.choice([
            "Tá subindo! 🚀", "Calma que sobe...", "É pump! 😂", "SafeMode ativado ☕"
        ]))

# -------------------------------------------------------------------------
# 10. MAIN LOOP
# -------------------------------------------------------------------------
async def main():
    md = MarketData()
    nt = NeuralTrend(seq_len=Config.LOOKBACK, features=1)
    ml = MetaLearner()
    dm = DriftDetector()
    rm = RiskManager()
    dash = AsciiDash()
    last_gen = datetime.now(timezone.utc) - timedelta(hours=Config.GEN_PERIOD_HOURS)
    wins = losses = 0
    equity = Config.INITIAL_CAPITAL

    while True:
        # fetch multi-timeframe cross-exchange
        dfs = []
        for tf in Config.TIMEFRAMES:
            df = await md.fetch(Config.SYMBOLS[0], tf, Config.LOOKBACK)
            if df is None: continue
            df["tf"] = tf
            dfs.append(df)
        if not dfs:
            await asyncio.sleep(5)
            continue
        df = pd.concat(dfs, axis=1).ffill().bfill()

        # compute signals
        close_series = df["c_binance"].values[-Config.LOOKBACK:]
        nt_sig = nt.predict(close_series.reshape(-1,1))
        qb_sig = QuantumBreak.wavelet_energy(close_series)
        ss_sig = SentimentScan.score()
        avg_sig = np.mean([nt_sig, qb_sig, ss_sig])

        # adapt hyper
        pnl_recent = equity - Config.INITIAL_CAPITAL
        ml.adapt(pnl_recent)
        thresh = ml.hyper["threshold"]

        # drift
        drift = dm.detect_ks(close_series)

        # generate new strategies
        if datetime.now(timezone.utc) - last_gen > timedelta(hours=Config.GEN_PERIOD_HOURS):
            strats = [StrategyGen.random_strategy() for _ in range(Config.N_GEN_STRATS)]
            results = [(s, StrategyGen.backtest(df, s)) for s in strats]
            best = max(results, key=lambda x: x[1])
            logger.info(f"New strat: {best[0]} ret={best[1]:.2f}")
            last_gen = datetime.now(timezone.utc)

        # decide trade
        side = None
        if avg_sig > thresh and not drift:
            side = "buy"
        elif avg_sig < 1-thresh:
            side = "sell"

        # place order simulate
        price = close_series[-1]
        if side=="buy":
            size = equity * Config.RISK_PER_TRADE / price
            equity -= size*price
            position = size
            logger.info(f"[SIM] BUY {size:.6f} @ {price:.2f}")
        elif side=="sell" and 'position' in locals() and position>0:
            equity += position*price
            pnl = equity - Config.INITIAL_CAPITAL
            wins += int(pnl>0); losses += int(pnl<=0)
            logger.info(f"[SIM] SELL {position:.6f} @ {price:.2f} pnl={pnl:.2f}")
            del position

        # safe mode
        if equity < Config.INITIAL_CAPITAL*(1-Config.SAFE_DRAWDOWN):
            logger.error("SafeMode triggered!")
            requests.post(Config.SLACK_WEBHOOK, json={"text":"It’s Over 9000!"})
            await asyncio.sleep(Config.SAFE_PAUSE_MIN*60)
            equity = Config.INITIAL_CAPITAL  # reset sim

        # render dashboard
        dash.render(equity, equity-Config.INITIAL_CAPITAL, wins, losses)

        await asyncio.sleep(Config.ASCII_REFRESH_SEC)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true", help="Run bot")
    args = parser.parse_args()
    if args.start:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("HyperStrike stopped by user")
    else:
        print("Use --start to run HyperStrike 9000")
