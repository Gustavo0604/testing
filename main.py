# advanced_bot.py

"""
Static Crypto Trading Bot
No incremental learning or retraining: same model from start to finish
"""

import os
import sys
import asyncio
import logging
import argparse
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from aiohttp import web
import requests
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES & UTILITÁRIOS
# -------------------------------------------------------------------------
class Config:
    SYMBOL                   = "BTC/USDT"
    TIMEFRAME                = "1m"
    LOOKBACK                 = 1000
    SIMULATE_MODE            = True
    INITIAL_CAPITAL          = 450.0
    MAX_POSITION_RISK        = 0.02
    FULL_MODEL_ROUNDS        = 100
    SIGNAL_THRESHOLD         = 0.55
    SAFE_MODE_LOSS_THRESHOLD = 0.005   # 0.5% drawdown
    SAFE_MODE_COOLDOWN       = 15      # minutes
    COOLDOWN_MINUTES         = 3
    MAX_VOLATILITY           = 0.02    # 2% std of returns
    VOLATILITY_SIZE_FACTOR   = 0.5
    LOG_FILE                 = "bot.log"
    SLACK_WEBHOOK_URL        = os.getenv("SLACK_WEBHOOK_URL", "")
    WEB_PORT                 = int(os.getenv("PORT", "8000"))


def setup_logging():
    logger = logging.getLogger("StaticBot")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()
console = Console()

def send_slack_alert(msg: str):
    if Config.SLACK_WEBHOOK_URL:
        try:
            requests.post(Config.SLACK_WEBHOOK_URL, json={"text": msg}, timeout=5)
        except Exception as e:
            logger.error(f"Slack alert error: {e}")

# -------------------------------------------------------------------------
# 2. DATA FETCH & PREPROCESSING
# -------------------------------------------------------------------------
async def safe_fetch_ohlcv(exchange, symbol, timeframe, limit):
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logger.warning(f"fetch_ohlcv failed: {e}")
        return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_sig"] = macd.macd(), macd.macd_signal()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"], df["bb_low"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    kc = KeltnerChannel(df["high"], df["low"], df["close"], window=20, window_atr=2)
    df["kc_high"], df["kc_low"] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df.dropna(inplace=True)
    return df


def prepare_ml_data(df: pd.DataFrame):
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["label"] = (df["future_close"] > df["close"]).astype(int)
    df.dropna(inplace=True)
    feats = ["rsi","macd","macd_sig","bb_high","bb_low","atr","hl_range","oc_change"]
    return df[feats].values, df["label"].values

# -------------------------------------------------------------------------
# 3. HYPERPARAMETER OPTIMIZATION (ONE-TIME)
# -------------------------------------------------------------------------
def hyperopt_objective(params):
    X, y = hyperopt_objective.data
    dtrain = xgb.DMatrix(X, label=y)
    cv = xgb.cv(params, dtrain, num_boost_round=50, nfold=3,
                metrics={"auc"}, as_pandas=True, seed=42)
    return {"loss": -cv["test-auc-mean"].iloc[-1], "status": STATUS_OK}


def optimize_hyperparams(X, y):
    space = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": hp.choice("md", np.arange(3,10,dtype=int)),
        "eta": hp.loguniform("eta", np.log(0.01), np.log(0.3)),
        "subsample": hp.uniform("ss", 0.5, 1),
        "colsample_bytree": hp.uniform("cb", 0.5, 1)
    }
    hyperopt_objective.data = (X, y)
    trials = Trials()
    best = fmin(hyperopt_objective, space, algo=tpe.suggest,
                max_evals=50, trials=trials)
    logger.info(f"Hyperopt best params: {best}")
    return best

# -------------------------------------------------------------------------
# 4. RISK MANAGEMENT
# -------------------------------------------------------------------------
def calculate_position_size(equity, price, risk_per_trade, atr):
    dollar_risk = equity * risk_per_trade
    size_by_risk = dollar_risk / atr
    max_size = equity / price
    return min(size_by_risk, max_size)

# -------------------------------------------------------------------------
# 5. ASYNC TRADER w/ SIMULAÇÃO
# -------------------------------------------------------------------------
class AsyncTrader:
    def __init__(self, simulate, initial_capital):
        self.simulate      = simulate
        self.exchange      = ccxt.binance({"enableRateLimit": True})
        self.exchange.set_sandbox_mode(True)
        base, quote       = Config.SYMBOL.split("/")
        self.balances     = {quote: initial_capital, base: 0.0}
        self.position     = 0
        self.last_price   = None
        self.last_buy_price = None

    async def update_last_price(self):
        try:
            ticker = await self.exchange.fetch_ticker(Config.SYMBOL)
            self.last_price = ticker["last"]
        except Exception as e:
            logger.warning(f"update_last_price failed: {e}")

    async def place_order(self, side, amount):
        price = self.last_price
        base, quote = Config.SYMBOL.split("/")
        if self.simulate and side == "buy":
            max_amt = self.balances[quote] / price
            amount = min(amount, max_amt)
        cost = amount * price

        if self.simulate:
            if side == "buy":
                self.balances[quote] -= cost
                self.balances[base]  += amount
                self.position = self.balances[base]
                self.last_buy_price = price
                logger.info(f"[SIM] BUY  {amount:.6f} BTC @ {price:.2f} -> USDT: {self.balances[quote]:.2f}")
                trade_events.append({"timestamp": datetime.now(timezone.utc).isoformat(), "side":"buy",  "price":price})
                return {"side":"buy","amount":amount,"price":price}
            else:
                self.balances[base] -= amount
                self.balances[quote]+= cost
                self.position = self.balances[base]
                logger.info(f"[SIM] SELL {amount:.6f} BTC @ {price:.2f} -> USDT: {self.balances[quote]:.2f}")
                trade_events.append({"timestamp": datetime.now(timezone.utc).isoformat(), "side":"sell","price":price})
                return {"side":"sell","amount":amount,"price":price,"buy_price":self.last_buy_price}

        try:
            o = await self.exchange.create_order(Config.SYMBOL, "market", side, amount)
            logger.info(f"REAL {side.upper()} {amount}@{Config.SYMBOL} status={o['status']}")
            return o
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

# -------------------------------------------------------------------------
# 6. DASHBOARD & SERVIDOR WEB
# -------------------------------------------------------------------------
candle_data  = deque(maxlen=60)
trade_events = []

async def dashboard_handler(request):
    # ... (mesmo HTML de antes)
    return web.Response(text="...", content_type="text/html")

async def data_handler(request):
    return web.json_response({"candles": list(candle_data), "trades": trade_events})

async def start_web_dashboard():
    app = web.Application()
    app.router.add_get("/", dashboard_handler)
    app.router.add_get("/data", data_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", Config.WEB_PORT)
    await site.start()
    logger.info(f"Web dashboard running on port {Config.WEB_PORT}")

class Dashboard:
    def __init__(self):
        layout = Layout()
        layout.split_column(
            Layout(name="hdr", size=3),
            Layout(name="main", ratio=1),
            Layout(name="ftr", size=3)
        )
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        self.layout = layout

    def render(self, equity, pnl, wins, losses, open_pos):
        tbl = Table(expand=True)
        for k,v in [
            ("Equity",   f"${equity:,.2f}"),
            ("P&L",      f"${pnl:,.2f}"),
            ("Wins",     str(wins)),
            ("Losses",   str(losses)),
            ("Open Pos", str(open_pos))
        ]:
            tbl.add_row(k, v)
        self.layout["hdr"].update(Panel("[b]Advanced Crypto Bot Dashboard[/b]"))
        self.layout["left"].update(tbl)
        self.layout["right"].update(Panel(f"Last Update:\n{datetime.now(timezone.utc).isoformat()}"))
        self.layout["ftr"].update(Panel("[green]Running...[/green]"))
        return self.layout

# -------------------------------------------------------------------------
# 7. LOOP PRINCIPAL
# -------------------------------------------------------------------------
async def main_loop():
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.set_sandbox_mode(True)
    trader = AsyncTrader(Config.SIMULATE_MODE, Config.INITIAL_CAPITAL)
    last_trade_time = datetime.now(timezone.utc) - timedelta(minutes=Config.COOLDOWN_MINUTES)

    # setup inicial e treino único
    df0 = await safe_fetch_ohlcv(exchange, Config.SYMBOL, Config.TIMEFRAME, Config.LOOKBACK)
    df0 = preprocess(df0)
    X0, y0 = prepare_ml_data(df0)
    best = optimize_hyperparams(X0, y0)
    xgb_params = {
        "eta":           float(best["eta"]),
        "max_depth":     int(best["md"]),
        "subsample":     float(best["ss"]),
        "colsample_bytree": float(best["cb"]),
        "objective":     "binary:logistic",
        "eval_metric":   "auc"
    }
    xgb_model = xgb.train(xgb_params, xgb.DMatrix(X0, label=y0), num_boost_round=Config.FULL_MODEL_ROUNDS)
    sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
    sgd_model.partial_fit(X0, y0, classes=[0,1])

    wins = losses = 0
    equity = Config.INITIAL_CAPITAL
    pnl = 0.0
    itr = 0
    dash = Dashboard()

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            await trader.update_last_price()
            df = await safe_fetch_ohlcv(trader.exchange, Config.SYMBOL, Config.TIMEFRAME, 60)
            if df is None:
                await asyncio.sleep(5)
                continue
            df = preprocess(df)

            candle_data.clear()
            for ts, row in df.iterrows():
                candle_data.append({
                    "timestamp": ts.isoformat(),
                    "open":      row["open"],
                    "high":      row["high"],
                    "low":       row["low"],
                    "close":     row["close"]
                })

            last = df.iloc[-1]
            feats = last[["rsi","macd","macd_sig","bb_high","bb_low",
                          "atr","hl_range","oc_change"]].values.reshape(1, -1)

            prob_xgb = xgb_model.predict(xgb.DMatrix(feats))[0]
            prob_sgd = sgd_model.predict_proba(feats)[0][1]
            avg_prob = (prob_xgb + prob_sgd) / 2.0

            now = datetime.now(timezone.utc)
            can_trade = (now - last_trade_time).total_seconds() >= Config.COOLDOWN_MINUTES * 60

            sig = 0
            order = None

            if can_trade:
                if avg_prob > Config.SIGNAL_THRESHOLD:
                    sig = 1
                elif avg_prob < 1 - Config.SIGNAL_THRESHOLD:
                    sig = -1

                if trader.position > 0:
                    stop_price = trader.last_buy_price - 2 * last["atr"]
                    if trader.last_price < stop_price:
                        order = await trader.place_order("sell", trader.position)
                        last_trade_time = now
                        sig = 0

                vola = df["oc_change"].rolling(20).std().iloc[-1]
                size = calculate_position_size(
                    equity, trader.last_price, Config.MAX_POSITION_RISK, last["atr"]
                )
                if vola and vola > Config.MAX_VOLATILITY:
                    size *= Config.VOLATILITY_SIZE_FACTOR

                if sig == 1 and trader.position == 0:
                    order = await trader.place_order("buy", size)
                    last_trade_time = now
                elif sig == -1 and trader.position > 0:
                    order = await trader.place_order("sell", trader.position)
                    last_trade_time = now

            if order and order.get("side") == "sell":
                buy_p  = order["buy_price"]
                sell_p = order["price"]
                label  = int(sell_p > buy_p)
                wins  += label
                losses+= 1 - label

            equity = (
                trader.balances["USDT"] + trader.balances["BTC"] * trader.last_price
                if Config.SIMULATE_MODE
                else Config.INITIAL_CAPITAL + trader.position * trader.last_price
            )
            pnl = equity - Config.INITIAL_CAPITAL

            itr += 1
            logger.info(f"[Loop {itr}] Equity=${equity:,.2f}  PnL=${pnl:,.2f}  Wins={wins} Losses={losses} OpenPos={int(trader.position>0)}")

            live.update(dash.render(equity, pnl, wins, losses, int(trader.position>0)))
            await asyncio.sleep(60)

# -------------------------------------------------------------------------
# 8. RUN BOTH BOT AND WEB DASHBOARD
# -------------------------------------------------------------------------
async def run_all():
    web_task = asyncio.create_task(start_web_dashboard())
    bot_task = asyncio.create_task(main_loop())
    await asyncio.gather(web_task, bot_task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Static Crypto Bot")
