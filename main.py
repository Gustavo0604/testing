# advanced_bot.py
"""
Advanced Crypto Trading Bot with Incremental Learning, Experience Buffer,
Trade Accuracy Scoring, Risk Management, Simulation Mode, Detailed Logging,
Alerts, and Terminal Dashboard.

Added metrics: Average Win, Average Loss, Expectancy.
"""

import os
import sys
import asyncio
import logging
import argparse
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import requests

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÕES & UTILITÁRIOS
# ------------------------------------------------------------------------------
class Config:
    SYMBOL            = os.getenv('SYMBOL', 'BTC/USDT')
    TIMEFRAME         = os.getenv('TIMEFRAME', '1m')
    LOOKBACK          = int(os.getenv('LOOKBACK', '1000'))
    SIMULATE_MODE     = os.getenv('SIMULATE_MODE', 'True') == 'True'
    INITIAL_CAPITAL   = float(os.getenv('INITIAL_CAPITAL', '450.0'))
    MAX_POSITION_RISK = float(os.getenv('MAX_POSITION_RISK', '0.02'))
    EXPERIENCE_BUFFER_SIZE   = int(os.getenv('EXPERIENCE_BUFFER_SIZE', '20'))
    INCREMENTAL_TRAIN_ROUNDS = int(os.getenv('INCREMENTAL_TRAIN_ROUNDS', '5'))
    RETRAIN_INTERVAL         = int(os.getenv('RETRAIN_INTERVAL', '60'))  # minutes
    LOG_FILE          = os.getenv('LOG_FILE', 'bot.log')
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

def setup_logging():
    logger = logging.getLogger('AdvancedBot')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
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
            requests.post(Config.SLACK_WEBHOOK_URL, json={'text': msg}, timeout=5)
        except Exception as e:
            logger.error(f"Slack alert error: {e}")

# ------------------------------------------------------------------------------
# 2. EXPERIENCE BUFFER FOR INCREMENTAL LEARNING
# ------------------------------------------------------------------------------
class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.X = []
        self.y = []

    def add(self, x_vec, label):
        self.X.append(x_vec)
        self.y.append(label)
        if len(self.y) >= self.capacity:
            batch_X = np.array(self.X)
            batch_y = np.array(self.y)
            self.X.clear()
            self.y.clear()
            return batch_X, batch_y
        return None, None

# ------------------------------------------------------------------------------
# 3. DATA FETCH & PREPROCESSING
# ------------------------------------------------------------------------------
async def fetch_ohlcv(exchange, symbol, timeframe, limit):
    data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'], df['macd_sig'] = macd.macd(), macd.macd_signal()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'], df['bb_low'] = bb.bollinger_hband(), bb.bollinger_lband()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=2)
    df['kc_high'], df['kc_low'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
    df['hl_range']  = df['high'] - df['low']
    df['oc_change'] = (df['close'] - df['open']) / df['open']
    df.dropna(inplace=True)
    return df

def prepare_ml_data(df: pd.DataFrame):
    df = df.copy()
    df['future_close'] = df['close'].shift(-1)
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    feats = ['rsi','macd','macd_sig','bb_high','bb_low','atr','hl_range','oc_change']
    return df[feats].values, df['label'].values

# ------------------------------------------------------------------------------
# 4. HYPERPARAMETER OPTIMIZATION (ONE-TIME)
# ------------------------------------------------------------------------------
def hyperopt_objective(params):
    X, y = hyperopt_objective.data
    dtrain = xgb.DMatrix(X, label=y)
    cv = xgb.cv(params, dtrain, num_boost_round=50, nfold=3,
                metrics={'auc'}, as_pandas=True, seed=42)
    return {'loss': -cv['test-auc-mean'].iloc[-1], 'status': STATUS_OK}

def optimize_hyperparams(X, y):
    space = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': hp.choice('md', np.arange(3,10,dtype=int)),
        'eta': hp.loguniform('eta', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('ss', 0.5, 1),
        'colsample_bytree': hp.uniform('cb', 0.5, 1)
    }
    hyperopt_objective.data = (X, y)
    trials = Trials()
    best = fmin(hyperopt_objective, space, algo=tpe.suggest,
                max_evals=50, trials=trials)
    logger.info(f"Hyperopt best params: {best}")
    return best

# ------------------------------------------------------------------------------
# 5. RISK MANAGEMENT
# ------------------------------------------------------------------------------
def calculate_position_size(equity, price, risk_per_trade, atr):
    dollar_risk = equity * risk_per_trade
    size_by_risk = dollar_risk / atr
    max_size = equity / price
    return min(size_by_risk, max_size)

# ------------------------------------------------------------------------------
# 6. ASYNC TRADER w/ SIMULATION & EXPERIENCE
# ------------------------------------------------------------------------------
class AsyncTrader:
    def __init__(self, simulate, initial_capital):
        self.simulate = simulate
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.exchange.set_sandbox_mode(True)
        base, quote = Config.SYMBOL.split('/')
        self.balances = {quote: initial_capital, base: 0.0}
        self.position = 0
        self.last_price = None
        self.last_buy_price = None

    async def update_last_price(self):
        t = await self.exchange.fetch_ticker(Config.SYMBOL)
        self.last_price = t['last']

    async def place_order(self, side, amount):
        price = self.last_price
        base, quote = Config.SYMBOL.split('/')
        if self.simulate and side == 'buy':
            max_amt = self.balances[quote] / price
            amount = min(amount, max_amt)
        cost = amount * price

        if self.simulate:
            if side == 'buy':
                self.balances[quote] -= cost
                self.balances[base]  += amount
                self.position = self.balances[base]
                self.last_buy_price = price
                logger.info(f"[SIM] BUY  {amount:.6f} {base} @ {price:.2f} -> {quote}: {self.balances[quote]:.2f}")
                return {'side':'buy','amount':amount,'price':price}
            else:
                self.balances[base]  -= amount
                self.balances[quote] += cost
                self.position = self.balances[base]
                logger.info(f"[SIM] SELL {amount:.6f} {base} @ {price:.2f} -> {quote}: {self.balances[quote]:.2f}")
                return {'side':'sell','amount':amount,'price':price,'buy_price':self.last_buy_price}

        try:
            o = await self.exchange.create_order(Config.SYMBOL,'market',side,amount)
            logger.info(f"REAL {side.upper()} {amount}@{Config.SYMBOL} status={o['status']}")
            return o
        except Exception as e:
            msg = str(e).lower()
            if 'insufficient balance' in msg:
                logger.warning("Order skipped – insufficient balance")
                return None
            logger.error(f"Order error: {e}")
            return None

    async def close(self):
        await self.exchange.close()

# ------------------------------------------------------------------------------
# 7. DASHBOARD WITH ADDITIONAL METRICS
# ------------------------------------------------------------------------------
class Dashboard:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

    def render(self, equity, pnl, wins, losses, total_win_amount, total_loss_amount, open_pos):
        avg_win = total_win_amount / wins if wins else 0.0
        avg_loss = total_loss_amount / losses if losses else 0.0
        win_rate = wins / (wins + losses) if (wins+losses) else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        tbl = Table(expand=True)
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        tbl.add_row("Equity", f"${equity:,.2f}")
        tbl.add_row("P&L", f"${pnl:,.2f}")
        tbl.add_row("Wins", str(wins))
        tbl.add_row("Losses", str(losses))
        tbl.add_row("Win Rate", f"{win_rate*100:.2f}%")
        tbl.add_row("Avg Win", f"${avg_win:.4f}")
        tbl.add_row("Avg Loss", f"${avg_loss:.4f}")
        tbl.add_row("Expectancy", f"${expectancy:.4f}")
        tbl.add_row("Open Positions", str(open_pos))
        self.layout["header"].update(Panel("[b]Advanced Crypto Bot Dashboard[/b]"))
        self.layout["left"].update(tbl)
        now = datetime.now(timezone.utc).isoformat()
        self.layout["right"].update(Panel(f"Last Update: {now}"))
        self.layout["footer"].update(Panel("[green]Running...[/green]"))
        return self.layout

# ------------------------------------------------------------------------------
# 8. MAIN LOOP
# ------------------------------------------------------------------------------
async def main_loop():
    # Initial data & model
    exchange = ccxt.binance({'enableRateLimit':True})
    exchange.set_sandbox_mode(True)
    df0 = await fetch_ohlcv(exchange, Config.SYMBOL, Config.TIMEFRAME, Config.LOOKBACK)
    df0 = preprocess(df0)
    X0, y0 = prepare_ml_data(df0)
    best = optimize_hyperparams(X0, y0)
    xgb_params = {
        'eta':             float(best['eta']),
        'max_depth':       int(best['md']),
        'subsample':       float(best['ss']),
        'colsample_bytree':float(best['cb']),
        'objective':'binary:logistic',
        'eval_metric':'auc'
    }
    dtrain0 = xgb.DMatrix(X0, label=y0)
    model = xgb.train(xgb_params, dtrain0, num_boost_round=100)

    # Setup
    trader = AsyncTrader(Config.SIMULATE_MODE, Config.INITIAL_CAPITAL)
    buffer = ExperienceBuffer(Config.EXPERIENCE_BUFFER_SIZE)
    wins = losses = 0
    total_win_amount = total_loss_amount = 0.0
    dash = Dashboard()
    equity = Config.INITIAL_CAPITAL
    pnl    = 0.0
    itr    = 0

    with Live(console=console, refresh_per_second=1) as live:
        try:
            while True:
                await trader.update_last_price()
                df = await fetch_ohlcv(trader.exchange, Config.SYMBOL, Config.TIMEFRAME, Config.LOOKBACK)
                df = preprocess(df)
                last = df.iloc[-1]
                feats = np.array([[last['rsi'], last['macd'], last['macd_sig'],
                                   last['bb_high'], last['bb_low'],
                                   last['atr'], last['hl_range'], last['oc_change']]])
                prob   = model.predict(xgb.DMatrix(feats))[0]
                signal = 1 if prob > 0.5 else -1

                size = calculate_position_size(equity, trader.last_price,
                                               Config.MAX_POSITION_RISK, last['atr'])

                order = None
                if signal == 1 and trader.position == 0:
                    order = await trader.place_order('buy', size)
                elif signal == -1 and trader.position > 0:
                    order = await trader.place_order('sell', trader.position)

                if order and order.get('side') == 'sell':
                    buy_p  = order['buy_price']
                    sell_p = order['price']
                    profit = (sell_p - buy_p) * order['amount']
                    if sell_p > buy_p:
                        wins += 1
                        total_win_amount += profit
                    else:
                        losses += 1
                        total_loss_amount += abs(profit)
                    batch_X, batch_y = buffer.add(feats.flatten(), int(sell_p>buy_p))
                    if batch_y is not None:
                        dnew = xgb.DMatrix(batch_X, label=batch_y)
                        model = xgb.train(xgb_params, dnew,
                                          num_boost_round=Config.INCREMENTAL_TRAIN_ROUNDS,
                                          xgb_model=model)
                        logger.info(f"Incremental training on {len(batch_y)} samples complete.")

                equity = (trader.balances['USDT'] +
                          trader.balances['BTC'] * trader.last_price
                          if Config.SIMULATE_MODE
                          else Config.INITIAL_CAPITAL + trader.position * trader.last_price)
                pnl = equity - Config.INITIAL_CAPITAL

                itr += 1
                if itr % Config.RETRAIN_INTERVAL == 0:
                    logger.info("Scheduled retrain...")
                    Xr, yr = prepare_ml_data(df)
                    dtrain_r = xgb.DMatrix(Xr, label=yr)
                    model = xgb.train(xgb_params, dtrain_r,
                                      num_boost_round=100, xgb_model=model)
                    send_slack_alert("Scheduled retrain complete.")

                layout = dash.render(
                    equity, pnl,
                    wins, losses,
                    total_win_amount, total_loss_amount,
                    int(trader.position>0)
                )
                live.update(layout)
                await asyncio.sleep(60)
        finally:
            await trader.close()
            await exchange.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Advanced Crypto Bot")
    parser.add_argument('--start', action='store_true', help='Start bot')
    args = parser.parse_args()
    if args.start:
        try:
            asyncio.run(main_loop())
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            send_slack_alert(f"Bot crashed: {e}")
    else:
        print("Use --start to run the bot")
