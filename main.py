# advanced_bot.py
"""
Advanced Crypto Trading Bot with Incremental Learning, Experience Buffer,
Trade Accuracy Scoring, Risk Management, Simulation Mode, Detailed Logging,
Alerts, Terminal Dashboard (Rich), and Web Dashboard.
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
from collections import deque
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import xgboost as xgb
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from aiohttp import web
import requests

# ------------------------------------------------------------------------------
# Globals for web dashboard
# ------------------------------------------------------------------------------
candle_data = deque(maxlen=60)
trade_events = []

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÕES & UTILITÁRIOS
# ------------------------------------------------------------------------------
class Config:
    SYMBOL            = 'BTC/USDT'
    TIMEFRAME         = '1m'
    LOOKBACK          = 1000
    SIMULATE_MODE     = True
    INITIAL_CAPITAL   = 450.0
    MAX_POSITION_RISK = 0.02
    EXPERIENCE_BUFFER_SIZE   = 20
    INCREMENTAL_TRAIN_ROUNDS = 5
    RETRAIN_INTERVAL         = 60
    LOG_FILE          = 'bot.log'
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
    WEB_PORT          = int(os.getenv('PORT', '8000'))

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
            batch_X, batch_y = np.array(self.X), np.array(self.y)
            self.X.clear()
            self.y.clear()
            return batch_X, batch_y
        return None, None

# ------------------------------------------------------------------------------
# 3. DATA FETCH & PREPROCESSING
# ------------------------------------------------------------------------------
async def safe_fetch_ohlcv(exchange, symbol, timeframe, limit):
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.warning(f"fetch_ohlcv failed: {e}")
        return None

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
        'objective': 'binary:logistic', 'eval_metric': 'auc',
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
        try:
            ticker = await self.exchange.fetch_ticker(Config.SYMBOL)
            self.last_price = ticker['last']
        except Exception as e:
            logger.warning(f"update_last_price failed: {e}")

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
                logger.info(f"[SIM] BUY  {amount:.6f} BTC @ {price:.2f} -> USDT: {self.balances[quote]:.2f}")
                trade_events.append({'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'side':'buy','price':price})
                return {'side':'buy','amount':amount,'price':price}
            else:
                self.balances[base]  -= amount
                self.balances[quote] += cost
                self.position = self.balances[base]
                logger.info(f"[SIM] SELL {amount:.6f} BTC @ {price:.2f} -> USDT: {self.balances[quote]:.2f}")
                trade_events.append({'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'side':'sell','price':price})
                return {'side':'sell','amount':amount,'price':price,'buy_price':self.last_buy_price}

        try:
            o = await self.exchange.create_order(Config.SYMBOL,'market',side,amount)
            logger.info(f"REAL {side.upper()} {amount}@{Config.SYMBOL} status={o['status']}")
            return o
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    async def close(self):
        await self.exchange.close()

# ------------------------------------------------------------------------------
# 7a. WEB DASHBOARD HANDLERS
# ------------------------------------------------------------------------------
async def dashboard_handler(request):
    return web.Response(text="""
<!DOCTYPE html><html><head><title>Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>
</head><body>
<canvas id="chart" width="800" height="400"></canvas>
<script>
async function load() {
  const resp = await fetch('/data'); const json = await resp.json();
  const candles = json.candles.map(c=>({x:new Date(c.timestamp),o:c.open,h:c.high,l:c.low,c:c.close}));
  const buys = json.trades.filter(t=>t.side==='buy').map(t=>({x:new Date(t.timestamp),y:t.price}));
  const sells= json.trades.filter(t=>t.side==='sell').map(t=>({x:new Date(t.timestamp),y:t.price}));
  window.ch = new Chart(document.getElementById('chart').getContext('2d'),{
    type:'candlestick',data:{datasets:[
      {label:'Price',data:candles},
      {label:'Buys',type:'scatter',data:buys,backgroundColor:'green'},
      {label:'Sells',type:'scatter',data:sells,backgroundColor:'red'}
    ]},options:{scales:{x:{type:'time'}}}
  });
}
async function refresh(){
  const resp=await fetch('/data'), json=await resp.json();
  window.ch.data.datasets[0].data=json.candles.map(c=>({x:new Date(c.timestamp),o:c.open,h:c.high,l:c.low,c:c.close}));
  window.ch.data.datasets[1].data=json.trades.filter(t=>t.side==='buy').map(t=>({x:new Date(t.timestamp),y:t.price}));
  window.ch.data.datasets[2].data=json.trades.filter(t=>t.side==='sell').map(t=>({x:new Date(t.timestamp),y:t.price}));
  window.ch.update();
}
load(); setInterval(refresh,5000);
</script>
</body></html>
""", content_type='text/html')

async def data_handler(request):
    return web.json_response({'candles': list(candle_data), 'trades': trade_events})

async def start_web_dashboard():
    app = web.Application()
    app.router.add_get('/', dashboard_handler)
    app.router.add_get('/data', data_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', Config.WEB_PORT)
    await site.start()
    logger.info(f"Web dashboard running on port {Config.WEB_PORT}")

# ------------------------------------------------------------------------------
# 7b. DASHBOARD WITH RICH
# ------------------------------------------------------------------------------
class Dashboard:
    def __init__(self):
        layout = Layout()
        layout.split_column(Layout(name="hdr",size=3),Layout(name="main",ratio=1),Layout(name="ftr",size=3))
        layout["main"].split_row(Layout(name="left"),Layout(name="right"))
        self.layout = layout

    def render(self, equity, pnl, wins, losses, open_pos):
        tbl = Table(expand=True)
        for k,v in [("Equity",f"${equity:,.2f}"),("P&L",f"${pnl:,.2f}"),("Wins",str(wins)),
                    ("Losses",str(losses)),("Open Pos",str(open_pos))]:
            tbl.add_row(k,v)
        self.layout["hdr"].update(Panel("[b]Advanced Crypto Bot Dashboard[/b]"))
        self.layout["left"].update(tbl)
        self.layout["right"].update(Panel(f"Last Update:\n{datetime.now(timezone.utc).isoformat()}"))
        self.layout["ftr"].update(Panel("[green]Running...[/green]"))
        return self.layout

# ------------------------------------------------------------------------------
# 8. MAIN LOOP
# ------------------------------------------------------------------------------
async def main_loop():
    # initial data & model setup
    exchange = ccxt.binance({'enableRateLimit':True})
    exchange.set_sandbox_mode(True)
    df0 = await safe_fetch_ohlcv(exchange, Config.SYMBOL, Config.TIMEFRAME, Config.LOOKBACK)
    df0 = preprocess(df0)
    X0,y0 = prepare_ml_data(df0)
    best = optimize_hyperparams(X0,y0)
    xgb_params = {
        'eta':float(best['eta']),'max_depth':int(best['md']),
        'subsample':float(best['ss']),'colsample_bytree':float(best['cb']),
        'objective':'binary:logistic','eval_metric':'auc'
    }
    model = xgb.train(xgb_params, xgb.DMatrix(X0,label=y0), num_boost_round=100)

    trader = AsyncTrader(Config.SIMULATE_MODE, Config.INITIAL_CAPITAL)
    buffer = ExperienceBuffer(Config.EXPERIENCE_BUFFER_SIZE)
    wins=losses=0; equity=Config.INITIAL_CAPITAL; pnl=0.0; itr=0
    dash=Dashboard()

    async with exchange:
        async with trader.exchange:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    # update price and candles
                    await trader.update_last_price()
                    df = await safe_fetch_ohlcv(trader.exchange, Config.SYMBOL, Config.TIMEFRAME, 60)
                    if df is None:
                        await asyncio.sleep(5); continue
                    df = preprocess(df)
                    candle_data.clear()
                    for ts,row in df.iterrows():
                        candle_data.append({'timestamp':ts.isoformat(),
                                            'open':row['open'],'high':row['high'],
                                            'low':row['low'],'close':row['close']})
                    # predict & trade
                    last=df.iloc[-1]
                    feats=np.array([[last['rsi'],last['macd'],last['macd_sig'],
                                     last['bb_high'],last['bb_low'],
                                     last['atr'],last['hl_range'],last['oc_change']]])
                    prob=model.predict(xgb.DMatrix(feats))[0]
                    sig=1 if prob>0.5 else -1
                    size=calculate_position_size(equity,trader.last_price,
                                                  Config.MAX_POSITION_RISK,last['atr'])
                    order=None
                    if sig==1 and trader.position==0:
                        order=await trader.place_order('buy',size)
                    elif sig==-1 and trader.position>0:
                        order=await trader.place_order('sell',trader.position)
                    if order and order.get('side')=='sell':
                        buy_p, sell_p = order['buy_price'], order['price']
                        label=int(sell_p>buy_p)
                        wins+=label; losses+=1-label
                        batch_X,batch_y=buffer.add(feats.flatten(),label)
                        if batch_y is not None:
                            model=xgb.train(xgb_params,
                                            xgb.DMatrix(batch_X,label=batch_y),
                                            num_boost_round=Config.INCREMENTAL_TRAIN_ROUNDS,
                                            xgb_model=model)
                            logger.info(f"Incremental train on {len(batch_y)} samples")
                    # equity & pnl
                    equity=(trader.balances['USDT']+trader.balances['BTC']*trader.last_price
                            if Config.SIMULATE_MODE
                            else Config.INITIAL_CAPITAL+trader.position*trader.last_price)
                    pnl=equity-Config.INITIAL_CAPITAL
                    # periodic retrain
                    itr+=1
                    if itr % Config.RETRAIN_INTERVAL == 0:
                        logger.info("Scheduled retrain...")
                        # prepara dados de retrain
                        Xr, yr = prepare_ml_data(df)
                        dtrain_r = xgb.DMatrix(Xr, label=yr)
                        # re-treina incrementalmente sobre o modelo existente
                        model = xgb.train(
                            xgb_params,
                            dtrain_r,
                            num_boost_round=100,
                            xgb_model=model
                        )
                        send_slack_alert("Scheduled retrain complete.")
                    # update rich dashboard
                    live.update(dash.render(equity,pnl,wins,losses,int(trader.position>0)))
                    await asyncio.sleep(60)

# ------------------------------------------------------------------------------
# 9. RUN BOTH BOT AND WEB DASHBOARD
# ------------------------------------------------------------------------------
async def run_all():
    web_task = asyncio.create_task(start_web_dashboard())
    bot_task = asyncio.create_task(main_loop())
    await asyncio.gather(web_task, bot_task)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',action='store_true')
    args=p.parse_args()
    if args.start:
        try:
            asyncio.run(run_all())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.exception(f"Error: {e}")
            send_slack_alert(f"Bot crashed: {e}")
    else:
        print("Use --start to run the bot")
