#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MASTER L/S QUANT BOT — hourly-capable, fast, risk-first, 50% target

Core design:
- Universe: ~200 liquid US names (broad large-cap stocks + AI/semis + factor/sector/broad ETFs).
- Signals (rank blend): 6m(12-1) momentum, MA(50/200) distances, MACD histogram, RSI preference (~55), volatility penalty.
- Filters: 200d trend filter (price>200d for longs, price<200d for shorts). Optional ETF-aware shorts toggle.
- Sizing: Inverse-vol per-name with caps. Portfolio-level volatility targeting around 35% annualized to
          support a 50% return target under good Sharpe (this is not a promise).
- Risk overlays:
    • Regime/VIX circuit-breakers to scale gross and pause adds under panic.
    • Trailing-drawdown kill switch + step-down scaling.
    • Per-position ATR stop advisory; time-stop advisory; gap-risk guard for fresh adds.
    • Max per-order notional; min-$ trade; buy-power (BP) buffer; optional market-neutral toggle.
- Broker: Alpaca PAPER only (DRY_RUN if keys missing/unauthorized). Basic order plumbing (2-phase: reduce risk, then add).
- Metrics: fast proxy backtest (daily rebalance heuristic) → Sharpe, Sortino, Vol, CAGR, MaxDD, Calmar.
- Safety & hygiene: solid auth preview & key redaction; ticker remaps; excludes known delists/mergers.

EDUCATIONAL USE ONLY. PAPER TRADING ONLY. NO FINANCIAL ADVICE.
"""

# ============================= Stable bootstrap =============================
import sys, subprocess, importlib

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])

def _bootstrap_stack():
    """Install compatible stack if imports fail (keeps it quick & stable)."""
    try:
        import alpaca_trade_api, yfinance, feedparser  # noqa: F401
    except Exception:
        # Pin a conservative HTTP stack that plays nicely with alpaca & yfinance
        _pip("six==1.16.0", "urllib3==1.26.18", "requests<2.32.0")
        _pip("alpaca-trade-api==3.2.0", "yfinance>=0.2.40", "feedparser", "python-dotenv")
    # purge half-loaded modules so imports reinitialize cleanly
    for m in ["urllib3","urllib3.util","urllib3.packages","urllib3.packages.six",
              "six","requests","alpaca_trade_api","yfinance","feedparser"]:
        if m in sys.modules: del sys.modules[m]
    importlib.invalidate_caches()

_bootstrap_stack()

# ============================= Imports & setup ===============================
import os, time, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
import feedparser
from alpaca_trade_api.rest import REST, APIError

# ============================= Keys (paste here) =============================
# Fill these with your PAPER keys. If you prefer env vars, set DIRECT_KEYS=None.
DIRECT_KEYS = {
    "APCA_API_KEY_ID":     "",
    "APCA_API_SECRET_KEY": "",
    "APCA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
}

# ============================= Config knobs ==================================
# Targets (portfolio-level)
TARGET_ANNUAL_RETURN     = 0.50         # aspirational; guided by vol targeting & Sharpe, not a promise
TARGET_ANNUAL_VOL        = 0.35         # key lever for gross scaling (try 0.30–0.45)
RISK_FREE_ANNUAL         = 0.03

# Portfolio & execution
PORTFOLIO_DOLLARS        = 100_000
TOP_N_LONG               = 30
TOP_N_SHORT              = 10
INCLUDE_ETFS_IN_SHORTS   = True         # allow shorting ETFs in the short leg selection
MARKET_NEUTRAL           = False        # if True: match |short gross| to long gross

# Gross exposures (pre-scaling; adjusted by risk + sentiment + vol-target)
LONG_GROSS               = 1.10
SHORT_GROSS              = 0.40

# Data windows
LOOKBACK_YEARS           = 1.5          # ~18m for daily signals; auto-tunes off holding-period input
VOL_WINDOW_DAYS          = 20
BACKTEST_DAYS            = 252          # 1y fast proxy (static selection heuristic)

# Per-name limits & trade plumbing
MAX_W_PER_NAME_LONG      = 0.08
MAX_W_PER_NAME_SHORT     = 0.04
MIN_DOLLAR_TRADE         = 200
MAX_NOTIONAL_PER_ORDER   = 12_500
BUYING_POWER_BUFFER      = 0.995
WAIT_AFTER_SELLS_S       = 6
SLEEP_BETWEEN_ORDERS_S   = 0.10
FORCE_TRADE_IF_CLOSED    = True         # still send orders in paper even if market is closed

# Risk overlays & guards
ENABLE_SENTIMENT         = True
SENTIMENT_REGION         = "US/Global"  # or "India", "Singapore" (uses local indices/VIX)
VIX_PANIC_LEVEL          = 30.0         # above → scale down adds; tilt defensive
VIX_HALT_LEVEL           = 40.0         # above → halt new adds; only risk-reducing
DRAWDOWN_KILL_SWITCH     = 0.12         # if trailing equity DD > 12% → near-flat
STEPDOWN_DD1, SCALE1     = 0.06, 0.65   # >6% DD → scale to 65%
STEPDOWN_DD2, SCALE2     = 0.09, 0.45   # >9% DD → scale to 45%

# Stops/time-stops/gap guard (advisories; ATR used for guidance)
ATR_WINDOW               = 14
ATR_STOP_MULT            = 3.0          # per-position stop ≈ 3*ATR from entry (advisory)
TIME_STOP_DAYS           = 20           # exit if stale after ~1 trading month (advisory)
GAP_GUARD_PCT            = 0.12         # skip new adds when last day move > 12% abs

# DRY-RUN flag auto-set if keys invalid/missing
DRY_RUN = False

# Known ticker fixes & excludes
# yfinance historical symbol remaps
TICKER_REMAP = {"FISV": "FI"}           # Fiserv → FI
EXCLUDE_TICKERS = {"ATVI", "PXD"}       # delisted/merged

# Broker symbol mapping (Alpaca uses dot for class B in BRK)
BROKER_TICKER_MAP = {"BRK-B": "BRK.B"}

def _to_broker(sym: str) -> str:
    return BROKER_TICKER_MAP.get(sym, sym)

# ============================= Universe ======================================
STOCKS_ALL = [
    # --- Mega & Large Cap Tech ---
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","ADBE","ORCL","CRM",
    "INTC","AMD","QCOM","CSCO","TXN","IBM","NFLX","SNOW","PANW","CRWD","ANET",
    "SMCI","S","CDNS","NTNX","HPE","UPST","MU","DOX",

    # --- AI, Robotics & Semiconductors ---
    "AI","BBAI","SOUN","IRBO","ROBO","ROBT","AIQ","BOTZ","SOXQ","CHPS","FDN",

    # --- Financials & Fintech ---
    "JPM","BAC","C","GS","MS","WFC","SCHW","SOFI","PYPL","BRK-B","AXP","COIN",

    # --- Energy, Commodities, Green Tech ---
    "XOM","CVX","SLB","COP","EOG","VLO","MPC","PSX","VGAS","CEG","FSLR","BHP","BTG","LAC","LTHM",

    # --- Healthcare / Pharma ---
    "UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","PFE","BMY","MDT","EHC","IFRX","TGTX",

    # --- Industrials & Materials ---
    "CAT","DE","HON","GE","LMT","RTX","BA","ETN","EMR","UPS","FDX","WM",
    "NUE","LIN","APD","SHW","CMI","ITW","GD",

    # --- Consumer / Retail ---
    "PG","KO","PEP","COST","WMT","MCD","SBUX","HD","LOW","NKE","TGT","EL","PM","CMG",

    # --- Discretionary / Misc ---
    "MA","V","ABNB","MAR","LYV","BKNG","RCL",

    # --- Telecoms & Media ---
    "T","VZ","TMUS","DIS","CMCSA","CHTR",

    # --- Other Sectors / Misc Stocks ---
    "REGN","VRTX","ZTS","IDXX","ISRG","SYK","GILD","IQV","HCA"
]

ETFS_ALL = [
    # --- Broad Index ETFs ---
    "SPY","VOO","IVV","VTI","SCHB","QQQ","IWM","DIA",
    # --- Sector ETFs ---
    "XLF","XLK","XLY","XLE","XLV","XLP","XLI","XLB","XLRE","XLC",
    # --- Factor / Value-Growth ETFs ---
    "VUG","VTV","VOE","VB","VO","QUAL","USMV","VLUE","SIZE","IWF","IWD",
    # --- Thematic / AI / Robotics ---
    "ARKK","AIQ","IRBO","ROBO","ROBT","BOTZ","SOXQ","CHPS","FDN","VGT",
    # --- Global & Bonds & Alts ---
    "EFA","IEFA","EEM","EMXC","VEA","VWO","BND","AGG","IEF","TLT",
    "HYG","LQD","IAU","GLD","SLV","DBC","FXI","PNQI"
]

# Build UNIVERSE cleanly (apply remaps & excludes; preserve order; drop dups)
UNIVERSE = []
_seen = set()
for t in (STOCKS_ALL + ETFS_ALL):
    if t in EXCLUDE_TICKERS:
        continue
    t2 = TICKER_REMAP.get(t, t)
    if t2 not in _seen:
        UNIVERSE.append(t2)
        _seen.add(t2)

# Helper ETF lookup for fast membership checks
ETF_SET = set(ETFS_ALL)

# ============================= Utilities & TA ================================
def _redact(s, keep=4):
    if not s: return "<none>"
    s = str(s).strip()
    return "*"*(len(s)-keep) + s[-keep:] if len(s) > keep else "*"*len(s)

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/period, adjust=False).mean()
    rd = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100/(1+rs))

def macd_hist(series, fast=12, slow=26, signal=9):
    ef, es = ema(series, fast), ema(series, slow)
    line = ef - es
    return line - ema(line, signal)

def zscore(x: pd.Series) -> pd.Series:
    out = (x - x.mean()) / (x.std() + 1e-12)
    return out.replace([np.inf,-np.inf], 0).fillna(0)

def dl_prices(tickers, years=LOOKBACK_YEARS):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365*years*1.1))
    df = yf.download(" ".join(tickers), start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, group_by="ticker")
    if df.empty:
        raise SystemExit("No price data downloaded. Try again later.")
    close, high, low, dropped = {}, {}, {}, []
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            c = sub["Close"].rename(t)
            h = sub["High"].rename(t)
            l = sub["Low"].rename(t)
            if c.dropna().empty:
                dropped.append(t); continue
            close[t], high[t], low[t] = c, h, l
        except Exception:
            dropped.append(t)
    if dropped:
        print(f"⚠️ Dropping {len(dropped)} tickers with missing data: {dropped[:12]}{' ...' if len(dropped)>12 else ''}")
    px  = pd.DataFrame(close).sort_index().ffill().dropna(axis=1, how="any")
    hi  = pd.DataFrame(high).reindex(px.index).ffill()[px.columns]
    lo  = pd.DataFrame(low).reindex(px.index).ffill()[px.columns]
    return px, hi, lo

def atr_from_hlc(h, l, c, n=ATR_WINDOW):
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def realized_vol(prices: pd.DataFrame, window=VOL_WINDOW_DAYS) -> pd.Series:
    rets = prices.pct_change()
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252)
    # fill outliers and NA with median of valid vols
    return vol.replace([np.inf,-np.inf], np.nan).fillna(vol.median())

def technical_score(prices: pd.DataFrame):
    """
    Blend of momentum, trend, MACD, RSI preference and volatility penalty.
    Returns: (score Series desc, sma200 Series)
    """
    last   = prices.iloc[-1]
    rets   = prices.pct_change()
    mom6_1 = prices.pct_change(126).shift(21).iloc[-1]           # 6m momentum, skip last month
    sma50  = prices.rolling(50).mean().iloc[-1]
    sma200 = prices.rolling(200).mean().iloc[-1]
    dist50 = (last - sma50)  / (sma50  + 1e-12)
    dist200= (last - sma200) / (sma200 + 1e-12)
    macdh  = prices.apply(macd_hist).iloc[-1]
    # prefer RSI around ~55 (mild positive trend with headroom)
    rsi14  = prices.apply(rsi).iloc[-1]
    rsi_pref = -((rsi14 - 55.0).abs())
    vol20  = rets.rolling(VOL_WINDOW_DAYS).std().iloc[-1] * np.sqrt(252)
    score = (
        0.35*zscore(mom6_1) +
        0.20*zscore(dist200) +
        0.15*zscore(dist50)  +
        0.20*zscore(macdh)   +
        0.10*zscore(rsi_pref) -
        0.15*zscore(vol20)
    ).sort_values(ascending=False)
    return score, sma200

# ============================= Regime/Sentiment ==============================
def market_sentiment(region="US/Global"):
    if region == "India":
        idx1, idx2, vol = "^NSEI","^NSEBANK","^INDIAVIX"
    elif region == "Singapore":
        idx1, idx2, vol = "^STI","^HSI","^VIX"
    else:
        idx1, idx2, vol = "^GSPC","^NDX","^VIX"
    packs = {k: yf.Ticker(k).history(period="6mo") for k in [idx1, idx2, vol]}
    def rr(df, n):
        try:
            return float(df["Close"].iloc[-1]/df["Close"].iloc[-n]-1) if len(df) > n else 0.0
        except Exception: return 0.0
    r1m = rr(packs[idx1], 21); r3m = rr(packs[idx2], 63)
    vix = float(packs[vol]["Close"].iloc[-1]) if not packs[vol].empty else 20.0
    # normalize to 0..1 (rough scaling)
    def nret(x): return 0.5 + x/0.20  # +/-10% → ~0..1
    vlow, vhigh = (14, 30)
    if vix <= vlow: vscore = 1.0
    elif vix >= vhigh: vscore = 0.2
    else: vscore = 1.0 - (vix - vlow) * (0.8/(vhigh - vlow))
    score = max(0, min(1, 0.65*((nret(r1m)+nret(r3m))/2.0) + 0.35*vscore))
    return score, vix

# ============================= Metrics & Backtest ============================
def perf_metrics(series_returns: pd.Series, rf_annual=RISK_FREE_ANNUAL):
    """Input: daily strategy returns (not cumulative)."""
    r = series_returns.dropna()
    if r.empty:
        return dict(days=0, vol=0, sharpe=0, sortino=0, cagr=0, maxdd=0, calmar=0)
    ann_fac = 252
    mu = r.mean()*ann_fac
    vol = r.std() * math.sqrt(ann_fac)
    downside = r[r < 0].std() * math.sqrt(ann_fac)
    sharpe = (mu - rf_annual) / (vol + 1e-12)
    sortino= (mu - rf_annual) / (downside + 1e-12)
    # equity curve
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq/peak - 1.0)
    maxdd = dd.min()
    years = len(r)/ann_fac
    cagr = eq.iloc[-1]**(1/years) - 1 if years > 0 else 0
    calmar = cagr/abs(maxdd) if maxdd < 0 else np.nan
    return dict(days=len(r), vol=vol, sharpe=sharpe, sortino=sortino, cagr=cagr, maxdd=maxdd, calmar=calmar)

def proxy_backtest(prices: pd.DataFrame, long_names, short_names,
                   vol_series: pd.Series, sma200: pd.Series,
                   is_etf_map: dict, hold_days=1):
    """
    Fast daily-rebalance proxy:
    - Uses *today's* selections and inverse-vol weights as static weights across the last BACKTEST_DAYS.
    - Applies 200d trend filter on the final selections.
    - Returns: (daily returns series, weights Series, long_final list, short_final list)
    """
    rets = prices.pct_change().fillna(0)
    cols = prices.columns.tolist()

    # static inverse-vol weights helper
    def inv_vol_w(names, cap, gross):
        if not names: return pd.Series(dtype=float)
        w = (1.0 / vol_series.reindex(names)).replace([np.inf,-np.inf], np.nan).fillna(0)
        if w.sum() <= 0: w = pd.Series(1/len(names), index=names)
        else:            w = w / w.sum()
        w = w.clip(upper=cap)
        return (w * (gross / (w.sum() + 1e-12))).reindex(cols).fillna(0)

    # Trend filter (live) on the candidate lists
    last_close = prices.iloc[-1]
    long_mask  = last_close.reindex(long_names)  > sma200.reindex(long_names)
    short_mask = last_close.reindex(short_names) < sma200.reindex(short_names)

    long_final  = list(pd.Index(long_names)[long_mask.fillna(False)])
    short_final = list(pd.Index(short_names)[short_mask.fillna(False)])

    wL = inv_vol_w(long_final,  MAX_W_PER_NAME_LONG,  LONG_GROSS)
    wS = inv_vol_w(short_final, MAX_W_PER_NAME_SHORT, -SHORT_GROSS)
    weights = (wL + wS)

    # optional market-neutral adjustment (match gross)
    if MARKET_NEUTRAL:
        long_gross  = float((weights.clip(lower=0)).sum())
        short_gross = float((-weights.clip(upper=0)).sum())
        if long_gross > 1e-9 and short_gross > 1e-9:
            scale = long_gross / short_gross
            weights[weights < 0] *= scale

    # Apply static weights across last BACKTEST_DAYS (fast proxy)
    window = min(BACKTEST_DAYS, len(rets))
    r_win = rets.iloc[-window:]
    port_r = (r_win * weights).sum(axis=1)

    return port_r, weights, long_final, short_final

# ============================= Broker & orders ===============================
APCA_DEFAULT_BASE = "https://paper-api.alpaca.markets"

class _DummyAPI:
    def __getattr__(self, _):
        def _noop(*a, **k): raise RuntimeError("DRY_RUN: set valid Alpaca PAPER keys.")
        return _noop

def _load_keys():
    def clean(x):
        if x is None: return None
        x = str(x).strip()
        return x if x else None
    if isinstance(DIRECT_KEYS, dict):
        k = clean(DIRECT_KEYS.get("APCA_API_KEY_ID"))
        s = clean(DIRECT_KEYS.get("APCA_API_SECRET_KEY"))
        b = clean(DIRECT_KEYS.get("APCA_PAPER_BASE_URL")) or APCA_DEFAULT_BASE
    else:
        k = clean(os.environ.get("APCA_API_KEY_ID"))
        s = clean(os.environ.get("APCA_API_SECRET_KEY"))
        b = clean(os.environ.get("APCA_PAPER_BASE_URL")) or APCA_DEFAULT_BASE
    return k, s, b

def connect_alpaca():
    global DRY_RUN
    key, sec, base = _load_keys()
    print("\n[Alpaca auth]")
    print(f"  Base : {base}")
    print(f"  Key  : {_redact(key)}")
    if not key or not sec:
        DRY_RUN = True
        print("⚠️ No keys → DRY_RUN=True (signals only).")
        return _DummyAPI(), PORTFOLIO_DOLLARS, False
    if "paper" not in base:
        print("⚠️ Forcing PAPER URL…")
        base = APCA_DEFAULT_BASE
    try:
        api = REST(key_id=key, secret_key=sec, base_url=base)
        acct = api.get_account()
        equity = float(getattr(acct, "equity", PORTFOLIO_DOLLARS))
        try: is_open = bool(api.get_clock().is_open)
        except Exception: is_open = False
        print(f"  Status: {getattr(acct, 'status', 'unknown')} | Equity ${equity:,.2f} | Open? {is_open}")
        return api, equity, is_open
    except APIError as e:
        DRY_RUN = True
        print(f"❌ APIError: {e} → DRY_RUN=True")
        return _DummyAPI(), PORTFOLIO_DOLLARS, False
    except Exception as e:
        DRY_RUN = True
        print(f"❌ Unexpected auth error: {repr(e)} → DRY_RUN=True")
        return _DummyAPI(), PORTFOLIO_DOLLARS, False

def cancel_open_orders(api):
    if DRY_RUN: return
    try:
        for o in api.list_orders(status="open"):
            try: api.cancel_order(o.id)
            except Exception: pass
        time.sleep(1.0)
    except Exception:
        pass

def refresh_positions_and_bp(api):
    if DRY_RUN: return {}, PORTFOLIO_DOLLARS
    acct = api.get_account()
    bp   = float(acct.buying_power)
    positions = {p.symbol: int(float(p.qty)) for p in api.list_positions()}
    return positions, bp

def is_shortable(api, symbol: str) -> bool:
    if DRY_RUN: return True
    try:
        asset = api.get_asset(symbol)
        return bool(getattr(asset, "shortable", False))
    except Exception:
        return False

def dollars_to_shares(target_dollars: float, price: float) -> int:
    if price <= 0: return 0
    capped = np.sign(target_dollars) * min(abs(target_dollars), MAX_NOTIONAL_PER_ORDER)
    if abs(capped) < MIN_DOLLAR_TRADE: return 0
    qty = int(math.floor(abs(capped) / price))
    return qty if target_dollars > 0 else -qty

# ============================= Inputs & scaling ==============================
def get_user_inputs():
    print("\n=== STRATEGY SETUP (hourly-ready) ===")
    try:
        hold_input = input("Holding period months (1/3/6/12/24) [6]: ").strip()
        hold_months = int(hold_input) if hold_input else 6
        if hold_months not in [1,3,6,12,24]: hold_months = 6
    except Exception:
        hold_months = 6
    try:
        risk_input = input("Risk (low/medium/high) [medium]: ").strip().lower()
        if risk_input not in ["low","medium","high"]: risk_input = "medium"
    except Exception:
        risk_input = "medium"
    risk_map = {"low":(0.75,0.25), "medium":(1.10,0.40), "high":(1.35,0.55)}
    gL, gS = risk_map[risk_input]
    print(f"→ Hold={hold_months}m | Risk={risk_input.upper()} → LONG_GROSS={gL} SHORT_GROSS={gS}")
    return hold_months, risk_input, gL, gS

def vol_target_scale(realized_annual_vol, target_annual_vol=TARGET_ANNUAL_VOL):
    if realized_annual_vol <= 1e-6: return 1.0
    raw = target_annual_vol / realized_annual_vol
    return max(0.4, min(1.8, raw))  # safety clamp

# ============================= Main =========================================
def main():
    global LONG_GROSS, SHORT_GROSS, LOOKBACK_YEARS

    # Inputs
    hold_months, risk_level, gL, gS = get_user_inputs()
    LONG_GROSS, SHORT_GROSS = gL, gS
    LOOKBACK_YEARS = 1.0 if hold_months <= 3 else (1.5 if hold_months <= 6 else 2.0)

    # 1) Data
    px, hi, lo = dl_prices(UNIVERSE, LOOKBACK_YEARS)
    if px.shape[0] < 220:
        raise SystemExit("Not enough history (~220 days). Try later.")
    last = px.iloc[-1]

    # 2) Signals & filters
    tscore, sma200 = technical_score(px)
    is_etf = {t: (t in ETF_SET) for t in px.columns}
    vol = realized_vol(px, VOL_WINDOW_DAYS)

    # selection (pre-filter)
    longs_raw  = list(tscore.index[:TOP_N_LONG])
    shorts_raw = [t for t in tscore.index[::-1] if (INCLUDE_ETFS_IN_SHORTS or not is_etf.get(t, False))]
    shorts_raw = shorts_raw[:TOP_N_SHORT]

    # apply trend filter (live)
    last_close = last
    long_names  = [t for t in longs_raw  if last_close.get(t, np.nan) > sma200.get(t, np.nan)]
    short_names = [t for t in shorts_raw if last_close.get(t, np.nan) < sma200.get(t, np.nan)]

    # 3) Regime controls (sentiment + VIX)
    LONG_scale = SHORT_scale = 1.0
    vix = 20.0
    if ENABLE_SENTIMENT:
        sent, vix = market_sentiment(SENTIMENT_REGION)
        scale = 0.6 + 0.8 * sent
        LONG_scale  = max(0.4, min(1.4, scale))
        SHORT_scale = max(0.4, min(1.4, (2.0 - scale)))
    if vix >= VIX_PANIC_LEVEL:
        LONG_scale *= 0.75; SHORT_scale *= 1.10
    if vix >= VIX_HALT_LEVEL:
        LONG_scale *= 0.50; SHORT_scale *= 1.20

    # 4) Proxy backtest + portfolio vol targeting
    port_r, weights0, long_final, short_final = proxy_backtest(
        px, long_names, short_names, vol, sma200, is_etf, hold_days=1
    )
    # realized vol (annual) from proxy
    realized_vol_ann = port_r.std() * math.sqrt(252)
    vt_scale = vol_target_scale(realized_vol_ann, TARGET_ANNUAL_VOL)

    # drawdown step-downs (equity from proxy)
    eq = (1 + port_r).cumprod()
    peak = eq.cummax()
    trailing_dd = float((eq.iloc[-1]/peak.iloc[-1]) - 1.0)
    gross_scale_dd = 1.0
    if trailing_dd <= -STEPDOWN_DD1: gross_scale_dd = min(gross_scale_dd, SCALE1)
    if trailing_dd <= -STEPDOWN_DD2: gross_scale_dd = min(gross_scale_dd, SCALE2)
    if trailing_dd <= -DRAWDOWN_KILL_SWITCH: gross_scale_dd = 0.10  # near-flat

    gross_scale_total = vt_scale * gross_scale_dd
    LONG_scale  *= gross_scale_total
    SHORT_scale *= gross_scale_total

    # 5) Final weights (inverse vol, capped)
    def inv_vol_w(names, cap, gross):
        if not names: return pd.Series(dtype=float)
        w = (1.0 / vol.reindex(names)).replace([np.inf,-np.inf], np.nan).fillna(0)
        if w.sum() <= 0: w = pd.Series(1/len(names), index=names)
        else:            w = w / w.sum()
        w = w.clip(upper=cap)
        return w * (gross / (w.sum() + 1e-12))

    wL = inv_vol_w(long_final,  MAX_W_PER_NAME_LONG,  LONG_GROSS * LONG_scale)
    wS = inv_vol_w(short_final, MAX_W_PER_NAME_SHORT, SHORT_GROSS * SHORT_scale)
    weights_live = (wL.reindex(px.columns).fillna(0) + (-wS.reindex(px.columns).fillna(0)))  # shorts negative

    # optional market-neutral adjustment on LIVE weights
    if MARKET_NEUTRAL:
        long_gross  = float(weights_live.clip(lower=0).sum())
        short_gross = float((-weights_live.clip(upper=0)).sum())
        if long_gross > 1e-9 and short_gross > 1e-9:
            scale = long_gross / short_gross
            weights_live[weights_live < 0] *= scale

    # 6) Live metrics summary (proxy backtest + “recent/live” slice)
    back = perf_metrics(port_r)
    recent_window = min(60, len(port_r))
    live = perf_metrics(port_r.iloc[-recent_window:]) if recent_window > 10 else perf_metrics(pd.Series(dtype=float))
    print("\n=== METRICS (proxy daily rebalance) ===")
    print(f"Backtest Days: {back['days']} | Vol: {back['vol']*100:5.2f}% | Sharpe: {back['sharpe']:.2f} | Sortino: {back['sortino']:.2f} | MaxDD: {back['maxdd']*100:5.2f}% | CAGR: {back['cagr']*100:5.2f}%")
    print(f"Recent  Days: {live['days']} | Vol: {live['vol']*100:5.2f}% | Sharpe: {live['sharpe']:.2f} | Sortino: {live['sortino']:.2f}")
    print(f"Regime   VIX: {vix:.2f} | VolTarget scale: {vt_scale:.2f} | DD scale: {gross_scale_dd:.2f}")

    # 7) Broker connect & positions (orders guarded by DRY_RUN/market hours)
    api, equity, is_open = connect_alpaca()
    if equity <= 0: equity = PORTFOLIO_DOLLARS
    cancel_open_orders(api)
    positions, buying_power = refresh_positions_and_bp(api)

    # 8) Risk guards for live entries: GAP guard & no new adds in halt condition
    def skip_new_adds(sym):
        p_hist = px[sym].iloc[-2:]
        if len(p_hist) < 2: return True
        gap = abs(p_hist.iloc[-1]/p_hist.iloc[-2] - 1.0)
        return gap > GAP_GUARD_PCT or (vix >= VIX_HALT_LEVEL)

    # 9) Targets → shares
    targets_usd = (weights_live * equity).to_dict()
    desired = {}
    for sym, dollars in targets_usd.items():
        price = float(last.get(sym, np.nan))
        if np.isnan(price): continue
        if dollars > 0 and skip_new_adds(sym):
            continue  # skip fresh long adds on large gap or VIX halt
        if dollars < 0 and skip_new_adds(sym):
            continue  # skip fresh short adds similarly
        qty = dollars_to_shares(dollars, price)
        if qty != 0:
            desired[_to_broker(sym)] = qty  # map to broker symbol if needed

    # flatten anything currently held but not in desired set
    target_set = set(desired.keys())
    for sym, curr in list(positions.items()):
        if sym not in target_set:
            desired.setdefault(sym, 0)

    # 10) Stops & time stops (advisory)
    atr_map = {}
    for t in list(set(long_final + short_final)):
        try:
            series_atr = atr_from_hlc(hi[t], lo[t], px[t], ATR_WINDOW)
            atr_map[t] = float(series_atr.iloc[-1])
        except Exception:
            atr_map[t] = None

    # 11) Phase 1: risk reductions first (sell/cover)
    phase1 = []
    for sym, curr in positions.items():
        want = desired.get(sym, 0)
        delta = want - curr
        if delta < 0:
            phase1.append((sym, abs(delta)))
    if not DRY_RUN and phase1 and (is_open or FORCE_TRADE_IF_CLOSED):
        print(f"\nPhase 1: SELL/COVER {len(phase1)} orders…")
        sent=errs=0
        for sym, qty in phase1:
            try:
                api.submit_order(symbol=sym, side="sell", qty=int(qty), type="market", time_in_force="day")
                sent += 1; time.sleep(SLEEP_BETWEEN_ORDERS_S)
            except Exception as e:
                print("  Sell/Cover error:", sym, repr(e)); errs+=1
        print(f"Phase 1 done. Sent={sent} | Errors={errs}")
    else:
        print("\nPhase 1: none (or DRY_RUN).")

    if not DRY_RUN:
        time.sleep(WAIT_AFTER_SELLS_S)
    positions, buying_power = refresh_positions_and_bp(api)

    # 12) Phase 2: buys/new shorts (BP-aware scaling)
    orders, need = [], 0.0
    for sym, want in desired.items():
        curr = positions.get(sym, 0)
        delta = want - curr
        if delta == 0: continue
        side = "buy" if delta > 0 else "sell"
        # fetch a price from last close map; if missing, skip
        base_sym = sym.replace("BRK.B", "BRK-B")  # for reporting/lookup in 'last'
        ref_price_sym = base_sym if base_sym in last.index else sym
        price = float(last.get(ref_price_sym, np.nan))
        if np.isnan(price): continue
        qty = abs(int(delta))
        orders.append((sym, side, qty, price))
        need += price * qty

    if not DRY_RUN and orders and (is_open or FORCE_TRADE_IF_CLOSED):
        cap = buying_power * BUYING_POWER_BUFFER
        scale = 1.0 if need <= cap + 1e-6 else max(0.0, cap/(need + 1e-9))
        print(f"\nPhase 2: BP=${buying_power:,.2f} | Need≈${need:,.2f} | scale={scale:.3f}")
        sent=errs=0
        for sym, side, qty, price in orders:
            adj = int(max(0, math.floor(qty * scale)))
            if adj <= 0: continue
            try:
                api.submit_order(symbol=sym, side=side, qty=adj, type="market", time_in_force="day")
                sent += 1; time.sleep(SLEEP_BETWEEN_ORDERS_S)
            except Exception as e:
                print("  Order error:", sym, repr(e)); errs+=1
        print(f"Phase 2 done. Sent={sent} | Errors={errs}")
    else:
        print("\nPhase 2: none (or DRY_RUN).")

    # 13) Summary + advisory stops
    positions_after, bp_after = refresh_positions_and_bp(api)
    print("\n================ SUMMARY ================")
    print(f"Universe used: {len(px.columns)}")
    print(f"Longs  ({len([w for w in weights_live[weights_live>0].index])}): "
          f"{dict(weights_live[weights_live>0].round(4).sort_values(ascending=False)[:10])}")
    print(f"Shorts ({len([w for w in weights_live[weights_live<0].index])}): "
          f"{dict((-weights_live[weights_live<0]).round(4).sort_values(ascending=False)[:10])}")
    print(f"Positions held: {len([q for q in positions_after.values() if q!=0])}")
    print(f"Buying power : ${bp_after:,.2f}")

    print("\nPer-name ATR stops (advisory):")
    for s in list(weights_live[weights_live!=0].index)[:10]:
        p = float(last.get(s, np.nan)); a = atr_map.get(s)
        if not np.isnan(p) and a:
            stop = p - ATR_STOP_MULT*a if weights_live[s] > 0 else p + ATR_STOP_MULT*a
            side = "LONG" if weights_live[s] > 0 else "SHORT"
            print(f"  {s}: price={p:.2f} | ATR≈{a:.2f} | {side} stop≈{stop:.2f}")

    # 14) News heads (optional quick read)
    try:
        for s in list(weights_live.abs().sort_values(ascending=False).index)[:5]:
            url = f"https://news.google.com/rss/search?q={s}+when:7d&hl=en-US&gl=US&ceid=US:en"
            fp = feedparser.parse(url)
            heads = [getattr(e, "title", "").strip() for e in fp.entries[:2] if getattr(e, "title", "").strip()]
            if heads:
                print(f"\nNews {s}:")
                for t in heads: print(f"  - {t}")
            time.sleep(0.05)
    except Exception:
        pass

    print("\nNotes:")
    print("- Volatility targeting + step-downs aim to keep realized vol near target for compounding.")
    print("- Trend filters, VIX guards, gap guard, ATR/time-stops reduce left-tail risk.")
    print("- Proxy backtest is a fast heuristic (static selection); use it to monitor regime/health.")
    print("- Re-run hourly; code is stateless and adjusts exposures under risk budget.")
    print("- PAPER ONLY. Past performance metrics are not predictive.")

if __name__ == "__main__":
    main()
