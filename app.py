"""
ADAMILL SNIPER — Streamlit + MetaTrader 5 starter application
Educational trading research tool. It does NOT place trades.
Live MT5 data requires the official MetaTrader5 Python package and a logged-in
MT5 desktop terminal on Windows. Economic calendar alerts require a provider/API.
"""
from __future__ import annotations

import os
import math
import html
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

st.set_page_config(
    page_title="ADAMILL SNIPER",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {background:#0d0714;color:#f4f0fa}
[data-testid="stSidebar"] {background:#120a1f}
div[data-testid="stMetric"] {background:#160d23;border:1px solid #4a1d96;padding:12px;border-radius:10px}
.signal-card {background:linear-gradient(135deg,#1b1b2f,#161b22);padding:16px;border-radius:12px;border:1px solid #4a1d96}
.small-note {color:#aaa3b8;font-size:0.88rem}
</style>
""", unsafe_allow_html=True)

TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M10": "TIMEFRAME_M10",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}

STRATEGIES = [
    "SMC (proxy rules)",
    "Trend + Pullback",
    "Supply/Demand (proxy rules)",
    "Fibonacci Retracement",
    "Price Action Confluence",
    "BTMM (manual confirmation required)",
    "Multi-strategy Confluence",
]

st.title("🎯 ADAMILL SNIPER")
st.caption("MT5 live-market research • multi-timeframe scanner • chart levels • macro-alert integration")
st.warning(
    "Research/decision-support only. This build does not execute orders and does not claim "
    "that a signal is profitable. A displayed score is a rule-agreement score, NOT a calibrated "
    "probability until a walk-forward backtest has been completed."
)

def mt5_ready() -> tuple[bool, str]:
    if mt5 is None:
        return False, "MetaTrader5 Python package is not installed."
    try:
        if not mt5.initialize():
            return False, f"MT5 initialize failed: {mt5.last_error()}"
        return True, "MT5 connection initialized."
    except Exception as exc:
        return False, f"MT5 connection error: {exc}"

def timeframe_constant(label: str):
    return getattr(mt5, TIMEFRAMES[label]) if mt5 else None

def get_rates(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("Install MetaTrader5 and run the MT5 desktop terminal first.")
    rates = mt5.copy_rates_from_pos(symbol, timeframe_constant(timeframe), 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned for {symbol} / {timeframe}. Check broker symbol and MT5 login.")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["atr"] = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs(),
    ], axis=1).max(axis=1).rolling(14).mean()
    out["swing_high"] = out["high"].rolling(20).max().shift(1)
    out["swing_low"] = out["low"].rolling(20).min().shift(1)
    return out

def build_signal(df: pd.DataFrame, strategy: str, rr: float = 2.0) -> dict[str, Any]:
    """
    Transparent baseline rules only. SMC/Supply-Demand/BTMM labels are approximations,
    not full implementations of proprietary or subjective methodologies.
    """
    d = indicators(df).dropna()
    if len(d) < 5:
        raise ValueError("Not enough bars to calculate indicators.")
    last = d.iloc[-1]
    prev = d.iloc[-2]
    close = float(last["close"])
    atr = float(last["atr"])
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError("ATR is not available.")
    buy_votes, sell_votes, reasons = [], [], []

    uptrend = last["ema20"] > last["ema50"] and last["ema50"] > last["ema200"]
    downtrend = last["ema20"] < last["ema50"] and last["ema50"] < last["ema200"]
    if uptrend:
        buy_votes.append("EMA trend bullish")
    if downtrend:
        sell_votes.append("EMA trend bearish")

    if close > last["swing_high"]:
        buy_votes.append("Close broke prior 20-bar high")
    if close < last["swing_low"]:
        sell_votes.append("Close broke prior 20-bar low")

    if last["low"] <= last["ema20"] and close > last["ema20"] and uptrend:
        buy_votes.append("Bullish EMA20 pullback reclaim")
    if last["high"] >= last["ema20"] and close < last["ema20"] and downtrend:
        sell_votes.append("Bearish EMA20 pullback rejection")

    if close > prev["high"] and close > last["ema20"]:
        buy_votes.append("Bullish close beyond previous candle high")
    if close < prev["low"] and close < last["ema20"]:
        sell_votes.append("Bearish close beyond previous candle low")

    # These are rule-agreement scores, deliberately not called probabilities.
    buy_score = round(100 * len(buy_votes) / 4)
    sell_score = round(100 * len(sell_votes) / 4)
    if buy_score >= 50 and buy_score > sell_score:
        direction = "BUY"
        entry = close
        sl = min(float(last["low"]), entry - atr * 1.2)
        if sl >= entry:
            sl = entry - atr * 1.2
        tp = entry + (entry - sl) * rr
        score = buy_score
        reasons = buy_votes
    elif sell_score >= 50 and sell_score > buy_score:
        direction = "SELL"
        entry = close
        sl = max(float(last["high"]), entry + atr * 1.2)
        if sl <= entry:
            sl = entry + atr * 1.2
        tp = entry - (sl - entry) * rr
        score = sell_score
        reasons = sell_votes
    else:
        direction, entry, sl, tp, score = "NO TRADE", close, np.nan, np.nan, max(buy_score, sell_score)
        reasons = ["No sufficiently strong directional agreement from baseline rules."]

    return {
        "direction": direction, "entry": entry, "sl": sl, "tp": tp,
        "score": score, "buy_score": buy_score, "sell_score": sell_score,
        "reasons": reasons, "atr": atr, "bar_time": last["time"],
    }

def draw_chart(df: pd.DataFrame, sig: dict[str, Any], symbol: str, tf: str):
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=symbol,
    )])
    fig.add_trace(go.Scatter(x=df["time"], y=ema(df["close"], 20), name="EMA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["time"], y=ema(df["close"], 50), name="EMA 50", line=dict(width=1)))
    if sig["direction"] in ("BUY", "SELL"):
        for price, label, dash in [
            (sig["entry"], "ENTRY", "solid"),
            (sig["sl"], "STOP LOSS", "dash"),
            (sig["tp"], "TAKE PROFIT", "dot"),
        ]:
            fig.add_hline(y=price, line_dash=dash, annotation_text=f"{label}: {price:.5f}")
    fig.update_layout(
        template="plotly_dark", height=610, xaxis_rangeslider_visible=False,
        title=f"{symbol} • {tf} • {sig['direction']}",
        paper_bgcolor="#0d0714", plot_bgcolor="#0d0714",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig

with st.sidebar:
    st.header("Scanner Settings")
    connect_clicked = st.button("Initialize / refresh MT5 connection", use_container_width=True)
    if connect_clicked:
        ok, message = mt5_ready()
        st.session_state["mt5_status"] = (ok, message)
    status = st.session_state.get("mt5_status", (False, "Not connected in this session."))
    st.caption(("🟢 " if status[0] else "🟠 ") + status[1])
    if mt5 is not None and status[0]:
        all_symbols = mt5.symbols_get()
        symbol_names = [s.name for s in all_symbols] if all_symbols else []
    else:
        symbol_names = []
    manual_symbol = st.text_input("Broker symbol (exact MT5 name)", value="XAUUSD")
    symbol = st.selectbox("Visible MT5 symbols (if connected)", ["Use manual symbol"] + symbol_names) if symbol_names else "Use manual symbol"
    if symbol == "Use manual symbol":
        symbol = manual_symbol.strip()
    tf = st.selectbox("Chart timeframe", list(TIMEFRAMES.keys()), index=1)
    strategy = st.selectbox("Strategy", STRATEGIES)
    bars = st.slider("Bars to load", min_value=200, max_value=1500, value=500, step=100)
    rr = st.slider("Risk : Reward target", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    run_scan = st.button("Scan selected market", type="primary", use_container_width=True)

tab_scan, tab_multi, tab_macro, tab_risk, tab_chat, tab_setup = st.tabs([
    "📊 Signal & Chart", "🔎 Multi-timeframe", "🗓️ NFP / CPI", "🛡️ Risk", "💬 Chat / Notes", "⚙️ Setup"
])

with tab_scan:
    st.subheader("Selected market analysis")
    if run_scan:
        try:
            if mt5 is None:
                st.error("MetaTrader5 package is missing. See Setup tab.")
            else:
                if not mt5.initialize():
                    st.error(f"Could not initialize MT5: {mt5.last_error()}")
                else:
                    if not mt5.symbol_select(symbol, True):
                        st.error(f"Cannot select symbol '{symbol}'. Use the exact name from Market Watch.")
                    else:
                        raw = get_rates(symbol, tf, bars)
                        signal = build_signal(raw, strategy, rr)
                        st.session_state["last_scan"] = {
                            "symbol": symbol, "tf": tf, "strategy": strategy,
                            "df": raw, "signal": signal,
                        }
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
    result = st.session_state.get("last_scan")
    if result:
        sig = result["signal"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Direction", sig["direction"])
        c2.metric("Rule agreement score", f"{sig['score']}%")
        c3.metric("Entry", f"{sig['entry']:.5f}")
        c4.metric("ATR(14)", f"{sig['atr']:.5f}")
        if sig["direction"] in ("BUY", "SELL"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Stop Loss", f"{sig['sl']:.5f}")
            c2.metric("Take Profit", f"{sig['tp']:.5f}")
            c3.metric("Risk : Reward", f"1 : {rr:.1f}")
        st.plotly_chart(draw_chart(result["df"], sig, result["symbol"], result["tf"]), use_container_width=True)
        st.markdown("**Reasons / rule evidence**")
        for reason in sig["reasons"]:
            st.write("• " + reason)
        st.caption(
            "The score measures agreement among this app's simple baseline rules. It is not a "
            "win probability and is not a full SMC/BTMM implementation."
        )
    else:
        st.info("Initialize MT5, choose a broker symbol, then click Scan selected market.")

with tab_multi:
    st.subheader("Multi-timeframe scan")
    selected_tfs = st.multiselect("Timeframes to scan", list(TIMEFRAMES.keys()), default=["M5", "M15", "H1", "H4", "D1"])
    if st.button("Run multi-timeframe scan"):
        rows = []
        try:
            if mt5 is None or not mt5.initialize():
                st.error("MT5 is not connected.")
            elif not mt5.symbol_select(symbol, True):
                st.error(f"Symbol '{symbol}' not found. Check broker suffix/prefix.")
            else:
                for selected_tf in selected_tfs:
                    try:
                        d = get_rates(symbol, selected_tf, bars)
                        s = build_signal(d, strategy, rr)
                        rows.append({
                            "Symbol": symbol, "Timeframe": selected_tf, "Signal": s["direction"],
                            "Rule score (%)": s["score"], "Entry": s["entry"],
                            "SL": s["sl"], "TP": s["tp"], "Bar time (UTC)": s["bar_time"],
                        })
                    except Exception as exc:
                        rows.append({"Symbol": symbol, "Timeframe": selected_tf, "Signal": "ERROR", "Details": str(exc)})
                st.session_state["multi_results"] = pd.DataFrame(rows)
        except Exception as exc:
            st.error(str(exc))
    if "multi_results" in st.session_state:
        st.dataframe(st.session_state["multi_results"], use_container_width=True, hide_index=True)

with tab_macro:
    st.subheader("NFP / CPI event alerts")
    st.info(
        "No economic calendar is connected in this starter build. Do not treat the example below as a live event feed. "
        "Connect a provider API before enabling scheduled alerts."
    )
    st.markdown("""
**Provider integration contract**
- Fetch upcoming US economic events with event name, country/currency, impact, scheduled UTC time, forecast and previous.
- Match event names for Non-Farm Payrolls (NFP) and Consumer Price Index (CPI).
- Persist each event ID and send one alert at each configured lead time (60/30/15 minutes); deduplicate by event ID + lead time.
- Display the provider timestamp and convert to Africa/Dar_es_Salaam for the UI.
- Never invent an Actual value before release. “Expected market reaction” must be labelled scenario analysis, not a guaranteed signal.
""")
    provider = st.selectbox("Calendar provider (integration to be configured)", [
        "Trading Economics API", "Financial Modeling Prep API", "Other provider with economic calendar API"
    ])
    lead_times = st.multiselect("Alert lead times (minutes before event)", [120, 60, 30, 15, 5], default=[60, 30, 15])
    st.write("Chosen provider:", provider)
    st.write("Lead times:", ", ".join(map(str, lead_times)), "minutes")
    st.caption("API keys should be stored in Streamlit secrets or environment variables, never hard-coded.")

with tab_risk:
    st.subheader("Risk & position sizing")
    balance = st.number_input("Account equity (account currency)", min_value=0.0, value=6000.0, step=100.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, 0.1)
    st.caption("Use the exact MT5 symbol's tick value/size and contract specs. This simplified section calculates money risk only.")
    money_risk = balance * risk_pct / 100
    st.metric("Money risk budget", f"{money_risk:,.2f}")
    st.warning("Lot-size calculation is intentionally not guessed. It must use MT5 symbol_info, tick_value, tick_size, volume_step, min/max volume and account currency.")

with tab_chat:
    st.subheader("Trading room — notes and checklist")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "SNIPER", "text": "Karibu. Andika bias, setup, entry confirmation na sababu za trade yako."}
        ]
    for message in st.session_state.chat_messages:
        with st.chat_message("assistant" if message["role"] == "SNIPER" else "user"):
            st.write(message["text"])
    user_text = st.chat_input("Andika uchambuzi au notes zako...")
    if user_text:
        st.session_state.chat_messages.append({"role": "Trader", "text": user_text})
        st.rerun()
    st.caption("Chat hii ni notes za ndani ya session. Kuifanya ihifadhiwe milele au ipatikane kwa watumiaji wengi kunahitaji database/login.")

with tab_setup:
    st.subheader("Installation and integration")
    st.markdown("""
### Requirements
- Windows PC/VPS with the MT5 desktop terminal installed and logged in to Deriv or HFM.
- Python 3.10/3.11 recommended for compatibility with the MT5 Python package.
- Broker symbols must be selected from Market Watch; names can include suffixes such as `.a`, `m`, or `-pro`.

### Install
```bash
py -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

### MT5 constraints
The official `MetaTrader5` Python package connects to a local MT5 terminal; this is not a hosted web API. A Streamlit deployment in a cloud container generally cannot see your local terminal. Run the app on the same Windows PC/VPS as MT5, or build a separate authenticated bridge service.

### Needed credentials / APIs
1. **Market prices:** official MetaTrader5 Python package + MT5 terminal logged in to each broker account. Deriv and HFM sessions may need separate terminals/instances; symbols and feeds differ by account.
2. **NFP/CPI calendar:** a calendar API provider such as Trading Economics or Financial Modeling Prep, with an API key and terms that permit your use case.
3. **Notifications:** Telegram Bot API (bot token + chat/channel ID) or an email/webhook provider. Not connected until credentials and send/response checks are implemented.
4. **Calibrated probability:** historical candles + explicit entry/exit/SL/TP rules, spread/commission/slippage, out-of-sample and walk-forward backtesting. Until then, the UI intentionally labels scores as rule agreement rather than probability.
5. **Persistent chat / signal history:** SQLite for one-user local use, or PostgreSQL for multi-user hosting.

### Current limitations
- Baseline strategies are transparent approximations. Full SMC, BTMM, supply/demand and FVG implementations need precise rule definitions and testing.
- The app never places orders.
- There is no connected economic calendar, Telegram bot, or calibrated probability model in this build.
- Do not use the output as a sole basis for live trades.
""")
