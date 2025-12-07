import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

# --- הגדרות עמוד ועיצוב ---
st.set_page_config(page_title="AI Sniper Ultimate", layout="wide", page_icon="🦅")

# CSS לעיצוב יוקרתי וגדול
st.markdown("""
<style>
    /* הגדלת הפונט בטבלה */
    .stDataFrame { font-size: 1.2rem !important; }
    
    /* עיצוב תיבת הדוח הטלגרמי */
    .telegram-box {
        background-color: #101010;
        color: #00ff41;
        padding: 25px;
        border-radius: 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #333;
        white-space: pre-wrap;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* כותרות */
    h1 { color: #4F8BF9; }
    h3 { border-bottom: 2px solid #444; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🦅 AI Sniper Ultimate - Trading Floor Edition")

# --- רשימת ברירת מחדל ---
DEFAULT_LIST = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציות ליבה ---
def fix_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    if 'Close' not in df.columns: return None
    return df

# --- סורק טבלה (מהיר) ---
@st.cache_data(ttl=600)
def scan_table(tickers_list):
    results = []
    chunk_size = 20
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    
    prog = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(chunk, period="1mo", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            for t in chunk:
                try:
                    df = data[t] if len(chunk) > 1 else data
                    df = fix_data(df)
                    if df is None or len(df) < 20: continue
                    
                    curr = df.iloc[-1]
                    rsi = ta.rsi(df['Close']).iloc[-1]
                    sma200 = ta.sma(df['Close'], length=200).iloc[-1] if len(df) > 200 else 0
                    
                    # Trend Determination
                    trend = "🟢 Bull" if curr['Close'] > sma200 else "🔴 Bear"
                    
                    results.append({
                        'Symbol': t,
                        'Price': curr['Close'],
                        'RSI': rsi,
                        'Trend': trend,
                        'Volume': curr['Volume'] / 1000000 # In Millions
                    })
                except: continue
        except: continue
        prog.progress((i+1)/len(chunks))
    prog.empty()
    return pd.DataFrame(results)

# --- מנוע הניתוח המלא (Deep Dive) ---
def analyze_monster(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        df = fix_data(df)
        if df is None: return None
        
        # === אינדיקטורים מלאים ===
        # SMA
        mas = {}
        for m in [5, 8, 12, 20, 50, 100, 150, 200]:
            df[f'SMA_{m}'] = ta.sma(df['Close'], length=m)
            mas[f'SMA{m}'] = df[f'SMA_{m}'].iloc[-1]
            
        # EMA
        for e in [5, 8, 12, 20, 26, 50]:
            df[f'EMA_{e}'] = ta.ema(df['Close'], length=e)
            mas[f'EMA{e}'] = df[f'EMA_{e}'].iloc[-1]
            
        curr = df.iloc[-1]
        close = curr['Close']
        
        # Distances
        dists = {
            'SMA20': ((close - mas['SMA20'])/mas['SMA20'])*100,
            'SMA50': ((close - mas['SMA50'])/mas['SMA50'])*100,
            'SMA200': ((close - mas['SMA200'])/mas['SMA200'])*100,
        }
        
        # Oscillators
        rsi = {
            '7': ta.rsi(df['Close'], 7).iloc[-1],
            '14': ta.rsi(df['Close'], 14).iloc[-1],
            '21': ta.rsi(df['Close'], 21).iloc[-1]
        }
        
        macd = ta.macd(df['Close'])
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        aroon = ta.aroon(df['High'], df['Low'])
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        bb = ta.bbands(df['Close'], 20, 2)
        mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        cci = ta.cci(df['High'], df['Low'], df['Close'])
        vwap = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
        
        # ATR Supreme
        atr14 = ta.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        atr20 = ta.atr(df['High'], df['Low'], df['Close'], 20).iloc[-1]
        atr28 = ta.atr(df['High'], df['Low'], df['Close'], 28).iloc[-1]
        atr_avg = (atr14 + atr20 + atr28) / 3
        
        # ML Prediction (Boosting)
        df_ml = df.dropna().copy()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        feats = ['Close', 'RSI_7', 'SMA_5']
        X = df_ml[feats].iloc[:-1]
        y = df_ml['Target'].iloc[:-1]
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X, y)
        pred = model.predict(df_ml[feats].iloc[[-1]])[0]
        acc = model.score(X, y) * 100
        
        # Forecast Logic
        trend_score = 50
        if close > mas['SMA200']: trend_score += 20
        if rsi['14'] < 30: trend_score += 15
        if pred > close: trend_score += 15
        
        sentiment = "NEUTRAL"
        if trend_score >= 80: sentiment = "STRONG BUY 🚀"
        elif trend_score >= 60: sentiment = "BUY 🟢"
        elif trend_score <= 40: sentiment = "SELL 🔴"
        
        # Pivots & Fibs
        h, l = df['High'].max(), df['Low'].min()
        fib618 = h - 0.618 * (h - l)
        p = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = 2*p - curr['Low']
        
        return {
            'Symbol': ticker, 'Price': close, 'Senti': sentiment, 'Score': trend_score,
            'Mas': mas, 'Dist': dists, 'RSI': rsi, 'Pred': pred, 'Acc': acc,
            'ATR': {'14': atr14, '20': atr20, '28': atr28, 'Avg': atr_avg},
            'Data': curr, 'VWAP': vwap, 'MACD': macd.iloc[-1], 'ADX': adx.iloc[-1],
            'Aroon': aroon.iloc[-1], 'BB': bb.iloc[-1], 'MFI': mfi.iloc[-1], 'CCI': cci.iloc[-1],
            'Pivot': p, 'R1': r1, 'Fib': fib618,
            'Vol': curr['Volume'], 'AvgVol': df['Volume'].mean(),
            'YHigh': h, 'YLow': l,
            'ChgPct': df['Close'].pct_change().iloc[-1] * 100,
            'ChgUSD': close - df['Close'].iloc[-2]
        }
        
    except Exception as e:
        return None

# --- ממשק משתמש ---
with st.sidebar:
    st.header("⚙️ פאנל שליטה")
    tickers_input = st.text_area("הכנס רשימת מניות:", DEFAULT_LIST, height=300)
    run_btn = st.button("🚀 הפעל סורק שוק")

# --- לוגיקה ---
if run_btn:
    t_list = [x.strip().upper() for x in tickers_input.replace('\n', ',').split(',') if x.strip()]
    st.session_state['scan'] = scan_table(t_list)

if 'scan' in st.session_state and st.session_state['scan'] is not None:
    df = st.session_state['scan']
    
    if df.empty:
        st.error("לא נמצאו נתונים. בדוק את הרשימה.")
    else:
        st.subheader("📊 טבלת סריקה (בחר מניה לניתוח עומק)")
        
        # טבלה מעוצבת וגדולה
        st.dataframe(
            df.style.format({'Price': '{:.2f}', 'RSI': '{:.1f}', 'Volume': '{:.2f}M'}),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        st.header("🔬 ניתוח עומק (Deep Dive)")
        
        sel_ticker = st.selectbox("בחר מניה:", df['Symbol'].unique())
        
        if st.button(f"הפעל ניתוח מלא על {sel_ticker}"):
            with st.spinner("מבצע חישובים..."):
                d = analyze_monster(sel_ticker)
                
            if d:
                # --- הדוח המפלצתי והמדויק ---
                report = f"""
⭐️ **{d['Symbol']} Corporation**
Sentiment: {d['Senti']} | Trend Score: {d['Score']}/100
══════════════════
💰 **Price & Change**
• Price: {d['Price']:.2f}$ ({'🟢' if d['ChgPct']>0 else '🔴'} {d['ChgPct']:.2f}% | {d['ChgUSD']:.2f}$)
• H/L: {d['Data']['High']:.2f}$ / {d['Data']['Low']:.2f}$
• 52W H/L: {d['YHigh']:.2f}$ / {d['YLow']:.2f}$
🔊 Vol Day: {d['Vol']/1000000:.2f}M | Avg Vol: {d['AvgVol']/1000000:.2f}M
• Ratio: {d['Vol']/d['AvgVol']:.2f}x
══════════════════
📊 **Moving Averages (Comprehensive)**
• SMA-5: {d['Mas']['SMA5']:.2f}$ | SMA-8: {d['Mas']['SMA8']:.2f}$ | SMA-12: {d['Mas']['SMA12']:.2f}$
• SMA-20: {d['Mas']['SMA20']:.2f}$ | SMA-50: {d['Mas']['SMA50']:.2f}$ | SMA-100: {d['Mas']['SMA100']:.2f}$
• SMA-150: {d['Mas']['SMA150']:.2f}$ | SMA-200: {d['Mas']['SMA200']:.2f}$

• EMA-5: {d['Mas']['EMA5']:.2f}$ | EMA-8: {d['Mas']['EMA8']:.2f}$ | EMA-20: {d['Mas']['EMA20']:.2f}$
• EMA-26: {d['Mas']['EMA26']:.2f}$ | EMA-50: {d['Mas']['EMA50']:.2f}$
→→→→→→→→→→→→→→→→→
• Distance:
  P→SMA20: {d['Dist']['SMA20']:.2f}% | P→SMA50: {d['Dist']['SMA50']:.2f}%
  P→SMA200: {d['Dist']['SMA200']:.2f}%
• VWAP-Day: {d['VWAP']:.2f}$
═════════════════
⚡️ **Momentum & Oscillators**
• RSI-7: {d['RSI']['7']:.1f} | RSI-14: {d['RSI']['14']:.1f} | RSI-21: {d['RSI']['21']:.1f}
• MACD: {d['MACD']['MACD_12_26_9']:.2f} | Sig: {d['MACD']['MACDs_12_26_9']:.2f} | Hist: {d['MACD']['MACDh_12_26_9']:.2f}
• ADX: {d['ADX']['ADX_14']:.2f} (Strength)
• Stoch %K/%D: {d['Stoch']['STOCHk_14_3_3']:.1f}/{d['Stoch']['STOCHd_14_3_3']:.1f}
• BB Width: {d['BB']['BBB_5_2.0']:.2f}%
• Aroon ↑/↓: {d['Aroon']['AROONU_14']:.0f} / {d['Aroon']['AROOND_14']:.0f}
• MFI: {d['MFI']:.1f} | CCI: {d['CCI']:.2f}
═════════════════
🎯 **AI Predictions (ML Based)**
• Tomorrow Forecast: ${d['Pred']:.2f}
• Model Accuracy: {d['Acc']:.1f}%
═════════════════
🌊 **ATR Supreme Analysis**
• ATR(14/20/28): {d['ATR']['14']:.2f} / {d['ATR']['20']:.2f} / {d['ATR']['28']:.2f}
• ATR Average: {d['ATR']['Avg']:.2f}
═════════════════
📐 **Support/Resistance & Pivots**
• Pivot: ${d['Pivot']:.2f}
• R1: ${d['R1']:.2f} | Golden Fib (61.8%): ${d['Fib']:.2f}
═════════════════
🏛 **Recommendation**
• Entry: ${d['Price']:.2f}
• Stop Loss (2xATR): ${d['Price'] - 2*d['ATR']['Avg']:.2f}
• Target (R1): ${d['R1']:.2f}
═══════════════════
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.markdown(f'<div class="telegram-box">{report}</div>', unsafe_allow_html=True)
