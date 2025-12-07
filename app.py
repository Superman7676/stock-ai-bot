import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

# --- הגדרות ---
st.set_page_config(page_title="AI Sniper Ultimate", layout="wide", page_icon="🦅")
st.title("🦅 AI Sniper Ultimate - מערכת הניתוח המלאה")

# --- עיצוב הדוח (כמו בטלגרם) ---
st.markdown("""
<style>
    .telegram-box {
        background-color: #151515;
        color: #00ff41;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.5;
        border: 1px solid #333;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# --- רשימת ברירת מחדל (דוגמה) ---
DEFAULT_LIST = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציית תיקון נתונים ---
def fix_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    if 'Close' not in df.columns: return None
    return df

# --- מנוע סריקה מהיר (לסינון ראשוני) ---
@st.cache_data(ttl=300)
def scan_fast(tickers_list):
    results = []
    chunk_size = 30
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    
    prog = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(chunk, period="1mo", group_by='ticker', threads=True, progress=False)
            for t in chunk:
                try:
                    df = data[t] if len(chunk) > 1 else data
                    df = fix_data(df)
                    if df is None or len(df) < 20: continue
                    
                    curr = df.iloc[-1]
                    rsi = ta.rsi(df['Close']).iloc[-1]
                    results.append({'Symbol': t, 'Price': curr['Close'], 'RSI': rsi})
                except: continue
        except: continue
        prog.progress((i+1)/len(chunks))
    prog.empty()
    return pd.DataFrame(results)

# --- מנוע הניתוח העמוק (The Deep Dive) ---
def analyze_full_stock(ticker):
    try:
        # הורדת שנתיים של נתונים לחישובים ארוכים
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        df = fix_data(df)
        if df is None: return None
        
        # === 1. חישוב כל הממוצעים והמרחקים ===
        # SMA
        for m in [5, 8, 12, 20, 50, 100, 150, 200]:
            df[f'SMA_{m}'] = ta.sma(df['Close'], length=m)
        # EMA
        for e in [5, 8, 12, 20, 26, 50]:
            df[f'EMA_{e}'] = ta.ema(df['Close'], length=e)
            
        curr = df.iloc[-1]
        close = curr['Close']
        
        # מרחקים באחוזים
        dist = {}
        for m in [20, 50, 150, 200]:
            sma_val = curr[f'SMA_{m}']
            dist[f'SMA{m}'] = ((close - sma_val) / sma_val) * 100

        # === 2. מתנדים מתקדמים ===
        df['RSI_7'] = ta.rsi(df['Close'], length=7)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_21'] = ta.rsi(df['Close'], length=21)
        
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx['ADX_14']
        
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Width'] = bb['BBB_5_2.0']
        df['BB_Pct'] = bb['BBP_5_2.0']
        
        aroon = ta.aroon(df['High'], df['Low'])
        df['Aroon_Up'] = aroon['AROONU_14']
        df['Aroon_Down'] = aroon['AROOND_14']
        
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # === 3. ATR Supreme Analysis ===
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['ATR_28'] = ta.atr(df['High'], df['Low'], df['Close'], length=28)
        
        atr_avg = (df['ATR_14'].iloc[-1] + df['ATR_20'].iloc[-1] + df['ATR_28'].iloc[-1]) / 3
        atr_rel = (atr_avg / close) * 100 # באחוזים
        
        # === 4. פיבונאצ'י ופיבוטים ===
        y_high = df['High'][-252:].max()
        y_low = df['Low'][-252:].min()
        diff = y_high - y_low
        
        fibs = {
            '23.6': y_high - 0.236 * diff,
            '38.2': y_high - 0.382 * diff,
            '50.0': y_high - 0.5 * diff,
            '61.8': y_high - 0.618 * diff,
            'Ext_127': y_high + 0.272 * diff,
            'Ext_161': y_high + 0.618 * diff
        }
        
        # Pivot Points
        p = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = 2*p - curr['Low']
        r2 = p + (curr['High'] - curr['Low'])
        r3 = curr['High'] + 2*(p - curr['Low'])
        s1 = 2*p - curr['High']
        s2 = p - (curr['High'] - curr['Low'])
        s3 = curr['Low'] - 2*(curr['High'] - p)
        
        # === 5. מודל חיזוי (Gradient Boosting - כתחליף ל-LSTM) ===
        df_ml = df.dropna().copy()
        df_ml['Target'] = df_ml['Close'].shift(-1)
        features = ['Close', 'RSI_14', 'SMA_5', 'MACD']
        X = df_ml[features].iloc[:-1]
        y = df_ml['Target'].iloc[:-1]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        pred_tmrw = model.predict(df_ml[features].iloc[[-1]])[0]
        acc = model.score(X, y) * 100
        
        # חיזוי לשבוע הבא (לינארי פשוט על בסיס המומנטום)
        trend_slope = (pred_tmrw - close)
        pred_week = pred_tmrw + (trend_slope * 5)
        
        # === 6. ניתוח משטר שוק (Market Regime) ===
        regime = "Consolidation"
        if close > curr['SMA_200'] and curr['SMA_50'] > curr['SMA_200']:
            regime = "Bullish Uptrend 🐂"
        elif close < curr['SMA_200'] and curr['SMA_50'] < curr['SMA_200']:
            regime = "Bearish Downtrend 🐻"
            
        # ציון סופי
        score = 50
        if close > curr['SMA_200']: score += 20
        if curr['RSI_14'] < 30: score += 15
        if curr['MACD_Hist'] > 0: score += 10
        if curr['ADX_14'] > 25: score += 5
        
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 40: rec = "SELL 🔴"
        
        # הכנת המילון להדפסה
        return {
            'Symbol': ticker, 'Price': close, 'Rec': rec, 'Score': score,
            'Regime': regime, 'Vol': curr['Volume'], 'AvgVol': df['Volume'].mean(),
            'High': curr['High'], 'Low': curr['Low'],
            'Year_High': y_high, 'Year_Low': y_low,
            'MAs': curr, 'Dist': dist, # כל הממוצעים
            'RSI': {'7': curr['RSI_7'], '14': curr['RSI_14'], '21': curr['RSI_21']},
            'MACD': curr, 'ADX': curr['ADX_14'], 'Stoch': curr, 'BB': curr, 'Aroon': curr,
            'MFI': curr['MFI'], 'CCI': curr['CCI'], 'VWAP': curr['VWAP'],
            'ATR': {'14': curr['ATR_14'], '20': curr['ATR_20'], '28': curr['ATR_28'], 'Avg': atr_avg, 'Rel': atr_rel},
            'Pivots': {'P': p, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3},
            'Fibs': fibs,
            'AI': {'Tmrw': pred_tmrw, 'Week': pred_week, 'Acc': acc},
            'Change_Pct': df['Close'].pct_change().iloc[-1] * 100,
            'Change_USD': close - df['Close'].iloc[-2]
        }
        
    except Exception as e:
        return None

# --- UI ---
with st.sidebar:
    st.header("הגדרות")
    tickers_input = st.text_area("רשימת מניות:", DEFAULT_LIST, height=200)
    run_scan = st.button("🚀 הפעל סריקה")

# לוגיקה ראשית
if run_scan:
    t_list = [x.strip().upper() for x in tickers_input.replace('\n', ',').split(',') if x.strip()]
    st.session_state['scan_data'] = scan_fast(t_list)

if 'scan_data' in st.session_state and st.session_state['scan_data'] is not None:
    df_res = st.session_state['scan_data']
    
    if not df_res.empty:
        st.subheader(f"📊 תוצאות סריקה ({len(df_res)} מניות)")
        
        # טבלה לחיצה
        event = st.dataframe(
            df_res.sort_values('RSI').style.format({'Price': '{:.2f}', 'RSI': '{:.1f}'}),
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True
        )
        
        selected_row = event.selection.rows
        if selected_row:
            ticker = df_res.iloc[selected_row[0]]['Symbol']
            
            with st.spinner(f"מפעיל ניתוח עומק על {ticker}..."):
                d = analyze_full_stock(ticker)
                
            if d:
                # === יצירת הדוח המפלצתי ===
                report = f"""
⭐️ **{d['Symbol']} Corporation**
Sector: Technology | Regime: {d['Regime']} | Trend Score: {d['Score']}/100
══════════════════
💰 **Price & Change**
• Price: {d['Price']:.2f}$ ({'🟢' if d['Change_Pct']>0 else '🔴'} {d['Change_Pct']:.2f}% | {d['Change_USD']:.2f}$)
• H/L: {d['High']:.2f}$ / {d['Low']:.2f}$
• 52W H/L: {d['Year_High']:.2f}$ / {d['Year_Low']:.2f}$
🔊 Vol Day: {d['Vol']/1000000:.2f}M | Avg: {d['AvgVol']/1000000:.2f}M | Ratio: {d['Vol']/d['AvgVol']:.2f}x
• ATR14: {d['ATR']['14']:.2f}$ ({d['ATR']['Rel']:.2f}%)
══════════════════
📊 **Moving Averages**
• SMA-5: {d['MAs']['SMA_5']:.2f}$ | SMA-20: {d['MAs']['SMA_20']:.2f}$ | SMA-50: {d['MAs']['SMA_50']:.2f}$
• SMA-100: {d['MAs']['SMA_100']:.2f}$ | SMA-150: {d['MAs']['SMA_150']:.2f}$ | SMA-200: {d['MAs']['SMA_200']:.2f}$
• EMA-5: {d['MAs']['EMA_5']:.2f}$ | EMA-20: {d['MAs']['EMA_20']:.2f}$ | EMA-50: {d['MAs']['EMA_50']:.2f}$
→→→→→→→→→→→→→→→→→
• Distance:
  P→SMA20: {d['Dist']['SMA20']:.2f}% | P→SMA50: {d['Dist']['SMA50']:.2f}%
  P→SMA150: {d['Dist']['SMA150']:.2f}% | P→SMA200: {d['Dist']['SMA200']:.2f}%
• VWAP-Day: {d['VWAP']:.2f}$
═════════════════
⚡️ **Momentum & Oscillators**
• RSI-7: {d['RSI']['7']:.1f} | RSI-14: {d['RSI']['14']:.1f} | RSI-21: {d['RSI']['21']:.1f}
• MACD: {d['MACD']['MACD']:.2f} | Sig: {d['MACD']['MACD_Signal']:.2f} | Hist: {d['MACD']['MACD_Hist']:.2f}
• ADX: {d['ADX']:.2f} | Aroon Up/Dn: {d['Aroon']['Aroon_Up']:.0f}/{d['Aroon']['Aroon_Down']:.0f}
• Stoch %K/%D: {d['Stoch']['Stoch_K']:.1f}/{d['Stoch']['Stoch_D']:.1f}
• BB Width: {d['BB']['BB_Width']:.2f}% | MFI: {d['MFI']:.1f} | CCI: {d['CCI']:.2f}
═════════════════
🎯 **AI Predictions (Gradient Boosting)**
• Tomorrow: ${d['AI']['Tmrw']:.2f}
• Next Week: ${d['AI']['Week']:.2f}
• Model Accuracy: {d['AI']['Acc']:.1f}%
🧠 AI Signal Score: {d['Score']} ({d['Rec']})
═════════════════
📐 **Support/Resistance & Pivots**
• Pivot: ${d['Pivots']['P']:.2f}
• R1: ${d['Pivots']['R1']:.2f} | R2: ${d['Pivots']['R2']:.2f} | R3: ${d['Pivots']['R3']:.2f}
• S1: ${d['Pivots']['S1']:.2f} | S2: ${d['Pivots']['S2']:.2f} | S3: ${d['Pivots']['S3']:.2f}
═════════════════
🔢 **Fibonacci Levels**
• Fib-23.6%: ${d['Fibs']['23.6']:.2f} | Fib-38.2%: ${d['Fibs']['38.2']:.2f}
• Fib-50%: ${d['Fibs']['50.0']:.2f} | Fib-61.8%: ${d['Fibs']['61.8']:.2f} 🌟
• Ext-127.2%: ${d['Fibs']['Ext_127']:.2f} | Ext-161.8%: ${d['Fibs']['Ext_161']:.2f} 🌟
═════════════════
🌊 **ATR Supreme Analysis**
• ATR(14/20/28): {d['ATR']['14']:.2f} / {d['ATR']['20']:.2f} / {d['ATR']['28']:.2f}
• ATR Average: {d['ATR']['Avg']:.2f}
═════════════════
🏛 **Recommendation**
Entry: ${d['Price']:.2f} | Stop: ${d['Price'] - 2*d['ATR']['Avg']:.2f}
Targets: T1=${d['Pivots']['R1']:.2f} · T2=${d['Fibs']['Ext_161']:.2f}
═══════════════════
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.code(report, language="text")
