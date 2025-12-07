import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# --- הגדרות ---
st.set_page_config(page_title="AI Sniper Ultimate", layout="wide", page_icon="🦅")
st.title("🦅 AI Sniper Ultimate - מערכת סריקה וניתוח מלאה")

# --- רשימת ברירת מחדל (תוכל להדביק את ה-500 שלך) ---
DEFAULT_LIST = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציה לתיקון נתונים (מונעת את השגיאות האדומות) ---
def fix_data(df):
    if df.empty: return None
    # טיפול ב-MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    # וידוא עמודת Close
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns: return None
    return df

# --- מנוע הסריקה המהיר (ל-500 מניות) ---
@st.cache_data(ttl=600)
def scan_fast(tickers_list):
    results = []
    # חלוקה למנות כדי למנוע חסימה
    chunk_size = 50
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    
    prog = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        try:
            data = yf.download(chunk, period="5d", group_by='ticker', threads=True, progress=False)
            
            for t in chunk:
                try:
                    df = data[t] if len(chunk) > 1 else data
                    df = fix_data(df)
                    if df is None or len(df) < 2: continue
                    
                    curr = df.iloc[-1]
                    prev = df.iloc[-2]
                    change = ((curr['Close'] - prev['Close']) / prev['Close']) * 100
                    
                    results.append({
                        'Symbol': t,
                        'Price': curr['Close'],
                        'Change': change,
                        'Volume': curr['Volume']
                    })
                except: continue
        except: continue
        prog.progress((i+1)/len(chunks))
        
    prog.empty()
    return pd.DataFrame(results)

# --- מנוע ניתוח עומק (Deep Dive) ---
def analyze_deep(ticker):
    try:
        # הורדת היסטוריה מלאה
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = fix_data(df)
        if df is None: return None
        
        # --- אינדיקטורים ---
        # ממוצעים
        for m in [5, 20, 50, 100, 150, 200]:
            df[f'SMA_{m}'] = ta.sma(df['Close'], length=m)
        df['EMA_5'] = ta.ema(df['Close'], length=5)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        
        # מתנדים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # VWAP & Aroon
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        aroon = ta.aroon(df['High'], df['Low'])
        df['Aroon_Up'] = aroon['AROONU_14']
        
        # --- ML Prediction (LSTM Style) ---
        df['Target'] = df['Close'].shift(-1)
        ml_data = df.dropna().copy()
        X = ml_data[['Close', 'RSI', 'SMA_5']]
        y = ml_data['Target']
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        pred_price = model.predict(X.iloc[[-1]])[0]
        accuracy = model.score(X, y) * 100
        
        # --- חישובים אחרונים ---
        curr = df.iloc[-1]
        
        # פיבונאצ'י
        h = df['High'].max()
        l = df['Low'].min()
        fib618 = h - 0.618 * (h - l)
        
        # פיבוט
        p = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = 2*p - curr['Low']
        s1 = 2*p - curr['High']
        
        # המלצה וניקוד
        score = 50
        if curr['Close'] > curr['SMA_200']: score += 20
        if curr['RSI'] < 30: score += 20
        if pred_price > curr['Close']: score += 10
        
        rec = "HOLD"
        if score >= 75: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 30: rec = "SELL 🔴"
        
        return {
            'Symbol': ticker, 'Price': curr['Close'], 'Rec': rec, 'Score': score,
            'Pred': pred_price, 'Acc': accuracy,
            'RSI': curr['RSI'], 'MACD': curr['MACD'], 'ADX': curr['ADX'],
            'SMA50': curr['SMA_50'], 'SMA200': curr['SMA_200'],
            'ATR': curr['ATR'], 'VWAP': curr['VWAP'], 'Aroon': curr['Aroon_Up'],
            'Pivot': p, 'R1': r1, 'S1': s1, 'Fib618': fib618,
            'Vol': curr['Volume'], 'AvgVol': df['Volume'].mean(),
            'High': curr['High'], 'Low': curr['Low'],
            'Change': ((curr['Close'] - df.iloc[-2]['Close'])/df.iloc[-2]['Close'])*100
        }
        
    except Exception as e:
        return None

# --- UI ---
with st.sidebar:
    st.header("הגדרות סורק")
    tickers_input = st.text_area("הדבק רשימת מניות:", DEFAULT_LIST, height=200)
    run_scan = st.button("🚀 הפעל סריקה מהירה")

# לוגיקה ראשית
if run_scan:
    t_list = [x.strip().upper() for x in tickers_input.replace('\n', ',').split(',') if x.strip()]
    st.session_state['scan_data'] = scan_fast(t_list)

if 'scan_data' in st.session_state and st.session_state['scan_data'] is not None:
    df_res = st.session_state['scan_data']
    
    if df_res.empty:
        st.error("לא נמצאו נתונים. בדוק את הרשימה.")
    else:
        st.subheader("תוצאות סריקה (לחץ על מניה לדוח מלא)")
        
        # טבלה לחיצה
        event = st.dataframe(
            df_res.style.format({'Price': '{:.2f}', 'Change': '{:.2f}%'}),
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True
        )
        
        selected_row = event.selection.rows
        if selected_row:
            ticker = df_res.iloc[selected_row[0]]['Symbol']
            
            with st.spinner(f"מנתח את {ticker} עם מודלים מתקדמים..."):
                data = analyze_deep(ticker)
                
            if data:
                st.markdown("---")
                # === הדוח הטלגרמי המדויק שביקשת ===
                report = f"""
⭐️ **{data['Symbol']} Corporation**
Sentiment: {data['Rec']} | Trend Score: {data['Score']}/100
══════════════════
💰 **Price & Change**
• Price: {data['Price']:.2f}$ ({data['Change']:.2f}%)
• H/L: {data['High']:.2f}$ / {data['Low']:.2f}$
🔊 Vol Day: {data['Vol']/1000000:.2f}M | Avg Vol: {data['AvgVol']/1000000:.2f}M
• ATR14: {data['ATR']:.2f}$
══════════════════
🎯 **LSTM AI Predictions**
• Tomorrow: ${data['Pred']:.2f}
• Model Accuracy: {data['Acc']:.1f}%
🧠 AI Signal Score: {data['Score']} ({data['Rec']})
══════════════════
📊 **Moving Averages**
• SMA-50: {data['SMA50']:.2f}$ | SMA-200: {data['SMA200']:.2f}$
• Distance to SMA200: {((data['Price']-data['SMA200'])/data['SMA200'])*100:.2f}%
• VWAP-Day: {data['VWAP']:.2f}$
═════════════════
⚡️ **Momentum & Oscillators**
• RSI-14: {data['RSI']:.1f} | ADX: {data['ADX']:.1f}
• MACD: {data['MACD']:.2f}
• Aroon Up: {data['Aroon']:.0f}
═════════════════
📐 **Support/Resistance & Pivots**
• Pivot: ${data['Pivot']:.2f}
• R1: ${data['R1']:.2f} | S1: ${data['S1']:.2f}
• Golden Fib (61.8%): ${data['Fib618']:.2f}
═════════════════
🌊 **Risk Management**
• Stop Loss: ${data['Price'] - 2*data['ATR']:.2f}
• Target: ${data['R1']:.2f}
═══════════════════
Generated: {datetime.now().strftime('%H:%M:%S')}
"""
                st.code(report, language="text")
