import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Sniper Pro", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .report-box {background-color: #111; color: #0f0; padding: 15px; border-radius: 5px; font-family: monospace; border: 1px solid #333;}
    .metric-card {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🦅 AI Sniper Pro - המערכת המלאה")
st.caption(f"Yfinance Version: {yf.__version__}") # בדיקה שאנחנו בגרסה החדשה

# --- רשימת מניות (שים כאן את כל הרשימה שלך) ---
DEFAULT_TICKERS = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציות זיהוי תבניות (החלק שחסר לך) ---
def check_patterns(open_p, high, low, close):
    body = abs(close - open_p)
    full_range = high - low
    if full_range == 0: return "None"
    
    lower_wick = min(open_p, close) - low
    upper_wick = high - max(open_p, close)
    
    pat = []
    # Hammer
    if lower_wick > 2 * body and upper_wick < body: pat.append("Hammer 🔨")
    # Shooting Star
    if upper_wick > 2 * body and lower_wick < body: pat.append("Shooting Star 🌠")
    # Doji
    if body < 0.05 * full_range: pat.append("Doji ➕")
    # Marubozu
    if body > 0.9 * full_range: pat.append("Marubozu 💪")
    
    return ", ".join(pat) if pat else "Normal"

# --- פונקציה שמנסה להוריד בכוח ---
def get_stock_data(ticker):
    try:
        # ניסיון להוריד נתונים
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        
        # וידוא שיש נתונים
        if df.empty: return None, "Empty Data"
        
        # טיפול ב-MultiIndex של יאהו
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=0)
            except: pass
            
        # בדיקה נוספת
        if 'Close' not in df.columns: return None, "No Close Column"
        
        return df, "OK"
    except Exception as e:
        return None, str(e)

# --- המוח (חישוב אינדיקטורים) ---
def analyze(ticker, df):
    try:
        # 1. אינדיקטורים קלאסיים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # 2. MACD
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        
        # 3. בולינגר
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_U'] = bb['BBU_5_2.0']
        df['BB_L'] = bb['BBL_5_2.0']
        
        # 4. זיהוי נרות (על הנר האחרון)
        curr = df.iloc[-1]
        candle_type = check_patterns(curr['Open'], curr['High'], curr['Low'], curr['Close'])
        
        # 5. פיבונאצ'י
        high_y = df['High'].max()
        low_y = df['Low'].min()
        fib_618 = high_y - 0.618 * (high_y - low_y)
        
        # 6. ציון וסיגנל
        score = 50
        if curr['Close'] > curr['SMA_200']: score += 20
        if curr['RSI'] < 30: score += 20
        if curr['RSI'] > 75: score -= 15
        if curr['MACD'] > curr['MACD_Signal']: score += 10
        if "Hammer" in candle_type: score += 15
        
        score = min(max(score, 0), 100)
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 30: rec = "SELL 🔴"
        
        # חישוב פיבוט
        pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
        
        return {
            'Symbol': ticker,
            'Price': curr['Close'],
            'Rec': rec,
            'Score': score,
            'RSI': curr['RSI'],
            'Candle': candle_type,
            'Fib_618': fib_618,
            'Pivot': pivot,
            'SMA_200': curr['SMA_200'],
            'ATR': curr['ATR']
        }
    except Exception as e:
        return None

# --- UI ראשי ---
input_tickers = st.text_area("הכנס רשימת מניות (מופרד בפסיקים)", DEFAULT_TICKERS, height=100)

if st.button("🔥 הפעל סריקה מלאה"):
    tickers_list = [t.strip().upper() for t in input_tickers.split(',') if t.strip()]
    
    results = []
    errors = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, t in enumerate(tickers_list):
        status.text(f"בודק את {t}...")
        df, msg = get_stock_data(t)
        
        if df is not None:
            res = analyze(t, df)
            if res: results.append(res)
        else:
            errors.append(f"{t}: {msg}")
            
        progress.progress((i+1)/len(tickers_list))
    
    status.empty()
    progress.empty()
    
    # הצגת תוצאות
    if results:
        df_res = pd.DataFrame(results)
        
        st.subheader("🏆 תוצאות הסריקה")
        st.dataframe(df_res.sort_values("Score", ascending=False), use_container_width=True)
        
        st.divider()
        st.subheader("🔬 דוח מפורט (לחץ להעתקה)")
        
        selected = st.selectbox("בחר מניה לדוח:", df_res['Symbol'].tolist())
        row = df_res[df_res['Symbol'] == selected].iloc[0]
        
        # יצירת הדוח הטקסטואלי
        report = f"""
🚨 **{row['Symbol']} REPORT** 🚨
══════════════════════
💰 Price: ${row['Price']:.2f}
🚦 Signal: {row['Rec']} (Score: {row['Score']})
🕯️ Pattern: {row['Candle']}

📊 **Technical Data**
• RSI: {row['RSI']:.1f}
• Trend (SMA200): {'Bullish 🟢' if row['Price'] > row['SMA_200'] else 'Bearish 🔴'}
• Volatility (ATR): {row['ATR']:.2f}

🎯 **Key Levels**
• Pivot: ${row['Pivot']:.2f}
• Golden Fib (61.8%): ${row['Fib_618']:.2f}

🛡️ **Trade Setup**
• Stop Loss: ${row['Price'] - 2*row['ATR']:.2f}
• Target: ${row['Pivot'] + 2*row['ATR']:.2f}
══════════════════════
"""
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
        
    # הצגת שגיאות אם יש (כדי שנבין למה דברים לא עובדים)
    if errors:
        with st.expander("ראה שגיאות טכניות (DEBUG)"):
            st.write(errors)
