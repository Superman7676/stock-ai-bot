import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import time

# --- הגדרות עמוד ---
st.set_page_config(page_title="Final AI Sniper", layout="wide", page_icon="🎯")
st.title("🎯 AI Sniper - גרסה מתוקנת ומלאה")

# --- רשימת המניות (הדבק כאן את הרשימה המלאה שלך) ---
DEFAULT_TICKERS = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN,
UBER, MELI, DELL, HOOD, UPST, FICO, EQIX, SPY"""

# --- פונקציית הקסם לתיקון הנתונים ---
def fix_yahoo_data(df):
    # אם הטבלה ריקה
    if df.empty: return df
    
    # הורדת רמה אם יש MultiIndex (הבעיה שגרמה לקריסה)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # מנסים לשטח את הטבלה
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    # וידוא שיש עמודת Close
    # לפעמים זה מגיע כ- 'Close' ולפעמים כ- 'Adj Close'
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
        
    return df

# --- פונקציית זיהוי תבניות ---
def check_patterns(open_p, high, low, close):
    body = abs(close - open_p)
    full = high - low
    if full == 0: return "Flat"
    
    lower_wick = min(open_p, close) - low
    upper_wick = high - max(open_p, close)
    
    pat = []
    if lower_wick > 2 * body and upper_wick < body: pat.append("Hammer 🔨")
    if upper_wick > 2 * body and lower_wick < body: pat.append("Shooting Star 🌠")
    if body < 0.1 * full: pat.append("Doji ➕")
    if body > 0.8 * full and close > open_p: pat.append("Big Green 💪")
    
    return ", ".join(pat) if pat else "Normal"

# --- המוח (ניתוח מניה) ---
def analyze_stock(ticker):
    try:
        # הורדה
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        
        # --- התיקון הקריטי ---
        df = fix_yahoo_data(df)
        # ---------------------
        
        if df.empty or 'Close' not in df.columns or len(df) < 50:
            return None
            
        # חישובים טכניים (בזהירות)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        
        # נתונים אחרונים
        curr = df.iloc[-1]
        
        # זיהוי תבנית
        pattern = check_patterns(curr['Open'], curr['High'], curr['Low'], curr['Close'])
        
        # פיבונאצ'י
        high_y = df['High'].max()
        low_y = df['Low'].min()
        fib_618 = high_y - 0.618 * (high_y - low_y)
        
        # פיבוט
        pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = (2 * pivot) - curr['Low']
        
        # ציון
        score = 50
        if curr['Close'] > (curr['SMA_200'] if not pd.isna(curr['SMA_200']) else 0): score += 20
        if curr['RSI'] < 30: score += 20
        if curr['RSI'] > 75: score -= 15
        if "Hammer" in pattern: score += 15
        
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 30: rec = "SELL 🔴"
        
        return {
            'Symbol': ticker,
            'Price': curr['Close'],
            'Score': score,
            'Rec': rec,
            'RSI': curr['RSI'],
            'Pattern': pattern,
            'Fib_618': fib_618,
            'Pivot': pivot,
            'R1': r1,
            'ATR': curr['ATR']
        }
        
    except Exception as e:
        return None

# --- ממשק משתמש ---
user_input = st.text_area("הכנס רשימת מניות:", DEFAULT_TICKERS, height=100)

if st.button("🔥 הפעל סריקה"):
    tickers = [t.strip().upper() for t in user_input.split(',') if t.strip()]
    
    st.info(f"סורק {len(tickers)} מניות... (עובר אחת אחת למניעת תקלות)")
    
    results = []
    bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        data = analyze_stock(t)
        if data:
            results.append(data)
        else:
            # אם נכשל, ננסה שוב עם השהייה קטנה
            time.sleep(0.5)
            data_retry = analyze_stock(t)
            if data_retry: results.append(data_retry)
            
        bar.progress((i+1)/len(tickers))
        
    bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        
        # טבלה ראשית
        st.success(f"נמצאו נתונים ל-{len(df_res)} מניות!")
        st.dataframe(
            df_res.sort_values('Score', ascending=False).style.format({"Price": "{:.2f}", "RSI": "{:.1f}"}),
            use_container_width=True
        )
        
        st.divider()
        
        # מחולל דוחות
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("📝 בחר מניה לדוח")
            selected = st.radio("רשימה:", df_res['Symbol'].tolist(), label_visibility="collapsed")
            
        with col2:
            row = df_res[df_res['Symbol'] == selected].iloc[0]
            stop_loss = row['Price'] - 2 * row['ATR']
            
            report = f"""
🚨 **{row['Symbol']} SIGNAL REPORT** 🚨
══════════════════════
💰 Price: ${row['Price']:.2f}
🚦 Signal: {row['Rec']} (Score: {row['Score']})
🕯️ Pattern: {row['Pattern']}

📊 **Technical Stats**
• RSI: {row['RSI']:.1f}
• Pivot Point: ${row['Pivot']:.2f}
• Resistance (R1): ${row['R1']:.2f}

🎯 **Key Levels**
• Golden Fib (61.8%): ${row['Fib_618']:.2f}
• Stop Loss: ${stop_loss:.2f}
══════════════════════
"""
            st.code(report, language="text")
            
    else:
        st.error("עדיין לא נמצאו נתונים. הבעיה כנראה חסימה חמורה של ה-IP בשרת.")
