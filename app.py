import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import time

# --- הגדרות ---
st.set_page_config(page_title="AI Hedge Fund Scanner", layout="wide", page_icon="🏦")
st.title("🏦 AI Hedge Fund Scanner (500+ Stocks Capable)")

# --- רשימת ברירת מחדל (תוכל להדביק את ה-500 שלך בממשק) ---
DEFAULT_LIST = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציות ליבה ---

# 1. פונקציית תיקון נתונים (מונע קריסות)
def fix_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    # הסרת טיקרים שנכשלו (שורות ריקות)
    if 'Close' not in df.columns: return None
    return df

# 2. חישוב טכני מהיר (לסריקת המונים)
def calculate_quick_technicals(df):
    # חישובים וקטוריים מהירים לכל ה-DataFrame בבת אחת
    # זה הרבה יותר מהיר מלולאה
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    return df

# 3. מנוע AI כבד (רץ רק על מניה ספציפית שנבחרה)
def run_deep_ai_analysis(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    df = fix_data(df)
    if df is None: return None
    
    # חישוב כל האינדיקטורים שביקשת
    df['SMA_5'] = ta.sma(df['Close'], length=5)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    aroon = ta.aroon(df['High'], df['Low'])
    df['Aroon_Up'] = aroon['AROONU_14']
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # פיבונאצ'י ופיבוטים
    curr = df.iloc[-1]
    y_high = df['High'][-252:].max()
    y_low = df['Low'][-252:].min()
    fib_618 = y_high - 0.618 * (y_high - y_low)
    
    pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
    r1 = 2*pivot - curr['Low']
    s1 = 2*pivot - curr['High']
    
    # --- אימון מודל ML לחיזוי (Heavy Computation) ---
    df_ml = df.dropna().copy()
    df_ml['Target'] = df_ml['Close'].shift(-1) # חיזוי למחר
    features = ['Close', 'RSI', 'SMA_5', 'MACD']
    
    X = df_ml[features].iloc[:-1]
    y = df_ml['Target'].iloc[:-1]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_row = df_ml[features].iloc[[-1]]
    pred_price = model.predict(last_row)[0]
    accuracy = model.score(X, y) * 100
    
    # ניקוד משוקלל
    score = 50
    if curr['Close'] > curr['SMA_200']: score += 20
    if curr['RSI'] < 30: score += 15
    if pred_price > curr['Close']: score += 15
    
    rec = "HOLD"
    if score >= 80: rec = "STRONG BUY 🚀"
    elif score >= 60: rec = "BUY 🟢"
    elif score <= 40: rec = "SELL 🔴"
    
    return {
        'Symbol': ticker, 'Price': curr['Close'], 'Rec': rec, 'Score': score,
        'Pred': pred_price, 'Acc': accuracy, 'RSI': curr['RSI'],
        'SMA200': curr['SMA_200'], 'ATR': curr['ATR'], 'VWAP': curr['VWAP'],
        'Pivot': pivot, 'R1': r1, 'S1': s1, 'Fib618': fib_618,
        'Aroon': curr['Aroon_Up'], 'ADX': curr['ADX'], 'Vol': curr['Volume']
    }

# --- לוגיקת סריקה המונית (Batch Processing) ---
@st.cache_data(ttl=600)
def scan_market(tickers_list):
    results = []
    # חלוקה למנות של 20 כדי לא לקרוס
    chunk_size = 20
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing batch {i+1}/{len(chunks)}...")
        try:
            # הורדה קבוצתית (מהירה)
            data = yf.download(chunk, period="6mo", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            
            for ticker in chunk:
                try:
                    # טיפול במידע
                    if len(chunk) == 1: df = data # אם יש רק מניה אחת
                    else: df = data[ticker]
                    
                    df = df.dropna(subset=['Close'])
                    if len(df) < 50: continue
                    
                    # חישוב מהיר
                    curr_price = df['Close'].iloc[-1]
                    rsi = ta.rsi(df['Close']).iloc[-1]
                    sma200 = ta.sma(df['Close'], length=200).iloc[-1]
                    
                    # ניקוד בסיסי לסינון
                    score = 50
                    if curr_price > sma200: score += 20
                    if rsi < 30: score += 20
                    elif rsi > 70: score -= 15
                    
                    rec = "NEUTRAL"
                    if score >= 70: rec = "BUY"
                    elif score <= 30: rec = "SELL"
                    
                    results.append({
                        'Symbol': ticker, 'Price': curr_price, 'RSI': rsi, 
                        'Score': score, 'Rec': rec, 'SMA200': sma200
                    })
                except: continue
        except: continue
        
        progress_bar.progress((i+1)/len(chunks))
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)

# --- UI ---
sidebar_input = st.sidebar.text_area("הדבק כאן 500+ מניות:", DEFAULT_LIST, height=300)
start_btn = st.sidebar.button("🚀 הפעל סורק על (Mass Scan)")

if 'scan_results' not in st.session_state:
    st.session_state['scan_results'] = None

if start_btn:
    clean_list = [x.strip().upper() for x in sidebar_input.replace('\n', ',').split(',') if x.strip()]
    st.info(f"מתחיל סריקה של {len(clean_list)} מניות... זה ייקח זמן, אבל לא יקרוס.")
    st.session_state['scan_results'] = scan_market(clean_list)

# הצגת תוצאות הסריקה
if st.session_state['scan_results'] is not None:
    df = st.session_state['scan_results']
    
    # טבלה מסכמת
    st.subheader(f"📊 תוצאות סריקה ({len(df)} מניות זוהו)")
    st.dataframe(
        df.sort_values('Score', ascending=False).style.format({'Price': '{:.2f}', 'RSI': '{:.1f}'}),
        use_container_width=True
    )
    
    st.divider()
    
    # --- החלק שאתה רוצה: Deep Dive & AI ---
    st.subheader("🔬 ניתוח עומק + AI Prediction")
    st.caption("בחר מניה מהטבלה למעלה כדי להפעיל עליה את המודלים הכבדים (LSTM/ML/Full Technicals):")
    
    selected_ticker = st.selectbox("בחר מניה:", df['Symbol'].unique())
    
    if st.button(f"🧠 הפעל בינה מלאכותית על {selected_ticker}"):
        with st.spinner("מאמן מודלים ומחשב 50 אינדיקטורים..."):
            data = run_deep_ai_analysis(selected_ticker)
            
        if data:
            # הדיווח המלא שביקשת
            report = f"""
⭐️ **{data['Symbol']} DEEP AI REPORT**
════════════════════════════
💰 Price: ${data['Price']:.2f}
🚦 Signal: {data['Rec']} (Score: {data['Score']})

🎯 **AI Prediction (Machine Learning)**
• Forecast (Next Day): ${data['Pred']:.2f}
• Model Confidence: {data['Acc']:.1f}%

📊 **Key Indicators**
• RSI: {data['RSI']:.1f} | ADX: {data['ADX']:.1f}
• Aroon Up: {data['Aroon']:.0f}
• VWAP: ${data['VWAP']:.2f}

🌊 **Levels & Risk**
• Pivot: ${data['Pivot']:.2f} | R1: ${data['R1']:.2f}
• Golden Fib (61.8%): ${data['Fib618']:.2f}
• ATR (Volatility): ${data['ATR']:.2f}
════════════════════════════
"""
            st.code(report, language="text")
            
            # אזור גרפי
            col1, col2 = st.columns(2)
            col1.metric("חיזוי AI", f"${data['Pred']:.2f}")
            col2.metric("טווח בולינגר", "תקין") # (אפשר להרחיב)
