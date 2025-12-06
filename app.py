import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# --- הגדרות ---
st.set_page_config(page_title="AI Future Predictor", layout="wide", page_icon="🔮")
st.title("🔮 AI Future Predictor & Deep Analysis")

# --- רשימת מניות ---
DEFAULT_TICKERS = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציית תיקון נתונים ---
def fix_yahoo_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df

# --- מודל חיזוי וניבוי (The AI Core) ---
def predict_price(df, days_ahead=5):
    # הכנת הנתונים ללמידה
    df = df.copy().dropna()
    df['Day'] = np.arange(len(df))
    
    # 1. רגרסיה לינארית (חישוב המגמה המתמטית)
    X = df[['Day']].tail(30) # לומד מה-30 יום האחרונים
    y = df['Close'].tail(30)
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = df['Day'].iloc[-1] + days_ahead
    prediction_lin = model.predict([[next_day]])[0]
    
    # 2. חישוב תנודתיות לטווח חיזוי (Confidence Interval)
    volatility = df['Close'].pct_change().std()
    current_price = df['Close'].iloc[-1]
    
    # טווח עליון ותחתון משוער (Monte Carlo style simplification)
    upper_bound = prediction_lin * (1 + (volatility * np.sqrt(days_ahead)))
    lower_bound = prediction_lin * (1 - (volatility * np.sqrt(days_ahead)))
    
    return round(prediction_lin, 2), round(upper_bound, 2), round(lower_bound, 2)

# --- זיהוי תבניות מתקדם ---
def get_advanced_patterns(df):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    open_p, close, high, low = curr['Open'], curr['Close'], curr['High'], curr['Low']
    
    patterns = []
    
    # גוף ונרות
    body = abs(close - open_p)
    full_range = high - low
    
    # Hammer
    if (min(open_p, close) - low) > 2 * body and (high - max(open_p, close)) < body:
        patterns.append("Hammer 🔨 (Possible Reversal)")
        
    # Engulfing Bullish
    if close > open_p and prev['Close'] < prev['Open'] and close > prev['Open'] and open_p < prev['Close']:
        patterns.append("Bullish Engulfing 🐮 (Strong Buy Signal)")
        
    # Golden Cross (SMA50 חוצה את SMA200)
    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] and df['SMA_50'].iloc[-2] < df['SMA_200'].iloc[-2]:
        patterns.append("Golden Cross ✨ (Major Uptrend)")
        
    return ", ".join(patterns) if patterns else "No Clear Pattern"

# --- המנתח הראשי ---
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        df = fix_yahoo_data(df)
        
        if df.empty or len(df) < 200: return None
        
        # אינדיקטורים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # חיזוי לעוד 5 ימים
        pred_price, pred_high, pred_low = predict_price(df, days_ahead=7)
        
        # נתונים נוכחיים
        curr = df.iloc[-1]
        
        # ציון AI משוקלל
        score = 50
        # מומנטום
        if curr['Close'] > curr['SMA_200']: score += 15
        if curr['RSI'] < 30: score += 20
        if pred_price > curr['Close']: score += 15 # המודל צופה עליה
        
        # תבניות
        patterns = get_advanced_patterns(df)
        if "Bullish" in patterns or "Hammer" in patterns: score += 10
        
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 30: rec = "SELL 🔴"
        
        return {
            'Symbol': ticker,
            'Price': curr['Close'],
            'Rec': rec,
            'Score': score,
            'Predicted_7d': pred_price,
            'Pred_Range': f"${pred_low} - ${pred_high}",
            'Upside%': round(((pred_price - curr['Close']) / curr['Close']) * 100, 2),
            'Pattern': patterns,
            'RSI': curr['RSI'],
            'SMA_200': curr['SMA_200'],
            'ATR': curr['ATR']
        }
    except Exception as e:
        return None

# --- UI ---
user_input = st.text_area("רשימת מניות:", DEFAULT_TICKERS, height=100)

if st.button("🔮 הפעל מודל חיזוי וניתוח"):
    tickers = [t.strip().upper() for t in user_input.split(',') if t.strip()]
    
    st.info(f"מריץ מודלים של חיזוי (Regression & Monte Carlo) על {len(tickers)} מניות...")
    
    results = []
    bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        res = analyze_stock(t)
        if res: results.append(res)
        else:
            time.sleep(0.5)
            res = analyze_stock(t) # Retry
            if res: results.append(res)
        bar.progress((i+1)/len(tickers))
        
    bar.empty()
    
    if results:
        df = pd.DataFrame(results)
        
        st.subheader("🤖 AI Forecast Results")
        # טבלה שמתמקדת בחיזוי
        st.dataframe(
            df[['Symbol', 'Price', 'Predicted_7d', 'Upside%', 'Rec', 'Score', 'Pattern']]
            .sort_values('Score', ascending=False)
            .style.format({"Price": "{:.2f}", "Predicted_7d": "{:.2f}", "Upside%": "{:.2f}%"}),
            use_container_width=True
        )
        
        st.divider()
        st.subheader("🧠 Deep Dive & Prediction Logic")
        
        sel = st.selectbox("בחר מניה לניתוח עומק:", df['Symbol'].tolist())
        row = df[df['Symbol'] == sel].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("מחיר נוכחי", f"${row['Price']:.2f}")
        col2.metric("חיזוי AI (7 ימים)", f"${row['Predicted_7d']:.2f}", f"{row['Upside%']}%")
        col3.metric("טווח צפוי", row['Pred_Range'])
        
        report = f"""
🔮 **AI PREDICTION REPORT: {row['Symbol']}**
══════════════════════════════════
🤖 **Model Forecast (Linear Regression):**
Based on the trend of the last 30 days, the model predicts
a price of **${row['Predicted_7d']}** within 7 days.
Potential Upside: **{row['Upside%']}%**

📊 **Technical Signal:**
• Recommendation: {row['Rec']} (AI Score: {row['Score']})
• Pattern Detected: {row['Pattern']}
• RSI Strength: {row['RSI']:.1f}

🛡️ **Risk Parameters:**
• Volatility (ATR): ${row['ATR']:.2f}
• Stop Loss Suggested: ${row['Price'] - 2*row['ATR']:.2f}
══════════════════════════════════
"""
        st.code(report, language="text")
        
    else:
        st.error("No data found. Check connection.")
