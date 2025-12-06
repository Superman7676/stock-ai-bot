import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# --- הגדרות ---
st.set_page_config(page_title="AI Sniper Ultimate", layout="wide", page_icon="🧠")
st.title("🧠 AI Sniper Ultimate - ML & Backtesting")

# --- רשימת מניות ברירת מחדל ---
DEFAULT_TICKERS = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציות עזר וטיפול בנתונים ---
def fix_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    # הסרת שורות ללא מידע
    df = df.dropna(subset=['Close'])
    return df

# --- מנוע Machine Learning (XGBoost Style) ---
def train_ai_model(df):
    # הכנת הדאטה ללמידה
    df_ml = df.copy()
    
    # יצירת פיצ'רים (Features) למודל ללמוד מהם
    df_ml['Returns'] = df_ml['Close'].pct_change()
    df_ml['SMA_Diff'] = df_ml['Close'] - ta.sma(df_ml['Close'], length=50)
    df_ml['RSI'] = ta.rsi(df_ml['Close'], length=14)
    df_ml['Volatility'] = ta.atr(df_ml['High'], df_ml['Low'], df_ml['Close'], length=14)
    
    # Target: אנחנו רוצים לחזות את המחיר בעוד 3 ימים
    df_ml['Target'] = df_ml['Close'].shift(-3)
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 50: return 0, 0 # אין מספיק דאטה ללמידה
    
    features = ['Close', 'Returns', 'SMA_Diff', 'RSI', 'Volatility']
    X = df_ml[features]
    y = df_ml['Target']
    
    # פיצול לאימון ומבחן
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # אימון מודל Gradient Boosting (דומה ל-XGBoost)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # ביצוע חיזוי על הנתונים העדכניים ביותר
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)[0]
    
    # דיוק המודל (R2 Score) על סט הבדיקה
    accuracy_score = model.score(X_test, y_test)
    
    return prediction, accuracy_score

# --- מנוע Backtesting (בדיקה לאחור) ---
def run_backtest(df):
    # אסטרטגיה פשוטה לבדיקה: קנה כשה-RSI נמוך ומעל ממוצע 200, מכור כשה-RSI גבוה
    # זו סימולציה היסטורית
    df_bt = df.copy()
    df_bt['SMA_200'] = ta.sma(df_bt['Close'], length=200)
    df_bt['RSI'] = ta.rsi(df_bt['Close'], length=14)
    
    capital = 10000 # דולר התחלתי
    position = 0
    df_bt['Signal'] = 0 # 1=Buy, -1=Sell
    
    # לוגיקת מסחר וקטורית מהירה
    buy_cond = (df_bt['RSI'] < 40) & (df_bt['Close'] > df_bt['SMA_200'])
    sell_cond = (df_bt['RSI'] > 70)
    
    df_bt.loc[buy_cond, 'Signal'] = 1
    df_bt.loc[sell_cond, 'Signal'] = -1
    
    # חישוב תשואה
    df_bt['Market_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Market_Return'] * df_bt['Signal'].shift(1)
    
    total_return = (df_bt['Strategy_Return'].fillna(0) + 1).cumprod().iloc[-1] - 1
    market_return = (df_bt['Market_Return'].fillna(0) + 1).cumprod().iloc[-1] - 1
    
    return total_return * 100, market_return * 100 # באחוזים

# --- מנוע אינדיקטורים טכניים (כל מה שביקשת) ---
def calculate_technicals(df):
    # Aroon
    aroon = ta.aroon(df['High'], df['Low'], length=14)
    df['Aroon_Up'] = aroon['AROONU_14']
    df['Aroon_Down'] = aroon['AROOND_14']
    
    # MAs
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    
    # VWAP
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    
    # RSI & ADX
    df['RSI'] = ta.rsi(df['Close'], length=14)
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx['ADX_14']
    
    # Bollinger
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_U'] = bb['BBU_5_2.0']
    df['BB_L'] = bb['BBL_5_2.0']
    
    # ATR
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    return df

# --- מנתח מניה בודד ---
def analyze_stock_full(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        df = fix_data(df)
        
        if df.empty or len(df) < 200: return None
        
        # 1. חישוב כל האינדיקטורים
        df = calculate_technicals(df)
        curr = df.iloc[-1]
        
        # 2. הרצת מודל AI (חיזוי)
        ai_pred, ai_accuracy = train_ai_model(df)
        ai_upside = ((ai_pred - curr['Close']) / curr['Close']) * 100
        
        # 3. הרצת Backtest (היסטוריה)
        strat_perf, market_perf = run_backtest(df)
        
        # 4. זיהוי תבניות
        patterns = []
        body = abs(curr['Close'] - curr['Open'])
        full_range = curr['High'] - curr['Low']
        
        if curr['Close'] > curr['Open'] and body > 0.8 * full_range: patterns.append("Big Green Candle")
        if (min(curr['Close'], curr['Open']) - curr['Low']) > 2 * body: patterns.append("Hammer")
        if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] and df['SMA_50'].iloc[-2] < df['SMA_200'].iloc[-2]: patterns.append("Golden Cross")
        
        pattern_str = ", ".join(patterns) if patterns else "None"
        
        # 5. ניקוד משוקלל
        score = 50
        # טכני
        if curr['Close'] > curr['SMA_200']: score += 15
        if curr['Aroon_Up'] > 70: score += 10
        if curr['RSI'] < 30: score += 15
        if curr['VWAP'] < curr['Close']: score += 10
        # AI
        if ai_upside > 2: score += 15
        # Backtest
        if strat_perf > market_perf: score += 5
        
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY 🚀"
        elif score >= 60: rec = "BUY 🟢"
        elif score <= 30: rec = "SELL 🔴"
        
        # חישוב רמות
        pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
        
        return {
            'Symbol': ticker,
            'Price': curr['Close'],
            'Rec': rec,
            'Score': score,
            'AI_Pred': ai_pred,
            'AI_Upside': ai_upside,
            'AI_Conf': ai_accuracy * 100, # אחוז ביטחון של המודל
            'Backtest_Perf': strat_perf,
            'Market_Perf': market_perf,
            'RSI': curr['RSI'],
            'Aroon': curr['Aroon_Up'],
            'VWAP': curr['VWAP'],
            'ATR': curr['ATR'],
            'Pattern': pattern_str,
            'Pivot': pivot
        }
        
    except Exception as e:
        return None

# --- UI ---
user_input = st.sidebar.text_area("רשימת מניות:", DEFAULT_TICKERS, height=300)
run_btn = st.sidebar.button("🚀 הפעל ניתוח מלא (ML + Backtest)")

if run_btn:
    tickers = [t.strip().upper() for t in user_input.split(',') if t.strip()]
    
    st.info(f"מנתח {len(tickers)} מניות... מאמן מודלים ומריץ Backtest לכל אחת. אנא המתן.")
    
    results = []
    progress = st.progress(0)
    
    for i, t in enumerate(tickers):
        data = analyze_stock_full(t)
        if data: results.append(data)
        progress.progress((i+1)/len(tickers))
        
    progress.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        
        # טבלה ראשית
        st.subheader("🏆 AI & Backtest Results")
        st.dataframe(
            df_res[['Symbol', 'Price', 'Rec', 'Score', 'AI_Upside', 'Backtest_Perf', 'Pattern', 'RSI']]
            .sort_values('Score', ascending=False)
            .style.format({'Price': '{:.2f}', 'AI_Upside': '{:.2f}%', 'Backtest_Perf': '{:.2f}%', 'RSI': '{:.1f}'}),
            use_container_width=True
        )
        
        st.divider()
        st.subheader("🔬 דוח עומק (כולל מודלים)")
        
        sel = st.selectbox("בחר מניה:", df_res['Symbol'].tolist())
        row = df_res[df_res['Symbol'] == sel].iloc[0]
        
        # דוח טלגרם מלא
        report = f"""
🧠 **{row['Symbol']} DEEP AI ANALYSIS** 🧠
══════════════════════════════════
💰 Price: ${row['Price']:.2f}
🚦 Signal: {row['Rec']} (Score: {row['Score']})

🤖 **Machine Learning Model (Gradient Boosting)**
• Prediction (3 Days): ${row['AI_Pred']:.2f}
• Potential Upside: {row['AI_Upside']:.2f}%
• Model Confidence (R2): {row['AI_Conf']:.1f}%

🔙 **Backtesting (1 Year Strategy)**
• Strategy Return: {row['Backtest_Perf']:.2f}%
• Buy & Hold Return: {row['Market_Perf']:.2f}%
• Alpha: {row['Backtest_Perf'] - row['Market_Perf']:.2f}%

📊 **Advanced Indicators**
• Aroon Up: {row['Aroon']:.0f} (Trend Strength)
• VWAP: ${row['VWAP']:.2f}
• RSI: {row['RSI']:.1f} | ATR: ${row['ATR']:.2f}
• Pattern: {row['Pattern']}

🎯 **Key Levels**
• Pivot Point: ${row['Pivot']:.2f}
• Stop Loss: ${row['Price'] - 2*row['ATR']:.2f}
══════════════════════════════════
"""
        st.code(report, language="text")
        
    else:
        st.error("לא נמצאו נתונים. נסה שוב.")
