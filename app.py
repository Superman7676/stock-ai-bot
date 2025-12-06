import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Sniper Elite", layout="wide", page_icon="🦅")
st.title("🦅 AI Sniper Elite - Full Technical Analysis")
st.markdown("""
**מערכת סריקה מלאה:** זיהוי תבניות נרות (Candles) | פיבונאצ'י | ניהול סיכונים | כל האינדיקטורים
""")

# --- פונקציה לזיהוי תבניות נרות (Candlestick Patterns) ---
def analyze_candles(open_p, high, low, close, prev_open, prev_close):
    body = abs(close - open_p)
    range_len = high - low
    if range_len == 0: return "Flat"
    
    upper_wick = high - max(close, open_p)
    lower_wick = min(close, open_p) - low
    
    pattern = "Normal"
    color = "🟢" if close > open_p else "🔴"
    
    # 1. Doji (נר של אי ודאות)
    if body <= 0.1 * range_len:
        pattern = "Doji ➕"
    
    # 2. Hammer (פטיש - סימן להיפוך למעלה)
    elif lower_wick > 2 * body and upper_wick < body:
        pattern = "Hammer 🔨 (Reversal?)"
        
    # 3. Shooting Star (כוכב נופל - סימן להיפוך למטה)
    elif upper_wick > 2 * body and lower_wick < body:
        pattern = "Shooting Star 🌠 (Bearish)"
        
    # 4. Marubozu (נר חזק בלי זנבות)
    elif body > 0.85 * range_len:
        pattern = "Marubozu 💪"
        
    # 5. Engulfing (בולען)
    prev_body = abs(prev_close - prev_open)
    if body > prev_body:
        if close > open_p and prev_close < prev_open: # ירוק בולע אדום
             pattern = "Bullish Engulfing 🐮"
        elif close < open_p and prev_close > prev_open: # אדום בולע ירוק
             pattern = "Bearish Engulfing 🐻"

    return f"{color} {pattern}"

# --- פונקציה ראשית לניתוח מניה בודדת ---
def analyze_stock(ticker):
    try:
        # הורדת נתונים (שנה אחורה)
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        # טיפול במבנה נתונים (MultiIndex fix)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=0)
            except:
                pass 

        if df.empty or len(df) < 200: return None

        # --- 1. חישוב אינדיקטורים (הכל) ---
        # ממוצעים
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # מתנדים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_H'] = macd['MACDh_12_26_9']
        
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx['ADX_14']
        
        # בולינגר
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_U'] = bb['BBU_5_2.0']
        df['BB_L'] = bb['BBL_5_2.0']
        
        # ATR (תנודתיות)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # נתונים נוכחיים
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 2. זיהוי תבניות נרות ---
        candle_pattern = analyze_candles(curr['Open'], curr['High'], curr['Low'], curr['Close'], prev['Open'], prev['Close'])

        # --- 3. פיבונאצ'י ופיבוטים ---
        # פיבונאצ'י שנתי
        year_high = df['High'].max()
        year_low = df['Low'].min()
        fib_618 = year_high - (0.618 * (year_high - year_low))
        
        # פיבוטים קלאסיים
        pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = (2 * pivot) - curr['Low']
        s1 = (2 * pivot) - curr['High']
        
        # --- 4. ניקוד AI ---
        score = 50
        trend = "Neutral"
        
        # מגמה
        if curr['Close'] > curr['SMA_200']: 
            score += 15
            trend = "Bullish 📈"
        else:
            score -= 10
            trend = "Bearish 📉"
            
        # RSI
        if curr['RSI'] < 30: score += 20
        elif curr['RSI'] > 75: score -= 15
        
        # MACD
        if curr['MACD_H'] > 0 and curr['MACD_H'] > prev['MACD_H']: score += 10 # מומנטום עולה
        
        # ADX (עוצמת מגמה)
        if curr['ADX'] > 25: score += 5 
        
        # נרות
        if "Bullish" in candle_pattern or "Hammer" in candle_pattern: score += 10
        if "Bearish" in candle_pattern or "Shooting" in candle_pattern: score -= 10

        final_score = min(max(score, 0), 100)
        
        rec = "HOLD"
        if final_score >= 80: rec = "STRONG BUY 🚀"
        elif final_score >= 65: rec = "BUY 🟢"
        elif final_score <= 35: rec = "SELL 🔴"
        
        return {
            'Symbol': ticker,
            'Price': round(curr['Close'], 2),
            'Change%': round(((curr['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            'Rec': rec,
            'Score': int(final_score),
            'Candle': candle_pattern,
            'Trend': trend,
            'RSI': round(curr['RSI'], 1),
            'MACD': round(curr['MACD'], 2),
            'ADX': round(curr['ADX'], 1),
            'SMA_200': round(curr['SMA_200'], 2),
            'Dist_SMA200': round(((curr['Close'] - curr['SMA_200'])/curr['SMA_200'])*100, 1),
            'ATR': round(curr['ATR'], 2),
            'VWAP': round(curr['VWAP'], 2),
            'Pivot': round(pivot, 2),
            'R1': round(r1, 2),
            'S1': round(s1, 2),
            'Fib_618': round(fib_618, 2),
            'Vol_M': round(curr['Volume'] / 1000000, 2)
        }
    except Exception as e:
        return None

# --- רשימת המניות (מלאה) ---
ALL_TICKERS = [
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 'MELI', 'DELL',
    'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AVGO', 'CRM', 'ORCL', 'INTU', 'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST',
    'FICO', 'EQIX', 'SPY', 'AXON', 'SNPS', 'TLN', 'ETN', 'RDDT', 'SNOW', 'PANW',
    'ICLR', 'VST', 'LRCX', 'DDOG', 'TWLO', 'BSX', 'NBIS', 'RBLX', 'AFRM', 'CELH',
    'JD', 'TTD', 'KVUE', 'NET', 'DKNG', 'CVNA', 'ZS', 'CRWD', 'SITM', 'POWL', 'STRL'
]
# הערה: לשימוש אמיתי תוסיף כאן את שאר הרשימה שלך, כרגע שמתי ~60 כדי שזה ירוץ מהר להדגמה

if st.button('🔥 הפעל סריקה (Deep Scan)'):
    st.write("מתחיל לעבד מניות... אנא המתן, זה לוקח זמן כי אנחנו מחשבים המון נתונים.")
    
    results = []
    prog_bar = st.progress(0)
    status = st.empty()
    
    # לולאה בטוחה (אחת אחת) למניעת קריסות
    for i, ticker in enumerate(ALL_TICKERS):
        status.text(f"בודק את {ticker} ({i+1}/{len(ALL_TICKERS)})...")
        res = analyze_stock(ticker)
        if res:
            results.append(res)
        
        prog_bar.progress((i + 1) / len(ALL_TICKERS))
    
    status.success("✅ הסריקה הושלמה!")
    prog_bar.empty()
    
    if results:
        df = pd.DataFrame(results)
        
        # --- 1. Top Opportunities ---
        st.subheader("🏆 ההזדמנויות הטובות ביותר (Top 5)")
        st.dataframe(df.sort_values('Score', ascending=False).head(5), use_container_width=True)
        
        # --- 2. כרטיס מניה מפורט (Telegram Style) ---
        st.divider()
        st.subheader("🔬 כרטיס ניתוח מלא (כפי שביקשת)")
        
        selected = st.selectbox("בחר מניה להצגת דוח מלא:", df['Symbol'].tolist())
        row = df[df['Symbol'] == selected].iloc[0]
        
        # חישוב יעד וסטופ
        stop_loss = row['Price'] - (2 * row['ATR'])
        target = row['R1']
        
        report = f"""
🚨 **{row['Symbol']} - TECHNICAL REPORT** 🚨
════════════════════════════════
💰 **Price:** ${row['Price']} ({row['Change%']}%)
🚦 **Signal:** {row['Rec']} (Score: {row['Score']})
🕯️ **Candle:** {row['Candle']}

📊 **Trend & Momentum**
• Trend: {row['Trend']} (vs SMA200)
• RSI: {row['RSI']} | ADX: {row['ADX']} (Strength)
• MACD: {row['MACD']}
• VWAP: ${row['VWAP']}

🎯 **Targets & Levels**
• Pivot Point: ${row['Pivot']}
• Resistance (R1): ${row['R1']}
• Support (S1): ${row['S1']}
• Golden Fib (61.8%): ${row['Fib_618']}

🛡️ **Risk Management**
• Volatility (ATR): ${row['ATR']}
• Suggested Stop: ${stop_loss:.2f}
• Next Target: ${target:.2f}
════════════════════════════════
"""
        st.info(report) # מציג את הדוח בתוך קופסה כחולה יפה
        st.code(report, language="text") # מציג את הדוח כטקסט להעתקה
        
        # --- 3. טבלה מלאה להורדה ---
        st.divider()
        st.subheader("📥 כל הנתונים")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("הורד דוח Excel מלא", csv, "ai_sniper_report.csv", "text/csv")
        
    else:
        st.error("לא נמצאו נתונים. בדוק את החיבור לאינטרנט או נסה שוב.")
