import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Pro", layout="wide", page_icon="💎")

st.markdown("""
<style>
    .report-box {
        background-color: #0e1117;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 AI Trading Command Center")
st.markdown("מערכת סריקה יציבה | זיהוי תבניות | פיבונאצ'י | ללא קריסות")

# --- רשימת המניות המלאה שלך (כטקסט כדי לא להעמיס על הזיכרון בהתחלה) ---
DEFAULT_LIST = """NVDA, ALAB, CLSK, PLTR, AMD, TSLA, MSFT, UBER, MELI, DELL,
VRT, COHR, LITE, SMCI, MDB, SOFI, GOOGL, AMZN, META, NFLX,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN, HOOD, UPST,
FICO, EQIX, SPY, AXON, SNPS, TLN, ETN, RDDT, SNOW, PANW,
ICLR, VST, LRCX, DDOG, TWLO, BSX, NBIS, RBLX, AFARM, CELH"""

# --- סרגל צד לשליטה ---
st.sidebar.header("הגדרות סריקה")
user_tickers = st.sidebar.text_area("רשימת מניות (מופרדות בפסיק)", DEFAULT_LIST, height=300)
scan_button = st.sidebar.button("🚀 הפעל סריקה עכשיו")

# --- פונקציות ניתוח ---
def identify_candle(open_p, high, low, close):
    body = abs(close - open_p)
    wick_upper = high - max(close, open_p)
    wick_lower = min(close, open_p) - low
    
    if body < 0.1 * (high - low): return "Doji ➕"
    if wick_lower > 2 * body and wick_upper < body: return "Hammer 🔨"
    if wick_upper > 2 * body and wick_lower < body: return "Shooting Star 🌠"
    if body > 0.8 * (high - low) and close > open_p: return "Big Green 💪"
    return "Normal"

def analyze_stock_safe(ticker):
    try:
        # משיכת נתונים עם השהייה למניעת חסימה
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        time.sleep(0.1) # נותן לשרת לנשום

        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=0)
            except: pass
            
        if df.empty or len(df) < 50: return None

        # אינדיקטורים
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        
        # Bollinger
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_U'] = bb['BBU_5_2.0']

        curr = df.iloc[-1]
        
        # זיהוי נרות
        candle = identify_candle(curr['Open'], curr['High'], curr['Low'], curr['Close'])
        
        # פיבונאצ'י שנתי
        year_high = df['High'].max()
        year_low = df['Low'].min()
        fib_618 = year_high - (0.618 * (year_high - year_low))
        
        # פיבוטים
        pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = (2 * pivot) - curr['Low']
        s1 = (2 * pivot) - curr['High']

        # ציון
        score = 50
        if curr['Close'] > curr['SMA_200']: score += 20
        if curr['RSI'] < 30: score += 20
        if curr['RSI'] > 70: score -= 15
        if curr['MACD'] > 0: score += 10
        if "Hammer" in candle: score += 10
        
        final_score = min(max(score, 0), 100)
        
        rec = "HOLD"
        if final_score >= 80: rec = "STRONG BUY 🚀"
        elif final_score >= 60: rec = "BUY 🟢"
        elif final_score <= 30: rec = "SELL 🔴"

        return {
            'Symbol': ticker.strip().upper(),
            'Price': curr['Close'],
            'Score': final_score,
            'Rec': rec,
            'RSI': curr['RSI'],
            'Candle': candle,
            'Pivot': pivot,
            'R1': r1, 'S1': s1,
            'Fib_618': fib_618,
            'ATR': curr['ATR'],
            'SMA_200': curr['SMA_200']
        }
    except:
        return None

# --- הלוגיקה הראשית ---
if scan_button:
    tickers = [t.strip() for t in user_tickers.split(',') if t.strip()]
    
    st.info(f"מתחיל סריקה של {len(tickers)} מניות... זה ייקח קצת זמן כדי לא לקרוס.")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"בודק: {ticker} ({i+1}/{len(tickers)})")
        data = analyze_stock_safe(ticker)
        if data:
            results.append(data)
        progress_bar.progress((i + 1) / len(tickers))
        
    status_text.empty()
    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        
        # חלק עליון - TOP 5
        st.subheader("🏆 Top Opportunities")
        st.dataframe(df_res.sort_values('Score', ascending=False).head(5), use_container_width=True)
        
        # חלק תחתון - יוצר הדוחות
        st.divider()
        st.subheader("📝 מחולל דוחות (Telegram Style)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_stock = st.radio("בחר מניה לדוח:", df_res['Symbol'].tolist())
            
        with col2:
            if selected_stock:
                row = df_res[df_res['Symbol'] == selected_stock].iloc[0]
                
                # חישוב סטופ ויעד
                stop = row['Price'] - (2 * row['ATR'])
                target = row['R1']
                
                report = f"""
🚨 **{row['Symbol']} REPORT** 🚨
══════════════════════
💰 Price: ${row['Price']:.2f}
🚦 Signal: {row['Rec']} (Score: {row['Score']})
🕯️ Candle: {row['Candle']}

📊 **Technicals**
• RSI: {row['RSI']:.1f}
• vs SMA200: {'Above 🟢' if row['Price'] > row['SMA_200'] else 'Below 🔴'}
• Pivot: ${row['Pivot']:.2f}

🎯 **Levels**
• Resistance (R1): ${row['R1']:.2f}
• Support (S1): ${row['S1']:.2f}
• Golden Fib: ${row['Fib_618']:.2f}

🛡️ **Trade Setup**
• Stop Loss: ${stop:.2f}
• Target: ${target:.2f}
══════════════════════
"""
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                st.code(report, language="text") # להעתקה קלה
        
        # הורדה
        st.divider()
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 הורד אקסל מלא", csv, "full_report.csv", "text/csv")
        
    else:
        st.error("לא הצלחנו למשוך נתונים. נסה שוב או צמצם את הרשימה.")

else:
    st.write("👈 ערוך את רשימת המניות משמאל ולחץ על 'הפעל סריקה'")
