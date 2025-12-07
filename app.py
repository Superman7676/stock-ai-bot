import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# --- הגדרות עמוד (מסך מלא) ---
st.set_page_config(page_title="Mega AI Scanner", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    /* הרחבת הטבלה למקסימום */
    .block-container { max-width: 95% !important; padding-top: 1rem; }
    /* עיצוב כותרות הטבלה */
    th { font-size: 16px !important; color: #4F8BF9 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 AI Hedge Fund - The Mega Table")

# --- רשימת ברירת מחדל ---
DEFAULT_LIST = """NVDA, TSLA, AMD, PLTR, MSFT, GOOGL, AMZN, META,
ALAB, CLSK, COHR, VRT, LITE, SMCI, MDB, SOFI,
AVGO, CRM, ORCL, INTU, RIVN, MARA, RIOT, IREN"""

# --- פונקציות ליבה ---
def get_slope(series):
    """חישוב שיפוע המגמה (Linear Regression)"""
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

def analyze_stock_fast(df, ticker):
    try:
        # נתונים בסיסיים
        curr = df.iloc[-1]
        close = curr['Close']
        
        # --- אינדיקטורים ---
        rsi = ta.rsi(df['Close'], length=14).iloc[-1]
        sma50 = ta.sma(df['Close'], length=50).iloc[-1]
        sma200 = ta.sma(df['Close'], length=200).iloc[-1]
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        vwap = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
        
        # MACD
        macd = ta.macd(df['Close'])
        macd_val = macd['MACD_12_26_9'].iloc[-1]
        
        # ADX
        adx = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14'].iloc[-1]
        
        # --- AI Prediction (Regression Based) ---
        # חיזוי ל-5 ימים קדימה על בסיס המומנטום האחרון
        slope = get_slope(df['Close'].tail(14))
        pred_price = close + (slope * 5)
        upside = ((pred_price - close) / close) * 100
        
        # --- Levels ---
        p = (curr['High'] + curr['Low'] + curr['Close']) / 3
        r1 = 2*p - curr['Low']
        
        # פיבונאצ'י שנתי
        h = df['High'][-252:].max()
        l = df['Low'][-252:].min()
        fib618 = h - 0.618 * (h - l)
        
        # --- Scoring & Logic ---
        score = 50
        if close > sma200: score += 20
        if rsi < 35: score += 15
        if macd_val > 0: score += 10
        if upside > 2: score += 15
        
        rec = "HOLD"
        if score >= 80: rec = "STRONG BUY"
        elif score >= 65: rec = "BUY"
        elif score <= 35: rec = "SELL"
        
        return {
            "Symbol": ticker,
            "Price": close,
            "Change%": ((close - df.iloc[-2]['Close'])/df.iloc[-2]['Close'])*100,
            "Signal": rec,
            "Score": score,
            "AI_Pred": pred_price,
            "Upside%": upside,
            "RSI": rsi,
            "Trend": "Bull 🟢" if close > sma200 else "Bear 🔴",
            "ADX": adx,
            "ATR": atr,
            "Pivot": p,
            "Res_R1": r1,
            "Fib_618": fib618,
            "VWAP": vwap,
            "Volume": curr['Volume'] / 1000000
        }
    except:
        return None

# --- מנוע הסריקה ---
def run_mega_scan(tickers_list):
    results = []
    chunk_size = 50 
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    
    prog = st.progress(0)
    status = st.empty()
    
    for i, chunk in enumerate(chunks):
        status.text(f"סורק קבוצה {i+1} מתוך {len(chunks)}...")
        try:
            # הורדה קבוצתית (מהירה ויעילה)
            data = yf.download(chunk, period="2y", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            
            for t in chunk:
                try:
                    df = data[t] if len(chunk) > 1 else data
                    if df.empty or len(df) < 50: continue
                    
                    # תיקון נתונים
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=['Close'])
                    
                    res = analyze_stock_fast(df, t)
                    if res: results.append(res)
                except: continue
        except: continue
        prog.progress((i+1)/len(chunks))
        
    prog.empty()
    status.empty()
    return pd.DataFrame(results)

# --- UI ---
with st.sidebar:
    st.header("📥 הזנת מניות")
    tickers_input = st.text_area("הדבק רשימה כאן:", DEFAULT_LIST, height=300)
    run_btn = st.button("🚀 הפעל סריקת על")

if run_btn:
    t_list = [x.strip().upper() for x in tickers_input.replace('\n', ',').split(',') if x.strip()]
    st.toast(f"מתחיל לעבד {len(t_list)} מניות בטבלה אחת...", icon="🔥")
    
    df = run_mega_scan(t_list)
    
    if not df.empty:
        st.success(f"הסריקה הושלמה! {len(df)} מניות נותחו.")
        
        # --- הגדרת תצוגת הטבלה (Column Config) ---
        st.dataframe(
            df.style.background_gradient(subset=['Score'], cmap='RdYlGn'),
            column_config={
                "Symbol": st.column_config.TextColumn("Ticker", width="small"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Change%": st.column_config.NumberColumn("Chg%", format="%.2f%%"),
                "Signal": st.column_config.TextColumn("Signal", width="small"),
                "Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%d"),
                "AI_Pred": st.column_config.NumberColumn("AI Target (5D)", format="$%.2f", help="Linear Regression Forecast"),
                "Upside%": st.column_config.NumberColumn("Upside", format="%.2f%%"),
                "RSI": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                "Trend": st.column_config.TextColumn("Trend (SMA200)"),
                "ATR": st.column_config.NumberColumn("ATR (Vol)", format="$%.2f"),
                "Pivot": st.column_config.NumberColumn("Pivot", format="$%.2f"),
                "Res_R1": st.column_config.NumberColumn("Resistance R1", format="$%.2f"),
                "Fib_618": st.column_config.NumberColumn("Golden Fib", format="$%.2f"),
                "Volume": st.column_config.NumberColumn("Vol (M)", format="%.1fM"),
            },
            use_container_width=True,
            height=800, # טבלה ארוכה
            hide_index=True
        )
        
        # כפתור הורדה
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 הורד את כל הטבלה לאקסל", csv, "mega_scan.csv", "text/csv")
        
    else:
        st.error("לא התקבלו נתונים. נסה שוב.")
