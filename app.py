import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Debug", layout="wide", page_icon="🛠️")

st.title("🛠️ AI Trading - מצב דיאגנוסטיקה")

# רשימה קצרה לבדיקה ראשונית - כדי לראות שהכל עובד
TICKERS = ['NVDA', 'TSLA', 'AMD', 'PLTR', 'GOOGL']

if st.button('🚀 הפעל סריקה עכשיו'):
    st.write("מתחיל בתהליך הסריקה...")
    
    # יצירת אזור לדיווח
    status_text = st.empty()
    progress_bar = st.progress(0)
    results = []
    errors = []

    # לולאה על המניות
    for i, ticker in enumerate(TICKERS):
        try:
            # עדכון סטטוס למשתמש
            status_text.text(f"בודק את מניית: {ticker} ({i+1}/{len(TICKERS)})")
            progress_bar.progress((i + 1) / len(TICKERS))
            
            # משיכת נתונים
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            if df.empty:
                errors.append(f"{ticker}: הגיע קובץ ריק מ-Yahoo Finance")
                continue

            # בדיקה שיש מספיק נתונים לחישובים
            if len(df) < 20:
                errors.append(f"{ticker}: אין מספיק היסטוריה (פחות מ-20 יום)")
                continue

            # חישוב אינדיקטורים (החלק הטכני)
            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=14)
            # בדיקה שהחישוב הצליח
            if df['RSI'].isnull().all():
                errors.append(f"{ticker}: נכשל בחישוב RSI")
                continue

            last_rsi = df['RSI'].iloc[-1]
            last_price = df['Close'].iloc[-1]
            
            # הוספה לתוצאות
            results.append({
                'Symbol': ticker,
                'Price': round(last_price, 2),
                'RSI': round(last_rsi, 2),
                'Status': 'OK'
            })
            
        except Exception as e:
            errors.append(f"שגיאה ב-{ticker}: {str(e)}")
            continue

    # סיום וניקוי
    status_text.empty()
    progress_bar.empty()

    # --- הצגת תוצאות ---
    if results:
        st.success(f"הסריקה הושלמה! נמצאו נתונים ל-{len(results)} מניות.")
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        
        # הצגת הטופ 1
        best_stock = df_res.sort_values('RSI').iloc[0]
        st.metric(label=f"המניה עם ה-RSI הכי נמוך: {best_stock['Symbol']}", value=best_stock['RSI'])
    else:
        st.error("לא הצלחנו למשוך נתונים לאף מניה. ראה שגיאות למטה.")

    # --- הצגת שגיאות (אם יש) ---
    if errors:
        with st.expander("ראה דוח שגיאות"):
            for err in errors:
                st.write(f"❌ {err}")

else:
    st.info("לחץ על הכפתור למעלה כדי להתחיל בדיקה.")

# בדיקת ספריות
with st.expander("בדיקת גרסאות מערכת"):
    st.write(f"Pandas version: {pd.__version__}")
    st.write(f"Yfinance version: {yf.__version__}")
