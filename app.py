import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Pro", layout="wide", page_icon="🚀")
st.title("🚀 AI Trading Command Center")

# רשימת המניות (מקוצרת לבדיקה - אם זה עובד, תוסיף את השאר אח"כ)
# כרגע שמתי את ה-50 החשובות ביותר כדי לוודא יציבות
TICKERS = [
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 'MELI', 'DELL',
    'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AVGO', 'CRM', 'ORCL', 'INTU', 'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST',
    'FICO', 'EQIX', 'SPY', 'AXON', 'SNPS', 'TLN', 'ETN', 'RDDT', 'SNOW', 'PANW',
    'ICLR', 'VST', 'LRCX', 'DDOG', 'TWLO', 'BSX', 'NBIS', 'RBLX', 'AFRM', 'CELH'
]

if st.button('🚀 הפעל סריקת שוק (מצב חכם)'):
    status = st.empty()
    status.write("🔄 מתחבר ל-Yahoo Finance ומוריד נתונים בבת אחת (Batch)...")
    
    try:
        # הורדה קבוצתית - טריק למניעת חסימות
        # מורידים את כל הנתונים במכה אחת
        data = yf.download(TICKERS, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
        
        if data.empty:
            st.error("❌ התקבל קובץ ריק מ-Yahoo. ייתכן שיש חסימת IP זמנית.")
            st.stop()
            
        status.write("✅ הנתונים ירדו! מתחיל ניתוח טכני...")
        
        results = []
        debug_errors = []
        
        # לולאה על המניות בתוך המבנה שהתקבל
        for ticker in TICKERS:
            try:
                # שליפת המידע למניה ספציפית
                # בודקים אם המניה קיימת בנתונים שהורדו
                if ticker not in data.columns.levels[0]:
                    continue
                    
                df = data[ticker].copy()
                
                # ניקוי שורות ריקות
                df.dropna(subset=['Close'], inplace=True)
                
                if len(df) < 20:
                    continue

                # --- ניתוח טכני ---
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # בולינגר
                bb = ta.bbands(df['Close'], length=20)
                if bb is not None:
                    df = pd.concat([df, bb], axis=1)
                
                # ממוצעים
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                
                # נתונים אחרונים
                curr = df.iloc[-1]
                
                # --- ניקוד ---
                score = 0
                signals = []
                
                # RSI Logic
                if curr['RSI'] < 30:
                    score += 25
                    signals.append("Oversold")
                elif curr['RSI'] > 70:
                    score -= 20
                    signals.append("Overbought")
                
                # Bollinger Logic
                # (משתמשים בשמות ברירת המחדל של פנדס-TA)
                if 'BBU_20_2.0' in df.columns and curr['Close'] > curr['BBU_20_2.0']:
                    score += 10
                    signals.append("Bollinger Break")

                # Trend Logic
                if curr['SMA_50'] > 0 and curr['Close'] > curr['SMA_50']:
                    score += 20
                    
                # נרמול
                final_score = min(max(score, 0), 100)
                
                rec = "HOLD"
                if final_score >= 60: rec = "BUY 🟢"
                if final_score >= 80: rec = "STRONG BUY 🚀"
                if final_score <= 20: rec = "SELL 🔴"

                results.append({
                    'Symbol': ticker,
                    'Price': round(curr['Close'], 2),
                    'RSI': round(curr['RSI'], 1),
                    'Score': final_score,
                    'Rec': rec,
                    'Signals': ", ".join(signals)
                })

            except Exception as e:
                debug_errors.append(f"{ticker}: {str(e)}")
                continue

        # --- הצגת תוצאות ---
        status.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            
            # הצגת Top 5
            st.subheader("🏆 Top 5 Opportunities")
            st.dataframe(df_res.sort_values('Score', ascending=False).head(5), use_container_width=True)
            
            # הצגת כל הטבלה
            with st.expander("ראה טבלה מלאה"):
                st.dataframe(df_res)
                
            # כפתור הורדה
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📥 הורד דוח Excel", csv, "market_report.csv", "text/csv")
            
        else:
            st.warning("לא הצלחנו לייצר תוצאות. ראה שגיאות למטה.")
            if debug_errors:
                st.write(debug_errors[:5]) # מציג 5 שגיאות ראשונות

    except Exception as e:
        st.error(f"שגיאה כללית במערכת: {e}")

else:
    st.info("המערכת מוכנה. לחץ על הכפתור כדי להתחיל.")
