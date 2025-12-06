import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- הגדרות עמוד ---
st.set_page_config(page_title="Ultimate Trading PRO", layout="wide", page_icon="💎")

# CSS לעיצוב נקי
st.markdown("""
<style>
    .report-box {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        border: 1px solid #333;
    }
    .stMetric {background-color: #f0f2f6; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("💎 Ultimate AI Trading - Institutional Grade")
st.markdown("מערכת ניתוח מוסדית | Fibonacci, Pivots, Multi-MA, Risk Management")

# רשימת המניות (מקוצרת להדגמה - תוסיף את כל ה-500 שלך כאן)
TICKERS = [
    'NVDA', 'TSLA', 'AMD', 'PLTR', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'ALAB', 'CLSK', 'COHR', 'VRT', 'LITE', 'SMCI', 'MDB', 'SOFI',
    'FICO', 'EQIX', 'SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SLV'
]

# פונקציה לחישוב פיבונאצ'י
def calc_fibonacci(high, low):
    diff = high - low
    levels = {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        'Ext_127.2%': high + 0.272 * diff, # Extensions for targets
        'Ext_161.8%': high + 0.618 * diff
    }
    return levels

# פונקציה לחישוב פיבוטים (Standard Pivot Points)
def calc_pivots(high, low, close):
    p = (high + low + close) / 3
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    return p, r1, r2, s1, s2

if st.button('🚀 הפעל סריקת מאסטר (Full Calculation)'):
    status = st.empty()
    status.info("🔄 מוריד נתונים ומבצע חישובים מורכבים (ADX, Aroon, Fibs)...")
    
    try:
        # הורדת נתונים (שנה אחורה כדי לחשב פיבונאצ'י ארוך טווח)
        data = yf.download(TICKERS, period="1y", group_by='ticker', auto_adjust=True, threads=True)
        
        if data.empty:
            st.error("❌ שגיאה במשיכת נתונים.")
            st.stop()
            
        final_results = []
        
        # סרגל התקדמות
        prog_bar = st.progress(0)
        
        for i, ticker in enumerate(TICKERS):
            prog_bar.progress((i + 1) / len(TICKERS))
            
            try:
                # בדיקה אם המניה קיימת בנתונים
                if ticker not in data.columns.levels[0]: continue
                
                df = data[ticker].copy()
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 200: continue # חייבים היסטוריה לממוצע 200

                # --- 1. אינדיקטורים בסיסיים ---
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 2. ממוצעים נעים (הכל) ---
                for ma in [5, 20, 50, 100, 150, 200]:
                    df[f'SMA_{ma}'] = ta.sma(df['Close'], length=ma)
                
                for ema in [5, 8, 12, 26, 50]:
                    df[f'EMA_{ema}'] = ta.ema(df['Close'], length=ema)
                    
                # VWAP (Rolling)
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

                # --- 3. מומנטום ומתנדים ---
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # MACD
                macd = ta.macd(df['Close'])
                df = pd.concat([df, macd], axis=1)
                
                # ADX (מגמה)
                adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                df = pd.concat([df, adx], axis=1)
                
                # Aroon
                aroon = ta.aroon(df['High'], df['Low'], length=14)
                df = pd.concat([df, aroon], axis=1)
                
                # Bollinger
                bb = ta.bbands(df['Close'], length=20, std=2)
                df = pd.concat([df, bb], axis=1)
                
                # ATR (Vol)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                
                # --- 4. חישובים גיאומטריים (Fibonacci & Pivots) ---
                # פיבונאצ'י שנתי (High/Low של השנה האחרונה)
                year_high = df['High'].max()
                year_low = df['Low'].min()
                fibs = calc_fibonacci(year_high, year_low)
                
                # פיבוטים (מבוסס על הנר האחרון)
                pivot, r1, r2, s1, s2 = calc_pivots(curr['High'], curr['Low'], curr['Close'])
                
                # --- 5. ניקוד ואסטרטגיה ---
                score = 50
                rec = "HOLD"
                
                # לוגיקת ניקוד
                if curr['Close'] > df['SMA_200'].iloc[-1]: score += 15
                if df['RSI'].iloc[-1] < 30: score += 20
                if df[f'ADX_14'].iloc[-1] > 25: score += 10 # מגמה חזקה
                
                final_score = min(max(score, 0), 100)
                if final_score >= 70: rec = "BUY"
                elif final_score <= 30: rec = "SELL"
                
                # --- שמירת כל המידע המטורף הזה ---
                # שים לב: אנחנו שומרים את האובייקט המלא כדי להציג אותו אח"כ
                res_obj = {
                    'Symbol': ticker,
                    'Price': curr['Close'],
                    'Change_Pct': ((curr['Close'] - prev['Close']) / prev['Close']) * 100,
                    'Change_USD': curr['Close'] - prev['Close'],
                    'High': curr['High'], 'Low': curr['Low'],
                    'Volume': curr['Volume'],
                    'Avg_Vol': df['Volume'].mean(),
                    'ATR': df['ATR'].iloc[-1],
                    # MAs
                    'SMA_5': df['SMA_5'].iloc[-1], 'SMA_200': df['SMA_200'].iloc[-1],
                    'EMA_8': df['EMA_8'].iloc[-1], 'EMA_26': df['EMA_26'].iloc[-1],
                    'VWAP': df['VWAP'].iloc[-1] if 'VWAP' in df else 0,
                    # Momentum
                    'RSI': df['RSI'].iloc[-1],
                    'MACD': df[df.columns[df.columns.str.startswith('MACD_')][0]].iloc[-1],
                    'ADX': df['ADX_14'].iloc[-1],
                    'Aroon_Up': df['AROONU_14'].iloc[-1],
                    'Aroon_Down': df['AROOND_14'].iloc[-1],
                    'BB_Width': df['BBB_5_2.0'].iloc[-1] if 'BBB_5_2.0' in df else 0,
                    # Pivots & Fibs
                    'Pivot': pivot, 'R1': r1, 'R2': r2, 'S1': s1, 'S2': s2,
                    'Fibs': fibs,
                    # Score
                    'Score': final_score,
                    'Rec': rec
                }
                final_results.append(res_obj)

            except Exception as e:
                continue

        prog_bar.empty()
        status.empty()
        
        if final_results:
            df_res = pd.DataFrame(final_results)
            
            # --- חלק א: טבלת שליטה ראשית (נקייה) ---
            st.subheader("📋 לוח בקרה ראשי (סינון מהיר)")
            
            # עיצוב הטבלה הראשית (רק נתונים קריטיים)
            main_view = df_res[['Symbol', 'Price', 'Change_Pct', 'Score', 'Rec', 'RSI', 'ADX', 'ATR']]
            st.dataframe(
                main_view.sort_values('Score', ascending=False).style.format({
                    'Price': '{:.2f}', 'Change_Pct': '{:.2f}%', 
                    'RSI': '{:.2f}', 'ADX': '{:.2f}', 'ATR': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # --- חלק ב: Deep Dive - כרטיס המניה (החלק שביקשת) ---
            st.divider()
            st.header("🔍 ניתוח עומק - כרטיס מניה (Telegram Style)")
            
            selected_ticker = st.selectbox("בחר מניה לקבלת דוח מלא ומפורט:", df_res['Symbol'].tolist())
            
            if selected_ticker:
                # שליפת הנתונים של המניה שנבחרה
                row = df_res[df_res['Symbol'] == selected_ticker].iloc[0]
                fibs = row['Fibs']
                
                # חישוב יעדים (Stop Loss & Targets)
                stop_loss = row['Price'] - (2 * row['ATR']) # סטופ מבוסס תנודתיות
                target1 = row['R1']
                target2 = fibs['Ext_127.2%']
                
                # יצירת הטקסט המפורט (בדיוק בפורמט שביקשת)
                report_text = f"""
• Price: ${row['Price']:.2f} ({row['Change_Pct']:.2f}% | ${row['Change_USD']:.2f})
• H/L: ${row['High']:.2f} / ${row['Low']:.2f}
• Vol: {row['Volume']/1000000:.2f}M | Avg: {row['Avg_Vol']/1000000:.2f}M
• ATR14: ${row['ATR']:.2f}
══════════════════
📊 Moving Averages
• SMA 5/200: ${row['SMA_5']:.2f} / ${row['SMA_200']:.2f}
• EMA 8/26: ${row['EMA_8']:.2f} / ${row['EMA_26']:.2f}
• VWAP: ${row['VWAP']:.2f}
═════════════════
⚡️ Momentum & Oscillators
• RSI(14): {row['RSI']:.2f} | MACD: {row['MACD']:.2f} | ADX: {row['ADX']:.2f}
• Aroon ↗/↘: {row['Aroon_Up']:.0f} / {row['Aroon_Down']:.0f}
══════════════════
📐 Support/Resistance & Pivots
• Pivot: ${row['Pivot']:.2f}
• R1: ${row['R1']:.2f} | R2: ${row['R2']:.2f}
• S1: ${row['S1']:.2f} | S2: ${row['S2']:.2f}
══════════════════
🔢 Fibonacci (Yearly)
• 38.2%: ${fibs['38.2%']:.2f}
• 50.0%: ${fibs['50.0%']:.2f}
• 61.8%: ${fibs['61.8%']:.2f}
• Ext 127.2%: ${fibs['Ext_127.2%']:.2f} (Target)
═════════════════
🎯 Recommendation: {row['Rec']}
Entry: ${row['Price']:.2f} | Stop: ${stop_loss:.2f}
Targets: T1=${target1:.2f} · T2=${target2:.2f}
═══════════════════
Composite Score: {row['Score']}/100
"""
                # הצגת הדוח בתוך קופסה שחורה מעוצבת
                st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)

            # --- כפתור הורדה ---
            st.divider()
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📥 הורד קובץ Excel מלא (כל הפרמטרים)", csv, "mega_report.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("המערכת מוכנה. לחץ על הכפתור כדי לייצר את דוח המאסטר.")
