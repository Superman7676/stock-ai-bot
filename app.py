import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Pro", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stDataFrame"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Trading Command Center - PRO Version")
st.markdown("מערכת ניתוח טכני מלאה | כולל MACD, Bollinger, SMA, RSI | **נתונים בזמן אמת**")

# רשימת המניות המלאה שלך
TICKERS = [
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 'MELI', 'DELL',
    'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AVGO', 'CRM', 'ORCL', 'INTU', 'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST',
    'FICO', 'EQIX', 'SPY', 'AXON', 'SNPS', 'TLN', 'ETN', 'RDDT', 'SNOW', 'PANW',
    'ICLR', 'VST', 'LRCX', 'DDOG', 'TWLO', 'BSX', 'NBIS', 'RBLX', 'AFRM', 'CELH',
    'JD', 'TTD', 'APP', 'CART', 'KVUE', 'NET', 'DKNG', 'CVNA', 'ZS', 'CRWD'
]

if st.button('🚀 הפעל סריקת עומק (Deep Scan)'):
    status = st.empty()
    status.info("🔄 מושך נתונים מורחבים מ-Yahoo Finance (Batch Download)...")
    
    try:
        # הורדת נתונים חכמה
        data = yf.download(TICKERS, period="1y", group_by='ticker', auto_adjust=True, threads=True)
        
        if data.empty:
            st.error("❌ התקבל קובץ ריק. נסה שוב בעוד דקה.")
            st.stop()
            
        status.success("✅ נתונים התקבלו! מבצע חישובים טכניים מורכבים...")
        
        results = []
        
        # סרגל התקדמות
        prog_bar = st.progress(0)
        
        for i, ticker in enumerate(TICKERS):
            prog_bar.progress((i + 1) / len(TICKERS))
            
            try:
                if ticker not in data.columns.levels[0]: continue
                
                df = data[ticker].copy()
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 200: continue # חייבים היסטוריה לממוצע 200

                # --- חישוב כל האינדיקטורים שביקשת ---
                
                # 1. מגמות (Trends)
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                
                # 2. מומנטום (Oscillators)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # MACD
                macd = ta.macd(df['Close'])
                df = pd.concat([df, macd], axis=1)
                # שמות העמודות של MACD משתנים, ננרמל אותם
                macd_col = [c for c in df.columns if c.startswith('MACD_')][0]
                signal_col = [c for c in df.columns if c.startswith('MACDs_')][0]
                hist_col = [c for c in df.columns if c.startswith('MACDh_')][0]
                
                # 3. תנודתיות (Volatility)
                bb = ta.bbands(df['Close'], length=20, std=2)
                df = pd.concat([df, bb], axis=1)
                lower_col = [c for c in df.columns if c.startswith('BBL_')][0]
                upper_col = [c for c in df.columns if c.startswith('BBU_')][0]
                
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                
                # --- נתונים אחרונים ---
                curr = df.iloc[-1]
                
                # --- מנוע ניקוד (AI SCORING ENGINE V2) ---
                # בסיס הציון הוא 50 (ניטרלי)
                score = 50
                signals = []
                
                # ניתוח מגמה (הכי חשוב)
                trend = "NEUTRAL"
                if curr['Close'] > curr['SMA_200']:
                    score += 20
                    trend = "UP 🟢"
                    if curr['Close'] > curr['SMA_50']:
                        score += 10 # מגמה חזקה מאוד
                else:
                    score -= 20
                    trend = "DOWN 🔴"
                
                # ניתוח RSI
                rsi_status = "Neutral"
                if curr['RSI'] < 30:
                    score += 25
                    rsi_status = "Oversold (Buy) 🟢"
                elif curr['RSI'] > 75:
                    score -= 15
                    rsi_status = "Overbought (Risk) 🔴"
                elif 50 < curr['RSI'] < 70 and trend == "UP 🟢":
                    score += 10 # מומנטום חיובי בריא
                    
                # ניתוח MACD
                macd_val = curr[macd_col]
                sig_val = curr[signal_col]
                macd_status = "Bearish"
                if macd_val > sig_val:
                    score += 10
                    macd_status = "Bullish 🟢"
                
                # ניתוח בולינגר
                bb_status = "Inside"
                if curr['Close'] > curr[upper_col]:
                    bb_status = "Breakout 🚀"
                    score += 5
                elif curr['Close'] < curr[lower_col]:
                    bb_status = "Oversold Bounce ♻️"
                    score += 15
                    
                # ציון סופי
                final_score = min(max(score, 0), 100)
                
                # המלצה מילולית
                rec = "HOLD"
                if final_score >= 80: rec = "STRONG BUY 🔥"
                elif final_score >= 60: rec = "BUY ✅"
                elif final_score <= 30: rec = "SELL ❌"
                
                results.append({
                    'Symbol': ticker,
                    'Price': round(curr['Close'], 2),
                    'Score': int(final_score),
                    'Recommendation': rec,
                    'Trend (SMA200)': trend,
                    'RSI': round(curr['RSI'], 1),
                    'MACD': macd_status,
                    'Bollinger': bb_status,
                    'ATR (Vol)': round(curr['ATR'], 2),
                    'SMA_50': round(curr['SMA_50'], 2),
                    'SMA_200': round(curr['SMA_200'], 2)
                })

            except Exception as e:
                continue
        
        prog_bar.empty()
        status.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            
            # --- 1. הצגת היהלומים (Top Picks) ---
            st.header("🏆 Top Opportunities (Score > 75)")
            
            # סינון: רק מניות עם ציון גבוה ו-Buy
            top_picks = df_res[df_res['Score'] >= 75].sort_values('Score', ascending=False).head(5)
            
            if not top_picks.empty:
                cols = st.columns(len(top_picks))
                for i, (idx, row) in enumerate(top_picks.iterrows()):
                    with cols[i]:
                        st.metric(label=row['Symbol'], value=f"${row['Price']}", delta=f"Score: {row['Score']}")
                        st.success(f"{row['Recommendation']}")
                        st.caption(f"RSI: {row['RSI']} | Trend: {row['Trend (SMA200)']}")
            else:
                st.warning("⚠️ השוק קשה כרגע. האלגוריתם לא מצא מניות בדירוג 'Strong Buy' מובהק (ציון מעל 75). בדוק את טבלת ה-Buy למטה.")

            # --- 2. הטבלה המלאה והמפורטת ---
            st.divider()
            st.subheader("📋 דוח טכני מלא ומפורט (כל האינדיקטורים)")
            
            # צביעת הטבלה
            def color_rec(val):
                if 'STRONG BUY' in val: return 'background-color: #90ee90; color: black; font-weight: bold'
                if 'BUY' in val: return 'color: green; font-weight: bold'
                if 'SELL' in val: return 'color: red'
                return ''

            st.dataframe(
                df_res.sort_values('Score', ascending=False).style.applymap(color_rec, subset=['Recommendation']),
                column_order=['Symbol', 'Price', 'Score', 'Recommendation', 'RSI', 'Trend (SMA200)', 'MACD', 'Bollinger', 'ATR (Vol)', 'SMA_50', 'SMA_200'],
                use_container_width=True,
                height=800
            )
            
            # כפתור הורדה
            st.download_button(
                "📥 הורד את הדוח המלא לאקסל",
                df_res.to_csv(index=False).encode('utf-8'),
                "pro_market_report.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"שגיאה: {str(e)}")

else:
    st.info("המערכת מוכנה. לחץ על הכפתור למעלה כדי להתחיל.")
