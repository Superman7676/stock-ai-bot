import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Sniper Pro", layout="wide", page_icon="🎯")
st.title("🎯 AI Sniper Pro - The Ultimate Scanner")
st.markdown("מערכת סריקה אולטימטיבית: זיהוי תבניות, נרות, פיבונאצ'י וכל האינדיקטורים בטבלה אחת.")

# רשימת המניות (חלקית להדגמה - תוסיף את כל ה-500 שלך כאן)
TICKERS = [
    'NVDA', 'TSLA', 'AMD', 'PLTR', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'ALAB', 'CLSK', 'COHR', 'VRT', 'LITE', 'SMCI', 'MDB', 'SOFI',
    'FICO', 'EQIX', 'SPY', 'QQQ', 'INTU', 'AVGO', 'CRM', 'UBER'
]

# --- פונקציות עזר לחישובים מורכבים ---

def identify_candle_pattern(open_p, high, low, close):
    """זיהוי תבניות נרות יפניים בסיסיות"""
    body = abs(close - open_p)
    wick_upper = high - max(close, open_p)
    wick_lower = min(close, open_p) - low
    
    pattern = "Normal"
    
    # Doji
    if body <= 0.03 * (high - low):
        pattern = "Doji ➕"
    # Hammer (פטיש)
    elif wick_lower > 2 * body and wick_upper < body:
        pattern = "Hammer 🔨"
    # Shooting Star (כוכב נופל)
    elif wick_upper > 2 * body and wick_lower < body:
        pattern = "Shooting Star 🌠"
    # Marubozu (נר מלא וחזק)
    elif body > 0.8 * (high - low):
        pattern = "Marubozu 💪"
        
    return pattern

def get_trend_strength(adx, aroon_up, aroon_down):
    if adx < 20: return "Weak/Range"
    if aroon_up > 70 and aroon_down < 30: return "Strong Up 🔥"
    if aroon_down > 70 and aroon_up < 30: return "Strong Down ❄️"
    return "Trending"

if st.button('🔥 הפעל סריקת עומק מלאה (כל הפרמטרים)'):
    status = st.empty()
    status.info("⏳ מוריד נתונים, מחשב 50+ אינדיקטורים, מזהה תבניות... זה ייקח רגע.")
    
    try:
        # 1. הורדת נתונים (Batch)
        data = yf.download(TICKERS, period="1y", group_by='ticker', auto_adjust=True, threads=True)
        
        if data.empty:
            st.error("❌ תקלה בהורדת הנתונים.")
            st.stop()
            
        results = []
        prog_bar = st.progress(0)
        
        for i, ticker in enumerate(TICKERS):
            prog_bar.progress((i + 1) / len(TICKERS))
            
            try:
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].copy()
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 200: continue

                # === חישוב אינדיקטורים (המסה הגדולה) ===
                
                # ממוצעים נעים (MAs)
                df['SMA_20'] = ta.sma(df['Close'], length=20)
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['EMA_9'] = ta.ema(df['Close'], length=9)
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
                
                # מתנדים (Oscillators)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                df['Stoch_K'] = stoch['STOCHk_14_3_3']
                df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
                
                # מומנטום ומגמה
                macd = ta.macd(df['Close'])
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                
                adx = ta.adx(df['High'], df['Low'], df['Close'])
                df['ADX'] = adx['ADX_14']
                
                aroon = ta.aroon(df['High'], df['Low'])
                df['Aroon_Up'] = aroon['AROONU_14']
                df['Aroon_Down'] = aroon['AROOND_14']
                
                # בולינגר ו-ATR
                bb = ta.bbands(df['Close'], length=20, std=2)
                df['BB_Upper'] = bb['BBU_5_2.0']
                df['BB_Lower'] = bb['BBL_5_2.0']
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                
                # Donchian Channels (High/Low של 20 יום)
                df['Donchian_High'] = df['High'].rolling(20).max()
                df['Donchian_Low'] = df['Low'].rolling(20).min()

                # === נתונים נוכחיים לניתוח ===
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # === זיהוי תבניות גרף ונרות ===
                candle_pat = identify_candle_pattern(curr['Open'], curr['High'], curr['Low'], curr['Close'])
                
                # זיהוי Engulfing (בולען)
                engulfing = ""
                if curr['Close'] > curr['Open'] and prev['Close'] < prev['Open']: # ירוק אחרי אדום
                    if curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
                        engulfing = "Bullish Engulfing 🐮"
                elif curr['Close'] < curr['Open'] and prev['Close'] > prev['Open']: # אדום אחרי ירוק
                    if curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']:
                        engulfing = "Bearish Engulfing 🐻"

                final_pattern = engulfing if engulfing else candle_pat

                # === חישוב פיבוטים ופיבונאצ'י ===
                pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
                r1 = 2 * pivot - curr['Low']
                s1 = 2 * pivot - curr['High']
                
                year_high = df['High'][-252:].max() # 52 weeks
                year_low = df['Low'][-252:].min()
                fib_618 = year_high - (0.618 * (year_high - year_low))

                # === ציון AI משוקלל (Score) ===
                score = 50
                # מגמה
                if curr['Close'] > curr['SMA_200']: score += 15
                if curr['Close'] > curr['SMA_50']: score += 10
                if curr['ADX'] > 25 and curr['Aroon_Up'] > 70: score += 10
                # מתנדים
                if curr['RSI'] < 30: score += 20 # מכירת יתר
                if curr['RSI'] > 70: score -= 15 # קניית יתר
                if curr['MACD'] > curr['MACD_Signal']: score += 10
                if curr['MFI'] < 20: score += 10 # כסף חכם נכנס?
                
                final_score = min(max(score, 0), 100)
                
                rec = "HOLD"
                if final_score >= 80: rec = "STRONG BUY 🚀"
                elif final_score >= 65: rec = "BUY 🟢"
                elif final_score <= 30: rec = "SELL 🔴"

                # === בניית השורה לטבלה ===
                results.append({
                    'Symbol': ticker,
                    'Price': round(curr['Close'], 2),
                    'Change%': round(((curr['Close'] - prev['Close']) / prev['Close']) * 100, 2),
                    'Rec': rec,
                    'Score': int(final_score),
                    'Pattern': final_pattern,
                    'Trend_Str': get_trend_strength(curr['ADX'], curr['Aroon_Up'], curr['Aroon_Down']),
                    'RSI': round(curr['RSI'], 1),
                    'MFI': round(curr['MFI'], 1),
                    'MACD_Hist': round(curr['MACD'] - curr['MACD_Signal'], 2),
                    'SMA_200': round(curr['SMA_200'], 2),
                    'Dist_SMA200%': round(((curr['Close'] - curr['SMA_200']) / curr['SMA_200']) * 100, 1),
                    'VWAP': round(curr['VWAP'], 2),
                    'ATR': round(curr['ATR'], 2),
                    'Pivot': round(pivot, 2),
                    'R1': round(r1, 2),
                    'S1': round(s1, 2),
                    'Fib_61.8%': round(fib_618, 2),
                    'Donchian_H': round(curr['Donchian_High'], 2),
                    'Donchian_L': round(curr['Donchian_Low'], 2),
                    'Vol_Ratio': round(curr['Volume'] / df['Volume'][-20:].mean(), 2)
                })

            except Exception as e:
                continue
        
        status.empty()
        prog_bar.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            
            # === תצוגה ראשית ===
            st.success(f"✅ הסריקה הושלמה. נותחו {len(df_res)} מניות.")
            
            # סינון Top 5
            st.subheader("🏆 Top 5 AI Picks (Highest Score)")
            st.dataframe(df_res.sort_values('Score', ascending=False).head(5), use_container_width=True)

            # === כרטיס מניה מפורט (כמו שביקשת) ===
            st.divider()
            st.subheader("🔬 ניתוח מניה בודדת - כל הפרמטרים")
            
            sel = st.selectbox("בחר מניה להצגת כרטיס מלא:", df_res['Symbol'].tolist())
            row = df_res[df_res['Symbol'] == sel].iloc[0]
            
            # עיצוב טקסט מיוחד (Telegram Style)
            report = f"""
💰 **{row['Symbol']}** | Price: ${row['Price']} ({row['Change%']}%)
🚦 **Recommendation:** {row['Rec']} (Score: {row['Score']}/100)
📊 **Pattern:** {row['Pattern']} | Trend: {row['Trend_Str']}

**Momentum & Oscillators:**
• RSI: {row['RSI']} | MFI: {row['MFI']} (Money Flow)
• MACD Histogram: {row['MACD_Hist']} (Positive=Bullish)

**Moving Averages:**
• Price vs SMA200: {row['Dist_SMA200%']}% distance
• VWAP: ${row['VWAP']} (Institutional Benchmark)

**Support & Resistance (Levels):**
• Pivot: ${row['Pivot']}
• Support (S1): ${row['S1']} | Resistance (R1): ${row['R1']}
• Golden Pocket (Fib 61.8%): ${row['Fib_61.8%']}
• Donchian Channel: ${row['Donchian_L']} - ${row['Donchian_H']}

**Risk Management:**
• Volatility (ATR): ${row['ATR']}
• Volume Ratio: {row['Vol_Ratio']}x (relative to avg)
            """
            st.info(report)

            # === טבלה מלאה להורדה ===
            st.divider()
            st.subheader("📥 הורדת הדוח המלא (Excel)")
            st.markdown("הטבלה הזו מכילה את **כל** העמודות והאינדיקטורים שחושבו.")
            
            st.dataframe(df_res)
            
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("הורד קובץ CSV מלא", csv, "full_market_scan.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("מוכן לסריקה.")
