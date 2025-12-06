import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Pro", layout="wide", page_icon="📈")

# --- רשימת המניות שלך (ניתן להוסיף את כל ה-500) ---
# שמתי כאן רשימה מייצגת של "היהלומים" שדיברנו עליהם
TICKERS = [
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 
    'MELI', 'DELL', 'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI',
    'GOOGL', 'AMZN', 'META', 'NFLX', 'AVGO', 'CRM', 'ORCL', 'INTU',
    'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST'
]

# --- פונקציות ניתוח (ה"מוח") ---
@st.cache_data(ttl=300) # רענון נתונים כל 5 דקות אוטומטית
def get_data(tickers):
    data = []
    for ticker in tickers:
        try:
            # משיכת היסטוריה
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if df.empty: continue
            
            # חישוב אינדיקטורים טכניים (TA)
            # 1. RSI
            df['RSI'] = ta.rsi(df['Close'], length=14)
            # 2. Bollinger Bands
            bb = ta.bbands(df['Close'], length=20)
            df = pd.concat([df, bb], axis=1)
            # 3. MACD
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            # 4. ATR (תנודתיות)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            # 5. SMA (מגמה)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)

            # נתונים עדכניים (שורה אחרונה)
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- אלגוריתם הניקוד (AI Score Logic) ---
            score = 0
            reasons = []
            
            # בדיקת RSI
            if curr['RSI'] < 30: 
                score += 25
                reasons.append("Oversold (RSI<30)")
            elif curr['RSI'] > 70: 
                score -= 20
                reasons.append("Overbought (RSI>70)")
            elif 50 <= curr['RSI'] <= 65:
                score += 10 # מומנטום בריא
                
            # בדיקת מגמה (מעל ממוצעים)
            if curr['Close'] > curr['SMA_50']: score += 15
            if curr['Close'] > curr['SMA_200']: score += 15
            
            # בדיקת MACD (חצייה)
            if curr['MACD_12_26_9'] > curr['MACDs_12_26_9']: 
                score += 15
                reasons.append("MACD Bullish")
                
            # בדיקת Bollinger (פריצה)
            if curr['Close'] > curr['BBU_5_2.0']: 
                score += 10
                reasons.append("Bollinger Breakout")
            
            # בדיקת ווליום (האם נכנס כסף?)
            if curr['Volume'] > df['Volume'].mean() * 1.5:
                score += 10
                reasons.append("High Volume")

            # נרמול ציון (0-100)
            final_score = min(max(score, 0), 100)
            
            # קביעת המלצה סופית
            recommendation = "HOLD"
            if final_score >= 75: recommendation = "STRONG BUY 🚀"
            elif final_score >= 60: recommendation = "BUY 🟢"
            elif final_score <= 20: recommendation = "SELL 🔴"
            
            data.append({
                'Symbol': ticker,
                'Price': round(curr['Close'], 2),
                'Change%': round(((curr['Close'] - prev['Close']) / prev['Close']) * 100, 2),
                'RSI': round(curr['RSI'], 1),
                'Score': final_score,
                'Rec': recommendation,
                'Reasons': ", ".join(reasons),
                'ATR': round(curr['ATR'], 2),
                'Volume_Ratio': round(curr['Volume'] / df['Volume'].mean(), 1)
            })
            
        except Exception as e:
            continue
            
    return pd.DataFrame(data)

# --- ממשק המשתמש (UI) ---

st.title("🧠 AI Trading Command Center")
st.markdown(f"**עדכון אחרון:** {datetime.now().strftime('%H:%M:%S')} | **מצב שוק:** פעיל")

if st.button('🔄 סרוק שוק עכשיו'):
    st.rerun()

# שלב 1: טעינת נתונים
with st.spinner('מנתח מניות, מחשב אינדיקטורים ומבצע סימולציות AI...'):
    df_results = get_data(TICKERS)

# שלב 2: הצגת Top 5 המומלצות ("היהלומים")
st.header("🏆 Top 5 המומלצות לקנייה (AI Ranked)")
if not df_results.empty:
    top_picks = df_results.sort_values(by='Score', ascending=False).head(5)
    
    cols = st.columns(5)
    for i, (index, row) in enumerate(top_picks.iterrows()):
        with cols[i]:
            st.metric(label=row['Symbol'], value=f"${row['Price']}", delta=f"{row['Change%']}%")
            st.info(f"ציון AI: **{row['Score']}**\n\n{row['Rec']}")

    # שלב 3: טבלה מפורטת עם כל הנתונים
    st.subheader("📊 דוח ניתוח מלא (כל המניות)")
    # צביעת שורות לפי המלצה
    def highlight_rec(val):
        color = 'red' if 'SELL' in val else 'green' if 'BUY' in val else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(df_results.style.applymap(highlight_rec, subset=['Rec']), use_container_width=True)

    # שלב 4: ניתוח מעמיק למניה נבחרת
    st.divider()
    st.header("🔍 מעבדה טכנית: ניתוח גרף עומק")
    selected_ticker = st.selectbox("בחר מניה לניתוח ויזואלי:", TICKERS)
    
    if selected_ticker:
        ticker_df = yf.download(selected_ticker, period="1y", interval="1d", progress=False)
        
        # בניית גרף נרות מקצועי
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ticker_df.index,
                        open=ticker_df['Open'], high=ticker_df['High'],
                        low=ticker_df['Low'], close=ticker_df['Close'], name='Price'))
        
        # הוספת בולינגר
        bb = ta.bbands(ticker_df['Close'], length=20)
        fig.add_trace(go.Scatter(x=ticker_df.index, y=bb['BBU_5_2.0'], line=dict(color='blue', width=1, dash='dot'), name='Upper BB'))
        fig.add_trace(go.Scatter(x=ticker_df.index, y=bb['BBL_5_2.0'], line=dict(color='blue', width=1, dash='dot'), name='Lower BB'))
        
        fig.update_layout(title=f"{selected_ticker} - ניתוח טכני מתקדם", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # תובנות מהירות
        last_rsi = ta.rsi(ticker_df['Close']).iloc[-1]
        st.write(f"**תובנת AI ל-{selected_ticker}:** ה-RSI עומד על {last_rsi:.1f}. " + 
                 ("המניה נמצאת באזור קניית יתר, היזהר מתיקון." if last_rsi > 70 else 
                  "המניה באזור מכירת יתר, הזדמנות אפשרית." if last_rsi < 30 else 
                  "המניה באזור ניטרלי, עקוב אחר פריצת בולינגר."))