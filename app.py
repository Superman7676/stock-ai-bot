import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
from datetime import datetime

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Trading Pro", layout="wide", page_icon="🚀")

st.title("🚀 AI Trading Command Center")
st.markdown("מערכת סריקה וניתוח מניות בזמן אמת | כולל זיהוי תבניות וניקוד AI")

# --- רשימת המניות המלאה (מתוך התמונות ששלחת) ---
TICKERS = [
    # Top Picks
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 'MELI', 'DELL',
    # Infrastructure & AI
    'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AVGO', 'CRM', 'ORCL', 'INTU', 'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST',
    'FICO', 'EQIX', 'IDXX', 'SPY', 'AXON', 'UTHR', 'SNPS', 'IESC', 'TLN', 'SITM',
    'POWL', 'ETN', 'STRL', 'SOXX', 'AVAV', 'ACN', 'LOW', 'RDDT', 'SNOW', 'JBL',
    'DAVE', 'EPAM', 'CIEN', 'PANW', 'SBAC', 'ICLR', 'BIIB', 'BWXT', 'VST', 'MZTI',
    'LRCX', 'ONTO', 'MNDY', 'DDOG', 'A', 'TMDX', 'APO', 'EMR', 'APH', 'ENVA',
    'TWLO', 'OLED', 'TFX', 'NOVT', 'DVA', 'ARKQ', 'QTUM', 'PCAR', 'SN', 'OKLO',
    'ROKU', 'BSX', 'NBIS', 'RBLX', 'SSNC', 'RBRK', 'CRCL', 'CORT', 'NEE', 'CNR',
    'AIR', 'IR', 'APTV', 'NEGG', 'KTOS', 'ESTC', 'AMBA', 'TTMI', 'SEZL', 'AFRM',
    'MCHP', 'LNTH', 'LIVN', 'MP', 'KOMP', 'O', 'INOD', 'CRSP', 'CECO', 'DOCS',
    'HNGE', 'SNY', 'URA', 'RKLB', 'BN', 'HROW', 'SKWD', 'DOCN', 'KARO', 'U',
    'TECK', 'EXEL', 'AMKR', 'INTA', 'SPNS', 'CELH', 'HUT', 'GLBE', 'GCT', 'TTD',
    'CGNX', 'VKTX', 'CNP', 'BP', 'TOST', 'NNE', 'BMNR', 'REZI', 'CHWY', 'KLAR',
    'PACS', 'BRBR', 'ARQQ', 'JD', 'LQDT', 'ALGM', 'RGTI', 'CRK', 'QBTS', 'RF',
    'TENB', 'AAOI', 'GLXY', 'OUST', 'CPRX', 'LYFT', 'SMR', 'ARCC', 'HSAI', 'ZIM',
    'WYFI', 'JHX', 'ZETA', 'SONO', 'SKYT', 'INFY', 'CLBT', 'ATEN', 'USAR', 'NU',
    'OSCR', 'CORZ', 'UUUU', 'JOBY', 'STNE', 'EOSE', 'GRRR', 'AEVA', 'EH', 'PONY',
    'ACHC', 'DLO', 'SERV', 'QUBT', 'QS', 'PL', 'SOUN', 'OSPN', 'AMPX', 'STLA',
    'GILT', 'LUNR', 'DV', 'UMAC', 'OPFI', 'DCTH', 'RZLT', 'DNA', 'TSSI', 'ONDS',
    'ACHR', 'LUMN', 'QMCO', 'AMCR', 'SHLS', 'MOB', 'TMC', 'CCC', 'OPEN', 'EVTL',
    'BBAI', 'ASPI', 'BTQ', 'PTON', 'POET', 'PDYN', 'MNKD', 'SLDP', 'VERI', 'EVEX',
    'KSCP', 'RXRX', 'RIG', 'RR', 'SPCE', 'ABAT', 'NUAI', 'VTEX', 'ARAI', 'MSOS',
    'PYPD', 'MVST', 'DGXX', 'EVGO', 'HIVE', 'WOOF', 'BITF', 'HNST', 'LIDR', 'KOPN',
    'ORBS', 'SRFM', 'BTBT', 'BTAI', 'CRNT', 'SLNH', 'ALTS', 'QSI', 'INVZ', 'PSTV',
    'NVNO', 'APLD', 'CRWV', 'RZLV', 'RCAT', 'NVTS', 'IONQ', 'BKSY', 'MNTS', 'ASTS',
    'PSTG', 'CIFR', 'UAMY', 'FIG', 'AQMS', 'KVUE'
]

# הסרת כפילויות למקרה שיש
TICKERS = list(set(TICKERS))

@st.cache_data(ttl=600)
def analyze_market(tickers):
    results = []
    errors = []
    
    # אזור תצוגה להתקדמות
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            # עדכון בר התקדמות
            progress_bar.progress((i + 1) / total)
            progress_text.text(f"סורק: {ticker} ({i+1}/{total})")
            
            # משיכת נתונים - התיקון החשוב לפורמט החדש
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
            
            # תיקון לבעיית ה-MultiIndex שגרמה לשגיאות הקודמות
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if df.empty or len(df) < 30:
                continue

            # המרת נתונים למספרים (למניעת שגיאות)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)

            # חישוב אינדיקטורים
            df['RSI'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            df = pd.concat([df, bb], axis=1)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            # נתונים נוכחיים
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # לוגיקת AI לניקוד
            score = 0
            signals = []
            
            # 1. RSI
            if curr['RSI'] < 30:
                score += 25
                signals.append("RSI Oversold (Buy)")
            elif curr['RSI'] > 70:
                score -= 20
                signals.append("RSI Overbought (Sell)")
            elif 50 <= curr['RSI'] <= 60:
                score += 10
                
            # 2. מגמה (SMA)
            if curr['Close'] > curr['SMA_50']: score += 15
            if curr['Close'] > curr['SMA_200']: score += 15
            
            # 3. מומנטום (MACD)
            if curr['MACD_12_26_9'] > curr['MACDs_12_26_9']:
                score += 15
                signals.append("MACD Cross")
                
            # 4. בולינגר
            if curr['Close'] > curr['BBU_5_2.0']:
                score += 10
                signals.append("Bollinger Breakout")
            elif curr['Close'] < curr['BBL_5_2.0']:
                score += 10
                signals.append("Bollinger Bounce")
                
            final_score = min(max(score, 0), 100)
            
            rec = "HOLD"
            if final_score >= 75: rec = "STRONG BUY 🚀"
            elif final_score >= 60: rec = "BUY 🟢"
            elif final_score <= 20: rec = "SELL 🔴"
            
            results.append({
                'Symbol': ticker,
                'Price': round(curr['Close'], 2),
                'Change%': round(((curr['Close'] - prev['Close']) / prev['Close']) * 100, 2),
                'RSI': round(curr['RSI'], 1),
                'Score': final_score,
                'Recommendation': rec,
                'Signals': ", ".join(signals),
                'ATR': round(curr['ATR'], 2)
            })
            
        except Exception as e:
            # מתעלמים משגיאות בודדות כדי לא לעצור את הריצה
            continue
            
    progress_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)

# --- כפתור הפעלה ראשי ---
if st.button('🚀 הפעל סריקת שוק מלאה (200+ מניות)'):
    with st.spinner('מעבד נתונים... זה עשוי לקחת כדקה בגלל כמות המניות...'):
        df = analyze_market(TICKERS)
        
        if not df.empty:
            # 1. הצגת 5 המומלצות
            st.success(f"הסריקה הושלמה! עובדו {len(df)} מניות בהצלחה.")
            
            st.subheader("🏆 Top 5 - המומלצות ביותר עכשיו")
            top_stocks = df.sort_values('Score', ascending=False).head(5)
            
            cols = st.columns(5)
            for i, (idx, row) in enumerate(top_stocks.iterrows()):
                with cols[i]:
                    st.metric(label=row['Symbol'], value=f"${row['Price']}", delta=f"{row['Change%']}%")
                    st.caption(f"Score: {row['Score']}")
                    st.write(row['Recommendation'])
            
            # 2. דוח מלא להורדה (Excel)
            st.divider()
            st.subheader("📥 הורדת דוח מלא")
            
            # המרה ל-CSV להורדה
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="הורד קובץ Excel (CSV) מלא",
                data=csv,
                file_name=f'market_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )
            
            # 3. טבלה אינטראקטיבית
            st.subheader("📊 כל הנתונים")
            st.dataframe(df.sort_values('Score', ascending=False), use_container_width=True)
            
        else:
            st.error("לא נמצאו נתונים. נסה שוב מאוחר יותר.")
else:
    st.info("לחץ על הכפתור למעלה כדי להתחיל.")
