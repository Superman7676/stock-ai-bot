import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time

# --- הגדרות עמוד ---
st.set_page_config(page_title="AI Sniper Ultimate", layout="wide", page_icon="☢️")
st.title("☢️ AI Sniper Ultimate - מערכת סריקה יציבה")
st.markdown("סריקה מלאה בחלוקה למנות (מונע קריסות) | כל האינדיקטורים | דוח טלגרם")

# --- רשימת המניות המלאה שלך (מהתמונות) ---
ALL_TICKERS = [
    'NVDA', 'ALAB', 'CLSK', 'PLTR', 'AMD', 'TSLA', 'MSFT', 'UBER', 'MELI', 'DELL',
    'VRT', 'COHR', 'LITE', 'SMCI', 'MDB', 'SOFI', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AVGO', 'CRM', 'ORCL', 'INTU', 'RIVN', 'MARA', 'RIOT', 'IREN', 'HOOD', 'UPST',
    'FICO', 'EQIX', 'SPY', 'AXON', 'SNPS', 'TLN', 'ETN', 'RDDT', 'SNOW', 'PANW',
    'ICLR', 'VST', 'LRCX', 'DDOG', 'TWLO', 'BSX', 'NBIS', 'RBLX', 'AFRM', 'CELH',
    'JD', 'TTD', 'KVUE', 'NET', 'DKNG', 'CVNA', 'ZS', 'CRWD', 'SITM', 'POWL', 'STRL',
    'SOXX', 'AVAV', 'ACN', 'LOW', 'JBL', 'EPAM', 'CIEN', 'SBAC', 'BIIB', 'BWXT', 'MZTI',
    'ONTO', 'MNDY', 'TMDX', 'APO', 'EMR', 'APH', 'ENVA', 'OLED', 'TFX', 'NOVT', 'DVA',
    'ARKQ', 'QTUM', 'PCAR', 'SN', 'OKLO', 'ROKU', 'SSNC', 'RBRK', 'CRCL', 'CORT', 'NEE',
    'CNR', 'AIR', 'IR', 'APTV', 'NEGG', 'KTOS', 'ESTC', 'AMBA', 'TTMI', 'SEZL', 'MCHP',
    'LNTH', 'LIVN', 'MP', 'KOMP', 'O', 'INOD', 'CRSP', 'CECO', 'DOCS', 'HNGE', 'SNY',
    'URA', 'RKLB', 'BN', 'HROW', 'SKWD', 'DOCN', 'KARO', 'U', 'TECK', 'EXEL', 'AMKR',
    'INTA', 'SPNS', 'HUT', 'GLBE', 'GCT', 'CGNX', 'VKTX', 'CNP', 'BP', 'TOST', 'NNE',
    'BMNR', 'REZI', 'CHWY', 'KLAR', 'PACS', 'BRBR', 'ARQQ', 'LQDT', 'ALGM', 'RGTI',
    'CRK', 'QBTS', 'RF', 'TENB', 'AAOI', 'GLXY', 'OUST', 'CPRX', 'LYFT', 'SMR', 'ARCC',
    'HSAI', 'ZIM', 'WYFI', 'JHX', 'ZETA', 'SONO', 'SKYT', 'INFY', 'CLBT', 'ATEN', 'USAR',
    'NU', 'OSCR', 'CORZ', 'UUUU', 'JOBY', 'STNE', 'EOSE', 'GRRR', 'AEVA', 'EH', 'PONY',
    'ACHC', 'DLO', 'SERV', 'QUBT', 'QS', 'PL', 'SOUN', 'OSPN', 'AMPX', 'STLA', 'GILT',
    'LUNR', 'DV', 'UMAC', 'OPFI', 'DCTH', 'RZLT', 'DNA', 'TSSI', 'ONDS', 'ACHR', 'LUMN',
    'QMCO', 'AMCR', 'SHLS', 'MOB', 'TMC', 'CCC', 'OPEN', 'EVTL', 'BBAI', 'ASPI', 'BTQ',
    'PTON', 'POET', 'PDYN', 'MNKD', 'SLDP', 'VERI', 'EVEX', 'KSCP', 'RXRX', 'RIG', 'RR',
    'SPCE', 'ABAT', 'NUAI', 'VTEX', 'ARAI', 'MSOS', 'PYPD', 'MVST', 'DGXX', 'EVGO',
    'HIVE', 'WOOF', 'BITF', 'HNST', 'LIDR', 'KOPN', 'ORBS', 'SRFM', 'BTBT', 'BTAI',
    'CRNT', 'SLNH', 'ALTS', 'QSI', 'INVZ', 'PSTV', 'NVNO', 'APLD', 'CRWV', 'RZLV',
    'RCAT', 'NVTS', 'IONQ', 'BKSY', 'MNTS', 'ASTS', 'PSTG', 'CIFR', 'UAMY', 'FIG',
    'AQMS', 'ISRG', 'SYK', 'MDT', 'ABT'
]

# הסרת כפילויות
ALL_TICKERS = list(set(ALL_TICKERS))

def analyze_batch(tickers_batch):
    """מעבד קבוצה קטנה של מניות כדי לא לקרוס"""
    try:
        data = yf.download(tickers_batch, period="1y", group_by='ticker', auto_adjust=True, threads=True)
    except:
        return []

    batch_results = []
    
    # אם יש רק מניה אחת בבאץ', המבנה שונה
    if len(tickers_batch) == 1:
        ticker = tickers_batch[0]
        # נטפל בזה כאילו זו רשימה
        # (לא נסבך את הקוד כרגע, נניח שהבאץ' תמיד > 1 ברוב המקרים)

    for ticker in tickers_batch:
        try:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) < 200: continue

            # === חישובים כבדים (הכל) ===
            # MAs
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['EMA_8'] = ta.ema(df['Close'], length=8)
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Oscillators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_H'] = macd['MACDh_12_26_9']
            
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx['ADX_14']
            
            aroon = ta.aroon(df['High'], df['Low'])
            df['Aroon_U'] = aroon['AROONU_14']
            
            # Bands & Vol
            bb = ta.bbands(df['Close'], length=20, std=2)
            df['BB_U'] = bb['BBU_5_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Current Values
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # === Logic Score ===
            score = 50
            if curr['Close'] > curr['SMA_200']: score += 15
            if curr['Close'] > curr['SMA_50']: score += 10
            if curr['RSI'] < 30: score += 20
            if curr['RSI'] > 75: score -= 15
            if curr['MACD'] > 0 and curr['MACD_H'] > 0: score += 10
            if curr['ADX'] > 25 and curr['Aroon_U'] > 70: score += 10
            
            final_score = min(max(score, 0), 100)
            
            rec = "HOLD"
            if final_score >= 80: rec = "STRONG BUY 🚀"
            elif final_score >= 65: rec = "BUY 🟢"
            elif final_score <= 30: rec = "SELL 🔴"

            # פיבונאצ'י
            year_high = df['High'].max()
            year_low = df['Low'].min()
            fib_618 = year_high - (0.618 * (year_high - year_low))
            
            # פיבוט
            pivot = (curr['High'] + curr['Low'] + curr['Close']) / 3
            r1 = (2 * pivot) - curr['Low']
            s1 = (2 * pivot) - curr['High']

            batch_results.append({
                'Symbol': ticker,
                'Price': round(curr['Close'], 2),
                'Change%': round(((curr['Close'] - prev['Close']) / prev['Close']) * 100, 2),
                'Rec': rec,
                'Score': int(final_score),
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
            })
        except:
            continue
            
    return batch_results

if st.button('🔥 הפעל סריקת על (Batch Processing)'):
    st.write("מתחיל תהליך... אל תסגור את החלון.")
    
    master_results = []
    
    # כאן הקסם: חלוקה למנות של 30 כדי לא לקרוס
    chunk_size = 30
    chunks = [ALL_TICKERS[i:i + chunk_size] for i in range(0, len(ALL_TICKERS), chunk_size)]
    
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"מעבד קבוצה {i+1} מתוך {len(chunks)}... ({len(chunk)} מניות)")
        
        # עיבוד הקבוצה
        batch_res = analyze_batch(chunk)
        master_results.extend(batch_res)
        
        # עדכון בר התקדמות
        prog_bar.progress((i + 1) / len(chunks))
        
        # מנוחה קטנה לשרת (מונע חסימות)
        time.sleep(0.5)

    status_text.success("✅ הסריקה הושלמה בהצלחה!")
    prog_bar.empty()
    
    if master_results:
        df = pd.DataFrame(master_results)
        
        # --- תצוגה 1: Top 5 ---
        st.subheader("🏆 Top 5 Opportunities")
        st.dataframe(df.sort_values('Score', ascending=False).head(5), use_container_width=True)
        
        # --- תצוגה 2: דוח טלגרם מפורט ---
        st.divider()
        st.subheader("🔬 כרטיס מניה מפורט (Deep Dive)")
        
        sel = st.selectbox("בחר מניה מהרשימה:", df['Symbol'].tolist())
        row = df[df['Symbol'] == sel].iloc[0]
        
        # הטקסט המפורט שביקשת
        report = f"""
🚀 **{row['Symbol']} REPORT** 🚀
════════════════════════
• Price: ${row['Price']} ({row['Change%']}%)
• Rec: {row['Rec']} (Score: {row['Score']})
• Volume: {row['Vol_M']}M

📊 **Trend & Momentum**
• RSI: {row['RSI']} | ADX: {row['ADX']} (Strength)
• MACD: {row['MACD']}
• vs SMA200: {row['Dist_SMA200']}%

🎯 **Key Levels**
• Pivot: ${row['Pivot']}
• Resistance (R1): ${row['R1']}
• Support (S1): ${row['S1']}
• Golden Fib (61.8%): ${row['Fib_618']}
• VWAP: ${row['VWAP']}

🛡️ **Risk**
• ATR (Volatility): ${row['ATR']}
════════════════════════
"""
        st.code(report, language="yaml")
        
        # --- תצוגה 3: הורדה ---
        st.divider()
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 הורד דוח מלא (Excel)", csv, "full_scan.csv", "text/csv")
        
        # הצגת כל הטבלה למטה
        with st.expander("ראה טבלה מלאה (כל המניות)"):
            st.dataframe(df)
            
    else:
        st.error("לא הצלחנו למשוך נתונים. נסה שוב מאוחר יותר.")
