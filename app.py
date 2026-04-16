import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
import google.generativeai as genai

# --- 0. Gemini AI 설정 ---
GEMINI_API_KEY = "AQ.Ab8RN6LCuzeVpFq2twhVD4-96Fc06eCeaTgU1qCuPVKRn8EJuw"

# 복잡한 if문 없이 바로 AI를 켭니다.
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 1. 페이지 설정 및 Apple iOS 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal iOS", layout="wide")

st.markdown("""
<style>
    /* 폰트 및 배경 설정 (iOS 스타일) */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {
        font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;
    }
    .stApp {
        background-color: #F2F2F7; /* iOS 기본 백그라운드 그레이 */
    }
    .block-container {
        padding-top: 3rem; 
        max-width: 1400px;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* iOS 위젯 스타일 카드 (Metric) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        border: none;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #1C1C1E;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #8E8E93;
        font-weight: 500;
    }

    /* 탭 디자인 (iOS Segmented Control 스타일) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E5E5EA;
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        color: #8E8E93;
        font-weight: 600;
        border: none;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.12), 0 3px 1px rgba(0,0,0,0.04);
    }
    
    /* 데이터프레임 (표) 모서리 둥글게 */
    .stDataFrame {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* 버튼 스타일 (iOS 블루) */
    .stButton>button {
        background-color: #007AFF;
        color: white;
        border-radius: 14px;
        border: none;
        font-weight: 600;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    
    /* 입력창 스타일 */
    .stTextInput input {
        border-radius: 12px;
        border: 1px solid #E5E5EA;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 유니버스 (한/미 100선) ---
KR_STOCKS = {
    '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아', '035420.KS': 'NAVER',
    '035720.KS': '카카오', '068270.KS': '셀트리온', '005490.KS': 'POSCO홀딩스', '051910.KS': 'LG화학', '006400.KS': '삼성SDI',
    '105560.KS': 'KB금융', '055550.KS': '신한지주', '090430.KS': '아모레퍼시픽', '086520.KQ': '에코프로', '036570.KS': '엔씨소프트',
    '096770.KS': 'SK이노베이션', '066570.KS': 'LG전자', '028260.KS': '삼성물산', '015760.KS': '한국전력', '011200.KS': 'HMM',
    '033780.KS': 'KT&G', '086790.KS': '하나금융지주', '034220.KS': 'LG디스플레이', '003490.KS': '대한항공', '034020.KS': '두산에너빌리티',
    '267250.KS': 'HD현대', '032830.KS': '삼성생명', '138040.KS': '메리츠금융지주', '042660.KS': '한화오션', '259960.KS': '크래프톤',
    '035900.KQ': 'JYP Ent.', '251270.KS': '넷마블', '352820.KS': '하이브', '010950.KS': 'S-Oil', '000810.KS': '삼성화재',
    '030200.KS': 'KT', '017670.KS': 'SK텔레콤', '036460.KS': '한국가스공사', '010130.KS': '고려아연', '011780.KS': '금호석유',
    '051900.KS': 'LG생활건강', '009150.KS': '삼성전기', '018260.KS': '대한제당', '000100.KS': '유한양행', '000080.KS': '하이트진로',
    '005830.KS': 'DB손해보험', '000720.KS': '현대건설', '012330.KS': '현대모비스', '004020.KS': '현대제철', '024110.KS': '기업은행'
}

US_STOCKS = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'GOOGL': 'Alphabet', 'AMZN': 'Amazon',
    'META': 'Meta', 'TSLA': 'Tesla', 'AVGO': 'Broadcom', 'LLY': 'Eli Lilly', 'JPM': 'JPMorgan',
    'V': 'Visa', 'WMT': 'Walmart', 'JNJ': 'J&J', 'PG': 'P&G', 'MA': 'Mastercard',
    'HD': 'Home Depot', 'CVX': 'Chevron', 'MRK': 'Merck', 'COST': 'Costco', 'ABBV': 'AbbVie',
    'PEP': 'PepsiCo', 'KO': 'Coca-Cola', 'ADBE': 'Adobe', 'ORCL': 'Oracle', 'BAC': 'BofA',
    'CRM': 'Salesforce', 'AMD': 'AMD', 'NFLX': 'Netflix', 'TMO': 'Thermo Fisher', 'CSCO': 'Cisco',
    'NKE': 'Nike', 'DIS': 'Disney', 'PFE': 'Pfizer', 'ABT': 'Abbott', 'DHR': 'Danaher',
    'QCOM': 'Qualcomm', 'CAT': 'Caterpillar', 'VZ': 'Verizon', 'TXN': 'TI', 'INTC': 'Intel',
    'AMAT': 'Applied Mat.', 'INTU': 'Intuit', 'IBM': 'IBM', 'LOW': 'Lowe\'s', 'NEE': 'NextEra',
    'UNP': 'Union Pacific', 'COP': 'ConocoPhillips', 'GE': 'GE', 'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley'
}

# --- 3. 정밀 분석 엔진 ---
@st.cache_data(ttl=3600)
def analyze_stock_detailed(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        h, l = bb.bollinger_hband(), bb.bollinger_lband()
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m, ms = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        
        curr_price = df['Close'].iloc[-1]
        bb_pos = (curr_price - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) * 100
        macd_trend = "상승" if m > ms else "하락"
        
        score = 50.0
        if m > ms: score += 15
        else: score -= 10
        if rsi <= 35: score += 20
        elif rsi >= 70: score -= 25
        if bb_pos <= 10: score += 15
        elif bb_pos >= 90: score -= 15
        
        final_score = int(max(0, min(100, score)))
        verdict = "🚀 적극 매수" if final_score >= 80 else "✅ 분할 매수" if final_score >= 60 else "🆘 위험/매도" if final_score <= 35 else "🟡 관망"
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 1),
            "MACD": macd_trend, "BB_Pos": f"{bb_pos:.1f}%", "Score": final_score, "Verdict": verdict, "df": df
        }
    except: return None

# --- 4. 메인 화면 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "TSLL": "TSLL"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800; margin-bottom: 2rem;'> Alpha Terminal</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["포트폴리오", "시장 스크리닝", "AI 브리핑"])

with tab1:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>나의 종목 관리</h3>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 2, 1])
    n_name = col_a.text_input("종목명 입력", placeholder="예: 삼성전자")
    n_ticker = col_b.text_input("티커 입력", placeholder="예: 005930.KS")
    st.markdown("""<style>div.stButton {margin-top: 28px;}</style>""", unsafe_allow_html=True)
    if col_c.button("추가하기"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()
    
    st.write(" ")
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_detailed(tk)
        if data:
            with p_cols[i % 2]:
                st.metric(f"{name}", f"{data['Price']:,.1f}", f"AI 점수: {data['Score']}점")
                st.markdown(f"<div style='color:#8E8E93; font-size:0.9rem; margin-bottom:10px;'>💡 <b>상태:</b> {data['Verdict']} <br> 📊 <b>데이터:</b> RSI {data['RSI']} | MACD {data['MACD']} | BB {data['BB_Pos']}</div>", unsafe_allow_html=True)
                if st.button(f"목록에서 삭제", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()
                
                # 미니멀 차트 (배경 투명화)
                fig = go.Figure(go.Candlestick(x=data['df'].index[-40:], open=data['df']['Open'][-40:], high=data['df']['High'][-40:], low=data['df']['Low'][-40:], close=data['df']['Close'][-40:]))
                fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>한·미 시장 유니버스 분석</h3>", unsafe_allow_html=True)
    c_kr, c_us = st.columns(2)
    
    def get_df(stock_dict):
        results = []
        for tk, name in stock_dict.items():
            res = analyze_stock_detailed(tk)
            if res:
                results.append({
                    "기업명": name, "AI 점수": res['Score'], "판단": res['Verdict'],
                    "RSI": res['RSI'], "MACD": res['MACD'], "BB": res['BB_Pos']
                })
        return pd.DataFrame(results)

    with c_kr:
        st.markdown("<h5 style='color: #1C1C1E;'>🇰🇷 KOSPI & KOSDAQ 50</h5>", unsafe_allow_html=True)
        st.dataframe(get_df(KR_STOCKS), use_container_width=True, hide_index=True, column_config={"AI 점수": st.column_config.ProgressColumn(min_value=0, max_value=100)})

    with c_us:
        st.markdown("<h5 style='color: #1C1C1E;'>🇺🇸 S&P 500 & NASDAQ 50</h5>", unsafe_allow_html=True)
        st.dataframe(get_df(US_STOCKS), use_container_width=True, hide_index=True, column_config={"AI 점수": st.column_config.ProgressColumn(min_value=0, max_value=100)})

with tab3:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>Gemini 데일리 브리핑</h3>", unsafe_allow_html=True)
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        st.info("API Key를 입력하여 실시간 브리핑을 활성화하세요.")
    else:
        st.success("시장 상황 요약 데이터를 수신했습니다.")
