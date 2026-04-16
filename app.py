import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
import google.generativeai as genai

# --- 0. Gemini AI 보안 설정 (Secrets 방식) ---
# 깃허브 코드에는 키를 직접 적지 않고, 스트림릿 금고에서 불러옵니다.
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# --- 1. 페이지 설정 및 Apple iOS 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal iOS", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F2F2F7;}
    .block-container {padding-top: 3rem; max-width: 1400px;}
    #MainMenu, footer, header {visibility: hidden;}

    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 700; color: #1C1C1E;}
    div[data-testid="stMetricLabel"] {font-size: 1rem; color: #8E8E93;}

    .stTabs [data-baseweb="tab-list"] {background-color: #E5E5EA; border-radius: 12px; padding: 4px;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px; color: #8E8E93; font-weight: 600;}
    .stTabs [aria-selected="true"] {background-color: #FFFFFF !important; color: #000000 !important;}
</style>
""", unsafe_allow_html=True)

# --- 2. 유니버스 데이터 ---
KR_STOCKS = {'005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '035420.KS': 'NAVER', '035720.KS': '카카오'}
US_STOCKS = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'TSLA': 'Tesla', 'AMZN': 'Amazon'}

# --- 3. 분석 엔진 ---
@st.cache_data(ttl=3600)
def analyze_stock_detailed(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        h, l = bb.bollinger_hband(), bb.bollinger_lband()
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        m, ms = ta.trend.macd(df['Close']).iloc[-1], ta.trend.macd_signal(df['Close']).iloc[-1]
        
        curr_price = df['Close'].iloc[-1]
        bb_pos = (curr_price - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) * 100
        
        score = 50 + (15 if m > ms else -10) + (20 if rsi <= 35 else -25 if rsi >= 70 else 0)
        final_score = int(max(0, min(100, score)))
        verdict = "🚀 적극 매수" if final_score >= 80 else "✅ 분할 매수" if final_score >= 60 else "🆘 위험/매도" if final_score <= 35 else "🟡 관망"
        
        return {"Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 1), "MACD": "상승" if m > ms else "하락", "BB_Pos": f"{bb_pos:.1f}%", "Score": final_score, "Verdict": verdict, "df": df}
    except: return None

# --- 4. UI 레이아웃 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "NVIDIA": "NVDA"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800;'> Alpha Terminal</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["포트폴리오", "시장 스크리닝", "AI 브리핑"])

with tab1:
    col_a, col_b, col_c = st.columns([2, 2, 1])
    n_name = col_a.text_input("종목명", placeholder="삼성전자")
    n_ticker = col_b.text_input("티커", placeholder="005930.KS")
    if col_c.button("추가") and n_name and n_ticker:
        st.session_state.my_portfolio[n_name] = n_ticker
        st.rerun()
    
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_detailed(tk)
        if data:
            with p_cols[i % 2]:
                st.metric(name, f"{data['Price']:,.1f}", f"Score: {data['Score']}")
                if st.button(f"삭제", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()
                
                # 전문가용 차트
                df_c = data['df'][-100:]
                bb_c = ta.volatility.BollingerBands(df_c['Close'])
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=bb_c.bollinger_hband(), line=dict(color='rgba(150,150,150,0.5)', width=1), name="BB"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=ta.momentum.rsi(df_c['Close']), line=dict(color='#007AFF', width=2), name="RSI"), row=2, col=1)
                fig.update_layout(height=350, margin=dict(l=20,r=20,t=10,b=10), plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, showlegend=False)
                fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
                fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(pd.DataFrame([analyze_stock_detailed(t) for t in KR_STOCKS] + [analyze_stock_detailed(t) for t in US_STOCKS]).dropna().drop(columns=['df']), use_container_width=True)

with tab3:
    if gemini_model:
        try:
            res = gemini_model.generate_content("현재 시장 상황(반도체 강세, 금리 변동성)을 고려해 투자 전략을 3줄 요약해줘.")
            st.success(res.text)
        except: st.error("AI 연결 상태를 확인하세요.")
    else:
        st.info("Streamlit Secrets에 GEMINI_API_KEY를 설정해주세요.")
