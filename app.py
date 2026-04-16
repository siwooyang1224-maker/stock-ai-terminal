import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta

# --- 1. 페이지 설정 및 프리미엄 UI CSS ---
st.set_page_config(page_title="Alpha Terminal Pro", layout="wide")

st.markdown("""
<style>
    /* 전체 배경 및 여백 설정 */
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px;}
    
    /* 헤더 및 푸터 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 탭 디자인 모던화 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #f0f2f6;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    
    /* 메트릭 카드(숫자) 디자인 */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem; 
        font-weight: 800; 
        color: #1f77b4;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 한/미 30선 기업명 매핑 ---
KR_STOCKS = {
    '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차',
    '000270.KS': '기아', '035420.KS': 'NAVER', '035720.KS': '카카오',
    '068270.KS': '셀트리온', '005490.KS': 'POSCO홀딩스', '051910.KS': 'LG화학',
    '006400.KS': '삼성SDI', '105560.KS': 'KB금융', '055550.KS': '신한지주',
    '090430.KS': '아모레퍼시픽', '086520.KQ': '에코프로', '036570.KS': '엔씨소프트'
}

US_STOCKS = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA',
    'GOOGL': 'Alphabet (Google)', 'AMZN': 'Amazon', 'META': 'Meta',
    'TSLA': 'Tesla', 'AVGO': 'Broadcom', 'LLY': 'Eli Lilly',
    'JPM': 'JPMorgan Chase', 'V': 'Visa', 'WMT': 'Walmart',
    'JNJ': 'Johnson & Johnson', 'PG': 'Procter & Gamble', 'MA': 'Mastercard'
}

# --- 3. AI 판단 엔진 (다이내믹 스코어링 V3) ---
@st.cache_data(ttl=3600) # 1시간마다 데이터 갱신 (로딩 속도 대폭 개선)
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None, 50, "데이터 부족"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_H'], df['BB_L'] = bb.bollinger_hband(), bb.bollinger_lband()
        df['RSI'] = ta.momentum.rsi(df['Close'])
        macd = ta.trend.MACD(df['Close'])
        df['M'], df['MS'] = macd.macd(), macd.macd_signal()
        
        curr = df.iloc[-1]
        score = 50.0 # 기본 점수
        
        # 1. 추세 강도 (MACD)
        if curr['M'] > curr['MS']: score += 15
        else: score -= 10
        
        # 2. 가격 메리트 (RSI) - 세분화된 점수 부여
        rsi = curr['RSI']
        if rsi <= 35: score += 20
        elif rsi <= 45: score += 10
        elif rsi >= 70: score -= 25
        elif rsi >= 60: score -= 10
            
        # 3. 볼린저 밴드 위치 (밴드 폭 대비 현재가 위치)
        bb_width = curr['BB_H'] - curr['BB_L']
        if bb_width > 0:
            bb_pos = (curr['Close'] - curr['BB_L']) / bb_width
            if bb_pos < 0.1: score += 15
            elif bb_pos < 0.3: score += 5
            elif bb_pos > 0.9: score -= 15
            elif bb_pos > 0.7: score -= 5
            
        # 점수 보정 (0 ~ 100 사이로 고정)
        final_score = int(max(0, min(100, score)))
        
        # 판단 기준 세분화
        if final_score >= 80: verdict = "🚀 강력 매수"
        elif final_score >= 60: verdict = "✅ 매수 검토"
        elif final_score <= 35: verdict = "🆘 분할 매도"
        else: verdict = "🟡 관망 (HOLD)"
        
        return df, final_score, verdict
    except: return None, 50, "오류"

# --- 4. 메인 화면 구성 ---
st.title("🏛️ 전략 투자 터미널 V3")
st.markdown("데이터 파이프라인 기반 **정량적 종목 스크리닝** 및 **AI 트레이딩 시그널**")

tab1, tab2, tab3 = st.tabs(["💎 포트폴리오 진단", "📊 한·미 전략 종목 30선", "🌍 매크로 리스크"])

# [Tab 1: 보유 종목 집중 분석]
with tab1:
    my_stocks = {"SK하이닉스": "000660.KS", "TSLL (Tesla 2x)": "TSLL"}
    cols = st.columns(2)
    for i, (name, tk) in enumerate(my_stocks.items()):
        df, score, verdict = analyze_stock(tk)
        with cols[i]:
            st.metric(f"{name}", f"{df['Close'].iloc[-1]:,.1f}", f"AI Score: {score}/100")
            st.markdown(f"**현재 AI 포지션 판단:** {verdict}")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df.index[-100:], open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-100:], y=df['BB_H'][-100:], line=dict(color='rgba(150,150,150,0.2)'), name="BB High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-100:], y=df['BB_L'][-100:], line=dict(color='rgba(150,150,150,0.2)'), fill='tonexty', name="BB Low"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-100:], y=df['RSI'][-100:], name="RSI", line=dict(color='#9467bd')), row=2, col=1)
            
            fig.update_layout(height=450, margin=dict(l=0,r=0,t=20,b=0), template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# [Tab 2: 추천 30선 리스트 (한/미 분리)]
with tab2:
    st.markdown("#### 🎯 AI 스크리닝: 양대 시장 주도주 30선")
    col_kr, col_us = st.columns(2)
    
    with col_kr:
        st.subheader("🇰🇷 KOSPI & KOSDAQ 15")
        kr_data = []
        for tk, name in KR_STOCKS.items():
            _, score, v = analyze_stock(tk)
            kr_data.append({"기업명": name, "AI 스코어": score, "매매 판단": v})
        
        df_kr = pd.DataFrame(kr_data)
        st.dataframe(df_kr, use_container_width=True, hide_index=True)

    with col_us:
        st.subheader("🇺🇸 S&P 500 & NASDAQ 15")
        us_data = []
        for tk, name in US_STOCKS.items():
            _, score, v = analyze_stock(tk)
            us_data.append({"기업명": name, "AI 스코어": score, "매매 판단": v})
            
        df_us = pd.DataFrame(us_data)
        st.dataframe(df_us, use_container_width=True, hide_index=True)

# [Tab 3: 거시경제 분석]
with tab3:
    st.markdown("#### 🌍 글로벌 리스크 매트릭스")
    st.info("📉 **금리 인하 지연 우려:** 미국 주요 경제지표(CPI, PCE) 발표에 따른 빅테크(Apple, Microsoft) 및 레버리지(TSLL) 변동성 주의")
    st.warning("🏭 **공급망 및 무역 갈등:** 반도체 장비 수출 통제 이슈가 SK하이닉스, 삼성전자 하반기 실적 가이던스에 미칠 영향 모니터링")
    st.success("💄 **소비재 반등 가능성:** 중국 경기 부양책 발표 시 아모레퍼시픽, LG화학 등 관련 섹터의 단기 모멘텀 부각 예상")
