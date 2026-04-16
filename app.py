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
@st.cache_data(ttl=3600)
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
