import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
from google import genai
import time

# --- 0. Gemini AI 보안 설정 ---
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# --- 1. 유니버스 데이터 (한국/미국 각 50종목) ---
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

# --- 2. 퀀트 터미널 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal Quant", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F2F2F7;}
    .quant-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #E5E5EA;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 정밀 퀀트 분석 엔진 ---
@st.cache_data(ttl=3600)
def analyze_stock_quant(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m_val, s_val = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        
        curr_price = df['Close'].iloc[-1]
        bb_h, bb_l = bb.bollinger_hband().iloc[-1], bb.bollinger_lband().iloc[-1]
        bb_pos = (curr_price - bb_l) / (bb_h - bb_l) * 100
        
        # --- 10단위 확률 모델 로직 ---
        score = 50.0 
        if rsi < 30: score += 30
        elif rsi < 40: score += 15
        elif rsi > 70: score -= 25
        elif rsi > 60: score -= 10
        
        if m_val > s_val: score += 15
        else: score -= 15
        
        if bb_pos < 10: score += 20
        elif bb_pos < 30: score += 10
        elif bb_pos > 90: score -= 20
        
        # 최종 승률(확률) 계산 (0-100)
        prob = int(max(0, min(100, score)))
        
        # 10단위 구간별 의사결정 매핑
        if prob >= 90: verdict = "Strong Conviction Buy (압도적 매수 구간)"
        elif prob >= 80: verdict = "Institutional Buy (기관형 강한 매수)"
        elif prob >= 70: verdict = "Accumulate (적극 분할 매수)"
        elif prob >= 60: verdict = "Watchful Buy (조심스러운 매수)"
        elif prob >= 50: verdict = "Hold (중립적 관망)"
        elif prob >= 40: verdict = "Caution (비중 축소 고려)"
        elif prob >= 30: verdict = "Weak Bearish (하락 압력 우세)"
        elif prob >= 20: verdict = "Strong Sell (강한 매도 신호)"
        elif prob >= 10: verdict = "Risk Off (적극 현금화)"
        else: verdict = "Panic Exit (기술적 붕괴 구간)"
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 1),
            "MACD_Trend": "Bullish (상승)" if m_val > s_val else "Bearish (하락)", 
            "BB_Pos_Val": bb_pos, "Probability": prob, "Verdict": verdict, "df": df
        }
    except: return None

# --- 4. 메인 대시보드 화면 ---
st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800;'> Alpha Terminal <span style='color:#007AFF;'>Quant</span></h1>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

tab1, tab2, tab3 = st.tabs(["Strategy Portfolio (내 포트폴리오)", "Market Universe (시장 스크리닝)", "AI Research (AI 리서치)"])

# [탭 1: 포트폴리오 및 10단위 확률 분석]
with tab1:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    n_name = col_in1.text_input("Asset Name (종목명)", placeholder="예: 삼성전자")
    n_ticker = col_in2.text_input("Ticker Symbol (티커)", placeholder="예: 005930.KS")
    if col_in3.button("Add (추가)"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                with st.container():
                    st.markdown(f"""
                    <div class="quant-card">
                        <div style="font-size: 14px; color: #8E8E93; font-weight: 600;">{name} ({tk})</div>
                        <div style="font-size: 32px; font-weight: 800; color: #1C1C1E; margin: 8px 0;">{data['Price']:,.2f}</div>
                        <div style="background-color: #007AFF15; color: #007AFF; padding: 12px; border-radius: 12px; font-weight: 700; text-align: center; font-size: 16px;">
                            승률 점수: {data['Probability']}/100 ➔ {data['Verdict']}
                        </div>
                        <div style="margin-top: 15px; font-size: 13px; color: #3A3A3C; line-height: 1.8;">
                            • <b>RSI (과열 지표):</b> {data['RSI']} (30↓ 과매도, 70↑ 과매수)<br>
                            • <b>MACD (추세 강도):</b> {data['MACD_Trend']}<br>
                            • <b>BB Position (가격 위치):</b> {data['BB_Pos_Val']:.1f}% (0% 바닥, 100% 천장)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"목록에서 제거", key=f"del_{tk}"):
                        del st.session_state.my_portfolio[name]
                        st.rerun()

# [탭 2: 시장 스크리닝 (한국/미국 유니버스 100선)]
with tab2:
    st.markdown("### 🔍 Market Universe Screening (시장별 50선 전수 조사)")
    c_kr, c_us = st.columns(2)
    
    def get_universe_df(stock_dict):
        results = []
        for tk, name in stock_dict.items():
            res = analyze_stock_quant(tk)
            if res:
                results.append({
                    "Asset (자산)": name, "Score (점수)": res['Probability'],
                    "Signal (신호)": res['Verdict'], "RSI": res['RSI'], "Trend (추세)": res['MACD_Trend']
                })
        return pd.DataFrame(results)

    with c_kr:
        st.markdown("🇰🇷 **KOSPI & KOSDAQ Top 50**")
        st.dataframe(get_universe_df(KR_STOCKS), use_container_width=True, hide_index=True, column_config={"Score (점수)": st.column_config.ProgressColumn(min_value=0, max_value=100)})

    with c_us:
        st.markdown("🇺🇸 **S&P 500 & NASDAQ Top 50**")
        st.dataframe(get_universe_df(US_STOCKS), use_container_width=True, hide_index=True, column_config={"Score (점수)": st.column_config.ProgressColumn(min_value=0, max_value=100)})

# [탭 3: AI Research (전문가용 리포트)]
with tab3:
    st.markdown("### 🏛️ Institutional Daily Research (기관급 전략 리포트)")
    if gemini_client:
        market_news = "미국 금리 인하 지연 가능성, AI 반도체 수급 불균형, 지정학적 에너지 가격 변동"
        if st.button("Generate Quant Report (리포트 생성)"):
            with st.spinner("Analyzing Global Data..."):
                prompt = f"""
                당신은 월스트리트의 시니어 퀀트 애널리스트입니다.
                다음 키워드[{market_news}]를 바탕으로 전문적인 'Market Insider Report'를 작성하세요.
                내용은 다음을 포함해야 합니다:
                1. 주요 기술적 지표 변동 상황 (RSI, 거래량 중심 분석)
                2. 통계적 확률에 기반한 진입/청산 전략
                3. 향후 리스크 시나리오 및 투자자 대응 가이드
                최대한 직관적이지만 전문 용어와 한국어 설명을 병기하여 작성해주세요.
                """
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Quota error or Server Busy: {e}")
