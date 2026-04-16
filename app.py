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
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        border: 1px solid #E5E5EA;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 논리적 분석 함수 (대학생 수준의 담백한 워딩) ---
def get_quant_analysis_html(data):
    rsi = data['RSI']
    bb_pos = data['BB_Pos_Val']
    macd_diff = data['MACD_Diff']
    prob = data['Probability']
    
    # RSI: 상대강도지수
    if rsi < 30: rsi_desc = "Technical Oversold (과매도: 통계적 저점 구간)"
    elif rsi < 45: rsi_desc = "Neutral Bearish (약세: 하방 경직성 탐색)"
    elif rsi < 55: rsi_desc = "Neutral Pivot (중립: 모멘텀 부재 구간)"
    elif rsi < 70: rsi_desc = "Neutral Bullish (강세: 추세 지속성 확인)"
    else: rsi_desc = "Technical Overbought (과매수: 단기 이격 조정 주의)"

    # 볼린저 밴드: 표준편차 기반 가격 위치
    if bb_pos < 10: bb_desc = "Lower Boundary (하단 이탈: 강한 하방 지지 예상)"
    elif bb_pos < 30: bb_desc = "Support Zone (지지권: 기술적 반등 유효 구간)"
    elif bb_pos < 70: bb_desc = "Fair Value (중심권: 평균 회귀 완료 및 횡보)"
    elif bb_pos < 90: bb_desc = "Resistance Zone (저항권: 상단 압력 점증 구간)"
    else: bb_desc = "Upper Boundary (상단 이탈: 추세 이격 조정 가능성)"

    prob_color = "#FF3B30" if prob < 40 else "#FF9500" if prob < 65 else "#34C759"

    html = f"""
    <div class="quant-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">
            <div style="font-size: 16px; font-weight: 700; color: #1C1C1E;">📑 Technical Scorecard (기술 분석 지표)</div>
            <div style="background-color: {prob_color}15; color: {prob_color}; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 13px;">
                Win Probability: {prob}%
            </div>
        </div>
        <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #F2F2F7;">
                <td style="padding: 10px 0; color: #8E8E93;">Oscillator RSI (14)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{rsi} → {rsi_desc}</td>
            </tr>
            <tr style="border-bottom: 1px solid #F2F2F7;">
                <td style="padding: 10px 0; color: #8E8E93;">Bollinger Band Pos (%)</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{bb_pos:.1f}% → {bb_desc}</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; color: #8E8E93;">Momentum MACD</td>
                <td style="padding: 10px 0; text-align: right; font-weight: 600;">{data['MACD_Trend']} (Spread: {macd_diff:.2f})</td>
            </tr>
        </table>
        <div style="margin-top: 18px; padding: 12px; background-color: #F8F8F9; border-radius: 8px; font-size: 13px; color: #3A3A3C;">
            <b>💡 Summary:</b> 현재 {data['Ticker']}는 기술적으로 <b>{prob}%</b>의 매수 우위 확률을 기록 중입니다. 
            지표간 상관관계를 고려할 때 <b>{data['Verdict']}</b> 포지션이 유효합니다.
        </div>
    </div>
    """
    return html

# --- 4. 정밀 퀀트 엔진 ---
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
        
        # --- 10단위 확률 로직 ---
        score = 50.0 
        if rsi < 30: score += 25
        elif rsi < 40: score += 10
        elif rsi > 70: score -= 20
        elif rsi > 60: score -= 10
        
        if m_val > s_val: score += 15
        else: score -= 15
        
        if bb_pos < 15: score += 20
        elif bb_pos < 35: score += 10
        elif bb_pos > 85: score -= 15
        
        final_prob = int(max(0, min(100, score)))
        final_prob = (final_prob // 10) * 10 # 10단위로 끊기
        
        # Verdict 매핑
        if final_prob >= 80: verdict = "Strong Buy (적극 매수 권장)"
        elif final_prob >= 60: verdict = "Accumulate (분할 매수 접근)"
        elif final_prob >= 40: verdict = "Hold (비중 유지/관망)"
        else: verdict = "Reduce (비중 축소/리스크 관리)"
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2),
            "MACD_Trend": "Bullish (상승 모멘텀)" if m_val > s_val else "Bearish (하락 모멘텀)", 
            "MACD_Diff": m_val - s_val,
            "BB_Pos_Val": bb_pos, "Probability": final_prob, "Verdict": verdict, "df": df
        }
    except: return None

# --- 5. 메인 대시보드 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800;'> Alpha Terminal <span style='color:#007AFF;'>Quant</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Strategy Portfolio", "Market Universe", "AI Research"])

# [탭 1: 개별 종목 분석]
with tab1:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    n_name = col_in1.text_input("Asset Name", placeholder="ex) 삼성전자")
    n_ticker = col_in2.text_input("Ticker Symbol", placeholder="ex) 005930.KS")
    if col_in3.button("Add to Monitor"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                st.markdown(f"#### {name} ({tk})")
                st.metric("Price", f"{data['Price']:,.2f}", f"Score: {data['Probability']}%")
                st.markdown(get_quant_analysis_html(data), unsafe_allow_html=True)
                
                if st.button(f"Delete Asset", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

# [탭 2: 시장 스크리닝 - 100종목 복구]
with tab2:
    st.markdown("### 🔍 Market Universe Screening")
    c_kr, c_us = st.columns(2)
    
    def get_df(stock_dict):
        results = []
        for tk, name in stock_dict.items():
            res = analyze_stock_quant(tk)
            if res:
                results.append({
                    "Name": name, "Prob (%)": res['Probability'], "Verdict": res['Verdict'],
                    "RSI": res['RSI'], "Trend": res['MACD_Trend']
                })
        return pd.DataFrame(results)

    with c_kr:
        st.markdown("🇰🇷 **KOSPI & KOSDAQ Top 50**")
        st.dataframe(get_df(KR_STOCKS), use_container_width=True, hide_index=True)

    with c_us:
        st.markdown("🇺🇸 **S&P 500 & NASDAQ Top 50**")
        st.dataframe(get_df(US_STOCKS), use_container_width=True, hide_index=True)

# [탭 3: AI Research]
with tab3:
    st.markdown("### 🏛️ Institutional Daily Research")
    if gemini_client:
        market_news = "미 국채 금리 변동성 상존, 기술주 이격 조정 가능성, 반도체 수급 재편 이슈"
        if st.button("Generate Quant Report"):
            with st.spinner("Analyzing Market Context..."):
                prompt = f"""
                당신은 월스트리트의 시니어 퀀트 애널리스트입니다. 
                현재 시장의 주요 키워드[{market_news}]를 바탕으로 전문적인 리서치 리포트를 작성하세요.
                내용은 다음을 포함해야 합니다:
                1. 주요 기술적 지표의 통계적 유의성 분석
                2. 리스크 관리를 위한 최적의 자산 배분 전략
                3. 향후 48시간 내 주목해야 할 변곡점
                논리적이고 명료한 한국어로 작성하되, 연세대 공대생이 읽기에 적합한 전문성을 유지하세요.
                """
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")
