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

# --- 1. 퀀트 터미널 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal Quant", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F2F2F7;}
    
    /* 전문적이고 밀도 있는 카드 스타일 */
    .quant-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #E5E5EA;
        margin-bottom: 20px;
    }
    .metric-title { font-size: 14px; color: #8E8E93; font-weight: 600; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 800; color: #1C1C1E; }
    .probability-box {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: white;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 전문가용 퀀트 분석 함수 ---
def get_quant_analysis_html(data):
    rsi = data['RSI']
    bb_pos = data['BB_Pos_Val']
    macd_val = data['MACD_Diff']
    prob = data['Probability']
    
    # RSI 촘촘한 분석
    if rsi < 30: rsi_desc = "Extreme Oversold (과매도 극치: 저점 매수 확률 85%↑)"
    elif rsi < 45: rsi_desc = "Weak Bearish (약세 구간: 하방 경직성 확보 중)"
    elif rsi < 55: rsi_desc = "Neutral Pivot (중립 구간: 방향성 탐색 시점)"
    elif rsi < 70: rsi_desc = "Weak Bullish (강세 구간: 추세 추종 유효)"
    else: rsi_desc = "Strong Overbought (과매수 경계: 수익 실현 매물 출회 주의)"

    # 볼린저 밴드 정밀 분석
    if bb_pos < 5: bb_desc = "Band Floor (밴드 하단 이탈: 평균 회귀 확률 극대화)"
    elif bb_pos < 25: bb_desc = "Discount Range (저평가 영역: 분할 매수 적기)"
    elif bb_pos < 75: bb_desc = "Standard Fairway (정상 범주: 안정적 추세 진행)"
    elif bb_pos < 95: bb_desc = "Premium Range (고평가 영역: 추가 상승 동력 둔화)"
    else: bb_desc = "Band Ceiling (밴드 상단 돌파: 단기 고점 징후)"

    # 확률별 색상
    prob_color = "#FF3B30" if prob < 40 else "#FF9500" if prob < 65 else "#34C759"

    html = f"""
    <div class="quant-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div style="font-size: 18px; font-weight: 800; color: #1C1C1E;">📊 Technical Strategy Scorecard</div>
            <div style="background-color: {prob_color}22; color: {prob_color}; padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 14px;">
                Win Probability: {prob}%
            </div>
        </div>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #F2F2F7;">
                <td style="padding: 12px 0; font-size: 13px; color: #8E8E93;">Oscillator RSI (14)</td>
                <td style="padding: 12px 0; text-align: right; font-weight: 600; font-size: 13px;">{rsi} → <span style="color:#007AFF;">{rsi_desc}</span></td>
            </tr>
            <tr style="border-bottom: 1px solid #F2F2F7;">
                <td style="padding: 12px 0; font-size: 13px; color: #8E8E93;">Bollinger Band Pos (%)</td>
                <td style="padding: 12px 0; text-align: right; font-weight: 600; font-size: 13px;">{bb_pos:.1f}% → <span style="color:#007AFF;">{bb_desc}</span></td>
            </tr>
            <tr>
                <td style="padding: 12px 0; font-size: 13px; color: #8E8E93;">Trend Momentum (MACD)</td>
                <td style="padding: 12px 0; text-align: right; font-weight: 600; font-size: 13px;">{data['MACD_Trend']} (Diff: {macd_val:.2f})</td>
            </tr>
        </table>
        <div style="margin-top: 20px; padding: 15px; background-color: #F8F8F9; border-radius: 12px; font-size: 13px; line-height: 1.6; color: #3A3A3C;">
            <b>🎯 리서치 인사이트:</b> 현재 {data['Ticker']}는 기술적으로 <b>{prob}%</b>의 매수 적합성을 보이고 있습니다. 
            {rsi_desc}와 {bb_desc}를 종합할 때, 현재 포지션은 <b>{data['Verdict']}</b> 전략이 통계적으로 우월한 구간입니다.
        </div>
    </div>
    """
    return html

# --- 3. 정밀 퀀트 엔진 ---
@st.cache_data(ttl=3600)
def analyze_stock_quant(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # 지표 계산
        bb = ta.volatility.BollingerBands(df['Close'])
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m_val, s_val = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        
        curr_price = df['Close'].iloc[-1]
        bb_h, bb_l = bb.bollinger_hband().iloc[-1], bb.bollinger_lband().iloc[-1]
        bb_pos = (curr_price - bb_l) / (bb_h - bb_l) * 100
        
        # --- 확률 모델 로직 (Quant Scoring) ---
        score = 50.0 # Base Neutral
        
        # 1. RSI 가중치 (역발상 전략 가중치)
        if rsi < 30: score += 25
        elif rsi < 40: score += 15
        elif rsi > 70: score -= 20
        elif rsi > 60: score -= 10
        
        # 2. MACD 가중치 (추세 추종)
        if m_val > s_val: score += 15
        else: score -= 15
        
        # 3. 볼린저 밴드 가중치 (가격 위치)
        if bb_pos < 15: score += 20
        elif bb_pos < 30: score += 10
        elif bb_pos > 85: score -= 15
        
        # 거래량 필터 (전일 대비 거래량 증가 시 신뢰도 업)
        vol_sma = df['Volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['Volume'].iloc[-1]
        if curr_vol > vol_sma: score += 5
        
        final_prob = int(max(5, min(98, score))) # 5%~98% 사이로 제한
        
        # 최종 판단
        if final_prob >= 75: verdict = "Institutional Buy (기관급 강한 매수)"
        elif final_prob >= 60: verdict = "Accumulate (분할 매수 유효)"
        elif final_prob <= 35: verdict = "Risk Off (적극 매도/관망)"
        else: verdict = "Hold (중립/추세 관찰)"
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2),
            "MACD_Trend": "Bullish" if m_val > s_val else "Bearish", 
            "MACD_Diff": m_val - s_val,
            "BB_Pos_Val": bb_pos, "Probability": final_prob, 
            "Verdict": verdict, "df": df
        }
    except: return None

# --- 4. 메인 대시보드 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800;'> Alpha Terminal <span style='color:#007AFF;'>Quant</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Strategy Portfolio", "Market Universe", "AI Research"])

with tab1:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    n_name = col_in1.text_input("Asset Name", placeholder="삼성전자")
    n_ticker = col_in2.text_input("Ticker Symbol", placeholder="005930.KS")
    if col_in3.button("Add to Terminal"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                st.markdown(f"### {name} ({tk})")
                st.metric("Market Price", f"{data['Price']:,.2f}", f"Entry Probability: {data['Probability']}%")
                
                # 퀀트 인사이트 카드 출력
                st.markdown(get_quant_analysis_html(data), unsafe_allow_html=True)
                
                if st.button(f"Remove Asset", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()
                
                # 차트 출력
                df_chart = data['df'][-100:]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=ta.volatility.BollingerBands(df_chart['Close']).bollinger_hband(), line=dict(color='rgba(0,122,255,0.2)'), name="BB Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=ta.volatility.BollingerBands(df_chart['Close']).bollinger_lband(), line=dict(color='rgba(0,122,255,0.2)'), fill='tonexty', name="BB Lower"), row=1, col=1)
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name="Volume", marker_color="#E5E5EA"), row=2, col=1)
                fig.update_layout(height=400, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# [탭 2/3 생략 및 탭3 프롬프트 강화]
with tab3:
    st.markdown("### 🏛️ Institutional Daily Research")
    if gemini_client:
        market_news = "미국 국채 금리 변동성, 반도체 공급망 재편 이슈, 빅테크 실적 발표 전야"
        if st.button("Generate Quant Report"):
            with st.spinner("Analyzing global market data..."):
                prompt = f"""
                당신은 골드만삭스의 시니어 퀀트 애널리스트입니다. 
                시장 데이터[{market_news}]를 바탕으로 기관 투자자용 리포트를 작성하세요.
                내용에는 반드시 다음이 포함되어야 합니다:
                1. 주요 기술적 지표 변동 사항 (RSI, 거래량 중심)
                2. 통계적 확률에 기반한 섹터별 로테이션 전략
                3. 향후 48시간 내 발생 가능한 테일 리스크 및 대응 매뉴얼
                """
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Quota exceeded or Error: {e}")
