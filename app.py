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

# --- 2. Bloomberg/IB 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal IB", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F9F9FB;}
    
    /* IB 리서치 카드 스타일 */
    .ib-card {
        background-color: #FFFFFF;
        border-radius: 4px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 4px solid #00529B; /* Bloomberg Blue */
        margin-bottom: 20px;
    }
    .decision-label { font-size: 14px; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 1px; }
    .decision-value { font-size: 28px; font-weight: 900; margin-bottom: 15px; }
    .data-table { width: 100%; font-size: 13px; border-top: 1px solid #EEEEEE; margin-top: 10px; }
    .data-table td { padding: 8px 0; border-bottom: 1px solid #F9F9FB; }
</style>
""", unsafe_allow_html=True)

# --- 3. 정밀 퀀트 엔진 ---
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
        
        # --- 논리적 의사결정 매핑 ---
        score = 50.0
        if rsi < 35: score += 20
        elif rsi > 65: score -= 20
        if m_val > s_val: score += 15
        else: score -= 15
        if bb_pos < 20: score += 15
        elif bb_pos > 80: score -= 15
        
        # 확률 대신 명확한 등급 시스템
        if score >= 70:
            verdict, color = "BUY (STRONG CONVICTION)", "#00873C"
        elif score >= 55:
            verdict, color = "ACCUMULATE (NEUTRAL BUY)", "#62B236"
        elif score >= 45:
            verdict, color = "HOLD (NEUTRAL)", "#8E8E93"
        elif score >= 30:
            verdict, color = "REDUCE (NEUTRAL SELL)", "#FF9500"
        else:
            verdict, color = "SELL (LIQUIDATE)", "#FF3B30"
            
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2),
            "MACD_Status": "Bullish Cross" if m_val > s_val else "Bearish Cross",
            "BB_Pos": round(bb_pos, 1), "Verdict": verdict, "Color": color, "df": df
        }
    except: return None

# --- 4. 메인 대시보드 ---
st.markdown("<h2 style='text-align: left; color: #1C1C1E; font-weight: 900; letter-spacing: -1px;'>ALPHA TERMINAL <span style='color:#00529B;'>QUANT-INSIGHT</span></h2>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

tab1, tab2, tab3 = st.tabs(["[1] ASSET STRATEGY", "[2] UNIVERSE SCREENING", "[3] RESEARCH"])

# [탭 1: 개별 종목 분석 - 좌/우 2열 배치]
with tab1:
    col_reg1, col_reg2, col_reg3 = st.columns([2, 2, 1])
    n_name = col_reg1.text_input("Asset Name", placeholder="삼성전자")
    n_ticker = col_reg2.text_input("Ticker Symbol", placeholder="005930.KS")
    if col_reg3.button("Register Asset"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    st.markdown("---")
    
    # 좌우 2열 배치 복구
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                st.markdown(f"""
                <div class="ib-card">
                    <div class="decision-label">{name} ({tk}) / Technical Status</div>
                    <div class="decision-value" style="color: {data['Color']};">{data['Verdict']}</div>
                    <table class="data-table">
                        <tr><td>Current Price</td><td style="text-align:right; font-weight:700;">{data['Price']:,.2f}</td></tr>
                        <tr><td>RSI (Relative Strength)</td><td style="text-align:right;">{data['RSI']}</td></tr>
                        <tr><td>MACD (Momentum)</td><td style="text-align:right;">{data['MACD_Status']}</td></tr>
                        <tr><td>BB (Standard Deviation Pos)</td><td style="text-align:right;">{data['BB_Pos']}%</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                # --- 차트 고도화 및 선 명시화 ---
                df_chart = data['df'][-120:]
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
                
                # 1. Price & BB (범례 명시)
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"), row=1, col=1)
                bb = ta.volatility.BollingerBands(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb.bollinger_hband(), line=dict(color='rgba(0,82,155,0.4)', width=1), name="BB Upper (2σ)"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb.bollinger_lband(), line=dict(color='rgba(0,82,155,0.4)', width=1), fill='tonexty', name="BB Lower (2σ)"), row=1, col=1)
                
                # 2. MACD (범례 명시)
                macd = ta.trend.MACD(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=macd.macd(), line=dict(color='#00529B', width=1.5), name="MACD Line"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=macd.macd_signal(), line=dict(color='#FF9500', width=1), name="Signal Line"), row=2, col=1)
                fig.add_trace(go.Bar(x=df_chart.index, y=macd.macd_diff(), marker_color='#E5E5EA', name="Histogram"), row=2, col=1)
                
                # 3. RSI (범례 명시)
                rsi_v = ta.momentum.rsi(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_v, line=dict(color='#AF52DE', width=1.5), name="RSI (14)"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#FF3B30", row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#34C759", row=3, col=1)
                
                fig.update_layout(height=650, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"Close Position {name}", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

# [탭 2: 한·미 유니버스 100선 복구]
with tab2:
    st.markdown("### Global Universe Screening (100 Assets)")
    def get_screen_df(stocks):
        res = []
        for t, n in stocks.items():
            d = analyze_stock_quant(t)
            if d: res.append({"Asset": n, "Ticker": t, "Verdict": d['Verdict'], "RSI": d['RSI'], "Trend": d['MACD_Status']})
        return pd.DataFrame(res)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("🇰🇷 **Korea Top 50**")
        st.dataframe(get_screen_df(KR_STOCKS), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("🇺🇸 **US Top 50**")
        st.dataframe(get_screen_df(US_STOCKS), use_container_width=True, hide_index=True)

# [탭 3: 리서치 리포트]
with tab3:
    st.markdown("### Institutional Alpha Research")
    if gemini_client:
        if st.button("Generate Senior Analyst Report"):
            with st.spinner("Accessing Terminal Data..."):
                prompt = "당신은 월스트리트의 시니어 애널리스트입니다. 현재 기술적 지표들을 기반으로, 연세대 경영/공학도 수준에서 납득 가능한 시장 대응 전략 5줄을 작성하세요. 불필요한 수식어는 배제하십시오."
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(f"<div style='background-color: #F0F4F8; padding: 20px; border-left: 5px solid #00529B;'>{res.text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
