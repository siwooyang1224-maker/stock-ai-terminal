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

# --- 1. 유니버스 데이터 복구 (한국/미국 각 50종목) ---
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

# --- 3. 정밀 퀀트 엔진 (Decision & Confidence 로직) ---
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
        
        # --- 스코어링 로직 ---
        score = 50.0 
        # RSI 가중치
        if rsi < 30: score += 25
        elif rsi < 40: score += 10
        elif rsi > 70: score -= 25
        elif rsi > 60: score -= 10
        # MACD 가중치
        if m_val > s_val: score += 15
        else: score -= 15
        # 볼린저 밴드 가중치
        if bb_pos < 15: score += 20
        elif bb_pos < 35: score += 10
        elif bb_pos > 85: score -= 20
        
        # --- Decision 가르마 타기 ---
        if score > 53:
            decision = "BUY (매수)"
            confidence = (score - 50) * 2
        elif score < 47:
            decision = "SELL (매도)"
            confidence = (50 - score) * 2
        else:
            decision = "HOLD (관망)"
            confidence = 100 - (abs(50 - score) * 10)
            
        final_confidence = int(max(10, min(100, confidence)) // 10 * 10)
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2),
            "MACD_Trend": "Bullish (상승)" if m_val > s_val else "Bearish (하락)",
            "MACD": macd, "BB": bb, "BB_Pos_Val": bb_pos, 
            "Decision": decision, "Confidence": final_confidence, "df": df
        }
    except: return None

# --- 4. 분석 결과 카드 UI ---
def get_analysis_card(data):
    color = "#34C759" if "BUY" in data['Decision'] else "#FF3B30" if "SELL" in data['Decision'] else "#8E8E93"
    
    return f"""
    <div class="quant-card">
        <div style="font-size: 14px; color: #8E8E93; font-weight: 600; margin-bottom: 10px;">{data['Ticker']} Analysis Research</div>
        <div style="display: flex; align-items: baseline; gap: 10px;">
            <span style="font-size: 36px; font-weight: 800; color: {color};">{data['Decision']}</span>
            <span style="font-size: 20px; font-weight: 600; color: #1C1C1E;">확률: {data['Confidence']}%</span>
        </div>
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #F2F2F7; font-size: 13px; color: #3A3A3C; line-height: 1.8;">
            <b>지표 근거:</b> 현재 RSI는 {data['RSI']}로 기술적 { '과매도' if data['RSI'] < 35 else '과매수' if data['RSI'] > 65 else '중립' } 상태이며, 
            가격은 볼린저 밴드 내 {data['BB_Pos_Val']:.1f}% 지점에 위치하여 { '이격 해소' if data['BB_Pos_Val'] < 20 else '조정 가능성' if data['BB_Pos_Val'] > 80 else '안정적 흐름' }을 보이고 있습니다.
        </div>
    </div>
    """

# --- 5. 대시보드 메인 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800;'> Alpha Terminal <span style='color:#007AFF;'>Quant</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Strategy Portfolio", "Market Universe (50/50)", "AI Research"])

with tab1:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    n_name = col_in1.text_input("Asset Name", placeholder="ex) 삼성전자")
    n_ticker = col_in2.text_input("Ticker Symbol", placeholder="ex) 005930.KS")
    if col_in3.button("Add to Monitor"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    for name, tk in st.session_state.my_portfolio.items():
        data = analyze_stock_quant(tk)
        if data:
            st.markdown(f"#### {name} ({tk})")
            st.markdown(get_analysis_card(data), unsafe_allow_html=True)
            
            # --- 고도화된 3단 차트 (Price/BB, MACD, RSI) ---
            df_chart = data['df'][-120:]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
            
            # 1. Price & Bollinger Bands
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"), row=1, col=1)
            bb_chart = ta.volatility.BollingerBands(df_chart['Close'])
            fig.add_trace(go.Scatter(x=df_chart.index, y=bb_chart.bollinger_hband(), line=dict(color='rgba(0,122,255,0.3)', width=1), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=bb_chart.bollinger_lband(), line=dict(color='rgba(0,122,255,0.3)', width=1), fill='tonexty', name="BB Lower"), row=1, col=1)
            
            # 2. MACD
            macd_chart = ta.trend.MACD(df_chart['Close'])
            fig.add_trace(go.Scatter(x=df_chart.index, y=macd_chart.macd(), line=dict(color='#007AFF', width=1.5), name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=macd_chart.macd_signal(), line=dict(color='#FF9500', width=1), name="Signal"), row=2, col=1)
            fig.add_trace(go.Bar(x=df_chart.index, y=macd_chart.macd_diff(), marker_color='gray', name="Histogram"), row=2, col=1)
            
            # 3. RSI
            rsi_values = ta.momentum.rsi(df_chart['Close'])
            fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_values, line=dict(color='#AF52DE', width=1.5), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(height=700, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button(f"Delete Asset {name}", key=f"del_{tk}"):
                del st.session_state.my_portfolio[name]
                st.rerun()
            st.markdown("---")

# [탭 2: 한국/미국 유니버스 50/50 복구]
with tab2:
    st.markdown("### 🔍 Global Market Universe Screening")
    c_kr, c_us = st.columns(2)
    
    def get_screening_df(stock_dict):
        res_list = []
        for tk, name in stock_dict.items():
            r = analyze_stock_quant(tk)
            if r:
                res_list.append({
                    "Asset": name, "Decision": r['Decision'], "Conf. (%)": r['Confidence'], 
                    "RSI": r['RSI'], "Trend": r['MACD_Trend']
                })
        return pd.DataFrame(res_list)

    with c_kr:
        st.markdown("🇰🇷 **KOSPI & KOSDAQ 50**")
        st.dataframe(get_screening_df(KR_STOCKS), use_container_width=True, hide_index=True)

    with c_us:
        st.markdown("🇺🇸 **S&P 500 & NASDAQ 50**")
        st.dataframe(get_screening_df(US_STOCKS), use_container_width=True, hide_index=True)

# [탭 3: AI Research]
with tab3:
    st.markdown("### 🏛️ Institutional Quant Report")
    if gemini_client:
        if st.button("Generate Alpha Research"):
            with st.spinner("Analyzing Global Factors..."):
                prompt = "당신은 월스트리트의 시니어 퀀트 애널리스트입니다. 현재 국채 금리 및 매크로 지표를 고려하여, 연세대 공학/경영학도 수준에서 논리적으로 납득 가능한 5줄 내외의 마켓 인사이트를 제공하세요."
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")
