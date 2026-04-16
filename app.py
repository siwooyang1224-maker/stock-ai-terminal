import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
from google import genai
import time
from streamlit_autorefresh import st_autorefresh

# --- 0. Gemini AI 보안 설정 ---
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

# --- 1. 유니버스 데이터 (KR/US Top 50) ---
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
st_autorefresh(interval=300000, key="datarefresh") # 5분 자동 새로고침

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F8F9FA;}
    .ib-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 5px solid #D71920; /* Aggressive Red 포인트 */
        margin-bottom: 25px;
    }
    .decision-label { font-size: 13px; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.5px; }
    .decision-value { font-size: 26px; font-weight: 900; margin: 10px 0; }
    .data-table { width: 100%; font-size: 13.5px; margin-top: 15px; border-collapse: collapse; }
    .data-table td { padding: 10px 0; border-bottom: 1px solid #F1F3F5; }
</style>
""", unsafe_allow_html=True)

# --- 3. 공격적 투자용 퀀트 분석 엔진 ---
@st.cache_data(ttl=300)
def analyze_stock_aggressive(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False).dropna()
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # 지표 산출
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        mfi = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
        adx = ta.trend.adx(df['High'], df['Low'], df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m_val, s_val = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        bb = ta.volatility.BollingerBands(df['Close'])
        bb_pos = (df['Close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) * 100
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).iloc[-1]

        # --- 공격적 투자자용 가중치 스코어링 (Aggressive Strategy) ---
        score = 50.0
        # 1. 추세 강도(ADX) 및 모멘텀(MACD): 60% 가중
        if adx > 25: # 추세 형성
            score += 15
            if m_val > s_val: score += 15 # 상승 모멘텀 일치
        else: # 횡보장
            score -= 10
            
        # 2. 자금 유입(MFI): 20% 가중
        if mfi > 60: score += 10
        elif mfi < 40: score -= 10
        
        # 3. 상대적 위치 및 강도(RSI/BB): 20% 가중
        if rsi > 70: score += 5 # 성장주는 과열권에서도 더 감 (돌파 매매)
        elif rsi < 30: score -= 5 # 하락 추세 성장주는 배제
        
        final_score = int(max(0, min(100, score)))
        
        # 방향성 및 확신도 판정
        if final_score >= 70:
            decision, color = "AGGRESSIVE BUY (공격적 매수)", "#D71920"
            confidence = int((final_score - 50) * 2)
        elif final_score >= 50:
            decision, color = "WATCH & ACCUMULATE (추세 관찰)", "#FF9500"
            confidence = int((final_score - 50) * 2 + 30)
        else:
            decision, color = "RISK OFF (적극 대피)", "#8E8E93"
            confidence = int((50 - final_score) * 2)

        return {
            "Ticker": ticker, "Price": df['Close'].iloc[-1], "RSI": round(rsi, 2),
            "MFI": round(mfi, 1), "ADX": round(adx, 1), "ATR": round(atr, 2),
            "Verdict": decision, "Confidence": min(100, confidence), "Color": color, "df": df, "MACD": macd, "BB": bb
        }
    except Exception as e:
        return None

# --- 4. 대시보드 메인 ---
st.markdown("<h2 style='text-align: left; color: #1C1C1E; font-weight: 900; letter-spacing: -1px;'>ALPHA TERMINAL <span style='color:#D71920;'>AGGRESSIVE-QUANT</span></h2>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "NVIDIA": "NVDA"}

tab1, tab2, tab3 = st.tabs(["[1] GROWTH STRATEGY (성장주 전략)", "[2] UNIVERSE SCAN (전수 조사)", "[3] RISK RESEARCH (리스크 연구)"])

# [탭 1: 공격적 자산 전략]
with tab1:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    n_name = col_in1.text_input("Growth Asset Name", placeholder="ex) 엔비디아")
    n_ticker = col_in2.text_input("Ticker Symbol", placeholder="ex) NVDA")
    if col_in3.button("Deploy Analysis"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_aggressive(tk)
        if data:
            with p_cols[i % 2]:
                st.markdown(f"""
                <div class="ib-card">
                    <div class="decision-label">{name} ({tk}) / Aggressive Decision Core</div>
                    <div class="decision-value" style="color: {data['Color']};">{data['Verdict']} (확신도: {data['Confidence']}%)</div>
                    <table class="data-table">
                        <tr><td>Current Price (현재가)</td><td style="text-align:right; font-weight:700;">{data['Price']:,.2f}</td></tr>
                        <tr><td>ADX (추세 강도)</td><td style="text-align:right; color:{'#D71920' if data['ADX'] > 25 else '#8E8E93'};">{data['ADX']} ({'추세 형성' if data['ADX'] > 25 else '횡보/약세'})</td></tr>
                        <tr><td>MFI (자금 유입 효율)</td><td style="text-align:right;">{data['MFI']} (Smart Money Flow)</td></tr>
                        <tr><td>ATR (변동성 지수)</td><td style="text-align:right;">{data['ATR']} (Expected Range)</td></tr>
                    </table>
                    <div style="margin-top: 15px; font-size: 13px; color: #495057; line-height: 1.6; background: #FFF5F5; padding: 15px; border-radius: 6px;">
                        <b>🔥 Aggressive Insight (공격적 분석):</b><br>
                        • {name}의 ADX는 {data['ADX']}로, 현재 {'강력한 추세 구간에 진입하여 추가 랠리 가능성이 높음' if data['ADX'] > 25 else '추세적 에너지가 부족하여 기간 조정이 예상됨'}을 시사합니다.<br>
                        • MFI 수치가 {data['MFI']}인 것으로 보아, 현재 가격 상승 시 {'대규모 자금이 동반 유입되는 고효율 구간' if data['MFI'] > 60 else '수급 뒷받침이 약한 기술적 반등 구간'}으로 판단됩니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 차트 출력 (3단 구성)
                df_c = data['df'][-120:]
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.25, 0.25])
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=data['BB'].bollinger_hband()[-120:], line=dict(color='rgba(215,25,32,0.2)', width=1), name="BB Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=data['BB'].bollinger_lband()[-120:], line=dict(color='rgba(215,25,32,0.2)', width=1), fill='tonexty', name="BB Lower"), row=1, col=1)
                
                # MFI 차트 (RSI 대신 자금유입 강조)
                mfi_c = ta.volume.money_flow_index(df_c['High'], df_c['Low'], df_c['Close'], df_c['Volume'])
                fig.add_trace(go.Scatter(x=df_c.index, y=mfi_c, line=dict(color='#D71920', width=1.5), name="MFI (자금유입)"), row=2, col=1)
                fig.add_hline(y=80, line_dash="dot", line_color="#D71920", row=2, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color="#34C759", row=2, col=1)
                
                # ADX 차트 (추세 강도 강조)
                adx_c = ta.trend.adx(df_c['High'], df_c['Low'], df_c['Close'])[-120:]
                fig.add_trace(go.Scatter(x=df_c.index, y=adx_c, line=dict(color='#00529B', width=1.5), name="ADX (추세강도)"), row=3, col=1)
                fig.add_hline(y=25, line_dash="dot", line_color="orange", row=3, col=1)

                fig.update_layout(height=650, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"Purge Asset {name}", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

# [탭 2: 유니버스 게이지 스캐너]
with tab2:
    st.markdown("### High-Growth Universe Screening (Top 100)")
    def get_growth_df(stocks):
        res = []
        for t, n in stocks.items():
            d = analyze_stock_aggressive(t)
            if d: res.append({
                "Asset": n, "Ticker": t, "Conf. (%)": d['Confidence'], "Verdict": d['Verdict'], "ADX (Trend)": d['ADX'], "MFI (Money)": d['MFI']
            })
        return pd.DataFrame(res)

    c1, c2 = st.columns(2)
    cfg = {"Conf. (%)": st.column_config.ProgressColumn("Growth Conviction", min_value=0, max_value=100)}
    with c1:
        st.markdown("🇰🇷 **K-Growth 50**")
        st.dataframe(get_growth_df(KR_STOCKS), use_container_width=True, hide_index=True, column_config=cfg)
    with c2:
        st.markdown("🇺🇸 **U.S. Growth 50**")
        st.dataframe(get_growth_df(US_STOCKS), use_container_width=True, hide_index=True, column_config=cfg)

# [탭 3: 공격적 리서치]
with tab3:
    st.markdown("### Senior Growth Strategist Report")
    if gemini_client:
        if st.button("Generate Alpha Strategy"):
            with st.spinner("Compiling High-Growth Intelligence..."):
                prompt = "당신은 월스트리트의 성장주 전문 전략가입니다. 현재 금리 환경과 기술주 섹터의 ADX/MFI 데이터를 기반으로, 공격적 투자자가 주목해야 할 5줄 이내의 핵심 전술 리포트를 작성하세요."
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(f"<div style='background-color: #FFF5F5; padding: 25px; border-left: 5px solid #D71920; line-height: 1.8;'>{res.text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
