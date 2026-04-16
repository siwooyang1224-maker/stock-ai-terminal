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
st_autorefresh(interval=300000, key="datarefresh")

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
        border-top: 5px solid #00529B;
        margin-bottom: 25px;
    }
    .macro-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-left: 5px solid #D71920;
        margin-bottom: 20px;
    }
    .decision-label { font-size: 13px; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.5px; }
    .decision-value { font-size: 26px; font-weight: 900; margin: 10px 0; display: flex; align-items: baseline; gap: 8px; }
    .decision-prob { font-size: 15px; font-weight: 600; padding: 4px 10px; border-radius: 6px; }
    .data-table { width: 100%; font-size: 13.5px; margin-top: 15px; border-collapse: collapse; }
    .data-table td { padding: 10px 0; border-bottom: 1px solid #F1F3F5; }
</style>
""", unsafe_allow_html=True)

# --- 3. 정밀 퀀트 엔진 ---
@st.cache_data(ttl=300)
def analyze_stock_quant(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False).dropna()
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m_val, s_val = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        curr_price = df['Close'].iloc[-1]
        bb_h, bb_l = bb.bollinger_hband().iloc[-1], bb.bollinger_lband().iloc[-1]
        bb_pos = (curr_price - bb_l) / (bb_h - bb_l) * 100 if (bb_h - bb_l) != 0 else 50.0
        
        adx = ta.trend.adx(df['High'], df['Low'], df['Close']).iloc[-1]
        mfi = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume']).iloc[-1]
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).iloc[-1]
        atr_pct = (atr / curr_price) * 100 if curr_price != 0 else 0
        
        score = 50.0 
        if m_val > s_val:
            score += 15
            if adx > 25: score += 15
        else:
            score -= 15
            if adx > 25: score -= 15
            
        if mfi > 70: score += 20
        elif mfi > 55: score += 10
        elif mfi < 30: score -= 20
        elif mfi < 45: score -= 10
        
        if rsi < 30: score += 10
        elif rsi > 70: score -= 10
        if bb_pos < 10: score += 5
        elif bb_pos > 90: score -= 5
        
        final_score = int(max(0, min(100, score)))
        
        if final_score >= 60:
            if final_score >= 80: verdict, color = "STRONG BUY (강력 매수)", "#00873C"
            else: verdict, color = "ACCUMULATE (분할 매수)", "#62B236"
            conf_str, conf_val, conf_bg = f"매수 확률: {final_score}%", final_score, "#E6F4EA"
        elif final_score <= 40:
            if final_score <= 20: verdict, color = "STRONG SELL (강력 매도)", "#FF3B30"
            else: verdict, color = "REDUCE (비중 축소)", "#FF9500"
            conf_str, conf_val, conf_bg = f"매도 확률: {100 - final_score}%", 100 - final_score, "#FCE8E6"
        else:
            verdict, color, conf_str, conf_val, conf_bg = "HOLD (중립 관망)", "#8E8E93", f"방향성 모호 (스코어: {final_score})", 50, "#F1F3F5"
            
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2), "MACD_Status": "Bullish Cross" if m_val > s_val else "Bearish Cross",
            "BB_Pos": round(bb_pos, 1), "ADX": round(adx, 1), "MFI": round(mfi, 1), "ATR": round(atr, 2), "ATR_Pct": round(atr_pct, 2),
            "Verdict": verdict, "Conf_Str": conf_str, "Conf_Val": conf_val, "Conf_Bg": conf_bg, "Score": final_score, "Color": color, "df": df
        }
    except: return None

# --- 매크로 데이터 패치 함수 (캐싱으로 속도 최적화) ---
@st.cache_data(ttl=600)
def get_macro_data():
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close']
        tnx = yf.Ticker("^TNX").history(period="5d")['Close']
        return {
            "VIX": vix.iloc[-1], "VIX_diff": vix.iloc[-1] - vix.iloc[-2],
            "TNX": tnx.iloc[-1], "TNX_diff": tnx.iloc[-1] - tnx.iloc[-2]
        }
    except: return None

# --- 4. 대시보드 메인 ---
st.markdown("<h2 style='text-align: left; color: #1C1C1E; font-weight: 900; letter-spacing: -1px;'>ALPHA TERMINAL <span style='color:#00529B;'>QUANT-INSIGHT</span></h2>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

tab1, tab2, tab3 = st.tabs(["[1] ASSET STRATEGY (자산 전략)", "[2] UNIVERSE SCREENING (전수 조사)", "[3] MACRO & RESEARCH (거시/정성 분석)"])

# [탭 1: 자산 전략]
with tab1:
    col_reg1, col_reg2, col_reg3 = st.columns([2, 2, 1])
    n_name = col_reg1.text_input("Asset Name (종목명)", placeholder="ex) 삼성전자")
    n_ticker = col_reg2.text_input("Ticker Symbol (티커)", placeholder="ex) 005930.KS")
    if col_reg3.button("Register Asset (등록)"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    st.markdown("---")
    
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                rsi_msg = f"RSI(심리 강도)가 {data['RSI']}입니다. 현재 {'과열권으로 단기 차익매물 출회 가능성' if data['RSI'] > 65 else '침체권으로 저가 매수세 유입 가능성' if data['RSI'] < 35 else '시장 심리가 안정된 중립 구간'}입니다."
                macd_msg = f"MACD 추세가 {data['MACD_Status']}입니다. 단기 이동평균이 장기 이동평균을 {'상향 돌파하여 상승 랠리' if 'Bullish' in data['MACD_Status'] else '하향 이탈하여 하락 압력'}가 형성 중입니다."
                bb_msg = f"BB(가격 편차 위치)가 {data['BB_Pos']}%입니다. 가격이 통계적 밴드의 {'상단을 뚫어 단기 조정이 예상됨' if data['BB_Pos'] > 85 else '하단에 닿아 기술적 반등이 기대됨' if data['BB_Pos'] < 15 else '정상 범위 안에서 움직임'}을 시사합니다."
                adx_msg = f"ADX(추세 강도)가 {data['ADX']}입니다. 25를 넘으면 추세가 강함을 뜻하며, 현재 {'명확한 방향성을 가지고 뻗어나가는 중' if data['ADX'] > 25 else '방향성이 뚜렷하지 않은 횡보장세'}입니다."
                mfi_msg = f"MFI(자금 유입)가 {data['MFI']}입니다. 거래량이 실린 스마트 머니가 {'강하게 유입되고 있어 추세 신뢰도가 높음' if data['MFI'] > 60 else '점차 빠져나가고 있어 보수적 접근 필요' if data['MFI'] < 40 else '균형을 이루고 있음'}을 의미합니다."
                
                atr_pct = data['ATR_Pct']
                if atr_pct >= 5.0: atr_msg = f"ATR(변동성)은 주가의 **{atr_pct}%**에 달하는 **[고변동성 종목]**입니다. 철저한 손절매 등 리스크 관리가 필수적입니다."
                elif atr_pct >= 2.0: atr_msg = f"ATR(변동성)은 주가의 **{atr_pct}%** 수준입니다. 일반적인 주식의 **[정상 변동폭]** 내에서 움직이고 있습니다."
                else: atr_msg = f"ATR(변동성)은 주가의 **{atr_pct}%**에 불과한 **[저변동성 방어주]** 성향을 보입니다. 중장기 관점이 어울립니다."

                st.markdown(f"""
                <div class="ib-card">
                    <div class="decision-label">{name} ({tk}) / Multi-Factor Intelligence</div>
                    <div class="decision-value" style="color: {data['Color']};">
                        {data['Verdict']}
                        <span class="decision-prob" style="background-color:{data['Conf_Bg']}; color:{data['Color']}; border: 1px solid {data['Color']}40;">{data['Conf_Str']}</span>
                    </div>
                    <table class="data-table">
                        <tr><td style="color:#8E8E93;">Current Price (현재 주가)</td><td style="text-align:right; font-weight:700;">{data['Price']:,.2f}</td></tr>
                        <tr><td style="color:#8E8E93;">RSI (심리 강도)</td><td style="text-align:right;">{data['RSI']}</td></tr>
                        <tr><td style="color:#8E8E93;">MACD Momentum (추세 모멘텀)</td><td style="text-align:right;">{data['MACD_Status']}</td></tr>
                        <tr><td style="color:#8E8E93;">BB Pos (가격 편차 위치)</td><td style="text-align:right;">{data['BB_Pos']}%</td></tr>
                        <tr style="border-top: 2px dashed #F1F3F5;"><td style="color:#8E8E93; font-weight:600;">ADX (추세 강도)</td><td style="text-align:right; font-weight:600; color:{'#D71920' if data['ADX'] > 25 else '#1C1C1E'};">{data['ADX']}</td></tr>
                        <tr><td style="color:#8E8E93; font-weight:600;">MFI (자금 유입 효율)</td><td style="text-align:right; font-weight:600;">{data['MFI']}</td></tr>
                        <tr><td style="color:#8E8E93;">ATR (변동성 비율)</td><td style="text-align:right; font-weight:600;">{data['ATR_Pct']}% (절대값: {data['ATR']:.2f})</td></tr>
                    </table>
                    <div style="margin-top: 20px; font-size: 13.5px; color: #3A3A3C; line-height: 1.75; background: #F8F9FA; padding: 18px; border-radius: 8px;">
                        <b>📉 퀀트 팩터 상세 분석:</b><br>
                        • {rsi_msg}<br>
                        • {macd_msg}<br>
                        • {bb_msg}<br>
                        • {adx_msg}<br>
                        • {mfi_msg}<br>
                        • {atr_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 4단 차트 출력
                df_chart = data['df'][-120:]
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2])
                
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price (주가)"), row=1, col=1)
                bb_ta = ta.volatility.BollingerBands(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb_ta.bollinger_hband(), line=dict(color='rgba(0,82,155,0.4)', width=1), name="BB Upper (상단)"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb_ta.bollinger_lband(), line=dict(color='rgba(0,82,155,0.4)', width=1), fill='tonexty', name="BB Lower (하단)"), row=1, col=1)
                
                macd_ta = ta.trend.MACD(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=macd_ta.macd(), line=dict(color='#00529B', width=1.5), name="MACD (추세)"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=macd_ta.macd_signal(), line=dict(color='#FF9500', width=1), name="Signal (신호)"), row=2, col=1)
                fig.add_trace(go.Bar(x=df_chart.index, y=macd_ta.macd_diff(), marker_color='#DEE2E6', name="Histogram"), row=2, col=1)
                
                rsi_ta = ta.momentum.rsi(df_chart['Close'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_ta, line=dict(color='#AF52DE', width=1.5), name="RSI (심리)"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#FF3B30", row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#34C759", row=3, col=1)
                
                adx_ta = ta.trend.adx(df_chart['High'], df_chart['Low'], df_chart['Close'])
                mfi_ta = ta.volume.money_flow_index(df_chart['High'], df_chart['Low'], df_chart['Close'], df_chart['Volume'])
                fig.add_trace(go.Scatter(x=df_chart.index, y=adx_ta, line=dict(color='#D71920', width=1.5), name="ADX (강도)"), row=4, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=mfi_ta, line=dict(color='#34C759', width=1.5), name="MFI (수급)"), row=4, col=1)
                fig.add_hline(y=25, line_dash="dot", line_color="orange", row=4, col=1) 

                fig.update_layout(height=800, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"Close Asset {name}", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

# [탭 2: 유니버스 스크리닝]
with tab2:
    st.markdown("### Global Universe Screening (100 Assets Summary)")
    st.info("💡 스크리닝 점수(Quant Score)는 MACD, BB, RSI, ADX, MFI 팩터를 모두 합산한 멀티 팩터 모델 결과입니다.")
    
    def get_screen_data(stocks):
        res = []
        for t, n in stocks.items():
            d = analyze_stock_quant(t)
            if d: res.append({
                "Asset (종목)": n, "Ticker": t, "Score (퀀트점수)": d['Score'], 
                "Verdict (의견)": d['Verdict'], "RSI": d['RSI'], "MACD": d['MACD_Status'], 
                "BB(%)": d['BB_Pos'], "ADX": d['ADX'], "MFI": d['MFI'], "ATR(%)": d['ATR_Pct']
            })
        return pd.DataFrame(res)

    c1, c2 = st.columns(2)
    column_cfg = {
        "Score (퀀트점수)": st.column_config.ProgressColumn("Quant Score (0-100)", min_value=0, max_value=100, format="%d")
    }
    with c1:
        st.markdown("🇰🇷 **KOSPI & KOSDAQ Top 50**")
        st.dataframe(get_screen_data(KR_STOCKS), use_container_width=True, hide_index=True, column_config=column_cfg)
    with c2:
        st.markdown("🇺🇸 **S&P 500 & NASDAQ Top 50**")
        st.dataframe(get_screen_data(US_STOCKS), use_container_width=True, hide_index=True, column_config=column_cfg)

# [탭 3: 매크로 및 정성 분석 (토큰 프리)]
with tab3:
    st.markdown("### 🏛️ Macro & Qualitative Intelligence")
    st.info("💡 거시 경제 지표 및 내 포트폴리오 종목의 최신 글로벌 뉴스를 API 제한 없이 실시간으로 제공합니다.")
    
    # 1. 매크로 지표 대시보드
    st.markdown("#### 🌐 Core Macro Indicators (실시간 거시 지표)")
    macro = get_macro_data()
    if macro:
        c_mac1, c_mac2, c_mac3 = st.columns(3)
        with c_mac1:
            vix_color = "red" if macro['VIX_diff'] > 0 else "green"
            st.markdown(f"""
            <div class="macro-card">
                <div style="font-size: 14px; color: #8E8E93; font-weight: 700;">VIX (시장 공포 지수)</div>
                <div style="font-size: 28px; font-weight: 900; margin-top: 5px;">{macro['VIX']:.2f}</div>
                <div style="color: {vix_color}; font-weight: 600;">{'+' if macro['VIX_diff'] > 0 else ''}{macro['VIX_diff']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_mac2:
            tnx_color = "red" if macro['TNX_diff'] > 0 else "green"
            st.markdown(f"""
            <div class="macro-card" style="border-left-color: #007AFF;">
                <div style="font-size: 14px; color: #8E8E93; font-weight: 700;">US 10-Yr Yield (미 국채 10년물 금리)</div>
                <div style="font-size: 28px; font-weight: 900; margin-top: 5px;">{macro['TNX']:.3f}%</div>
                <div style="color: {tnx_color}; font-weight: 600;">{'+' if macro['TNX_diff'] > 0 else ''}{macro['TNX_diff']:.3f}%p</div>
            </div>
            """, unsafe_allow_html=True)
        with c_mac3:
            st.markdown("""
            <div style="padding: 10px; font-size: 12.5px; color: #495057; line-height: 1.6;">
                <b>📌 지표 가이드:</b><br>
                • <b>VIX가 오르면</b>: 시장의 불확실성과 공포가 커지고 있음을 의미합니다 (보통 주가 하락).<br>
                • <b>국채 금리가 오르면</b>: 연준의 긴축(금리 인하 지연) 우려가 커져 기술/성장주에 악재로 작용합니다.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # 2. 내 포트폴리오 뉴스 피드 (Token Free)
    st.markdown("#### 📰 Portfolio News Feed (내 종목 최신 이벤트)")
    news_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        with news_cols[i % 2]:
            with st.expander(f"📌 {name} ({tk}) 주요 뉴스 확인", expanded=True):
                try:
                    # yfinance 내장 뉴스 기능 사용 (Gemini API 미사용)
                    stock_news = yf.Ticker(tk).news
                    if stock_news:
                        for n in stock_news[:4]: # 상위 4개 뉴스
                            title = n.get('title', '제목 없음')
                            publisher = n.get('publisher', '출처 미상')
                            link = n.get('link', '#')
                            
                            # 날짜 변환 (UNIX timestamp -> 일반 날짜)
                            pub_time = "최근"
                            if 'providerPublishTime' in n:
                                pub_time = time.strftime('%Y-%m-%d', time.localtime(n['providerPublishTime']))
                                
                            st.markdown(f"""
                            <div style="margin-bottom: 12px; font-size: 13.5px;">
                                <a href="{link}" target="_blank" style="text-decoration: none; color: #00529B; font-weight: 600; display: block; margin-bottom: 3px;">{title}</a>
                                <span style="font-size: 11.5px; color: #8E8E93;">{publisher} • {pub_time}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("최신 뉴스 데이터가 없습니다.")
                except Exception as e:
                    st.write("뉴스를 불러오지 못했습니다.")

    st.markdown("---")
    
    # 3. AI 심층 리서치 (기존 기능)
    st.markdown("#### 🤖 AI Senior Analyst (선택적 심층 브리핑)")
    if gemini_client:
        if st.button("Generate Senior Analyst Briefing (토큰 소모)"):
            with st.spinner("Accessing Terminal Meta-Data..."):
                prompt = "당신은 월스트리트의 시니어 애널리스트입니다. 현재 마켓의 거시경제 흐름과 주요 기술적 지표들을 기반으로, 연세대 경영/공학 대학생 수준에서 논리적으로 납득 가능한 투자 대응 전략 5줄을 마크다운 형식으로 작성하세요."
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(f"<div style='background-color: #FFFFFF; padding: 25px; border-left: 5px solid #00529B; line-height: 1.8; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>{res.text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"Error (토큰 제한 발생): {e}")
