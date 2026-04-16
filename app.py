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
    .decision-label { font-size: 13px; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.5px; }
    .decision-value { font-size: 26px; font-weight: 900; margin: 10px 0; display: flex; align-items: baseline; gap: 8px; }
    .decision-prob { font-size: 15px; font-weight: 600; padding: 4px 10px; border-radius: 6px; }
    .data-table { width: 100%; font-size: 13.5px; margin-top: 15px; border-collapse: collapse; }
    .data-table td { padding: 10px 0; border-bottom: 1px solid #F1F3F5; }
</style>
""", unsafe_allow_html=True)

# --- 3. 정밀 퀀트 엔진 (IB Best Practice Multi-Factor Model) ---
@st.cache_data(ttl=300)
def analyze_stock_quant(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False).dropna()
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        # 지표 산출
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
        
        # --- IB Best Practice Scoring (0~100) ---
        score = 50.0 # 시작 기준선
        
        # 1. Trend & Momentum (MACD + ADX)
        if m_val > s_val:
            score += 15
            if adx > 25: score += 15
        else:
            score -= 15
            if adx > 25: score -= 15
            
        # 2. Smart Money Flow (MFI)
        if mfi > 70: score += 20
        elif mfi > 55: score += 10
        elif mfi < 30: score -= 20
        elif mfi < 45: score -= 10
        
        # 3. Mean Reversion & Volatility (RSI + BB)
        if rsi < 30: score += 10
        elif rsi > 70: score -= 10
        if bb_pos < 10: score += 5
        elif bb_pos > 90: score -= 5
        
        final_score = int(max(0, min(100, score)))
        
        # 방향성 맵핑
        if final_score >= 60:
            if final_score >= 80: verdict, color = "STRONG BUY (강력 매수)", "#00873C"
            else: verdict, color = "ACCUMULATE (분할 매수)", "#62B236"
            conf_str = f"매수 확률: {final_score}%"
            conf_val = final_score
            conf_bg = "#E6F4EA"
        elif final_score <= 40:
            if final_score <= 20: verdict, color = "STRONG SELL (강력 매도)", "#FF3B30"
            else: verdict, color = "REDUCE (비중 축소)", "#FF9500"
            conf_str = f"매도 확률: {100 - final_score}%"
            conf_val = 100 - final_score
            conf_bg = "#FCE8E6"
        else:
            verdict, color = "HOLD (중립 관망)", "#8E8E93"
            conf_str = f"방향성 모호 (스코어: {final_score})"
            conf_val = 50
            conf_bg = "#F1F3F5"
            
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 2), "MACD_Status": "Bullish Cross" if m_val > s_val else "Bearish Cross",
            "BB_Pos": round(bb_pos, 1), "ADX": round(adx, 1), "MFI": round(mfi, 1), "ATR": round(atr, 2),
            "Verdict": verdict, "Conf_Str": conf_str, "Conf_Val": conf_val, "Conf_Bg": conf_bg, "Score": final_score, "Color": color, "df": df
        }
    except: return None

# --- 4. 대시보드 메인 ---
st.markdown("<h2 style='text-align: left; color: #1C1C1E; font-weight: 900; letter-spacing: -1px;'>ALPHA TERMINAL <span style='color:#00529B;'>QUANT-INSIGHT</span></h2>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "IonQ": "IONQ"}

tab1, tab2, tab3 = st.tabs(["[1] ASSET STRATEGY (자산 전략)", "[2] UNIVERSE SCREENING (전수 조사)", "[3] RESEARCH (리서치)"])

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
                # ATR 설명 문장 추가
                atr_msg = f"ATR(변동성 지수)가 {data['ATR']:.2f}입니다. 이는 최근 일일 평균 가격 변동폭을 의미하며, 해당 수치를 바탕으로 타이트한 손절선과 목표가를 설정하여 리스크를 관리해야 합니다."

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
                        <tr><td style="color:#8E8E93;">ATR (변동성 지수)</td><td style="text-align:right;">{data['ATR']:.2f}</td></tr>
                    </table>
                    <div style="margin-top: 20px; font-size: 13px; color: #495057; line-height: 1.7; background: #F8F9FA; padding: 15px; border-radius: 6px;">
                        <b>📉 퀀트 팩터 분석 요약:</b><br>
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

# [탭 2: 유니버스 스크리닝 - 모든 지표 출력]
with tab2:
    st.markdown("### Global Universe Screening (100 Assets Summary)")
    st.info("💡 스크리닝 점수(Quant Score)는 RSI, MACD, BB, ADX, MFI 데이터를 모두 합산하여 산출됩니다.")
    
    def get_screen_data(stocks):
        res = []
        for t, n in stocks.items():
            d = analyze_stock_quant(t)
            if d: res.append({
                "Asset (종목)": n, "Ticker": t, "Score (퀀트점수)": d['Score'], 
                "Verdict (의견)": d['Verdict'], "RSI": d['RSI'], "MACD": d['MACD_Status'], 
                "BB(%)": d['BB_Pos'], "ADX": d['ADX'], "MFI": d['MFI'], "ATR": d['ATR']
            })
        return pd.DataFrame(res)

    c1, c2 = st.columns(2)
    column_cfg = {
        "Score (퀀트점수)": st.column_config.ProgressColumn(
            "Quant Score (0-100)", min_value=0, max_value=100, format="%d"
        )
    }

    with c1:
        st.markdown("🇰🇷 **KOSPI & KOSDAQ Top 50**")
        st.dataframe(get_screen_data(KR_STOCKS), use_container_width=True, hide_index=True, column_config=column_cfg)
    with c2:
        st.markdown("🇺🇸 **S&P 500 & NASDAQ Top 50**")
        st.dataframe(get_screen_data(US_STOCKS), use_container_width=True, hide_index=True, column_config=column_cfg)

# [탭 3: 리서치 리포트]
with tab3:
    st.markdown("### Institutional Alpha Research (전문가 전략 보고)")
    if gemini_client:
        if st.button("Generate Senior Analyst Briefing (리포트 생성)"):
            with st.spinner("Accessing Terminal Meta-Data..."):
                prompt = "당신은 월스트리트의 시니어 애널리스트입니다. 현재 마켓의 주요 기술적 지표들을 기반으로, 연세대 경영/공학 대학생 수준에서 논리적으로 납득 가능한 투자 대응 전략 5줄을 마크다운 형식으로 작성하세요."
                try:
                    res = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    st.markdown(f"<div style='background-color: #FFFFFF; padding: 25px; border-left: 5px solid #00529B; line-height: 1.8; box-shadow: 0 2px 5px rgba(0,0,0,0.05);'>{res.text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
