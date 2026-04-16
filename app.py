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

# --- 3. 정밀 퀀트 엔진 ---
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
        
        # ATR을 현재 주가 대비 비율(%)로 변환 (리스크 체감용)
        atr_pct = (atr / curr_price) * 100 if curr_price != 0 else 0
        
        # --- IB Best Practice Scoring (0~100) ---
        score = 50.0 
        
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
            "BB_Pos": round(bb_pos, 1), "ADX": round(adx, 1), "MFI": round(mfi, 1), "ATR": round(atr, 2), "ATR_Pct": round(atr_pct, 2),
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
                
                # --- 동적이고 촘촘한 코멘트 생성 로직 ---
                
                # 1. RSI (5단계 세분화)
                rsi = data['RSI']
                if rsi >= 70: rsi_msg = f"RSI가 {rsi}로 **과열권(Overbought)**입니다. 단기 고점 징후가 있으니 신규 진입은 보수적으로 접근해야 합니다."
                elif rsi >= 60: rsi_msg = f"RSI가 {rsi}로 **강세 심리**가 유지 중입니다. 우상향 랠리를 이어갈 추가 동력이 남아있습니다."
                elif rsi <= 30: rsi_msg = f"RSI가 {rsi}로 **극단적 과매도(Oversold)**입니다. 투매가 진정되면 기술적 반등(Dead Cat Bounce)을 노릴 만합니다."
                elif rsi <= 45: rsi_msg = f"RSI가 {rsi}로 **약세 심리**가 지배적입니다. 아직 하방 지지선이 완전히 확인되지 않았습니다."
                else: rsi_msg = f"RSI가 {rsi}로 **팽팽한 중립** 상태입니다. 뚜렷한 매수/매도 우위가 없는 눈치싸움 구간입니다."
                
                # 2. MACD & BB
                macd_msg = f"MACD가 **{data['MACD_Status']}**입니다. 단기 이평선이 장기 이평선을 {'상향 돌파하여 **상승 모멘텀**이 작동 중' if 'Bullish' in data['MACD_Status'] else '하향 이탈하여 **하락 압력**이 작동 중'}입니다."
                bb_msg = f"BB 위치는 **{data['BB_Pos']}%**입니다. 통계적 밴드의 {'상단을 뚫어 평균 회귀(조정)가 우려됨' if data['BB_Pos'] > 85 else '하단에 닿아 기술적 반등이 기대됨' if data['BB_Pos'] < 15 else '중심부에서 정상적인 변동성 내에 머무르고 있음'}을 시사합니다."
                
                # 3. MFI & ADX
                adx = data['ADX']
                mfi = data['MFI']
                adx_msg = f"ADX(추세 강도)는 **{adx}**입니다. {'현재 방향(상승이든 하락이든)으로 **매우 강한 추세**를 형성 중' if adx > 25 else '**뚜렷한 방향성이 없는 횡보장**으로 박스권 매매가 유효'}합니다."
                
                if mfi >= 70: mfi_msg = f"MFI(자금 유입)는 **{mfi}**입니다. 스마트 머니(거대 자금)가 **공격적으로 유입**되고 있어 상승 신뢰도가 높습니다."
                elif mfi >= 55: mfi_msg = f"MFI는 **{mfi}**로 매수 자금이 **점진적으로 유입**되며 수급을 받쳐주고 있습니다."
                elif mfi <= 30: mfi_msg = f"MFI는 **{mfi}**로 자금이 **빠르게 유출**되고 있습니다. 추세 붕괴 리스크를 관리해야 합니다."
                elif mfi <= 45: mfi_msg = f"MFI는 **{mfi}**로 자금 유입이 **저조한 편**입니다. 가짜 반등에 속지 않도록 유의하세요."
                else: mfi_msg = f"MFI는 **{mfi}**로 자금의 유입과 유출이 **균형**을 이루고 있습니다."

                # 4. ATR (비율 기반 리스크 평가 - 핵심 수정 사항)
                atr_pct = data['ATR_Pct']
                if atr_pct >= 5.0:
                    atr_msg = f"ATR(변동성)은 {data['ATR']:.2f}로 주가의 **{atr_pct}%**에 달하는 **[고변동성 종목]**입니다. 하루에도 위아래 흔들림이 커 타이트한 손절매 등 철저한 리스크 관리가 필수적입니다."
                elif atr_pct >= 2.0:
                    atr_msg = f"ATR(변동성)은 {data['ATR']:.2f}로 주가의 **{atr_pct}%** 수준입니다. 일반적인 주식의 **[정상 변동폭]** 내에서 무난하게 움직이고 있습니다."
                else:
                    atr_msg = f"ATR(변동성)은 {data['ATR']:.2f}로 주가의 **{atr_pct}%** 불과한 **[저변동성/방어주]** 성향을 보입니다. 움직임이 무거워 단기 트레이딩보다는 중장기 관점이 어울립니다."

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
