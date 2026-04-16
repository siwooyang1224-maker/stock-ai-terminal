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
        border-left: 5px solid #00529B;
        margin-bottom: 15px;
    }
    .decision-label { font-size: 13px; font-weight: 700; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.5px; }
    .decision-value { font-size: 26px; font-weight: 900; margin: 10px 0; display: flex; align-items: baseline; gap: 8px; }
    .decision-prob { font-size: 15px; font-weight: 600; padding: 4px 10px; border-radius: 6px; }
    .data-table { width: 100%; font-size: 13.5px; margin-top: 15px; border-collapse: collapse; }
    .data-table td { padding: 10px 0; border-bottom: 1px solid #F1F3F5; }
    .streamlit-expanderHeader { font-weight: 700 !important; color: #1C1C1E !important; }
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

# --- 매크로 데이터 패치 함수 ---
@st.cache_data(ttl=600)
def get_macro_data():
    try:
        tickers = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB", "BTC": "BTC-USD"}
        data = {}
        for name, tk in tickers.items():
            hist = yf.Ticker(tk).history(period="5d")['Close']
            if not hist.empty and len(hist) >= 2:
                data[name] = {"val": hist.iloc[-1], "diff": hist.iloc[-1] - hist.iloc[-2]}
        return data
    except: return None

# --- 4. 대시보드 메인 ---
st.markdown("<h2 style='text-align: left; color: #1C1C1E; font-weight: 900; letter-spacing: -1px;'>ALPHA TERMINAL <span style='color:#00529B;'>QUANT-INSIGHT</span></h2>", unsafe_allow_html=True)

if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "TSLL": "TSLL"}

tab1, tab2, tab3 = st.tabs(["[1] ASSET STRATEGY", "[2] UNIVERSE SCREENING", "[3] MACRO & RESEARCH"])

with tab1:
    col_reg1, col_reg2, col_reg3 = st.columns([2, 2, 1])
    n_name = col_reg1.text_input("Asset Name", placeholder="ex) 삼성전자")
    n_ticker = col_reg2.text_input("Ticker Symbol", placeholder="ex) 005930.KS")
    if col_reg3.button("Register Asset"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()

    st.markdown("---")
    
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_quant(tk)
        if data:
            with p_cols[i % 2]:
                st.markdown(f"""
                <div class="ib-card">
                    <div class="decision-label">{name} ({tk}) / Intelligence Report</div>
                    <div class="decision-value" style="color: {data['Color']};">
                        {data['Verdict']}
                        <span class="decision-prob" style="background-color:{data['Conf_Bg']}; color:{data['Color']}; border: 1px solid {data['Color']}40;">{data['Conf_Str']}</span>
                    </div>
                    <table class="data-table">
                        <tr><td style="color:#8E8E93;">Current Price</td><td style="text-align:right; font-weight:700;">{data['Price']:,.2f}</td></tr>
                        <tr><td style="color:#8E8E93;">RSI (심리 강도)</td><td style="text-align:right;">{data['RSI']}</td></tr>
                        <tr><td style="color:#8E8E93;">MACD Momentum</td><td style="text-align:right;">{data['MACD_Status']}</td></tr>
                        <tr><td style="color:#8E8E93;">BB Pos (%)</td><td style="text-align:right;">{data['BB_Pos']}%</td></tr>
                        <tr style="border-top: 2px dashed #F1F3F5;"><td style="color:#8E8E93;">ADX (추세 강도)</td><td style="text-align:right; color:{'#D71920' if data['ADX'] > 25 else '#1C1C1E'};">{data['ADX']}</td></tr>
                        <tr><td style="color:#8E8E93;">MFI (자금 유입)</td><td style="text-align:right;">{data['MFI']}</td></tr>
                        <tr><td style="color:#8E8E93;">ATR (변동성 비율)</td><td style="text-align:right; font-weight:600;">{data['ATR_Pct']}% ({data['ATR']:.2f})</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Chart
                df_c = data['df'][-120:]
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price"), row=1, col=1)
                fig.update_layout(height=700, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"Close Asset {name}", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

with tab2:
    st.markdown("### Global Universe Screening (100 Assets)")
    def get_screen_data(stocks):
        res = []
        for t, n in stocks.items():
            d = analyze_stock_quant(t)
            if d: res.append({"Asset": n, "Ticker": t, "Score": d['Score'], "Verdict": d['Verdict'], "RSI": d['RSI'], "ATR(%)": d['ATR_Pct']})
        return pd.DataFrame(res)
    c1, c2 = st.columns(2)
    cfg = {"Score": st.column_config.ProgressColumn("Quant Score", min_value=0, max_value=100, format="%d")}
    with c1: st.dataframe(get_screen_data(KR_STOCKS), use_container_width=True, hide_index=True, column_config=cfg)
    with c2: st.dataframe(get_screen_data(US_STOCKS), use_container_width=True, hide_index=True, column_config=cfg)

with tab3:
    st.markdown("### 🏛️ Macro & Qualitative Analysis")
    macro = get_macro_data()
    if macro:
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        for i, (k, v) in enumerate(macro.items()):
            cols = [c_m1, c_m2, c_m3, c_m4]
            with cols[i]:
                st.markdown(f"""<div class="macro-card"><div style="font-size:12px;color:#8E8E93;">{k}</div><div style="font-size:22px;font-weight:900;">{v['val']:,.2f}</div><div style="color:{'red' if v['diff']>0 else 'green'};font-size:13px;">{'+' if v['diff']>0 else ''}{v['diff']:,.2f}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("#### 📰 Portfolio News Feed")
    n_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        with n_cols[i % 2]:
            with st.expander(f"{name} 최신 뉴스", expanded=True):
                try:
                    news = yf.Ticker(tk).news
                    for n in news[:3]:
                        st.markdown(f"""<div style="margin-bottom:10px;"><a href="{n['link']}" target="_blank" style="text-decoration:none;color:#00529B;font-weight:600;font-size:13px;">{n['title']}</a><br><span style="font-size:11px;color:#8E8E93;">{n.get('publisher','Yahoo')}</span></div>""", unsafe_allow_html=True)
                except: st.write("뉴스를 불러올 수 없습니다.")

    st.markdown("---")
    st.markdown("#### 🤖 Qualitative Prompt Generator (Copy & Paste)")
    port_str = ", ".join([f"{k}({v})" for k, v in st.session_state.my_portfolio.items()])
    m_str = ", ".join([f"{k}: {v['val']:.2f}" for k, v in macro.items()]) if macro else ""
    prompt_text = f"""당신은 월스트리트 시니어 매크로 전략가입니다. 아래 데이터를 기반으로 정성 분석 리포트를 작성하세요.
[상태] {m_str} / [포트폴리오] {port_str}
[요청] 1. 거시경제 변화가 성장주 밸류에이션에 미치는 영향 2. 지정학적 리스크와 반도체 공급망 분석 3. 기업 리더 동향 및 핵심 모멘텀 분석. 5줄 핵심 요약을 포함할 것."""
    st.code(prompt_text, language="markdown")
