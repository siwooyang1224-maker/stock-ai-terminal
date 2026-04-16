import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
import google.generativeai as genai

# --- 0. Gemini AI 보안 설정 ---
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None

# --- 1. 페이지 설정 및 Apple iOS 스타일 CSS ---
st.set_page_config(page_title="Alpha Terminal iOS", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {font-family: '-apple-system', 'BlinkMacSystemFont', 'Pretendard', sans-serif !important;}
    .stApp {background-color: #F2F2F7;}
    .block-container {padding-top: 3rem; max-width: 1400px;}
    #MainMenu, footer, header {visibility: hidden;}

    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        border: none;
    }
    div[data-testid="stMetricValue"] {font-size: 2.2rem; font-weight: 700; color: #1C1C1E; letter-spacing: -0.5px;}
    div[data-testid="stMetricLabel"] {font-size: 1rem; color: #8E8E93; font-weight: 500;}

    .stTabs [data-baseweb="tab-list"] {background-color: #E5E5EA; border-radius: 12px; padding: 4px; gap: 2px; border-bottom: none;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 8px 16px; color: #8E8E93; font-weight: 600; border: none; background-color: transparent;}
    .stTabs [aria-selected="true"] {background-color: #FFFFFF !important; color: #000000 !important; box-shadow: 0 3px 8px rgba(0,0,0,0.12), 0 3px 1px rgba(0,0,0,0.04);}
    
    .stDataFrame {background-color: #FFFFFF; border-radius: 20px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.03);}
    .stButton>button {background-color: #007AFF; color: white; border-radius: 14px; border: none; font-weight: 600; padding: 10px;}
    .stButton>button:hover {background-color: #0056b3; color: white;}
    .stTextInput input {border-radius: 12px; border: 1px solid #E5E5EA; padding: 12px;}
</style>
""", unsafe_allow_html=True)

# --- 2. 직관적인 AI 해설 생성 함수 ---
def get_easy_explanation(rsi, macd_trend, bb_val, verdict):
    rsi_text = "사람들이 공포에 질려 다 팔아버렸어요! (바닥권 진입 가능성)" if rsi <= 35 else "너도나도 사겠다고 몰려들어 거품이 조금 꼈습니다." if rsi >= 70 else "사는 사람과 파는 사람이 팽팽하게 눈치를 보고 있습니다."
    macd_text = "주가가 위로 올라가려는 '순풍'을 탔습니다." if macd_trend == "상승" else "지금은 주가가 아래로 밀리는 '역풍'이 불고 있네요."
    bb_text = "평소 놀던 가격대의 맨 밑바닥까지 떨어졌어요. 튀어 오를 자리를 찾고 있습니다." if bb_val <= 10 else "평소 가격대의 지붕을 뚫고 나갔습니다. 다시 내려올 확률이 높아요." if bb_val >= 90 else "평소 움직이는 정상적인 가격대 안에서 얌전히 움직이고 있습니다."
    
    html = f"""
    <div style="background-color: #F8F8F9; border-radius: 18px; padding: 20px; margin-top: 8px; margin-bottom: 24px; border: 1px solid #E5E5EA;">
        <div style="font-size: 15px; color: #1C1C1E; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center;">
            <span style="font-size: 18px; margin-right: 6px;">🤖</span> AI의 쉬운 차트 해설
        </div>
        <div style="font-size: 13.5px; color: #3A3A3C; line-height: 1.7;">
            <b>🔥 시장 심리:</b> {rsi_text}<br>
            <b>💨 주가 바람:</b> {macd_text}<br>
            <b>📏 가격 위치:</b> {bb_text}
        </div>
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #E5E5EA; color: #007AFF; font-weight: 700; font-size: 14px;">
            💡 종합 결론 ➔ {verdict}
        </div>
    </div>
    """
    return html

# --- 3. 유니버스 데이터 ---
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

# --- 4. 정밀 분석 엔진 ---
@st.cache_data(ttl=3600)
def analyze_stock_detailed(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        
        bb = ta.volatility.BollingerBands(df['Close'])
        h, l = bb.bollinger_hband(), bb.bollinger_lband()
        rsi = ta.momentum.rsi(df['Close']).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        m, ms = macd.macd().iloc[-1], macd.macd_signal().iloc[-1]
        
        curr_price = df['Close'].iloc[-1]
        bb_pos = (curr_price - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) * 100
        macd_trend = "상승" if m > ms else "하락"
        
        score = 50.0
        if m > ms: score += 15
        else: score -= 10
        if rsi <= 35: score += 20
        elif rsi >= 70: score -= 25
        if bb_pos <= 10: score += 15
        elif bb_pos >= 90: score -= 15
        
        final_score = int(max(0, min(100, score)))
        verdict = "🚀 적극 매수" if final_score >= 80 else "✅ 분할 매수" if final_score >= 60 else "🆘 위험/매도" if final_score <= 35 else "🟡 관망"
        
        return {
            "Ticker": ticker, "Price": curr_price, "RSI": round(rsi, 1),
            "MACD": macd_trend, "BB_Pos": f"{bb_pos:.1f}%", "BB_Pos_Val": bb_pos, 
            "Score": final_score, "Verdict": verdict, "df": df
        }
    except: return None

# --- 5. 메인 화면 ---
if 'my_portfolio' not in st.session_state:
    st.session_state.my_portfolio = {"SK하이닉스": "000660.KS", "NVIDIA": "NVDA"}

st.markdown("<h1 style='text-align: center; color: #1C1C1E; font-weight: 800; margin-bottom: 2rem;'> Alpha Terminal</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["포트폴리오", "시장 스크리닝", "AI 브리핑"])

# [탭 1: 포트폴리오 관리]
with tab1:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>나의 종목 관리</h3>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 2, 1])
    n_name = col_a.text_input("종목명 입력", placeholder="예: 삼성전자")
    n_ticker = col_b.text_input("티커 입력", placeholder="예: 005930.KS")
    st.markdown("""<style>div.stButton {margin-top: 28px;}</style>""", unsafe_allow_html=True)
    if col_c.button("추가하기"):
        if n_name and n_ticker:
            st.session_state.my_portfolio[n_name] = n_ticker
            st.rerun()
    
    st.write(" ")
    p_cols = st.columns(2)
    for i, (name, tk) in enumerate(st.session_state.my_portfolio.items()):
        data = analyze_stock_detailed(tk)
        if data:
            with p_cols[i % 2]:
                st.metric(f"{name}", f"{data['Price']:,.1f}", f"AI 점수: {data['Score']}점")
                
                if st.button(f"목록에서 삭제", key=f"del_{tk}"):
                    del st.session_state.my_portfolio[name]
                    st.rerun()

                # 직관적 해설 카드 렌더링
                explanation_html = get_easy_explanation(data['RSI'], data['MACD'], data['BB_Pos_Val'], data['Verdict'])
                st.markdown(explanation_html, unsafe_allow_html=True)
                
                df_chart = data['df'][-100:]
                bb_chart = ta.volatility.BollingerBands(df_chart['Close'])
                bb_h, bb_l = bb_chart.bollinger_hband(), bb_chart.bollinger_lband()
                rsi_chart = ta.momentum.rsi(df_chart['Close'])

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb_h, line=dict(color='rgba(150,150,150,0.5)', width=1), name="BB High"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=bb_l, line=dict(color='rgba(150,150,150,0.5)', width=1), fill='tonexty', name="BB Low"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=rsi_chart, line=dict(color='#007AFF', width=2), name="RSI"), row=2, col=1)
                
                fig.update_layout(
                    height=300, margin=dict(l=20, r=20, t=10, b=10),
                    plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_rangeslider_visible=False, showlegend=False
                )
                for row in [1, 2]:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)', row=row, col=1)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)', row=row, col=1)

                st.plotly_chart(fig, use_container_width=True)

# [탭 2: 시장 스크리닝 복구]
with tab2:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>한·미 시장 유니버스 분석</h3>", unsafe_allow_html=True)
    c_kr, c_us = st.columns(2)
    
    def get_df(stock_dict):
        results = []
        for tk, name in stock_dict.items():
            res = analyze_stock_detailed(tk)
            if res:
                results.append({
                    "기업명": name, "AI 점수": res['Score'], "판단": res['Verdict'],
                    "RSI": res['RSI'], "MACD": res['MACD'], "BB": res['BB_Pos']
                })
        return pd.DataFrame(results)

    with c_kr:
        st.markdown("<h5 style='color: #1C1C1E;'>🇰🇷 KOSPI & KOSDAQ 50</h5>", unsafe_allow_html=True)
        st.dataframe(get_df(KR_STOCKS), use_container_width=True, hide_index=True, column_config={"AI 점수": st.column_config.ProgressColumn(min_value=0, max_value=100)})

    with c_us:
        st.markdown("<h5 style='color: #1C1C1E;'>🇺🇸 S&P 500 & NASDAQ 50</h5>", unsafe_allow_html=True)
        st.dataframe(get_df(US_STOCKS), use_container_width=True, hide_index=True, column_config={"AI 점수": st.column_config.ProgressColumn(min_value=0, max_value=100)})

# [탭 3: AI 브리핑 복구]
with tab3:
    st.markdown("<h3 style='color: #1C1C1E; font-weight: 700;'>Gemini 데일리 브리핑</h3>", unsafe_allow_html=True)
    if gemini_model:
        market_news = "최근 기술주 전반에 조정이 오고 있으며, 반도체 섹터의 변동성이 큽니다. 금리 인하 기대감은 다소 후퇴했습니다."
        try:
            res = gemini_model.generate_content(f"투자 전략가로서 다음 시장 상황을 분석하고 비전공자도 이해할 수 있게 3줄로 요약해줘: {market_news}")
            st.success(res.text)
        except Exception as e:
            st.error(f"AI 브리핑 생성 중 오류가 발생했습니다: {e}")
    else:
        st.info("Streamlit Cloud 설정(Secrets)에서 GEMINI_API_KEY를 추가해주세요.")
