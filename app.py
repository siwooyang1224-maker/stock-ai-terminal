import os
import json
import math
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from textwrap import dedent
from html import escape

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    from supabase import create_client
except Exception:
    create_client = None

try:
    from google import genai
except Exception:
    genai = None


# =========================================================
# 0. PAGE / LOGGING
# =========================================================
st.set_page_config(
    page_title="Alpha Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)


# =========================================================
# 1. DEFAULT PORTFOLIO / STORAGE CONFIG
# =========================================================
DB_PATH = Path("data") / "portfolio.db"

DEFAULT_PORTFOLIO = {
    "SK하이닉스": {
        "ticker": "000660.KS",
        "avg_price": 0.0,
        "quantity": 0.0,
        "target_weight": 25.0,
        "max_loss_pct": 8.0,
        "memo": "AI 반도체 / 메모리 사이클"
    },
    "TSLL": {
        "ticker": "TSLL",
        "avg_price": 15.0764,
        "quantity": 0.0,
        "target_weight": 15.0,
        "max_loss_pct": 10.0,
        "memo": "TSLA 2배 레버리지 ETF"
    }
}


# =========================================================
# 2. KR / US 100 STOCK UNIVERSE
#    - 실시간 시총 순위 API가 아니라, yfinance 분석용 대형주/고유동성 유니버스입니다.
# =========================================================
KR_STOCKS = {
    '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '373220.KS': 'LG에너지솔루션', '207940.KS': '삼성바이오로직스',
    '005380.KS': '현대차', '000270.KS': '기아', '105560.KS': 'KB금융', '068270.KS': '셀트리온',
    '005490.KS': 'POSCO홀딩스', '035420.KS': 'NAVER', '055550.KS': '신한지주', '006400.KS': '삼성SDI',
    '051910.KS': 'LG화학', '028260.KS': '삼성물산', '012330.KS': '현대모비스', '138040.KS': '메리츠금융지주',
    '012450.KS': '한화에어로스페이스', '034020.KS': '두산에너빌리티', '329180.KS': 'HD현대중공업', '011200.KS': 'HMM',
    '003670.KS': '포스코퓨처엠', '066570.KS': 'LG전자', '035720.KS': '카카오', '032830.KS': '삼성생명',
    '086790.KS': '하나금융지주', '017670.KS': 'SK텔레콤', '033780.KS': 'KT&G', '015760.KS': '한국전력',
    '000810.KS': '삼성화재', '004020.KS': '현대제철', '267260.KS': 'HD현대일렉트릭', '096770.KS': 'SK이노베이션',
    '259960.KS': '크래프톤', '003550.KS': 'LG', '010130.KS': '고려아연', '086280.KS': '현대글로비스',
    '090430.KS': '아모레퍼시픽', '051900.KS': 'LG생활건강', '010950.KS': 'S-Oil', '316140.KS': '우리금융지주',
    '005830.KS': 'DB손해보험', '009150.KS': '삼성전기', '009540.KS': 'HD한국조선해양', '011170.KS': '롯데케미칼',
    '018260.KS': '삼성에스디에스', '042660.KS': '한화오션', '064350.KS': '현대로템', '097950.KS': 'CJ제일제당',
    '402340.KS': 'SK스퀘어', '021240.KS': '코웨이', '000100.KS': '유한양행', '271560.KS': '오리온',
    '036570.KS': '엔씨소프트', '323410.KS': '카카오뱅크', '377300.KS': '카카오페이', '251270.KS': '넷마블',
    '352820.KS': '하이브', '011070.KS': 'LG이노텍', '034730.KS': 'SK', '011790.KS': 'SKC',
    '010120.KS': 'LS ELECTRIC', '006260.KS': 'LS', '079550.KS': 'LIG넥스원', '000720.KS': '현대건설',
    '078930.KS': 'GS', '139480.KS': '이마트', '008770.KS': '호텔신라', '030000.KS': '제일기획',
    '128940.KS': '한미약품', '006280.KS': '녹십자', '010140.KS': '삼성중공업', '028670.KS': '팬오션',
    '003490.KS': '대한항공', '009830.KS': '한화솔루션', '454910.KS': '두산로보틱스', '000150.KS': '두산',
    '047050.KS': '포스코인터내셔널', '001570.KS': '금양', '011780.KS': '금호석유', '010060.KS': 'OCI홀딩스',
    '086520.KQ': '에코프로', '247540.KQ': '에코프로비엠', '196170.KQ': '알테오젠', '403870.KQ': 'HPSP',
    '035900.KQ': 'JYP Ent.', '041510.KQ': '에스엠', '066970.KS': '엘앤에프', '068760.KQ': '셀트리온제약',
    '141080.KQ': '리가켐바이오', '000250.KQ': '삼천당제약', '263750.KQ': '펄어비스', '293490.KQ': '카카오게임즈',
    '237690.KQ': '에스티팜', '214150.KQ': '클래시스', '277810.KQ': '레인보우로보틱스', '058470.KQ': '리노공업',
    '357780.KQ': '솔브레인', '095340.KQ': 'ISC', '240810.KQ': '원익IPS', '005290.KQ': '동진쎄미켐'
}

US_STOCKS = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'GOOGL': 'Alphabet Class A', 'GOOG': 'Alphabet Class C',
    'AMZN': 'Amazon', 'META': 'Meta', 'TSLA': 'Tesla', 'AVGO': 'Broadcom', 'BRK-B': 'Berkshire Hathaway',
    'LLY': 'Eli Lilly', 'JPM': 'JPMorgan Chase', 'V': 'Visa', 'UNH': 'UnitedHealth', 'XOM': 'Exxon Mobil',
    'MA': 'Mastercard', 'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart', 'PG': 'Procter & Gamble', 'HD': 'Home Depot',
    'COST': 'Costco', 'ORCL': 'Oracle', 'ABBV': 'AbbVie', 'BAC': 'Bank of America', 'KO': 'Coca-Cola',
    'NFLX': 'Netflix', 'MRK': 'Merck', 'CVX': 'Chevron', 'ADBE': 'Adobe', 'CRM': 'Salesforce',
    'AMD': 'AMD', 'PEP': 'PepsiCo', 'TMO': 'Thermo Fisher', 'LIN': 'Linde', 'ACN': 'Accenture',
    'MCD': "McDonald's", 'CSCO': 'Cisco', 'ABT': 'Abbott', 'WFC': 'Wells Fargo', 'DHR': 'Danaher',
    'QCOM': 'Qualcomm', 'INTU': 'Intuit', 'GE': 'GE Aerospace', 'TXN': 'Texas Instruments', 'AMAT': 'Applied Materials',
    'IBM': 'IBM', 'VZ': 'Verizon', 'PM': 'Philip Morris', 'NOW': 'ServiceNow', 'CAT': 'Caterpillar',
    'ISRG': 'Intuitive Surgical', 'DIS': 'Disney', 'NEE': 'NextEra Energy', 'RTX': 'RTX', 'UBER': 'Uber',
    'GS': 'Goldman Sachs', 'PFE': 'Pfizer', 'SPGI': 'S&P Global', 'LOW': "Lowe's", 'T': 'AT&T',
    'UNP': 'Union Pacific', 'PGR': 'Progressive', 'HON': 'Honeywell', 'BLK': 'BlackRock', 'BKNG': 'Booking Holdings',
    'ETN': 'Eaton', 'SYK': 'Stryker', 'TJX': 'TJX', 'BSX': 'Boston Scientific', 'C': 'Citigroup',
    'VRTX': 'Vertex', 'AMGN': 'Amgen', 'PANW': 'Palo Alto Networks', 'ADP': 'ADP', 'MDT': 'Medtronic',
    'COP': 'ConocoPhillips', 'LMT': 'Lockheed Martin', 'SCHW': 'Charles Schwab', 'CB': 'Chubb', 'MU': 'Micron',
    'ADI': 'Analog Devices', 'GILD': 'Gilead Sciences', 'MMC': 'Marsh & McLennan', 'PLD': 'Prologis', 'DE': 'Deere',
    'SBUX': 'Starbucks', 'LRCX': 'Lam Research', 'ELV': 'Elevance Health', 'BMY': 'Bristol Myers Squibb',
    'AMT': 'American Tower', 'SO': 'Southern Company', 'MO': 'Altria', 'CI': 'Cigna', 'DUK': 'Duke Energy',
    'KLAC': 'KLA', 'ANET': 'Arista Networks', 'MDLZ': 'Mondelez', 'ICE': 'Intercontinental Exchange',
    'SHW': 'Sherwin-Williams', 'ZTS': 'Zoetis'
}

LEVERAGED_ETFS = {
    "TSLL", "TSLQ", "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "LABU", "LABD", "NVDL", "NVDU", "NVDQ"
}


# =========================================================
# 3. CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: -apple-system, BlinkMacSystemFont, Pretendard, sans-serif !important; }
    .stApp { background-color: #F7F8FA; }
    .main-title { font-size: 31px; font-weight: 900; letter-spacing: -1px; color: #111827; margin-bottom: 0px; }
    .sub-title { font-size: 14px; color: #6B7280; margin-bottom: 24px; }
    .ib-card { background-color: white; border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(17,24,39,0.06); border: 1px solid #E5E7EB; margin-bottom: 18px; }
    .mini-card { background-color: white; border-radius: 10px; padding: 18px; border: 1px solid #E5E7EB; box-shadow: 0 1px 4px rgba(17,24,39,0.04); margin-bottom: 12px; }
    .label { font-size: 12px; font-weight: 800; color: #6B7280; letter-spacing: 0.4px; text-transform: uppercase; }
    .big-value { font-size: 27px; font-weight: 900; margin-top: 5px; margin-bottom: 5px; }
    .tag { display: inline-block; padding: 4px 9px; border-radius: 999px; font-size: 12px; font-weight: 800; margin-right: 4px; margin-top: 5px; }
    .tag-green { background-color: #E8F5E9; color: #137333; }
    .tag-red { background-color: #FCE8E6; color: #C5221F; }
    .tag-gray { background-color: #F3F4F6; color: #374151; }
    .tag-orange { background-color: #FFF4E5; color: #B45309; }
    .tag-blue { background-color: #E8F0FE; color: #1A73E8; }
    .explain-box { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px; padding: 15px; line-height: 1.7; font-size: 13.5px; color: #374151; margin-top: 14px; }
    .risk-box { background-color: #FFF7ED; border: 1px solid #FED7AA; border-radius: 10px; padding: 14px; line-height: 1.65; font-size: 13.5px; color: #7C2D12; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. GENERAL UTILITIES
# =========================================================
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


def clamp(value, low=0, high=100):
    try:
        if pd.isna(value):
            return 50
        return int(max(low, min(high, value)))
    except Exception:
        return 50


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def is_korea_ticker(ticker):
    ticker = str(ticker).upper()
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def is_leveraged(ticker):
    base = str(ticker).upper().replace(".KS", "").replace(".KQ", "")
    return base in LEVERAGED_ETFS


def format_price(ticker, value):
    value = safe_float(value)
    if is_korea_ticker(ticker):
        return f"₩{value:,.0f}"
    return f"${value:,.2f}"


def format_pct(value):
    return f"{safe_float(value):.2f}%"


def score_color(score):
    score = safe_float(score)
    if score >= 75: return "#137333"
    if score >= 60: return "#4D7C0F"
    if score >= 45: return "#6B7280"
    if score >= 35: return "#B45309"
    return "#C5221F"


def action_style(action):
    if action in ["ADD", "ACCUMULATE"]:
        return "#137333", "#E8F5E9"
    if action in ["HOLD", "WATCH"]:
        return "#374151", "#F3F4F6"
    if action == "TRIM":
        return "#B45309", "#FFF4E5"
    return "#C5221F", "#FCE8E6"


def benchmark_for(ticker):
    ticker = str(ticker).upper()
    if ticker.endswith(".KQ"):
        return "^KQ11"
    if ticker.endswith(".KS"):
        return "^KS11"
    if ticker == "TSLL":
        return "TSLA"
    if ticker in ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "AMAT", "LRCX", "KLAC", "MU", "SOXL", "SOXS", "NVDL"]:
        return "SOXX"
    if ticker in ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NFLX", "ADBE", "CRM", "NOW"]:
        return "QQQ"
    return "SPY"


# =========================================================
# 5. PERSISTENCE: SQLITE LOCAL + SUPABASE CLOUD
# =========================================================
def normalize_portfolio(raw):
    if not isinstance(raw, dict):
        return DEFAULT_PORTFOLIO.copy()

    normalized = {}
    for name, item in raw.items():
        if isinstance(item, str):
            normalized[name] = {
                "ticker": item.upper(), "avg_price": 0.0, "quantity": 0.0,
                "target_weight": 10.0, "max_loss_pct": 8.0, "memo": ""
            }
        elif isinstance(item, dict):
            ticker = str(item.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            normalized[str(name).strip()] = {
                "ticker": ticker,
                "avg_price": safe_float(item.get("avg_price", 0.0)),
                "quantity": safe_float(item.get("quantity", 0.0)),
                "target_weight": safe_float(item.get("target_weight", 10.0)),
                "max_loss_pct": safe_float(item.get("max_loss_pct", 8.0)),
                "memo": str(item.get("memo", "") or "")
            }
    return normalized if normalized else DEFAULT_PORTFOLIO.copy()


def portfolio_hash(portfolio):
    return json.dumps(normalize_portfolio(portfolio), ensure_ascii=False, sort_keys=True)


def use_supabase():
    return (
        create_client is not None
        and get_secret("SUPABASE_URL") is not None
        and get_secret("SUPABASE_KEY") is not None
    )


@st.cache_resource
def get_supabase_client():
    return create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY"))


def init_sqlite():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            name TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            avg_price REAL DEFAULT 0,
            quantity REAL DEFAULT 0,
            target_weight REAL DEFAULT 10,
            max_loss_pct REAL DEFAULT 8,
            memo TEXT DEFAULT '',
            updated_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        conn.commit()


def sqlite_is_initialized():
    init_sqlite()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT value FROM portfolio_meta WHERE key = 'initialized'").fetchone()
    return row is not None and row[0] == "1"


def save_portfolio_sqlite(portfolio):
    init_sqlite()
    portfolio = normalize_portfolio(portfolio)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM portfolio")
        for name, item in portfolio.items():
            conn.execute("""
            INSERT INTO portfolio (name, ticker, avg_price, quantity, target_weight, max_loss_pct, memo, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, item["ticker"], item["avg_price"], item["quantity"],
                item["target_weight"], item["max_loss_pct"], item["memo"],
                datetime.now().isoformat(timespec="seconds")
            ))
        conn.execute("INSERT OR REPLACE INTO portfolio_meta (key, value) VALUES ('initialized', '1')")
        conn.commit()


def load_portfolio_sqlite():
    init_sqlite()
    if not sqlite_is_initialized():
        save_portfolio_sqlite(DEFAULT_PORTFOLIO)
        return DEFAULT_PORTFOLIO.copy()

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
        SELECT name, ticker, avg_price, quantity, target_weight, max_loss_pct, memo
        FROM portfolio
        ORDER BY name
        """).fetchall()

    portfolio = {}
    for name, ticker, avg_price, quantity, target_weight, max_loss_pct, memo in rows:
        portfolio[name] = {
            "ticker": ticker,
            "avg_price": safe_float(avg_price),
            "quantity": safe_float(quantity),
            "target_weight": safe_float(target_weight, 10.0),
            "max_loss_pct": safe_float(max_loss_pct, 8.0),
            "memo": memo or ""
        }
    return portfolio if portfolio else DEFAULT_PORTFOLIO.copy()


def save_portfolio_supabase(portfolio):
    client = get_supabase_client()
    user_id = get_secret("PORTFOLIO_USER_ID", "default_user")
    portfolio = normalize_portfolio(portfolio)

    client.table("portfolio").delete().eq("user_id", user_id).execute()
    payload = []
    for name, item in portfolio.items():
        payload.append({
            "user_id": user_id,
            "name": name,
            "ticker": item["ticker"],
            "avg_price": item["avg_price"],
            "quantity": item["quantity"],
            "target_weight": item["target_weight"],
            "max_loss_pct": item["max_loss_pct"],
            "memo": item["memo"],
            "updated_at": datetime.now().isoformat(timespec="seconds")
        })
    if payload:
        client.table("portfolio").insert(payload).execute()


def load_portfolio_supabase():
    client = get_supabase_client()
    user_id = get_secret("PORTFOLIO_USER_ID", "default_user")
    response = client.table("portfolio").select("*").eq("user_id", user_id).execute()
    rows = response.data or []

    if not rows:
        save_portfolio_supabase(DEFAULT_PORTFOLIO)
        return DEFAULT_PORTFOLIO.copy()

    portfolio = {}
    for row in rows:
        portfolio[row["name"]] = {
            "ticker": row.get("ticker", ""),
            "avg_price": safe_float(row.get("avg_price", 0.0)),
            "quantity": safe_float(row.get("quantity", 0.0)),
            "target_weight": safe_float(row.get("target_weight", 10.0), 10.0),
            "max_loss_pct": safe_float(row.get("max_loss_pct", 8.0), 8.0),
            "memo": row.get("memo", "") or ""
        }
    return normalize_portfolio(portfolio)


def load_portfolio():
    try:
        if use_supabase():
            return load_portfolio_supabase()
        return load_portfolio_sqlite()
    except Exception as e:
        logging.exception(e)
        st.warning(f"포트폴리오 로딩 오류: {e}")
        return DEFAULT_PORTFOLIO.copy()


def save_portfolio(portfolio):
    try:
        if use_supabase():
            save_portfolio_supabase(portfolio)
        else:
            save_portfolio_sqlite(portfolio)
    except Exception as e:
        logging.exception(e)
        st.warning(f"포트폴리오 저장 오류: {e}")


def portfolio_to_df(portfolio):
    rows = []
    for name, item in normalize_portfolio(portfolio).items():
        rows.append({
            "Asset": name,
            "Ticker": item.get("ticker", ""),
            "Avg Price": item.get("avg_price", 0.0),
            "Quantity": item.get("quantity", 0.0),
            "Target Weight %": item.get("target_weight", 10.0),
            "Max Loss %": item.get("max_loss_pct", 8.0),
            "Memo": item.get("memo", "")
        })
    return pd.DataFrame(rows)


def df_to_portfolio(df):
    portfolio = {}
    if df is None or df.empty:
        return portfolio
    for _, row in df.iterrows():
        name = str(row.get("Asset", "")).strip()
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not name or not ticker or name.lower() == "nan" or ticker.lower() == "nan":
            continue
        portfolio[name] = {
            "ticker": ticker,
            "avg_price": safe_float(row.get("Avg Price", 0.0)),
            "quantity": safe_float(row.get("Quantity", 0.0)),
            "target_weight": safe_float(row.get("Target Weight %", 10.0), 10.0),
            "max_loss_pct": safe_float(row.get("Max Loss %", 8.0), 8.0),
            "memo": str(row.get("Memo", "") or "")
        }
    return normalize_portfolio(portfolio)


# =========================================================
# 6. DATA DOWNLOAD
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        return df.dropna(subset=required)
    except Exception as e:
        logging.exception(e)
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_macro_data():
    tickers = {
        "VIX": "^VIX", "US10Y": "^TNX", "DXY": "DX-Y.NYB", "QQQ": "QQQ",
        "SOXX": "SOXX", "HYG": "HYG", "USDKRW": "KRW=X", "WTI": "CL=F", "BTC": "BTC-USD"
    }
    data = {}
    for key, tk in tickers.items():
        try:
            hist = yf.Ticker(tk).history(period="3mo")["Close"].dropna()
            if len(hist) < 2:
                data[key] = {"ticker": tk, "value": np.nan, "diff": np.nan, "ret_20d": np.nan, "ma20": np.nan, "above_ma20": False}
                continue
            value = hist.iloc[-1]
            diff = hist.iloc[-1] - hist.iloc[-2]
            ret_20d = (hist.iloc[-1] / hist.iloc[-21] - 1) * 100 if len(hist) > 21 else np.nan
            ma20 = hist.rolling(20).mean().iloc[-1] if len(hist) >= 20 else np.nan
            if key == "US10Y" and value > 15:
                value = value / 10
                diff = diff / 10
            data[key] = {
                "ticker": tk,
                "value": float(value),
                "diff": float(diff),
                "ret_20d": float(ret_20d) if not pd.isna(ret_20d) else np.nan,
                "ma20": float(ma20) if not pd.isna(ma20) else np.nan,
                "above_ma20": bool(value > ma20) if not pd.isna(ma20) else False
            }
        except Exception as e:
            logging.exception(e)
            data[key] = {"ticker": tk, "value": np.nan, "diff": np.nan, "ret_20d": np.nan, "ma20": np.nan, "above_ma20": False}
    return data


# =========================================================
# 7. INDICATORS / SCORES
# =========================================================
def add_indicators(df):
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["Return"] = close.pct_change()
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA120"] = close.rolling(120).mean()
    df["MA200"] = close.rolling(200).mean()
    df["High20"] = high.rolling(20).max()
    df["Low20"] = low.rolling(20).min()
    df["High60"] = high.rolling(60).max()
    df["Low60"] = low.rolling(60).min()
    df["Volume_MA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_MA20"]

    df["RSI"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Pos"] = (close - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"]) * 100

    df["ADX"] = ta.trend.adx(high, low, close, window=14)
    df["DI_Plus"] = ta.trend.adx_pos(high, low, close, window=14)
    df["DI_Minus"] = ta.trend.adx_neg(high, low, close, window=14)

    df["MFI"] = ta.volume.money_flow_index(high, low, close, volume, window=14)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["ATR_Pct"] = df["ATR"] / close * 100
    df["Rolling_Max"] = close.cummax()
    df["Drawdown"] = close / df["Rolling_Max"] - 1
    df["Annual_Vol"] = df["Return"].rolling(60).std() * np.sqrt(252) * 100
    df["MDD_1Y"] = df["Drawdown"].rolling(252).min() * 100
    df["Ret_20D"] = close.pct_change(20) * 100
    df["Ret_60D"] = close.pct_change(60) * 100
    return df


def get_trend_score(row, prev):
    score = 50
    close = row["Close"]
    score += 7 if close > row["MA20"] else -5
    score += 9 if close > row["MA50"] else -7
    score += 7 if close > row["MA120"] else -5
    score += 12 if close > row["MA200"] else -14
    score += 8 if row["MA20"] > row["MA50"] else -6
    score += 8 if row["MA50"] > row["MA200"] else -6
    score += 9 if row["MACD"] > row["MACD_Signal"] else -9
    score += 4 if row["MACD_Hist"] > prev["MACD_Hist"] else -3
    if row["ADX"] > 25:
        score += 8 if row["DI_Plus"] > row["DI_Minus"] else -8
    return clamp(score)


def get_momentum_score(row, prev):
    score = 50
    rsi, bb_pos = row["RSI"], row["BB_Pos"]
    if 50 <= rsi <= 65: score += 14
    elif 65 < rsi <= 75: score += 5
    elif 40 <= rsi < 50: score += 1
    elif 30 <= rsi < 40: score -= 6
    elif rsi < 30: score += 3
    elif rsi > 75: score -= 12
    score += 7 if row["MACD_Hist"] > 0 else -7
    score += 6 if row["MACD_Hist"] > prev["MACD_Hist"] else -4
    if 20 <= bb_pos <= 80: score += 5
    elif 80 < bb_pos <= 95: score += 1
    elif bb_pos > 95: score -= 8
    elif 5 <= bb_pos < 20: score -= 2
    elif bb_pos < 5: score -= 6
    return clamp(score)


def get_volume_score(row, prev):
    score = 50
    price_up = row["Close"] > prev["Close"]
    vr = row["Volume_Ratio"]
    if vr > 1.5 and price_up: score += 14
    elif vr > 1.5 and not price_up: score -= 12
    elif vr > 1.1 and price_up: score += 6
    elif vr > 1.1 and not price_up: score -= 5
    if 50 <= row["MFI"] <= 70: score += 9
    elif 70 < row["MFI"] <= 85: score += 4
    elif row["MFI"] > 85: score -= 8
    elif 30 <= row["MFI"] < 50: score -= 4
    elif row["MFI"] < 30: score -= 12
    score += 8 if row["OBV"] > row["OBV_MA20"] else -8
    return clamp(score)


def get_risk_score(row, ticker):
    score = 85
    atr_pct, annual_vol, mdd = safe_float(row["ATR_Pct"]), safe_float(row["Annual_Vol"]), safe_float(row["MDD_1Y"])
    if atr_pct >= 10: score -= 40
    elif atr_pct >= 7: score -= 30
    elif atr_pct >= 5: score -= 22
    elif atr_pct >= 3: score -= 10
    elif atr_pct < 1.2: score += 4
    if annual_vol >= 100: score -= 30
    elif annual_vol >= 70: score -= 22
    elif annual_vol >= 50: score -= 14
    elif annual_vol >= 35: score -= 7
    if mdd <= -60: score -= 26
    elif mdd <= -45: score -= 20
    elif mdd <= -30: score -= 12
    elif mdd <= -20: score -= 6
    if is_leveraged(ticker): score -= 15
    return clamp(score)


def get_relative_strength_score(stock_df, benchmark_df):
    if stock_df.empty or benchmark_df.empty:
        return 50, {"bench_ret_20d": np.nan, "bench_ret_60d": np.nan, "rs_20d": np.nan, "rs_60d": np.nan}
    merged = pd.concat([stock_df["Close"].rename("stock"), benchmark_df["Close"].rename("bench")], axis=1).dropna()
    if len(merged) < 70:
        return 50, {"bench_ret_20d": np.nan, "bench_ret_60d": np.nan, "rs_20d": np.nan, "rs_60d": np.nan}
    stock_ret_20 = (merged["stock"].iloc[-1] / merged["stock"].iloc[-21] - 1) * 100
    bench_ret_20 = (merged["bench"].iloc[-1] / merged["bench"].iloc[-21] - 1) * 100
    stock_ret_60 = (merged["stock"].iloc[-1] / merged["stock"].iloc[-61] - 1) * 100
    bench_ret_60 = (merged["bench"].iloc[-1] / merged["bench"].iloc[-61] - 1) * 100
    rs_20, rs_60 = stock_ret_20 - bench_ret_20, stock_ret_60 - bench_ret_60
    return clamp(50 + rs_20 * 1.1 + rs_60 * 0.45), {
        "bench_ret_20d": bench_ret_20, "bench_ret_60d": bench_ret_60, "rs_20d": rs_20, "rs_60d": rs_60
    }


POSITIVE_KEYWORDS = [
    "beat", "beats", "upgrade", "upgraded", "surge", "rally", "record", "profit", "growth", "strong",
    "partnership", "approval", "buyback", "raises", "raised", "outperform", "bullish", "ai", "demand",
    "guidance", "contract", "expansion", "launch", "margin", "benefit", "win", "winner"
]
NEGATIVE_KEYWORDS = [
    "miss", "misses", "downgrade", "downgraded", "fall", "falls", "drop", "plunge", "loss", "weak",
    "lawsuit", "probe", "investigation", "recall", "delay", "bearish", "cut", "cuts", "risk", "warning",
    "slump", "fraud", "concern", "tariff", "ban", "restriction", "shortage", "margin pressure"
]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_items(ticker):
    try:
        news = yf.Ticker(ticker).news
        items = []
        if not isinstance(news, list):
            return items
        for n in news[:10]:
            content = n.get("content", {}) if isinstance(n, dict) else {}
            title = n.get("title") or content.get("title") or ""
            publisher = n.get("publisher") or content.get("provider", {}).get("displayName") or "Unknown"
            link = n.get("link") or content.get("clickThroughUrl", {}).get("url") or content.get("canonicalUrl", {}).get("url") or ""
            pub_time = n.get("providerPublishTime") or content.get("pubDate") or ""
            if title:
                items.append({"title": title, "publisher": publisher, "link": link, "pub_time": pub_time})
        return items
    except Exception as e:
        logging.exception(e)
        return []


def get_qualitative_score(news_items):
    if not news_items:
        return 50, {"positive_count": 0, "negative_count": 0, "neutral_count": 0, "summary": "최근 뉴스 데이터가 부족합니다. 정성 점수는 중립 처리했습니다."}
    pos, neg = 0, 0
    for item in news_items:
        title = item["title"].lower()
        pos_hit = any(k in title for k in POSITIVE_KEYWORDS)
        neg_hit = any(k in title for k in NEGATIVE_KEYWORDS)
        if pos_hit and not neg_hit: pos += 1
        elif neg_hit and not pos_hit: neg += 1
        elif pos_hit and neg_hit: neg += 1
    score = clamp(50 + pos * 6 - neg * 8)
    neutral = max(0, len(news_items) - pos - neg)
    if score >= 65:
        summary = "최근 뉴스 플로우는 대체로 긍정 이벤트가 우세합니다."
    elif score <= 40:
        summary = "최근 뉴스 플로우에는 부정 이벤트 또는 리스크 키워드가 많습니다."
    else:
        summary = "최근 뉴스 플로우는 방향성이 뚜렷하지 않아 중립에 가깝습니다."
    return score, {"positive_count": pos, "negative_count": neg, "neutral_count": neutral, "summary": summary}


def get_macro_score(macro):
    score, notes = 50, []
    vix = macro.get("VIX", {}).get("value", np.nan)
    tnx = macro.get("US10Y", {}).get("value", np.nan)
    tnx_diff = macro.get("US10Y", {}).get("diff", np.nan)
    dxy_ret = macro.get("DXY", {}).get("ret_20d", np.nan)
    qqq_ret = macro.get("QQQ", {}).get("ret_20d", np.nan)
    soxx_ret = macro.get("SOXX", {}).get("ret_20d", np.nan)
    hyg_above = macro.get("HYG", {}).get("above_ma20", False)
    usdkrw_ret = macro.get("USDKRW", {}).get("ret_20d", np.nan)

    if not pd.isna(vix):
        if vix < 16: score += 12; notes.append("VIX가 낮아 위험자산 선호 환경입니다.")
        elif vix < 22: score += 4; notes.append("VIX는 중립 구간입니다.")
        elif vix < 30: score -= 12; notes.append("VIX가 높아 변동성 리스크가 커졌습니다.")
        else: score -= 25; notes.append("VIX가 30 이상으로 리스크오프 경계가 필요합니다.")
    if not pd.isna(tnx):
        if tnx >= 4.8: score -= 14; notes.append("미국 10년물 금리가 높아 성장주 밸류에이션에는 부담입니다.")
        elif tnx >= 4.3: score -= 7; notes.append("미국 10년물 금리가 다소 높은 편입니다.")
        elif tnx < 3.8: score += 8; notes.append("금리 부담이 상대적으로 낮아 성장주에 우호적입니다.")
    if not pd.isna(tnx_diff):
        if tnx_diff > 0.12: score -= 6; notes.append("금리가 단기 상승 중입니다.")
        elif tnx_diff < -0.12: score += 5; notes.append("금리가 단기 하락 중입니다.")
    if not pd.isna(dxy_ret):
        if dxy_ret > 2: score -= 6; notes.append("달러 강세는 글로벌 위험자산과 환율 리스크에 부담입니다.")
        elif dxy_ret < -2: score += 4; notes.append("달러 약세는 위험자산에 우호적입니다.")
    if not pd.isna(qqq_ret):
        if qqq_ret > 3: score += 8; notes.append("QQQ가 최근 강세라 성장주 시장 체력이 좋습니다.")
        elif qqq_ret < -3: score -= 8; notes.append("QQQ가 약세라 성장주 진입에는 보수적 접근이 필요합니다.")
    if not pd.isna(soxx_ret):
        if soxx_ret > 4: score += 7; notes.append("반도체 섹터가 강세입니다.")
        elif soxx_ret < -4: score -= 7; notes.append("반도체 섹터가 약세입니다.")
    score += 4 if hyg_above else -4
    notes.append("하이일드 채권이 20일선 위에 있어 신용 리스크는 비교적 안정적입니다." if hyg_above else "하이일드 채권 흐름이 약해 신용 리스크를 점검해야 합니다.")
    if not pd.isna(usdkrw_ret):
        if usdkrw_ret > 2: score -= 4; notes.append("원/달러 환율 상승은 한국 투자자 관점에서 환율 변동 리스크를 키울 수 있습니다.")
        elif usdkrw_ret < -2: score += 3; notes.append("원/달러 환율 하락은 환율 부담을 낮춥니다.")
    score = clamp(score)
    regime = "RISK-ON" if score >= 65 else "RISK-OFF" if score <= 40 else "NEUTRAL"
    return score, regime, notes


def build_trade_plan(row, ticker, account_value, risk_per_trade_pct, target_weight, current_qty):
    price, atr = safe_float(row["Close"]), safe_float(row["ATR"])
    if price <= 0 or atr <= 0:
        return {"entry": price, "stop": np.nan, "target": np.nan, "risk_reward": np.nan, "risk_per_share": np.nan,
                "position_qty_by_risk": 0, "position_qty_by_weight": 0, "suggested_add_qty": 0,
                "risk_budget": account_value * risk_per_trade_pct / 100}

    support_20 = safe_float(row["Low20"], price - 2 * atr)
    support_60 = safe_float(row["Low60"], price - 3 * atr)
    resistance_20 = safe_float(row["High20"], price + 2 * atr)
    resistance_60 = safe_float(row["High60"], price + 3 * atr)

    stop_atr = price - 2.0 * atr
    stop_structure = min(support_20 * 0.985, support_60 * 0.995)
    stop = min(stop_atr, stop_structure)
    if (price - stop) / price * 100 > max(safe_float(row["ATR_Pct"]) * 2.8, 12):
        stop = stop_atr

    target = max(price + 2.5 * atr, resistance_20, resistance_60)
    risk_per_share = max(price - stop, 0.0001)
    reward_per_share = max(target - price, 0.0001)
    risk_reward = reward_per_share / risk_per_share
    risk_budget = account_value * risk_per_trade_pct / 100
    qty_by_risk = math.floor(risk_budget / risk_per_share)
    target_value = account_value * target_weight / 100
    current_value = current_qty * price
    remaining_value = max(target_value - current_value, 0)
    qty_by_weight = math.floor(remaining_value / price)
    suggested = max(0, min(qty_by_risk, qty_by_weight))
    return {"entry": price, "stop": stop, "target": target, "risk_reward": risk_reward, "risk_per_share": risk_per_share,
            "position_qty_by_risk": qty_by_risk, "position_qty_by_weight": qty_by_weight,
            "suggested_add_qty": suggested, "risk_budget": risk_budget}


def decide_action(total_score, risk_score, risk_reward, row, ticker, quantity):
    below_ma200 = bool(row["Close"] < row["MA200"]) if not pd.isna(row["MA200"]) else False
    if risk_score < 30:
        return "RISK-OFF", "변동성과 낙폭 리스크가 커서 신규 진입보다 리스크 관리가 우선입니다."
    if is_leveraged(ticker) and row["ATR_Pct"] >= 7 and total_score < 78:
        return "RISK-OFF", "레버리지 ETF이고 변동성이 높아 신호가 아주 강하지 않으면 비중 확대는 위험합니다."
    if total_score >= 78 and risk_reward >= 1.5 and not below_ma200:
        return "ADD", "추세, 상대강도, 손익비가 모두 양호하여 추가 매수 후보입니다."
    if total_score >= 68 and risk_reward >= 1.2:
        return "ACCUMULATE", "방향성은 우호적이나 한 번에 진입하기보다 분할 접근이 적절합니다."
    if total_score >= 55:
        return "HOLD", "핵심 신호는 중립 이상입니다. 기존 보유는 가능하나 신규 비중 확대는 신중해야 합니다."
    if total_score >= 45:
        return "WATCH", "방향성이 애매합니다. 지지선 접근 또는 거래량 동반 반전을 확인해야 합니다."
    if total_score >= 35:
        return ("TRIM", "기술적·상대강도 신호가 약해 일부 비중 축소를 검토할 구간입니다.") if quantity > 0 else ("AVOID", "신규 진입 매력도가 낮습니다.")
    return ("TRIM", "하락 신호가 우세합니다. 손절 기준과 비중 축소를 우선 검토해야 합니다.") if quantity > 0 else ("AVOID", "하락 신호가 강해 신규 진입을 피하는 것이 적절합니다.")


# =========================================================
# 8. MAIN ANALYSIS
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def analyze_stock(ticker, account_value, risk_per_trade_pct, target_weight, avg_price, quantity, include_news=True):
    df = fetch_price_data(ticker, period="2y")
    if df.empty or len(df) < 220:
        return None
    df = add_indicators(df).dropna(subset=["MA200", "RSI", "MACD", "ATR"])
    if len(df) < 2:
        return None
    row, prev = df.iloc[-1], df.iloc[-2]

    benchmark_ticker = benchmark_for(ticker)
    benchmark_df = fetch_price_data(benchmark_ticker, period="2y")

    trend_score = get_trend_score(row, prev)
    momentum_score = get_momentum_score(row, prev)
    volume_score = get_volume_score(row, prev)
    risk_score = get_risk_score(row, ticker)
    relative_score, rs_detail = get_relative_strength_score(df, benchmark_df)

    news_items = fetch_news_items(ticker) if include_news else []
    qualitative_score, qual_detail = get_qualitative_score(news_items) if include_news else (
        50, {"positive_count": 0, "negative_count": 0, "neutral_count": 0, "summary": "스크리닝 모드에서는 정성 점수를 중립 처리했습니다."}
    )

    macro = fetch_macro_data()
    macro_score, macro_regime, macro_notes = get_macro_score(macro)

    technical_score = trend_score * 0.40 + momentum_score * 0.30 + volume_score * 0.30
    total_score = clamp(technical_score * 0.35 + relative_score * 0.20 + risk_score * 0.20 + qualitative_score * 0.15 + macro_score * 0.10)

    trade_plan = build_trade_plan(row, ticker, account_value, risk_per_trade_pct, target_weight, quantity)
    action, action_reason = decide_action(total_score, risk_score, safe_float(trade_plan["risk_reward"]), row, ticker, quantity)

    price = float(row["Close"])
    market_value = price * quantity
    invested_value = avg_price * quantity if avg_price > 0 and quantity > 0 else 0
    pnl = market_value - invested_value if invested_value > 0 else 0
    pnl_pct = pnl / invested_value * 100 if invested_value > 0 else 0
    current_weight = market_value / account_value * 100 if account_value > 0 else 0

    return {
        "ticker": ticker, "benchmark": benchmark_ticker, "df": df, "row": row.to_dict(), "price": price,
        "trend_score": trend_score, "momentum_score": momentum_score, "volume_score": volume_score,
        "technical_score": clamp(technical_score), "relative_score": relative_score, "risk_score": risk_score,
        "qualitative_score": qualitative_score, "macro_score": macro_score, "macro_regime": macro_regime,
        "total_score": total_score, "action": action, "action_reason": action_reason,
        "trade_plan": trade_plan, "rs_detail": rs_detail, "news_items": news_items, "qual_detail": qual_detail,
        "macro_notes": macro_notes, "market_value": float(market_value), "invested_value": float(invested_value),
        "pnl": float(pnl), "pnl_pct": float(pnl_pct), "current_weight": float(current_weight), "is_leveraged": is_leveraged(ticker)
    }


# =========================================================
# 9. BACKTEST
# =========================================================
def calculate_point_score_for_backtest(df, i, ticker):
    row, prev = df.iloc[i], df.iloc[i - 1]
    needed = ["MA20", "MA50", "MA120", "MA200", "MACD", "MACD_Signal", "MACD_Hist", "RSI", "BB_Pos", "ADX", "DI_Plus", "DI_Minus", "MFI", "OBV_MA20", "ATR_Pct", "Annual_Vol", "MDD_1Y"]
    if any(pd.isna(row.get(c, np.nan)) for c in needed):
        return np.nan
    trend = get_trend_score(row, prev)
    momentum = get_momentum_score(row, prev)
    volume = get_volume_score(row, prev)
    risk = get_risk_score(row, ticker)
    technical = trend * 0.40 + momentum * 0.30 + volume * 0.30
    return clamp(technical * 0.75 + risk * 0.25)


@st.cache_data(ttl=300, show_spinner=False)
def backtest_technical_signal(ticker, horizon=20):
    df = fetch_price_data(ticker, period="3y")
    if df.empty or len(df) < 260:
        return None
    df = add_indicators(df).dropna()
    if len(df) < 220 + horizon:
        return None
    records = []
    for i in range(220, len(df) - horizon):
        score = calculate_point_score_for_backtest(df, i, ticker)
        if pd.isna(score):
            continue
        fwd_ret = (df["Close"].iloc[i + horizon] / df["Close"].iloc[i] - 1) * 100
        records.append({"date": df.index[i], "score": score, "fwd_ret": fwd_ret})
    bt = pd.DataFrame(records)
    if bt.empty:
        return None
    high_signal, low_signal = bt[bt["score"] >= 70], bt[bt["score"] <= 35]
    return {
        "sample_size": len(bt), "high_count": len(high_signal), "low_count": len(low_signal),
        "high_avg_ret": high_signal["fwd_ret"].mean() if len(high_signal) > 0 else np.nan,
        "high_hit_rate": (high_signal["fwd_ret"] > 0).mean() * 100 if len(high_signal) > 0 else np.nan,
        "low_avg_ret": low_signal["fwd_ret"].mean() if len(low_signal) > 0 else np.nan,
        "low_hit_rate_down": (low_signal["fwd_ret"] < 0).mean() * 100 if len(low_signal) > 0 else np.nan,
        "horizon": horizon
    }


# =========================================================
# 10. CHARTS / RENDERERS
# =========================================================
def make_stock_chart(result):
    df = result["df"].copy().tail(180)
    trade = result["trade_plan"]
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.42, 0.14, 0.16, 0.14, 0.14],
        subplot_titles=("Price / MA / Bollinger", "Volume", "MACD", "RSI", "ADX / MFI")
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    for col in ["MA20", "MA50", "MA200", "BB_High", "BB_Low"]:
        width = 1.4 if col == "MA200" else 1.0
        dash = "dot" if col.startswith("BB") else None
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=width, dash=dash)), row=1, col=1)
    if not pd.isna(trade["stop"]):
        fig.add_hline(y=trade["stop"], line_dash="dash", annotation_text="Stop", annotation_position="bottom right", row=1, col=1)
    if not pd.isna(trade["target"]):
        fig.add_hline(y=trade["target"], line_dash="dash", annotation_text="Target", annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.55), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(width=1.0)), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist", opacity=0.45), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(width=1.3)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(width=1.3)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MFI"], name="MFI", line=dict(width=1.3)), row=5, col=1)
    fig.add_hline(y=25, line_dash="dot", row=5, col=1)
    fig.update_layout(height=920, template="plotly_white", margin=dict(l=10, r=10, t=45, b=10), xaxis_rangeslider_visible=False, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig


def render_score(label, score):
    score = clamp(score)
    st.markdown(f"**{label}**")
    st.progress(score / 100)
    st.caption(f"{score}/100")


def render_macro_card(title, value, diff, unit="", inverse=False):
    if pd.isna(value):
        value_text, diff_text, color = "N/A", "", "#6B7280"
    else:
        value_text = f"{value:,.2f}{unit}"
        diff_text = f"{diff:+,.2f}{unit}" if not pd.isna(diff) else ""
        color = "#C5221F" if (diff > 0 and inverse) or (diff <= 0 and not inverse) else "#137333"
    st.markdown(f"""
    <div class="mini-card">
        <div class="label">{title}</div>
        <div class="big-value">{value_text}</div>
        <div style="font-weight:800; color:{color}; font-size:13px;">{diff_text}</div>
    </div>
    """, unsafe_allow_html=True)



def classify_score_band(score):
    """점수를 사용자가 바로 읽을 수 있는 해석 구간으로 변환합니다."""
    score = clamp(score)
    if score >= 75:
        return "강한 우호", "핵심 신호가 매우 좋습니다. 다만 과열 여부와 손익비를 함께 확인해야 합니다."
    if score >= 65:
        return "우호", "매수 후보로 볼 수 있지만 분할 진입과 손절 기준이 필요합니다."
    if score >= 55:
        return "중립 이상", "기존 보유는 가능하지만 신규 비중 확대는 신중한 구간입니다."
    if score >= 45:
        return "애매", "방향성이 뚜렷하지 않습니다. 가격·거래량 확인 후 판단하는 편이 좋습니다."
    if score >= 35:
        return "주의", "신호가 약합니다. 기존 보유자는 비중과 손절선을 먼저 점검해야 합니다."
    return "위험", "하락/리스크 신호가 강합니다. 신규 진입은 피하고 리스크 관리가 우선입니다."


def build_interpretation_memo(name, item, result):
    """
    각 종목 카드 아래에 표시할 '해석 메모'를 생성합니다.
    단순 점수 나열이 아니라, 왜 최종 액션이 나왔는지 투자자 관점 문장으로 설명합니다.
    """
    ticker = result["ticker"]
    row = result["row"]
    trade = result["trade_plan"]

    total_band, total_desc = classify_score_band(result["total_score"])
    tech_band, tech_desc = classify_score_band(result["technical_score"])
    rel_band, rel_desc = classify_score_band(result["relative_score"])
    risk_band, risk_desc = classify_score_band(result["risk_score"])
    news_band, news_desc = classify_score_band(result["qualitative_score"])
    macro_band, macro_desc = classify_score_band(result["macro_score"])

    strengths = []
    cautions = []
    checklist = []

    if result["technical_score"] >= 65:
        strengths.append("차트/모멘텀은 우호적입니다.")
    elif result["technical_score"] <= 45:
        cautions.append("차트 신호가 약해 추세 확인이 필요합니다.")

    if result["relative_score"] >= 70:
        strengths.append(f"벤치마크({result['benchmark']}) 대비 상대강도가 강합니다.")
    elif result["relative_score"] <= 40:
        cautions.append(f"벤치마크({result['benchmark']}) 대비 약한 흐름입니다.")

    if result["risk_score"] <= 40:
        cautions.append("Risk Quality가 낮아 변동성·낙폭 리스크가 큽니다. 비중 확대보다 손절선과 보유 비중을 먼저 봐야 합니다.")
    elif result["risk_score"] >= 70:
        strengths.append("변동성/낙폭 기준 리스크 품질은 양호합니다.")

    if result["qualitative_score"] <= 40:
        cautions.append("News/Event 점수가 낮습니다. 뉴스 제목 기반 점수라 과소/과대평가될 수 있으니 실제 악재인지 뉴스 탭에서 확인해야 합니다.")
    elif result["qualitative_score"] >= 65:
        strengths.append("최근 뉴스 플로우는 비교적 긍정적으로 잡힙니다.")

    if result["macro_score"] <= 45:
        cautions.append("매크로 환경은 우호적이지 않습니다. 금리·달러·변동성 지표가 부담일 수 있습니다.")
    elif result["macro_score"] >= 65:
        strengths.append("매크로 환경은 위험자산에 비교적 우호적입니다.")

    rr = safe_float(trade.get("risk_reward"))
    if rr >= 1.5:
        strengths.append(f"손익비 R/R이 {rr:.2f}로 매매 구조는 양호합니다.")
    elif rr > 0:
        cautions.append(f"손익비 R/R이 {rr:.2f}로 충분히 매력적이지 않을 수 있습니다.")

    if result.get("is_leveraged"):
        cautions.append("레버리지 ETF라 장기 보유 시 변동성 손실과 리밸런싱 디케이를 별도로 고려해야 합니다.")

    if row.get("Close", 0) < row.get("MA200", np.inf):
        checklist.append("200일선 회복 여부")
    else:
        checklist.append("200일선 이탈 여부")

    checklist.extend([
        "거래량이 20일 평균 대비 1.5배 이상 붙는지",
        "손절가를 종가 기준으로 깨는지",
        "뉴스/Event 점수가 낮다면 실제 악재인지",
        "Macro Fit이 50 이상으로 회복되는지"
    ])

    # 최종 한 줄 결론
    if result["action"] in ["ADD", "ACCUMULATE"]:
        one_liner = "추세와 상대강도는 우호적입니다. 다만 한 번에 몰아 사기보다 손절가와 목표 비중을 정하고 분할 접근하는 쪽이 낫습니다."
    elif result["action"] == "HOLD":
        one_liner = "기존 보유는 가능하지만, 신규 비중 확대는 리스크·뉴스·매크로 부담을 확인한 뒤 판단하는 구간입니다."
    elif result["action"] == "WATCH":
        one_liner = "아직 방향성이 애매합니다. 지지선 근처 반등이나 거래량 동반 돌파가 나올 때까지 관망하는 편이 좋습니다."
    elif result["action"] == "TRIM":
        one_liner = "보유 중이라면 일부 비중 축소나 손절선 재점검이 필요한 구간입니다. 신규 진입 매력은 낮습니다."
    elif result["action"] == "RISK-OFF":
        one_liner = "점수와 별개로 변동성/레버리지/매크로 리스크가 커서 비중 관리가 최우선입니다."
    else:
        one_liner = "신규 진입은 피하고 더 좋은 가격·신호·손익비가 나올 때까지 기다리는 편이 낫습니다."

    # 평단/보유 여부 기반 문장
    avg_price = safe_float(item.get("avg_price", 0))
    qty = safe_float(item.get("quantity", 0))
    holding_note = "평단/수량을 입력하면 보유자 관점의 손익 해석이 더 정확해집니다."
    if avg_price > 0 and qty > 0:
        if result["pnl_pct"] >= 15:
            holding_note = f"현재 수익률이 {result['pnl_pct']:.2f}%로 높은 편입니다. 추가매수보다 익절/트레일링 스탑 기준을 먼저 정하는 게 좋습니다."
        elif result["pnl_pct"] <= -10:
            holding_note = f"현재 수익률이 {result['pnl_pct']:.2f}%입니다. 물타기 전 손절가와 투자 thesis 훼손 여부를 먼저 확인해야 합니다."
        else:
            holding_note = f"현재 수익률은 {result['pnl_pct']:.2f}%입니다. 보유 지속 여부는 손절가와 목표비중 기준으로 판단하세요."

    return {
        "one_liner": one_liner,
        "holding_note": holding_note,
        "score_table": pd.DataFrame([
            ["Signal Score", result["total_score"], total_band, total_desc],
            ["Technical", result["technical_score"], tech_band, tech_desc],
            ["Relative", result["relative_score"], rel_band, rel_desc],
            ["Risk Quality", result["risk_score"], risk_band, risk_desc],
            ["News/Event", result["qualitative_score"], news_band, news_desc],
            ["Macro Fit", result["macro_score"], macro_band, macro_desc],
        ], columns=["항목", "점수", "판정", "해석"]),
        "strengths": strengths or ["뚜렷한 강점 신호가 아직 부족합니다."],
        "cautions": cautions or ["현재 점수상 큰 경고 신호는 제한적입니다."],
        "checklist": checklist,
    }


def render_interpretation_memo(name, item, result):
    memo = build_interpretation_memo(name, item, result)

    st.markdown("##### 📝 해석 메모")
    st.info(memo["one_liner"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**좋게 보는 근거**")
        for s in memo["strengths"][:5]:
            st.markdown(f"- {s}")

    with col_b:
        st.markdown("**주의할 부분**")
        for c in memo["cautions"][:6]:
            st.markdown(f"- {c}")

    st.caption(memo["holding_note"])

    with st.expander("점수별 해석 자세히 보기", expanded=False):
        st.dataframe(memo["score_table"], use_container_width=True, hide_index=True)
        st.markdown("**다음 체크포인트**")
        for point in memo["checklist"]:
            st.markdown(f"- {point}")

def render_asset_card(name, item, result):
    ticker, row, trade = result["ticker"], result["row"], result["trade_plan"]
    color, bg = action_style(result["action"])

    # Streamlit Markdown이 빈 줄/들여쓰기 HTML을 코드블록으로 오해하지 않도록
    # 1) 태그 문자열을 미리 만들고 2) dedent().strip()으로 정리해서 렌더링합니다.
    leveraged_tag = "<span class='tag tag-orange'>Leveraged ETF</span>" if result.get("is_leveraged") else ""
    asset_html = f"""
    <div class="ib-card">
        <div class="label">{escape(str(name))} / {escape(str(ticker))} / Benchmark: {escape(str(result['benchmark']))}</div>
        <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div>
                <div class="big-value" style="color:{color};">{escape(str(result['action']))}</div>
                <span class="tag" style="background:{bg}; color:{color};">Signal Score {result['total_score']}/100</span>
                <span class="tag tag-blue">Macro {escape(str(result['macro_regime']))}</span>{leveraged_tag}
            </div>
            <div style="text-align:right;">
                <div class="label">Current Price</div>
                <div style="font-size:24px; font-weight:900;">{escape(format_price(ticker, result['price']))}</div>
            </div>
        </div>
        <div class="explain-box">
            <b>판단 요약:</b> {escape(str(result['action_reason']))}<br>
            <b>주의:</b> 이 점수는 매수 확률이 아니라 기술적·상대강도·리스크·뉴스·매크로를 합친 의사결정 보조 점수입니다.
        </div>
    </div>
    """
    st.markdown(dedent(asset_html).strip(), unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_score("Technical", result["technical_score"])
    with c2: render_score("Relative", result["relative_score"])
    with c3: render_score("Risk Quality", result["risk_score"])
    with c4: render_score("News/Event", result["qualitative_score"])
    with c5: render_score("Macro Fit", result["macro_score"])

    st.markdown("##### Portfolio & Trade Plan")

    # 긴 숫자를 한 metric에 "손절가 / 목표가" 형태로 함께 넣으면
    # Streamlit이 좁은 컬럼에서 값을 말줄임표(...)로 잘라 표시합니다.
    # 그래서 포트폴리오 현황과 진입/청산 계획을 2줄로 나누어 표시합니다.
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("보유 평가금액", format_price(ticker, result["market_value"]), f"{result['current_weight']:.2f}% weight")
    p2.metric("평가손익", format_price(ticker, result["pnl"]), f"{result['pnl_pct']:.2f}%" if item.get("avg_price", 0) > 0 else "평단 미입력")
    p3.metric("현재 비중", f"{result['current_weight']:.2f}%", f"Target {item.get('target_weight', 0):.1f}%")
    p4.metric("추천 추가 수량", f"{trade['suggested_add_qty']:,}", f"Risk budget {format_price(ticker, trade['risk_budget'])}")

    st.markdown("##### Entry / Exit Plan")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("현재가", format_price(ticker, result["price"]))
    e2.metric("손절가", format_price(ticker, trade["stop"]), f"-{((result['price'] - trade['stop']) / result['price'] * 100) if result['price'] > 0 else 0:.2f}%")
    e3.metric("목표가", format_price(ticker, trade["target"]), f"+{((trade['target'] - result['price']) / result['price'] * 100) if result['price'] > 0 else 0:.2f}%")
    e4.metric("손익비", f"{safe_float(trade['risk_reward']):.2f}x", "1.5x 이상 선호")

    st.markdown(f"""
    <div class="risk-box">
        <b>리스크 관리:</b><br>
        • ATR 변동성: {format_pct(row['ATR_Pct'])}<br>
        • 60일 연율화 변동성: {format_pct(row['Annual_Vol'])}<br>
        • 1년 최대낙폭 추정: {format_pct(row['MDD_1Y'])}<br>
        • 1주당 리스크: {format_price(ticker, trade['risk_per_share'])}<br>
        • 목표 비중: {item.get('target_weight', 0):.1f}% / 현재 비중: {result['current_weight']:.1f}%<br>
        • 손절가는 ATR과 최근 지지선을 함께 고려하고, 목표가는 최근 저항선 또는 2.5ATR 기준을 함께 반영합니다.
    </div>
    """, unsafe_allow_html=True)

    factor_table = pd.DataFrame([
        ["RSI", f"{row['RSI']:.2f}", "50~65는 건강한 상승 모멘텀, 75 이상은 과열 가능성"],
        ["MACD", "Bullish" if row["MACD"] > row["MACD_Signal"] else "Bearish", "추세 모멘텀 확인"],
        ["MA Position", "Above MA200" if row["Close"] > row["MA200"] else "Below MA200", "중장기 추세 필터"],
        ["BB Position", f"{row['BB_Pos']:.1f}%", "밴드 내 가격 위치"],
        ["ADX", f"{row['ADX']:.2f}", "25 이상이면 추세 강도 존재"],
        ["MFI", f"{row['MFI']:.2f}", "가격과 거래량을 함께 본 자금 흐름"],
        ["Volume Ratio", f"{row['Volume_Ratio']:.2f}x", "20일 평균 거래량 대비 현재 거래량"],
        ["Relative 20D", f"{result['rs_detail']['rs_20d']:.2f}%", "벤치마크 대비 20일 초과수익률"],
        ["Relative 60D", f"{result['rs_detail']['rs_60d']:.2f}%", "벤치마크 대비 60일 초과수익률"],
    ], columns=["Factor", "Value", "Meaning"])
    st.dataframe(factor_table, use_container_width=True, hide_index=True)

    bt = backtest_technical_signal(ticker, horizon=20)
    if bt:
        st.markdown("##### Historical Signal Check")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("표본 수", f"{bt['sample_size']:,}")
        b2.metric("강한 신호 횟수", f"{bt['high_count']:,}")
        b3.metric("강한 신호 후 20D 평균", f"{safe_float(bt['high_avg_ret']):.2f}%")
        b4.metric("강한 신호 후 상승 비율", f"{safe_float(bt['high_hit_rate']):.1f}%")
        st.caption("백테스트는 기술적 점수만 단순 검증한 참고용입니다. 뉴스, 실적, 매크로, 슬리피지, 세금, 환율은 반영하지 않습니다.")

    st.plotly_chart(make_stock_chart(result), use_container_width=True)

    with st.expander("News / Qualitative Events", expanded=False):
        st.write(result["qual_detail"]["summary"])
        st.write(f"긍정 {result['qual_detail']['positive_count']}개 / 부정 {result['qual_detail']['negative_count']}개 / 중립 {result['qual_detail']['neutral_count']}개")
        if result["news_items"]:
            for n in result["news_items"][:6]:
                if n["link"]:
                    st.markdown(f"- [{n['title']}]({n['link']}) — {n['publisher']}")
                else:
                    st.markdown(f"- {n['title']} — {n['publisher']}")
        else:
            st.info("최근 뉴스 데이터를 찾지 못했습니다.")


# =========================================================
# 11. SESSION INIT / SIDEBAR
# =========================================================
if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = load_portfolio()
    st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)

st.sidebar.markdown("## Portfolio Settings")
account_value = st.sidebar.number_input("총 운용자산 입력", min_value=0.0, value=10000000.0, step=100000.0)
risk_per_trade_pct = st.sidebar.slider("한 종목당 허용 손실 비중", min_value=0.2, max_value=5.0, value=1.0, step=0.1)

enable_autorefresh = st.sidebar.toggle("Auto Refresh 켜기", value=False, help="배포 환경에서 streamlit_autorefresh 오류가 나면 끄세요.")
if enable_autorefresh and st_autorefresh is not None:
    st_autorefresh(interval=300000, key="datarefresh")
elif enable_autorefresh and st_autorefresh is None:
    st.sidebar.warning("streamlit_autorefresh를 불러오지 못했습니다. requirements.txt를 확인하세요.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Storage Backend: {'Supabase' if use_supabase() else 'Local SQLite'}")

if genai is not None and get_secret("GEMINI_API_KEY"):
    gemini_client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    st.sidebar.success("Gemini API 연결 가능")
else:
    gemini_client = None
    st.sidebar.caption("Gemini API 미사용 모드")


# =========================================================
# 12. MAIN UI
# =========================================================
st.markdown("""
<div class="main-title">ALPHA TERMINAL <span style="color:#00529B;">PRO</span></div>
<div class="sub-title">기술적 분석 + 상대강도 + 리스크 관리 + 포지션 사이징 + 매크로 + 뉴스 이벤트를 결합한 개인 투자 보조 터미널</div>
""", unsafe_allow_html=True)

st.caption("기존 JSON 저장 방식 대신 SQLite/Supabase 저장 구조를 사용합니다. Signal Score는 매수 확률이 아니라 의사결정 보조 점수입니다.")

tab1, tab2, tab3, tab4 = st.tabs(["[1] Portfolio Strategy", "[2] Universe Screening", "[3] Macro & Qualitative", "[4] Settings / Data"])


# =========================================================
# TAB 1. PORTFOLIO STRATEGY
# =========================================================
with tab1:
    with st.expander("📘 분석 결과 해석 가이드", expanded=True):
        st.markdown("""
        **1) 최상단 액션**  
        `ADD`는 추가매수 후보, `ACCUMULATE`는 분할매수, `HOLD`는 보유 유지, `WATCH`는 관망, `TRIM`은 일부 축소, `AVOID`는 신규진입 회피, `RISK-OFF`는 리스크 관리 우선입니다.

        **2) Signal Score**  
        매수 확률이 아니라 기술적 지표, 시장 대비 상대강도, 리스크, 뉴스, 매크로를 합산한 의사결정 보조 점수입니다. 70점 이상은 우호적, 55~69점은 중립 이상, 45~54점은 애매, 45점 미만은 보수적으로 해석합니다.

        **3) 세부 점수**  
        - **Technical**: 이동평균, MACD, RSI, 볼린저밴드, 거래량 기반 차트 상태입니다. 70 이상이면 차트는 우호적입니다.  
        - **Relative**: 벤치마크 대비 상대강도입니다. 한국 주식은 KOSPI/KOSDAQ, 미국 주식은 SPY/QQQ/SOXX 등과 비교합니다. 높을수록 시장보다 강합니다.  
        - **Risk Quality**: 변동성, ATR, 낙폭, 레버리지 여부를 반영한 안전도입니다. 높을수록 안전하고, 낮을수록 비중 확대를 조심해야 합니다.  
        - **News/Event**: 최근 뉴스 제목의 긍정/부정 키워드 기반 정성 점수입니다. 참고용이며 실제 공시와 실적을 함께 확인해야 합니다.  
        - **Macro Fit**: VIX, 미국 10년물, 달러, QQQ, SOXX, HYG, 환율 등을 본 매크로 적합도입니다.

        **4) Trade Plan**  
        손절가, 목표가, 손익비, 추천 추가 수량을 함께 봐야 합니다. Signal Score가 높아도 손익비가 낮거나 Risk Quality가 낮으면 신규 매수는 신중하게 해석합니다.
        """)

    st.markdown("### Portfolio Editor")
    st.caption("종목명, 티커, 평단가, 보유수량, 목표비중을 수정하면 저장됩니다. 로컬은 SQLite, 배포는 Supabase 설정 시 DB에 저장됩니다.")

    auto_save_portfolio = st.toggle("Auto-save portfolio changes", value=True)

    edited_portfolio_df = st.data_editor(
        portfolio_to_df(st.session_state.my_portfolio),
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Asset": st.column_config.TextColumn("Asset", required=True),
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Avg Price": st.column_config.NumberColumn("Avg Price", min_value=0.0, step=0.01),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=1.0),
            "Target Weight %": st.column_config.NumberColumn("Target Weight %", min_value=0.0, max_value=100.0, step=1.0),
            "Max Loss %": st.column_config.NumberColumn("Max Loss %", min_value=0.0, max_value=100.0, step=0.5),
            "Memo": st.column_config.TextColumn("Memo"),
        },
        key="portfolio_editor"
    )

    edited_portfolio = df_to_portfolio(edited_portfolio_df)
    edited_hash = portfolio_hash(edited_portfolio)
    if auto_save_portfolio and edited_hash != st.session_state.get("last_saved_portfolio_hash"):
        st.session_state.my_portfolio = edited_portfolio
        save_portfolio(st.session_state.my_portfolio)
        st.session_state.last_saved_portfolio_hash = edited_hash
        st.toast("포트폴리오 변경사항 자동 저장 완료")

    col_save, col_reload = st.columns([1, 1])
    with col_save:
        if st.button("Save Portfolio Now"):
            st.session_state.my_portfolio = edited_portfolio
            save_portfolio(st.session_state.my_portfolio)
            st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
            st.success("포트폴리오 저장 완료")
    with col_reload:
        if st.button("Reload Saved Portfolio"):
            st.session_state.my_portfolio = load_portfolio()
            st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
            st.rerun()

    st.markdown("---")
    results, portfolio_rows = {}, []

    for name, item in list(st.session_state.my_portfolio.items()):
        ticker = item.get("ticker", "").upper()
        if not ticker:
            continue
        with st.spinner(f"{name} 분석 중..."):
            result = analyze_stock(
                ticker=ticker,
                account_value=account_value,
                risk_per_trade_pct=risk_per_trade_pct,
                target_weight=item.get("target_weight", 10.0),
                avg_price=item.get("avg_price", 0.0),
                quantity=item.get("quantity", 0.0),
                include_news=True
            )
        if result is None:
            st.warning(f"{name}({ticker}) 데이터를 불러오지 못했습니다.")
            continue
        results[name] = result
        portfolio_rows.append({
            "Asset": name, "Ticker": ticker, "Action": result["action"], "Signal Score": result["total_score"],
            "Technical": result["technical_score"], "Relative": result["relative_score"], "Risk": result["risk_score"],
            "Price": result["price"], "Qty": item.get("quantity", 0.0), "Market Value": result["market_value"],
            "P&L %": result["pnl_pct"], "Weight %": result["current_weight"]
        })

    if portfolio_rows:
        st.markdown("### Portfolio Snapshot")
        snapshot = pd.DataFrame(portfolio_rows)
        st.dataframe(
            snapshot,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Signal Score": st.column_config.ProgressColumn("Signal Score", min_value=0, max_value=100, format="%d"),
                "Technical": st.column_config.ProgressColumn("Technical", min_value=0, max_value=100, format="%d"),
                "Relative": st.column_config.ProgressColumn("Relative", min_value=0, max_value=100, format="%d"),
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d")
            }
        )
        st.markdown("---")
        for name, item in list(st.session_state.my_portfolio.items()):
            if name in results:
                render_asset_card(name, item, results[name])
                if st.button(f"Remove {name}", key=f"remove_{name}_{item.get('ticker','')}"):
                    del st.session_state.my_portfolio[name]
                    save_portfolio(st.session_state.my_portfolio)
                    st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
                    st.rerun()
                st.markdown("---")
    else:
        st.info("아직 등록된 종목이 없습니다.")


# =========================================================
# TAB 2. UNIVERSE SCREENING
# =========================================================
with tab2:
    st.markdown("### Universe Screening: Korea 100 + US 100")
    st.info("스크리닝에서는 속도를 위해 뉴스 정성 점수를 중립으로 두고, 기술적 점수·상대강도·리스크·매크로 적합도를 중심으로 봅니다.")

    universe_type = st.radio("Universe", ["Korea 100", "US 100", "Custom"], horizontal=True)
    custom_tickers = ""
    if universe_type == "Custom":
        custom_tickers = st.text_area("티커를 쉼표로 입력", placeholder="ex) NVDA, TSLA, TSLL, 000660.KS")

    if st.button("Run Screening"):
        if universe_type == "Korea 100":
            universe = {t: n for t, n in KR_STOCKS.items()}
        elif universe_type == "US 100":
            universe = {t: n for t, n in US_STOCKS.items()}
        else:
            tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
            universe = {t: t for t in tickers}

        rows = []
        progress = st.progress(0)
        for idx, (ticker, name) in enumerate(universe.items()):
            progress.progress((idx + 1) / max(len(universe), 1))
            result = analyze_stock(
                ticker=ticker,
                account_value=account_value,
                risk_per_trade_pct=risk_per_trade_pct,
                target_weight=10.0,
                avg_price=0.0,
                quantity=0.0,
                include_news=False
            )
            if result is None:
                continue
            rows.append({
                "Asset": name, "Ticker": ticker, "Action": result["action"], "Signal Score": result["total_score"],
                "Technical": result["technical_score"], "Relative": result["relative_score"], "Risk": result["risk_score"],
                "Macro": result["macro_score"], "Price": result["price"], "ATR %": result["row"]["ATR_Pct"],
                "RSI": result["row"]["RSI"], "20D RS": result["rs_detail"]["rs_20d"],
                "60D RS": result["rs_detail"]["rs_60d"], "R/R": result["trade_plan"]["risk_reward"]
            })
        progress.empty()
        if rows:
            screen_df = pd.DataFrame(rows).sort_values("Signal Score", ascending=False)
            st.dataframe(
                screen_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Signal Score": st.column_config.ProgressColumn("Signal Score", min_value=0, max_value=100, format="%d"),
                    "Technical": st.column_config.ProgressColumn("Technical", min_value=0, max_value=100, format="%d"),
                    "Relative": st.column_config.ProgressColumn("Relative", min_value=0, max_value=100, format="%d"),
                    "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
                    "Macro": st.column_config.ProgressColumn("Macro", min_value=0, max_value=100, format="%d")
                }
            )
            st.markdown("#### Top Candidates")
            st.dataframe(screen_df.head(15)[["Asset", "Ticker", "Action", "Signal Score", "Technical", "Relative", "Risk", "ATR %", "R/R"]], use_container_width=True, hide_index=True)
        else:
            st.warning("스크리닝 결과가 없습니다. 티커나 데이터 소스를 확인하세요.")


# =========================================================
# TAB 3. MACRO / QUALITATIVE
# =========================================================
with tab3:
    st.markdown("### Macro Regime")
    macro = fetch_macro_data()
    macro_score, macro_regime, macro_notes = get_macro_score(macro)

    m1, m2, m3, m4 = st.columns(4)
    with m1: render_macro_card("VIX", macro["VIX"]["value"], macro["VIX"]["diff"], inverse=True)
    with m2: render_macro_card("US 10Y", macro["US10Y"]["value"], macro["US10Y"]["diff"], unit="%", inverse=True)
    with m3: render_macro_card("DXY", macro["DXY"]["value"], macro["DXY"]["diff"], inverse=True)
    with m4: render_macro_card("USDKRW", macro["USDKRW"]["value"], macro["USDKRW"]["diff"], inverse=True)

    m5, m6, m7, m8 = st.columns(4)
    with m5: render_macro_card("QQQ", macro["QQQ"]["value"], macro["QQQ"]["diff"])
    with m6: render_macro_card("SOXX", macro["SOXX"]["value"], macro["SOXX"]["diff"])
    with m7: render_macro_card("HYG", macro["HYG"]["value"], macro["HYG"]["diff"])
    with m8: render_macro_card("WTI", macro["WTI"]["value"], macro["WTI"]["diff"], inverse=True)

    st.markdown(f"""
    <div class="ib-card">
        <div class="label">Macro Regime</div>
        <div class="big-value" style="color:{score_color(macro_score)};">{macro_regime} / {macro_score}/100</div>
        <div class="explain-box">{"<br>".join(["• " + note for note in macro_notes])}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Portfolio News Summary")
    for name, item in st.session_state.my_portfolio.items():
        ticker = item.get("ticker", "").upper()
        news_items = fetch_news_items(ticker)
        q_score, q_detail = get_qualitative_score(news_items)
        with st.expander(f"{name} / {ticker} / News Event Score {q_score}/100", expanded=False):
            st.write(q_detail["summary"])
            st.write(f"긍정 {q_detail['positive_count']}개 / 부정 {q_detail['negative_count']}개 / 중립 {q_detail['neutral_count']}개")
            if news_items:
                for n in news_items[:8]:
                    st.markdown(f"- [{n['title']}]({n['link']}) — {n['publisher']}" if n["link"] else f"- {n['title']} — {n['publisher']}")
            else:
                st.info("뉴스 데이터 없음")

    st.markdown("---")
    st.markdown("### AI 정성 분석 프롬프트")
    portfolio_text = "\n".join([f"- {name}: {item.get('ticker')} / 평단 {item.get('avg_price')} / 수량 {item.get('quantity')} / 목표비중 {item.get('target_weight')}%" for name, item in st.session_state.my_portfolio.items()])
    macro_text = "\n".join([f"- {k}: {v.get('value', np.nan):.2f}, 1D diff {v.get('diff', np.nan):+.2f}, 20D return {v.get('ret_20d', np.nan):+.2f}%" for k, v in macro.items() if not pd.isna(v.get("value", np.nan))])
    generated_prompt = f"""
당신은 월스트리트 시니어 매크로/에쿼티 전략가입니다.
저는 한국과 미국 주식, 특히 성장주와 레버리지 ETF 일부를 함께 운용하는 개인 투자자입니다.

[내 포트폴리오]
{portfolio_text}

[현재 매크로 상태]
Macro Regime: {macro_regime}
Macro Score: {macro_score}/100

{macro_text}

[분석 요청]
1. 금리, 달러, VIX 흐름이 성장주와 레버리지 ETF에 미치는 영향
2. 반도체, AI, 전기차, 플랫폼 등 섹터별 모멘텀
3. 각 종목별 최근 뉴스/실적/가이던스/규제 리스크
4. 지금 추가매수하면 안 되는 종목과 이유
5. 손절 또는 비중축소를 먼저 고려해야 하는 종목
6. 향후 1~4주 동안 체크해야 할 핵심 이벤트
7. 최종적으로 종목별 액션 플랜을 표로 제시
"""
    st.code(generated_prompt.strip(), language="markdown")


# =========================================================
# TAB 4. SETTINGS / DATA
# =========================================================
with tab4:
    st.markdown("### Current Portfolio JSON")
    st.json(st.session_state.my_portfolio)

    col_reset1, col_reset2, col_reset3 = st.columns(3)
    with col_reset1:
        if st.button("Save Portfolio Now", key="settings_save"):
            save_portfolio(st.session_state.my_portfolio)
            st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
            st.success("저장 완료")
    with col_reset2:
        if st.button("Reload Saved Portfolio", key="settings_reload"):
            st.session_state.my_portfolio = load_portfolio()
            st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
            st.rerun()
    with col_reset3:
        if st.button("Reset to Default Portfolio"):
            st.session_state.my_portfolio = DEFAULT_PORTFOLIO.copy()
            save_portfolio(st.session_state.my_portfolio)
            st.session_state.last_saved_portfolio_hash = portfolio_hash(st.session_state.my_portfolio)
            st.success("기본 포트폴리오로 초기화 완료")
            st.rerun()

    st.markdown("---")
    st.markdown("### Storage Status")
    st.write(f"현재 저장 방식: **{'Supabase' if use_supabase() else 'Local SQLite'}**")
    if not use_supabase():
        st.caption(f"로컬 DB 경로: {DB_PATH}")

    st.markdown("---")
    st.markdown("### requirements.txt")
    st.code("""
streamlit
yfinance
plotly
pandas
numpy
ta
streamlit-autorefresh
supabase
google-genai
""".strip(), language="text")

    st.markdown("---")
    st.warning(
        "이 앱은 투자 판단을 보조하는 도구입니다. Signal Score는 매수 확률이 아니며, 특히 TSLL, SOXL, TQQQ 같은 레버리지 ETF는 장기 보유 시 변동성 손실과 리밸런싱 디케이가 발생할 수 있으므로 포지션 사이징과 손절 기준을 반드시 함께 봐야 합니다."
    )
