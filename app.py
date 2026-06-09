import math
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📈",
    layout="wide",
)

PRICE_FETCH_EVERY_SECONDS = 60
HISTORY_CACHE_SECONDS = 15 * 60

# 여기만 본인 포트폴리오에 맞게 수정하면 됩니다.
# avg_price가 0이면 평단 미입력으로 처리됩니다.
WATCHLIST = [
    {
        "asset": "AMDL",
        "ticker": "AMDL",
        "benchmark": "^IXIC",
        "qty": 0,
        "avg_price": 0,
        "target_weight": 0.25,
        "risk_budget": 100_000,
        "news_score": 50,
    },
    {
        "asset": "NVDL",
        "ticker": "NVDL",
        "benchmark": "^IXIC",
        "qty": 0,
        "avg_price": 0,
        "target_weight": 0.25,
        "risk_budget": 100_000,
        "news_score": 50,
    },
    {
        "asset": "RGTI",
        "ticker": "RGTI",
        "benchmark": "^IXIC",
        "qty": 0,
        "avg_price": 0,
        "target_weight": 0.25,
        "risk_budget": 100_000,
        "news_score": 50,
    },
    {
        "asset": "SK하이닉스",
        "ticker": "000660.KS",
        "benchmark": "^KS11",
        "qty": 0,
        "avg_price": 0,
        "target_weight": 0.25,
        "risk_budget": 100_000,
        "news_score": 50,
    },
]

SCORE_WEIGHTS = {
    "technical": 0.30,
    "relative": 0.25,
    "risk_quality": 0.25,
    "news_event": 0.10,
    "macro_fit": 0.10,
}

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .main-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }
    .small-label {font-size: 12px; color: #6b7280; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;}
    .big-title {font-size: 28px; font-weight: 900; margin: 4px 0 8px 0;}
    .risk-off {color: #dc2626;}
    .watch {color: #d97706;}
    .risk-on {color: #16a34a;}
    .badge-red {display:inline-block; padding:6px 10px; border-radius:999px; background:#fee2e2; color:#dc2626; font-weight:800; font-size:12px; margin-right:6px;}
    .badge-blue {display:inline-block; padding:6px 10px; border-radius:999px; background:#dbeafe; color:#2563eb; font-weight:800; font-size:12px; margin-right:6px;}
    .badge-gray {display:inline-block; padding:6px 10px; border-radius:999px; background:#f3f4f6; color:#374151; font-weight:800; font-size:12px; margin-right:6px;}
    .summary-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
        line-height: 1.65;
    }
    .score-wrap {margin-bottom: 14px;}
    .score-title {font-weight: 800; margin-bottom: 8px;}
    .bar-bg {height: 9px; background:#eef2f7; border-radius: 999px; overflow:hidden; border:1px solid #edf0f5;}
    .bar-fill {height: 100%; border-radius: 999px; background:#2563eb;}
    .metric-title {font-size: 14px; color:#111827; margin-bottom:6px;}
    .metric-value {font-size: 36px; font-weight: 500; color:#1f2937; line-height: 1.1;}
    .metric-sub-green {display:inline-block; margin-top:6px; padding:3px 8px; border-radius:999px; background:#dcfce7; color:#15803d; font-weight:700; font-size:13px;}
    .metric-sub-red {display:inline-block; margin-top:6px; padding:3px 8px; border-radius:999px; background:#fee2e2; color:#dc2626; font-weight:700; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 유틸 함수
# =========================================================
def _to_float_price(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        v = float(x)
        if not np.isfinite(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


def clip_score(x, low=0, high=100):
    try:
        if x is None or not np.isfinite(x):
            return 50
        return int(round(max(low, min(high, float(x)))))
    except Exception:
        return 50


def is_korean_ticker(ticker: str) -> bool:
    ticker = str(ticker).upper().strip()
    return ticker.endswith(".KS") or ticker.endswith(".KQ")


def extract_korean_code(ticker: str) -> str:
    return re.sub(r"\.(KS|KQ)$", "", str(ticker).upper().strip())


def format_price(price, ticker: str):
    if price is None or not np.isfinite(price):
        return "-"
    if is_korean_ticker(ticker):
        return f"₩{price:,.0f}"
    return f"${price:,.2f}"


def pct_text(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.2f}%"


# =========================================================
# 1분 단위 최신 가격 조회
# 핵심:
# - st_autorefresh 사용 안 함
# - minute_bucket이 1분마다 바뀜
# - 가격 API 캐시 키가 바뀌면서 가격 데이터만 새로 가져옴
# =========================================================
@st.cache_data(ttl=PRICE_FETCH_EVERY_SECONDS + 5, show_spinner=False)
def get_naver_current_price(ticker: str, minute_bucket: int):
    """
    한국 주식 현재가 조회.
    네이버 금융 now 값을 그대로 사용합니다.
    10을 곱하거나 나누지 않습니다.
    """
    code = extract_korean_code(ticker)
    url = f"https://api.finance.naver.com/service/itemSummary.nhn?itemcode={code}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.naver.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()

        price = _to_float_price(data.get("now"))

        if price is not None:
            return {
                "price": price,
                "source": "NAVER_NOW_1MIN",
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    except Exception as e:
        return {
            "price": None,
            "source": f"NAVER_FAILED: {type(e).__name__}",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    return {
        "price": None,
        "source": "NAVER_FAILED",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@st.cache_data(ttl=PRICE_FETCH_EVERY_SECONDS + 5, show_spinner=False)
def get_yfinance_latest_price(ticker: str, minute_bucket: int):
    """
    미국 주식/ETF 및 fallback용 최신가.
    1순위는 1분봉 마지막 가격입니다.
    """
    ticker = str(ticker).strip()

    try:
        t = yf.Ticker(ticker)

        # 1순위: 1분봉 마지막 가격
        try:
            intraday = t.history(period="1d", interval="1m", prepost=True, auto_adjust=False)

            if intraday is not None and not intraday.empty and "Close" in intraday.columns:
                close = intraday["Close"].dropna()

                if not close.empty:
                    price = _to_float_price(close.iloc[-1])

                    if price is not None:
                        return {
                            "price": price,
                            "source": "YF_1M_CLOSE_1MIN",
                            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

        except Exception:
            pass

        # 2순위: 5분봉 마지막 가격
        try:
            intraday = t.history(period="5d", interval="5m", prepost=True, auto_adjust=False)

            if intraday is not None and not intraday.empty and "Close" in intraday.columns:
                close = intraday["Close"].dropna()

                if not close.empty:
                    price = _to_float_price(close.iloc[-1])

                    if price is not None:
                        return {
                            "price": price,
                            "source": "YF_5M_CLOSE_FALLBACK",
                            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

        except Exception:
            pass

        # 3순위: fast_info
        try:
            fi = t.fast_info

            for key in ["last_price", "lastPrice", "regularMarketPrice", "regular_market_price"]:
                try:
                    price = _to_float_price(fi.get(key))
                    if price is not None:
                        return {
                            "price": price,
                            "source": f"YF_FAST_{key}",
                            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                except Exception:
                    pass

                try:
                    price = _to_float_price(getattr(fi, key))
                    if price is not None:
                        return {
                            "price": price,
                            "source": f"YF_FAST_{key}",
                            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                except Exception:
                    pass

        except Exception:
            pass

        # 4순위: 일봉 종가 fallback
        try:
            hist = t.history(period="5d", interval="1d", auto_adjust=False)

            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = hist["Close"].dropna()

                if not close.empty:
                    price = _to_float_price(close.iloc[-1])

                    if price is not None:
                        return {
                            "price": price,
                            "source": "YF_DAILY_CLOSE_FALLBACK",
                            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

        except Exception:
            pass

    except Exception as e:
        return {
            "price": None,
            "source": f"YF_FAILED: {type(e).__name__}",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    return {
        "price": None,
        "source": "PRICE_FAILED",
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_latest_price(ticker: str, minute_bucket: int):
    """
    최종 현재가 조회.
    한국 주식: 네이버 현재가 우선
    미국 주식/ETF: yfinance 1분봉 우선
    """
    if is_korean_ticker(ticker):
        naver = get_naver_current_price(ticker, minute_bucket)

        if naver.get("price") is not None:
            return naver

        return get_yfinance_latest_price(ticker, minute_bucket)

    return get_yfinance_latest_price(ticker, minute_bucket)


# =========================================================
# 히스토리 데이터
# =========================================================
@st.cache_data(ttl=HISTORY_CACHE_SECONDS, show_spinner=False)
def get_daily_history(ticker: str, period: str = "1y"):
    ticker = str(ticker).strip()

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = df["Close"]

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return df


def apply_latest_price_to_history(hist: pd.DataFrame, latest_price):
    """
    판단 점수 계산 전에 마지막 Close를 최신 가격으로 교체합니다.
    즉 Technical / Relative / Risk Quality / Signal Score가
    전일 종가가 아니라 최신 가격 기준으로 계산됩니다.
    """
    if hist is None or hist.empty:
        return hist

    latest_price = _to_float_price(latest_price)

    if latest_price is None:
        return hist

    hist = hist.copy()
    last_idx = hist.index[-1]

    prev_close = _to_float_price(hist.loc[last_idx, "Close"])

    if prev_close is None:
        prev_close = latest_price

    hist.loc[last_idx, "Close"] = latest_price

    if "High" in hist.columns:
        old_high = _to_float_price(hist.loc[last_idx, "High"])
        hist.loc[last_idx, "High"] = max(old_high or latest_price, latest_price, prev_close)

    if "Low" in hist.columns:
        old_low = _to_float_price(hist.loc[last_idx, "Low"])
        hist.loc[last_idx, "Low"] = min(old_low or latest_price, latest_price, prev_close)

    return hist


# =========================================================
# 지표 계산
# =========================================================
def calc_rsi(close: pd.Series, period: int = 14):
    close = close.dropna()

    if len(close) < period + 2:
        return pd.Series(index=close.index, dtype=float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14):
    if df is None or df.empty or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else None, dtype=float)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def last_return(close: pd.Series, days: int):
    close = close.dropna()

    if len(close) <= days:
        return 0.0

    base = close.iloc[-days]

    if base == 0 or pd.isna(base):
        return 0.0

    return float(close.iloc[-1] / base - 1)


def score_technical(hist: pd.DataFrame):
    if hist is None or hist.empty or len(hist) < 30:
        return 50

    close = hist["Close"].dropna()
    price = float(close.iloc[-1])

    ma_scores = []

    for window, multiplier in [(20, 700), (50, 550), (200, 350)]:
        if len(close) >= window:
            ma = close.rolling(window).mean().iloc[-1]

            if pd.notna(ma) and ma > 0:
                ma_scores.append(clip_score(50 + ((price / ma) - 1) * multiplier))

    trend_score = int(round(np.mean(ma_scores))) if ma_scores else 50

    mom20 = last_return(close, 20)
    mom60 = last_return(close, 60)

    momentum_score = clip_score(50 + mom20 * 300 + mom60 * 180)

    rsi_series = calc_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50

    if 45 <= rsi <= 65:
        rsi_score = 85
    elif 35 <= rsi < 45:
        rsi_score = 65
    elif 65 < rsi <= 75:
        rsi_score = 60
    elif 30 <= rsi < 35:
        rsi_score = 45
    elif 75 < rsi <= 82:
        rsi_score = 45
    else:
        rsi_score = 25

    return clip_score(0.45 * trend_score + 0.35 * momentum_score + 0.20 * rsi_score)


def score_risk_quality(hist: pd.DataFrame):
    """
    높을수록 안전한 점수입니다.
    낮다는 건 위험이 낮다는 뜻이 아니라,
    변동성/낙폭/ATR 부담이 크다는 뜻입니다.
    """
    if hist is None or hist.empty or len(hist) < 40:
        return 50

    close = hist["Close"].dropna()
    price = float(close.iloc[-1])
    ret = close.pct_change().dropna()

    if ret.empty:
        return 50

    vol20 = ret.tail(20).std() * math.sqrt(252)
    vol60 = ret.tail(60).std() * math.sqrt(252) if len(ret) >= 60 else vol20

    vol = np.nanmean([vol20, vol60])
    vol_score = clip_score(100 - vol * 115)

    rolling_high = close.tail(120).cummax()

    if rolling_high.empty or rolling_high.iloc[-1] == 0:
        dd_score = 50
    else:
        drawdown = price / rolling_high.iloc[-1] - 1
        dd_score = clip_score(100 + drawdown * 230)

    atr = calc_atr(hist).dropna()

    if atr.empty or price <= 0:
        atr_score = 50
    else:
        atr_pct = float(atr.iloc[-1] / price)
        atr_score = clip_score(100 - atr_pct * 1100)

    return clip_score(0.45 * vol_score + 0.35 * dd_score + 0.20 * atr_score)


def score_relative(asset_hist: pd.DataFrame, benchmark_hist: pd.DataFrame):
    if (
        asset_hist is None
        or benchmark_hist is None
        or asset_hist.empty
        or benchmark_hist.empty
        or len(asset_hist) < 30
        or len(benchmark_hist) < 30
    ):
        return 50

    joined = pd.concat(
        [
            asset_hist["Close"].rename("asset"),
            benchmark_hist["Close"].rename("bench"),
        ],
        axis=1,
    ).dropna()

    if len(joined) < 30:
        return 50

    asset_close = joined["asset"]
    bench_close = joined["bench"]

    alpha20 = last_return(asset_close, 20) - last_return(bench_close, 20)
    alpha60 = last_return(asset_close, 60) - last_return(bench_close, 60)

    rel_line = asset_close / bench_close.replace(0, np.nan)

    if len(rel_line.dropna()) >= 20:
        rel_ma = rel_line.rolling(20).mean().iloc[-1]
        rel_trend = float(rel_line.iloc[-1] / rel_ma - 1) if rel_ma and pd.notna(rel_ma) else 0
    else:
        rel_trend = 0

    s20 = clip_score(50 + alpha20 * 350)
    s60 = clip_score(50 + alpha60 * 240)
    srel = clip_score(50 + rel_trend * 500)

    return clip_score(0.40 * s20 + 0.40 * s60 + 0.20 * srel)


def score_macro_fit(benchmark_hist: pd.DataFrame):
    if benchmark_hist is None or benchmark_hist.empty or len(benchmark_hist) < 60:
        return 50

    close = benchmark_hist["Close"].dropna()
    price = float(close.iloc[-1])

    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    trend = 50

    if pd.notna(ma20) and ma20 > 0:
        trend += 25 if price > ma20 else -15

    if pd.notna(ma60) and ma60 > 0:
        trend += 25 if price > ma60 else -15

    mom20 = last_return(close, 20)
    momentum = clip_score(50 + mom20 * 250)

    ret = close.pct_change().dropna()
    vol = ret.tail(20).std() * math.sqrt(252) if not ret.empty else 0.2
    vol_penalty = max(0, (vol - 0.18) * 80)

    return clip_score(0.55 * trend + 0.35 * momentum - 0.10 * vol_penalty)


def calc_signal_score(technical, relative, risk_quality, news_event, macro_fit):
    raw = (
        SCORE_WEIGHTS["technical"] * technical
        + SCORE_WEIGHTS["relative"] * relative
        + SCORE_WEIGHTS["risk_quality"] * risk_quality
        + SCORE_WEIGHTS["news_event"] * news_event
        + SCORE_WEIGHTS["macro_fit"] * macro_fit
    )

    return clip_score(raw)


def decide_action(signal_score, risk_quality):
    if signal_score >= 70 and risk_quality >= 40:
        return "RISK-ON"

    if signal_score >= 65 and risk_quality >= 35:
        return "WATCH"

    return "RISK-OFF"


def action_class(action):
    if action == "RISK-ON":
        return "risk-on"

    if action == "WATCH":
        return "watch"

    return "risk-off"


# =========================================================
# 매매 계획
# =========================================================
def calc_trade_plan(hist: pd.DataFrame, current_price, risk_quality, risk_budget, action):
    current_price = _to_float_price(current_price)

    if current_price is None:
        return {
            "stop_price": None,
            "target_price": None,
            "stop_pct": None,
            "target_pct": None,
            "rr": 1.8,
            "recommended_qty": 0,
        }

    atr = calc_atr(hist).dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)

    if not atr.empty:
        atr_pct = float(atr.iloc[-1] / current_price)
        stop_pct = max(0.07, min(0.25, atr_pct * 2.0))
    else:
        if risk_quality >= 65:
            stop_pct = 0.08
        elif risk_quality >= 35:
            stop_pct = 0.12
        else:
            stop_pct = 0.15

    if risk_quality < 30:
        stop_pct = max(stop_pct, 0.14)

    rr = 1.8
    stop_price = current_price * (1 - stop_pct)
    target_price = current_price + (current_price - stop_price) * rr
    target_pct = (target_price / current_price - 1) if current_price else None

    risk_per_share = current_price - stop_price

    if action == "RISK-OFF" or risk_per_share <= 0:
        recommended_qty = 0
    else:
        recommended_qty = int(max(0, math.floor(float(risk_budget) / risk_per_share)))

    return {
        "stop_price": stop_price,
        "target_price": target_price,
        "stop_pct": -stop_pct * 100,
        "target_pct": target_pct * 100 if target_pct is not None else None,
        "rr": rr,
        "recommended_qty": recommended_qty,
    }


# =========================================================
# 해석 문구
# =========================================================
def explain_factor(name, score):
    if name == "Technical":
        if score >= 70:
            return "추세·모멘텀이 강해서 가격 흐름은 우호적입니다."
        if score >= 50:
            return "추세가 나쁘진 않지만, 강한 매수 신호까지는 아닙니다."
        return "이동평균·모멘텀 기준으로 가격 흐름이 약합니다."

    if name == "Relative":
        if score >= 70:
            return "벤치마크 대비 초과 성과가 강합니다."
        if score >= 50:
            return "벤치마크와 비슷하거나 소폭 우위입니다."
        return "벤치마크 대비 상대 성과가 약합니다."

    if name == "Risk Quality":
        if score >= 70:
            return "변동성·낙폭 부담이 낮아 리스크 품질이 좋습니다."
        if score >= 40:
            return "리스크가 중간 수준이라 비중 관리가 필요합니다."
        return "변동성·낙폭 부담이 커서 신규 진입보다 리스크 관리가 우선입니다."

    if name == "News/Event":
        if score >= 70:
            return "뉴스/이벤트 흐름이 우호적으로 반영된 상태입니다."
        if score >= 40:
            return "특별히 강한 호재/악재 없이 중립으로 반영됩니다."
        return "뉴스/이벤트 리스크를 보수적으로 반영합니다."

    if name == "Macro Fit":
        if score >= 70:
            return "시장/벤치마크 환경이 종목 진입에 우호적입니다."
        if score >= 40:
            return "시장 환경은 중립권입니다."
        return "시장 환경이 약해 보수적 접근이 필요합니다."

    return "보조 판단 점수입니다."


def build_summary(action, risk_quality):
    if action == "RISK-ON":
        return "판단 요약: 추세와 상대강도가 충분하고 리스크 품질도 버틸 만해 분할 진입을 검토할 수 있습니다."

    if action == "WATCH":
        return "판단 요약: 일부 지표는 우호적이지만 리스크 또는 시장 조건이 애매해 관망 후 확인 매수가 낫습니다."

    if risk_quality < 35:
        return "판단 요약: 변동성과 낙폭 리스크가 커서 신규 진입보다 리스크 관리가 우선입니다."

    return "판단 요약: 종합 점수가 충분하지 않아 신규 진입보다는 관망이 우선입니다."


def macro_label(score):
    if score >= 65:
        return "Macro GOOD"

    if score >= 40:
        return "Macro NEUTRAL"

    return "Macro WEAK"


# =========================================================
# 분석 실행
# =========================================================
def analyze_asset(asset: dict, minute_bucket: int):
    ticker = asset["ticker"]
    benchmark = asset.get("benchmark") or ("^KS11" if is_korean_ticker(ticker) else "^IXIC")

    price_payload = get_latest_price(ticker, minute_bucket)
    latest_price = price_payload.get("price")

    hist_raw = get_daily_history(ticker, "1y")
    hist = apply_latest_price_to_history(hist_raw, latest_price)

    if latest_price is None and hist is not None and not hist.empty:
        latest_price = _to_float_price(hist["Close"].iloc[-1])
        price_payload = {
            "price": latest_price,
            "source": "DAILY_CLOSE_LAST_RESORT",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    bench_hist = get_daily_history(benchmark, "1y")

    technical = score_technical(hist)
    relative = score_relative(hist, bench_hist)
    risk_quality = score_risk_quality(hist)
    news_event = clip_score(asset.get("news_score", 50))
    macro_fit = score_macro_fit(bench_hist)
    signal = calc_signal_score(technical, relative, risk_quality, news_event, macro_fit)
    action = decide_action(signal, risk_quality)

    qty = int(asset.get("qty", 0) or 0)
    avg_price = _to_float_price(asset.get("avg_price", 0)) or 0
    market_value = (latest_price or 0) * qty
    pnl_pct = ((latest_price / avg_price - 1) * 100) if avg_price and latest_price else 0

    trade_plan = calc_trade_plan(
        hist=hist,
        current_price=latest_price,
        risk_quality=risk_quality,
        risk_budget=asset.get("risk_budget", 100_000),
        action=action,
    )

    return {
        "asset": asset.get("asset", ticker),
        "ticker": ticker,
        "benchmark": benchmark,
        "action": action,
        "signal": signal,
        "technical": technical,
        "relative": relative,
        "risk_quality": risk_quality,
        "news_event": news_event,
        "macro_fit": macro_fit,
        "price": latest_price,
        "qty": qty,
        "avg_price": avg_price,
        "market_value": market_value,
        "pnl_pct": pnl_pct,
        "target_weight": float(asset.get("target_weight", 0) or 0),
        "risk_budget": float(asset.get("risk_budget", 100_000) or 100_000),
        "price_source": price_payload.get("source"),
        "fetched_at": price_payload.get("fetched_at"),
        "hist": hist,
        "trade_plan": trade_plan,
    }


# =========================================================
# UI 렌더링
# =========================================================
def render_score_bar(title, score):
    score = clip_score(score)

    st.markdown(
        f"""
        <div class="score-wrap">
            <div class="score-title">{title}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{score}%;"></div></div>
            <div style="margin-top:8px; color:#64748b;">{score}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_html(title, value, sub_text=None, positive=True):
    sub_class = "metric-sub-green" if positive else "metric-sub-red"

    if sub_text is None:
        sub_text = ""

    return f"""
    <div>
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="{sub_class}">{sub_text}</div>' if sub_text else ''}
    </div>
    """


def render_detail(selected):
    action = selected["action"]
    cls = action_class(action)
    price_str = format_price(selected["price"], selected["ticker"])
    signal = selected["signal"]

    st.markdown(
        f"""
        <div class="main-card">
            <div style="display:flex; justify-content:space-between; gap:20px; align-items:flex-start;">
                <div>
                    <div class="small-label">{selected['asset']} / {selected['ticker']} / Benchmark: {selected['benchmark']}</div>
                    <div class="big-title {cls}">{action}</div>
                    <span class="badge-red">Signal Score {signal}/100</span>
                    <span class="badge-blue">{macro_label(selected['macro_fit'])}</span>
                    <span class="badge-gray">Price source: {selected['price_source']}</span>
                </div>
                <div style="text-align:right; min-width:220px;">
                    <div class="small-label">Current Price</div>
                    <div style="font-size:26px; font-weight:900;">{price_str}</div>
                    <div style="font-size:12px; color:#6b7280; margin-top:6px;">Fetched: {selected['fetched_at']}</div>
                </div>
            </div>
            <div class="summary-box">
                <b>{build_summary(action, selected['risk_quality'])}</b><br>
                <b>주의:</b> 이 점수는 매수 확률이 아니라 기술적·상대강도·리스크·뉴스·매크로를 합친 의사결정 보조 점수입니다.<br>
                <b>Risk Quality:</b> 높을수록 안전성 우수, 낮을수록 변동성/낙폭 부담이 크다는 뜻입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        render_score_bar("Technical", selected["technical"])

    with c2:
        render_score_bar("Relative", selected["relative"])

    with c3:
        render_score_bar("Risk Quality", selected["risk_quality"])

    with c4:
        render_score_bar("News/Event", selected["news_event"])

    with c5:
        render_score_bar("Macro Fit", selected["macro_fit"])

    with st.expander("점수 해석 보기", expanded=False):
        rows = []

        for name, key in [
            ("Technical", "technical"),
            ("Relative", "relative"),
            ("Risk Quality", "risk_quality"),
            ("News/Event", "news_event"),
            ("Macro Fit", "macro_fit"),
        ]:
            rows.append(
                {
                    "Factor": name,
                    "Score": selected[key],
                    "Meaning": explain_factor(name, selected[key]),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Portfolio & Trade Plan")

    total_value = selected["market_value"]
    current_weight = 0.0
    tp = selected["trade_plan"]

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            metric_html(
                "보유 평가금액",
                format_price(total_value, selected["ticker"]),
                f"{selected['target_weight'] * 100:.2f}% target",
                True,
            ),
            unsafe_allow_html=True,
        )

    with p2:
        pnl_positive = selected["pnl_pct"] >= 0
        st.markdown(
            metric_html(
                "평가손익",
                pct_text(selected["pnl_pct"]),
                "평단 미입력" if selected["avg_price"] == 0 else "vs avg price",
                pnl_positive,
            ),
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            metric_html(
                "현재 비중",
                f"{current_weight:.2f}%",
                f"Target {selected['target_weight'] * 100:.1f}%",
                True,
            ),
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            metric_html(
                "추천 추가 수량",
                f"{tp['recommended_qty']:,}",
                f"Risk budget {format_price(selected['risk_budget'], selected['ticker'])}",
                True,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Entry / Exit Plan")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.markdown(
            metric_html("현재가", price_str),
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            metric_html(
                "손절가",
                format_price(tp["stop_price"], selected["ticker"]),
                pct_text(tp["stop_pct"]),
                False,
            ),
            unsafe_allow_html=True,
        )

    with e3:
        st.markdown(
            metric_html(
                "목표가",
                format_price(tp["target_price"], selected["ticker"]),
                f"+{tp['target_pct']:.2f}%" if tp["target_pct"] is not None else "-",
                True,
            ),
            unsafe_allow_html=True,
        )

    with e4:
        st.markdown(
            metric_html(
                "손익비",
                f"{tp['rr']:.2f}x",
                "1.5x 이상 선호",
                True,
            ),
            unsafe_allow_html=True,
        )


# =========================================================
# 전체 대시보드
# =========================================================
def render_dashboard_body():
    # 이 값이 1분마다 바뀌면서 가격 함수 캐시 키도 바뀝니다.
    # 따라서 가격 데이터는 1분 단위로 새로 가져옵니다.
    minute_bucket = int(time.time() // PRICE_FETCH_EVERY_SECONDS)

    analyses = [analyze_asset(asset, minute_bucket) for asset in WATCHLIST]
    total_mv = sum(x["market_value"] for x in analyses)

    st.title("Portfolio Risk Dashboard")

    st.caption(
        "가격 데이터는 1분 단위로 새로 가져오고, 판단 점수는 최신 가격을 마지막 Close에 반영해 계산합니다. "
        "전체 페이지 autorefresh는 사용하지 않습니다."
    )

    st.markdown("---")
    st.subheader("Portfolio Snapshot")

    snapshot_rows = []

    for x in analyses:
        weight = (x["market_value"] / total_mv * 100) if total_mv > 0 else 0

        snapshot_rows.append(
            {
                "Asset": x["asset"],
                "Ticker": x["ticker"],
                "Action": x["action"],
                "Signal Score": x["signal"],
                "Technical": x["technical"],
                "Relative": x["relative"],
                "Risk Quality": x["risk_quality"],
                "News/Event": x["news_event"],
                "Macro Fit": x["macro_fit"],
                "Price": x["price"],
                "Qty": x["qty"],
                "Market Value": x["market_value"],
                "P&L %": x["pnl_pct"],
                "Weight %": weight,
                "Price Source": x["price_source"],
            }
        )

    snap_df = pd.DataFrame(snapshot_rows)

    st.dataframe(
        snap_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal Score": st.column_config.ProgressColumn(
                "Signal Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Technical": st.column_config.ProgressColumn(
                "Technical",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Relative": st.column_config.ProgressColumn(
                "Relative",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Risk Quality": st.column_config.ProgressColumn(
                "Risk Quality",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "News/Event": st.column_config.ProgressColumn(
                "News/Event",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Macro Fit": st.column_config.ProgressColumn(
                "Macro Fit",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Price": st.column_config.NumberColumn("Price", format="%.2f"),
            "Market Value": st.column_config.NumberColumn("Market Value", format="%.2f"),
            "P&L %": st.column_config.NumberColumn("P&L %", format="%.2f"),
            "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f"),
        },
    )

    st.markdown("---")

    selected_name = st.selectbox(
        "상세 종목 선택",
        options=[x["asset"] for x in analyses],
        index=0,
        key="selected_asset_name",
    )

    selected = next(x for x in analyses if x["asset"] == selected_name)

    render_detail(selected)


# Streamlit 1.37+에서는 fragment만 60초마다 다시 실행됩니다.
# 전체 페이지를 강제로 새로고침하는 st_autorefresh는 쓰지 않습니다.
if hasattr(st, "fragment"):
    render_dashboard = st.fragment(run_every=f"{PRICE_FETCH_EVERY_SECONDS}s")(render_dashboard_body)
else:
    render_dashboard = render_dashboard_body
    st.warning(
        "현재 Streamlit 버전에는 st.fragment가 없어 자동 1분 데이터 재조회가 제한됩니다. "
        "터미널에서 `pip install --upgrade streamlit` 후 다시 실행하세요."
    )

render_dashboard()
