import time
import datetime
import logging

import numpy as np
import pandas as pd
import yfinance as yf
try:
    import pkg_resources
except ImportError:
    import importlib.metadata, types, sys
    _pkg = types.ModuleType("pkg_resources")
    def _get_dist(name):
        ns = types.SimpleNamespace()
        try:
            ns.version = importlib.metadata.version(name)
        except Exception:
            ns.version = "0.0.0"
        return ns
    def _resource_filename(package, resource):
        import importlib.util, os
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            return os.path.join(os.path.dirname(spec.origin), resource)
        return resource
    _pkg.get_distribution = _get_dist
    _pkg.resource_filename = _resource_filename
    sys.modules["pkg_resources"] = _pkg
from pykrx import stock as krx

from analysis import calc_cn, calc_indicators, stage_meaning

logger = logging.getLogger(__name__)

RECENTLY_CHANGED_DAYS = 7

# ─── Ticker 정의 ────────────────────────────────────────────────────────────

global_tickers = {
    "SPY":     "S&P500",
    "QQQ":     "NASDAQ",
    "IWM":     "RUSSELL2000",
    "EEM":     "EEM",
    "EWJ":     "JAPAN",
    "FXI":     "CHINA",
    "GLD":     "GOLD",
    "TLT":     "TLT",
    "USO":     "OIL",
    "BTC-USD": "BTC",
}

kospi_global = {"069500": "KOSPI"}

etf_tickers = {
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "091180": "KODEX 자동차",
    "102960": "KODEX 기계장비",
    "102970": "KODEX 증권",
    "117460": "KODEX 에너지화학",
    "117680": "KODEX 철강",
    "117700": "KODEX 건설",
    "122630": "KODEX 레버리지",
    "139220": "TIGER 200 건설",
    "139240": "TIGER 200 철강소재",
    "139250": "TIGER 200 에너지화학",
    "139270": "TIGER 200 금융",
    "140700": "KODEX 보험",
    "140710": "KODEX 운송",
    "147970": "TIGER 모멘텀",
    "157490": "TIGER 소프트웨어",
    "161510": "PLUS 고배당주",
    "227540": "TIGER 200 헬스케어",
    "228790": "TIGER 화장품",
    "228800": "TIGER 여행레저",
    "228810": "TIGER 미디어컨텐츠",
    "229200": "KODEX 코스닥150",
    "244580": "KODEX 바이오",
    "261070": "TIGER 코스닥150바이",
    "266360": "KODEX K콘텐츠",
    "266370": "KODEX IT",
    "266390": "KODEX 경기소비재",
    "266410": "KODEX 필수소비재",
    "266420": "KODEX 헬스케어",
    "300950": "KODEX 게임산업",
    "305720": "KODEX 2차전지산업",
    "307520": "TIGER 지주회사",
    "364980": "TIGER 2차전지TOP10",
    "364990": "TIGER 게임TOP10",
    "365000": "TIGER 인터넷TOP10",
}

kospi_tickers = {
    "000240": "한국앤컴퍼니",
    "000270": "기아",
    "000810": "삼성화재",
    "001120": "LX인터내셔널",
    "001270": "부국증권",
    "001430": "세아베스틸지주",
    "002310": "아세아제지",
    "003540": "대신증권",
    "003550": "LG",
    "003690": "코리안리",
    "004980": "성신양회",
    "005380": "현대차",
    "005830": "DB손해보험",
    "005940": "NH투자증권",
    "007340": "DN오토모티브",
    "009970": "영원무역홀딩스",
    "011780": "금호석유화학",
    "016360": "삼성증권",
    "017670": "SK텔레콤",
    "017800": "현대엘리베이터",
    "024110": "기업은행",
    "029780": "삼성카드",
    "030000": "제일기획",
    "030200": "KT",
    "032640": "LG유플러스",
    "032830": "삼성생명",
    "033780": "KT&G",
    "035250": "강원랜드",
    "036460": "한국가스공사",
    "055550": "신한지주",
    "071050": "한국금융지주",
    "078930": "GS",
    "086790": "하나금융지주",
    "105560": "KB금융",
    "138930": "BNK금융지주",
    "139130": "iM금융지주",
    "161390": "한국타이어앤테크놀로지",
    "175330": "JB금융지주",
    "183190": "아세아시멘트",
    "192080": "더블유게임즈",
    "214320": "이노션",
    "316140": "우리금융지주",
}

global_names_ordered = [
    "S&P500", "NASDAQ", "RUSSELL2000",
    "EEM", "KOSPI", "JAPAN", "CHINA",
    "GOLD", "TLT", "OIL", "BTC",
]

# ─── 공통 결과 추출 ──────────────────────────────────────────────────────────

def _build_result(name: str, df: pd.DataFrame) -> dict | None:
    today = datetime.datetime.today()
    five_years_ago = today - datetime.timedelta(days=5 * 365)
    one_year_ago   = today - datetime.timedelta(days=365)

    data5 = df[df['Date'] >= five_years_ago]
    data1 = df[df['Date'] >= one_year_ago]

    if data5.empty:
        return None

    # 1년 CN 상관계수
    corr = None
    if len(data1) > 10:
        try:
            c = data1[['CN', 'Close']].corr().iloc[0, 1]
            corr = None if np.isnan(c) else round(c, 2)
        except Exception:
            pass

    current           = data5.iloc[-1]
    stage             = current['stage']
    stage_change_date = current['stage_change_date']

    if stage_change_date is not None and pd.notna(stage_change_date):
        elapsed_days = (today - stage_change_date).days
    else:
        elapsed_days = None

    recently_changed = elapsed_days is not None and elapsed_days <= RECENTLY_CHANGED_DAYS

    return {
        "Asset":             name,
        "Price":             int(current['Close']),
        "CN":                round(current['CN'], 2),
        "투자비중":            round(100 - current['CN'], 2),
        "1Y Corr":           corr,
        "stage":             stage,
        "recently_changed":  recently_changed,
        "stage_change_date": stage_change_date,
        "elapsed_days":      elapsed_days,
        "_chart_data":       data5,
    }


# ─── 글로벌 데이터 로딩 ──────────────────────────────────────────────────────

def _fetch_global(ticker: str, name: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            dates = pd.to_datetime(df['Date'])
            df['Date'] = dates.dt.tz_convert(None) if dates.dt.tz is not None else dates
            df = df[['Date', 'Close', 'High', 'Low', 'Volume']].dropna()
            df = df.sort_values('Date').reset_index(drop=True)
            df = calc_cn(df)
            df = calc_indicators(df)
            if df is None:
                return None
            return _build_result(name, df)
        except Exception as e:
            logger.warning(f"[{name}] 시도 {attempt+1}/{retries} 실패: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─── 한국 데이터 로딩 ────────────────────────────────────────────────────────

def _fetch_korean(ticker: str, name: str, start: str, end: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            raw = krx.get_market_ohlcv(start, end, ticker)
            if raw.empty:
                return None

            raw = raw.reset_index()

            col_map = {}
            for col in raw.columns:
                s = str(col)
                if '날짜' in s or s == 'Date':
                    col_map[col] = 'Date'
                elif '고가' in s:
                    col_map[col] = 'High'
                elif '저가' in s:
                    col_map[col] = 'Low'
                elif '종가' in s:
                    col_map[col] = 'Close'
                elif '거래량' in s:
                    col_map[col] = 'Volume'
            raw = raw.rename(columns=col_map)

            if 'Date' not in raw.columns and raw.index.name in ['날짜', 'Date']:
                raw = raw.rename_axis('Date').reset_index()

            raw['Date'] = pd.to_datetime(raw['Date'])
            needed = [c for c in ['Date', 'Close', 'High', 'Low', 'Volume'] if c in raw.columns]
            if len(needed) < 5:
                return None

            raw = raw[needed].dropna()
            raw = raw.sort_values('Date').reset_index(drop=True)

            df = calc_cn(raw)
            df = calc_indicators(df)
            if df is None:
                return None
            return _build_result(name, df)

        except Exception as e:
            logger.warning(f"[{name}] 시도 {attempt+1}/{retries} 실패: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ─── 전체 로딩 ───────────────────────────────────────────────────────────────

def load_all_data(progress_callback=None):
    """
    results: list of dict
    charts:  dict { name -> DataFrame (5년치) }
    """
    today = datetime.datetime.today()
    pykrx_start = (today - datetime.timedelta(days=10 * 365)).strftime("%Y%m%d")
    pykrx_end   = today.strftime("%Y%m%d")

    results = []
    charts  = {}

    all_tickers = (
        list(global_tickers.items())
        + list(kospi_global.items())
        + list(etf_tickers.items())
        + list(kospi_tickers.items())
    )
    total = len(all_tickers)

    for idx, (ticker, name) in enumerate(all_tickers):
        if progress_callback:
            progress_callback(idx, total, name)

        if ticker in global_tickers:
            row = _fetch_global(ticker, name)
        else:
            row = _fetch_korean(ticker, name, pykrx_start, pykrx_end)

        if row:
            chart_data = row.pop("_chart_data")
            results.append(row)
            charts[name] = chart_data

    if progress_callback:
        progress_callback(total, total, "완료")

    return results, charts
