import pandas as pd
import numpy as np

stage_meaning = {
    "1단계 베이스": "매집 적기",
    "2단계 상승":   "상승 진행 중, 유지",
    "3단계 천장":   "비중 축소",
    "4단계 하락":   "관망",
}

STAGE_COLORS = {
    "1단계 베이스": "#2196F3",
    "2단계 상승":   "#4CAF50",
    "3단계 천장":   "#FF9800",
    "4단계 하락":   "#F44336",
}

MIN_ROWS = 150


def calc_cn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['거래대금'] = df['Volume'] * df['Close']
    df['전일종가'] = df['Close'].shift(1)
    df['평균회귀'] = np.where(
        df['Close'].values > df['전일종가'].values,
        -df['거래대금'].values,
        df['거래대금'].values,
    )
    df['평균회귀합'] = df['평균회귀'].cumsum()
    df['최고저점']  = df['평균회귀합'].cummin()
    df['매집수량']  = df['평균회귀합'] - df['최고저점']
    df['매집고점']  = df['매집수량'].cummax()
    df['CN'] = np.where(
        df['매집고점'] == 0,
        0,
        (1 - (df['매집수량'] / df['매집고점'])) * 100,
    )
    return df


def calc_weinstein_stages(df: pd.DataFrame):
    close_arr  = df['Close'].values
    ma150_arr  = df['MA150'].values
    nh100_arr  = df['신고가100'].values
    nl100_arr  = df['신저가100'].values
    nh50_arr   = df['신고가50'].values
    nl50_arr   = df['신저가50'].values
    dates_arr  = df['Date'].values

    n = len(df)
    stages             = [None] * n
    stage_change_dates = [None] * n

    for i in range(n):
        if np.isnan(ma150_arr[i]):
            stages[i]             = None
            stage_change_dates[i] = None
            continue

        close = close_arr[i]
        ma150 = ma150_arr[i]
        nh100 = nh100_arr[i]
        nl100 = nl100_arr[i]
        nh50  = nh50_arr[i]
        nl50  = nl50_arr[i]
        prev  = stages[i - 1] if i > 0 else None

        if close > ma150 and nh100:
            new_stage = "2단계 상승"
        elif close < ma150 and nl100:
            new_stage = "4단계 하락"
        elif prev == "2단계 상승" and (close < ma150 or nl50):
            new_stage = "3단계 천장"
        elif prev == "4단계 하락" and (close > ma150 or nh50):
            new_stage = "1단계 베이스"
        else:
            new_stage = prev  # None until first valid stage

        stages[i] = new_stage

        if i > 0 and prev and new_stage != prev:
            stage_change_dates[i] = pd.Timestamp(dates_arr[i])
        else:
            stage_change_dates[i] = stage_change_dates[i - 1] if i > 0 else None

    return stages, stage_change_dates


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    if len(df) < MIN_ROWS:
        return None
    df = df.copy()
    df['MA150']    = df['Close'].rolling(150).mean()
    df['고가99']   = df['High'].shift(1).rolling(99).max()
    df['저가99']   = df['Low'].shift(1).rolling(99).min()
    df['고가49']   = df['High'].shift(1).rolling(49).max()
    df['저가49']   = df['Low'].shift(1).rolling(49).min()
    df['신고가100'] = (df['High'] > df['고가99']).fillna(False)
    df['신저가100'] = (df['Low']  < df['저가99']).fillna(False)
    df['신고가50']  = (df['High'] > df['고가49']).fillna(False)
    df['신저가50']  = (df['Low']  < df['저가49']).fillna(False)
    df['stage'], df['stage_change_date'] = calc_weinstein_stages(df)
    return df
