import os
import json
import re
import math
from collections import Counter
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# =========================
# 固定仕様（変更禁止）
# =========================
START_ROW = 14
MAX_ROW = 2000

BENCHMARK_TICKERS = ["1308.T", "1305.T", "1348.T"]

# rebalance固定パラメータ
MAX_SINGLE_WEIGHT = 0.15
TARGET_SINGLE_WEIGHT = 0.08

# forecast clip
RET_CLIP_MIN = -0.30
RET_CLIP_MAX = 0.30

# health閾値
DD_EXIT = -0.30
DD_CAUTION = -0.20

# risk analytics parameters
TRADING_DAYS_PER_YEAR = 252
RISK_MIN_OBSERVATIONS = 126
RISK_MIN_COVERAGE = 0.80
RISK_FFILL_LIMIT = 3

# output ranges
DASH_RANGE = "A1:X11"
OUT_RANGE = f"E{START_ROW}:T{MAX_ROW}"  # E..P 既存出力 + Q..T 初回観測系4列

# observation columns (Q:T)
OBS_HEADER_RANGE = "Q13:T13"
OBS_HEADERS = ["初回観測日", "初回観測数量", "観測保有年月数", "3か月判定"]

# history ranges (AU:CG)
HIST_START_COL = "AU"
HIST_END_COL = "CG"
HIST_HEADER_RANGE = f"{HIST_START_COL}1:{HIST_END_COL}1"
HIST_DATA_RANGE = f"{HIST_START_COL}2:{HIST_END_COL}"

HIST_HEADERS = [
    "snapshot_date",                              # AU 実行日
    "updated_at_jst",                             # AV
    "total_market_value_jpy",                     # AW
    "total_cost_jpy",                             # AX
    "total_pnl_jpy",                              # AY
    "total_pnl_pct",                              # AZ
    "symbol_count",                               # BA
    "top1_weight",                                # BB
    "top3_weight",                                # BC
    "hhi",                                        # BD
    "exit_count",                                 # BE
    "caution_count",                              # BF
    "ok_count",                                   # BG
    "pf_ret_base",                                # BH
    "pf_ret_opt",                                 # BI
    "pf_ret_pess",                                # BJ
    "value_10y_base_jpy",                         # BK
    "value_10y_opt_jpy",                          # BL
    "value_10y_pess_jpy",                         # BM
    "unallocated_cash_jpy",                       # BN
    "pf_cum_1y",                                  # BO
    "topix_cum_1y",                               # BP
    "alpha_1y",                                   # BQ
    "portfolio_vol_annual",                       # BR
    "top1_risk_contributor_ticker",               # BS
    "top1_risk_contribution_pct",                 # BT
    "top3_risk_contribution_pct",                 # BU
    "risk_hhi",                                   # BV
    "effective_risk_symbol_count",                # BW
    "beta_to_topix",                              # BX
    "tracking_error_1y",                          # BY
    "information_ratio_1y",                       # BZ
    "downside_capture_1y",                        # CA
    "max_drawdown_1y",                            # CB
    "stress_topix_minus10_loss_pct",              # CC
    "stress_topix_minus20_loss_pct",              # CD
    "stress_top1_risk_minus20_loss_pct",          # CE
    "stress_top3_risk_minus20_loss_pct",          # CF
    "stress_topix_minus10_top3_minus15_loss_pct", # CG
]
HIST_COLS = len(HIST_HEADERS)

# observation DB ranges (CH:CU)
OBS_DB_START_COL = "CH"
OBS_DB_END_COL = "CU"
OBS_DB_HEADER_RANGE = f"{OBS_DB_START_COL}13:{OBS_DB_END_COL}13"
OBS_DB_DATA_RANGE = f"{OBS_DB_START_COL}14:{OBS_DB_END_COL}{MAX_ROW}"

OBS_DB_HEADERS = [
    "position_key",        # CH ticker + custody_type
    "ticker",              # CI
    "custody_type",        # CJ TOKUTEI / NISA
    "code",                # CK
    "name",                # CL
    "first_seen_date",     # CM
    "first_seen_qty",      # CN
    "first_seen_cost",     # CO
    "last_seen_date",      # CP
    "last_seen_qty",       # CQ
    "last_seen_cost",      # CR
    "active",              # CS
    "inactive_since",      # CT
    "last_updated_at_jst", # CU
]
OBS_DB_COLS = len(OBS_DB_HEADERS)

# dividend yield cache ranges (CV:CY)
# 価格系は毎回更新し、配当利回りは7日キャッシュでAPI負荷を抑える。
DIVIDEND_CACHE_TTL_DAYS = 7
DIVIDEND_OUTPUT_HEADER_RANGE = "O13"
DIVIDEND_CACHE_START_COL = "CV"
DIVIDEND_CACHE_END_COL = "CY"
DIVIDEND_CACHE_HEADER_RANGE = f"{DIVIDEND_CACHE_START_COL}13:{DIVIDEND_CACHE_END_COL}13"
DIVIDEND_CACHE_DATA_RANGE = f"{DIVIDEND_CACHE_START_COL}14:{DIVIDEND_CACHE_END_COL}{MAX_ROW}"
DIVIDEND_CACHE_HEADERS = [
    "ticker",
    "dividend_yield",
    "fetched_at_jst",
    "fetch_status",
]
DIVIDEND_CACHE_COLS = len(DIVIDEND_CACHE_HEADERS)

CUSTODY_TOKUTEI = "TOKUTEI"
CUSTODY_NISA = "NISA"

# =========================
# Utility
# =========================
def get_gspread_client():
    app_config_str = os.environ.get("APP_CONFIG")
    if not app_config_str:
        raise ValueError("APP_CONFIG is not set.")
    app_config = json.loads(app_config_str)
    sa_info = app_config.get("gcp_key")
    if not sa_info:
        raise ValueError("gcp_key is missing in APP_CONFIG.")
        
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def parse_ticker(code: str):
    code = str(code).strip().upper()

    # JPXの新証券コード体系に対応
    # 例: 8053 -> 8053.T / 456A -> 456A.T / 456A.T -> 456A.T
    if re.match(r"^[0-9A-Z]{4}$", code):
        return f"{code}.T"
    if re.match(r"^[0-9A-Z]{4}\.T$", code):
        return code
    return None


def parse_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def calc_cagr(close_series: pd.Series):
    """
    仕様固定：
      years = (date_last - date_first).days / 365.25
      CAGR = (p1/p0)^(1/years) - 1
    """
    if close_series is None:
        return None
    s = close_series.dropna()
    if len(s) < 2:
        return None
    p0 = float(s.iloc[0])
    p1 = float(s.iloc[-1])
    if p0 <= 0:
        return None
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return None
    return (p1 / p0) ** (1 / years) - 1


def extract_close_df(dl: pd.DataFrame, tickers: list):
    """
    yf.download の返り値を Close の DataFrame（列=ticker）に正規化する。
    """
    if dl is None or dl.empty:
        return pd.DataFrame()

    if isinstance(dl.columns, pd.MultiIndex):
        # パターン1：level0 に Close がある（一般的）
        if "Close" in dl.columns.get_level_values(0):
            close = dl["Close"].copy()
            # close columns: tickers
            return close
        # パターン2：level1 に Close がある
        if "Close" in dl.columns.get_level_values(1):
            close = dl.xs("Close", level=1, axis=1).copy()
            return close
        return pd.DataFrame()
    else:
        # 単一ティッカー
        if "Close" not in dl.columns:
            return pd.DataFrame()
        close = dl[["Close"]].copy()
        close.columns = [tickers[0]]
        return close


def extract_price_df(dl: pd.DataFrame, tickers: list, field: str, fallback_field: str = None):
    """
    yf.download の返り値から指定フィールドの DataFrame（列=ticker）を返す。
    Adj Close が取れない場合は fallback_field にフォールバックする。
    """
    if dl is None or dl.empty:
        return pd.DataFrame()

    if isinstance(dl.columns, pd.MultiIndex):
        if field in dl.columns.get_level_values(0):
            return dl[field].copy()
        if field in dl.columns.get_level_values(1):
            return dl.xs(field, level=1, axis=1).copy()
        if fallback_field:
            return extract_price_df(dl, tickers, fallback_field, None)
        return pd.DataFrame()

    if field in dl.columns:
        price = dl[[field]].copy()
        price.columns = [tickers[0]]
        return price
    if fallback_field and fallback_field in dl.columns:
        price = dl[[fallback_field]].copy()
        price.columns = [tickers[0]]
        return price
    return pd.DataFrame()


def build_dashboard_matrix():
    # 11 rows x 24 cols (A..X)
    return [["" for _ in range(24)] for __ in range(11)]


def set_cell(mat, a1: str, value):
    """
    A1形式（A1〜X11）だけを対象にセットする。
    """
    match = re.match(r"^([A-X])(\d+)$", str(a1).strip().upper())
    if not match:
        raise ValueError(f"Invalid dashboard cell: {a1}")

    col_letter = match.group(1)
    row_num = int(match.group(2))
    c = ord(col_letter) - ord("A")
    r = row_num - 1

    if not (0 <= r < 11 and 0 <= c < 24):
        raise ValueError(f"Cell out of dashboard range: {a1}")

    mat[r][c] = value


def pad_matrix_rows(rows, total_rows, col_count):
    """
    固定範囲の全件上書き用に、行数・列数を固定する。
    既存データを消すためのclearは使わない。
    """
    if len(rows) > total_rows:
        raise ValueError(f"FATAL:MATRIX_CAPACITY_EXCEEDED:{len(rows)}>{total_rows}")

    out = []
    for row in rows:
        normalized = list(row[:col_count])
        if len(normalized) < col_count:
            normalized += [""] * (col_count - len(normalized))
        out.append(normalized)

    out.extend([[""] * col_count for _ in range(total_rows - len(out))])
    return out


def display_name(name, ticker):
    name_value = "" if name is None else str(name).strip()
    ticker_value = "" if ticker is None else str(ticker).strip()
    return name_value if name_value else ticker_value


def clip(x, lo, hi):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(lo, min(hi, float(x)))


def normalize_hist_row(row, size):
    row = list(row[:size])
    if len(row) < size:
        row += [""] * (size - len(row))
    return row


def normalize_yyyymmdd(value):
    """
    yyyymmdd / yyyy-mm-dd / yyyy/mm/dd を yyyymmdd に正規化する。
    不正値は空文字にする。
    """
    s = "" if value is None else str(value).strip()
    if s == "":
        return ""
    digits = re.sub(r"\D", "", s)
    if len(digits) != 8:
        return ""
    try:
        datetime.strptime(digits, "%Y%m%d")
        return digits
    except Exception:
        return ""


def format_qty(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        v = float(value)
        if v.is_integer():
            return int(v)
        return v
    except Exception:
        return value


def calc_observed_months(first_seen_yyyymmdd: str, now_jst: datetime):
    first_seen_yyyymmdd = normalize_yyyymmdd(first_seen_yyyymmdd)
    if first_seen_yyyymmdd == "":
        return None
    try:
        d0 = datetime.strptime(first_seen_yyyymmdd, "%Y%m%d").date()
        d1 = now_jst.date()
        months = (d1.year - d0.year) * 12 + (d1.month - d0.month)
        if d1.day < d0.day:
            months -= 1
        return max(0, int(months))
    except Exception:
        return None


def format_observed_ym(observed_months):
    if observed_months is None:
        return "未観測"
    y = int(observed_months) // 12
    m = int(observed_months) % 12
    return f"{y}年{m}か月"


def calc_3mo_judgement(observed_months, pnl_pct):
    """
    3か月単位で、不調 / 普通 / 好調 を判定する。
    - 不調: 損益率 0%未満
    - 好調: 年率10%相当の3か月換算ライン以上
    - 普通: 0%以上、好調ライン未満
    """
    if observed_months is None:
        return "判定不能"
    if pnl_pct is None or pnl_pct == "":
        return "判定不能"

    try:
        m = max(0, int(observed_months))
        pnl_pct_value = float(pnl_pct) * 100.0
    except Exception:
        return "判定不能"

    if m <= 3:
        lower_month = 0
        upper_month = 3
    else:
        upper_month = int(math.ceil(m / 3.0) * 3)
        lower_month = upper_month - 3

    good_line = upper_month / 12.0 * 10.0

    if pnl_pct_value < 0:
        label = "不調"
    elif pnl_pct_value >= good_line:
        label = "好調"
    else:
        label = "普通"

    return f"{lower_month}〜{upper_month}か月：{label}"


def calc_valid_1y_return(series, min_ret=-0.5, max_ret=1.0):
    """
    1年リターンを計算する。
    分割未調整などで極端な値になった場合は None を返す。
    """
    if series is None:
        return None
    try:
        s = series.dropna()
        if len(s) < 2:
            return None
        p0 = float(s.iloc[0])
        p1 = float(s.iloc[-1])
        if p0 <= 0 or p1 <= 0:
            return None
        ret = (p1 / p0) - 1
        if not (min_ret < ret < max_ret):
            return None
        return float(ret)
    except Exception:
        return None


def pick_benchmark_return(return_df: pd.DataFrame, benchmark_tickers: list):
    """
    TOPIX系ETFの候補から、優先順に正常な1年リターンが取れた最初の銘柄を採用する。
    複数銘柄の平均は取らない。
    """
    if return_df is None or return_df.empty:
        return None, None

    for tk in benchmark_tickers:
        if tk not in return_df.columns:
            continue
        ret = calc_valid_1y_return(return_df[tk])
        if ret is not None:
            return tk, ret

    return None, None


def fetch_single_benchmark_return(ticker: str):
    """
    一括取得で取れなかった場合の保険。
    auto_adjust=True のCloseで1年リターンを計算する。
    """
    try:
        hist = yf.Ticker(ticker).history(
            period="1y",
            interval="1d",
            auto_adjust=True,
        )
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return calc_valid_1y_return(hist["Close"])
    except Exception:
        return None


def fetch_current_price_fast(ticker: str):
    """
    E列の現在値用。
    fast_info.last_price のみ取得する。
    失敗・不正値なら None を返し、呼び出し側で既存 close_5d fallback に落とす。
    """
    try:
        fi = yf.Ticker(ticker).fast_info

        try:
            value = fi.get("last_price")
        except Exception:
            value = fi["last_price"]

        if value is None:
            return None

        value = float(value)
        if math.isnan(value) or value <= 0:
            return None

        return value

    except Exception:
        return None



def empty_portfolio_analytics(status="DATA_NG:NOT_CALCULATED"):
    return {
        "status": status,
        "coverage": 0.0,
        "calc_symbol_count": 0,
        "portfolio_vol_annual": "",
        "top1_risk_contributor_ticker": "",
        "top1_risk_contributor_name": "",
        "top1_risk_contribution_pct": "",
        "top3_risk_contribution_pct": "",
        "risk_hhi": "",
        "effective_risk_symbol_count": "",
        "money_top1_difference": "",
        "correlation_concentration_memo": "",
        "beta_to_topix": "",
        "tracking_error_1y": "",
        "information_ratio_1y": "",
        "downside_capture_1y": "",
        "max_drawdown_1y": "",
        "stress_topix_minus10_loss_pct": "",
        "stress_topix_minus20_loss_pct": "",
        "stress_top1_risk_minus20_loss_pct": "",
        "stress_top3_risk_minus20_loss_pct": "",
        "stress_topix_minus10_top3_minus15_loss_pct": "",
        "top_entries": [],
    }


def calculate_portfolio_analytics(valid_price_rows, adjusted_close_df, selected_benchmark_ticker):
    """
    既存のauto_adjust=True・1年日足だけを使い、追加APIなしで計算する。
    特定/NISAの重複保有はticker単位で集約する。
    新規分析が失敗しても既存処理は止めず、空欄とstatusを返す。
    """
    result = empty_portfolio_analytics()

    if adjusted_close_df is None or adjusted_close_df.empty:
        result["status"] = "DATA_NG:NO_ADJUSTED_PRICE"
        return result
    if not selected_benchmark_ticker or selected_benchmark_ticker not in adjusted_close_df.columns:
        result["status"] = "DATA_NG:BENCHMARK_MISSING"
        return result

    positions = {}
    for row in valid_price_rows:
        ticker = str(row.get("ticker", "")).strip()
        if ticker == "":
            continue
        market_value = parse_float(row.get("market_value"))
        if market_value is None or market_value <= 0:
            continue

        entry = positions.setdefault(
            ticker,
            {
                "ticker": ticker,
                "name": "",
                "market_value": 0.0,
            },
        )
        entry["market_value"] += float(market_value)
        if entry["name"] == "":
            entry["name"] = str(row.get("name", "")).strip()

    total_market_value = sum(x["market_value"] for x in positions.values())
    if total_market_value <= 0:
        result["status"] = "DATA_NG:NO_VALID_POSITION"
        return result

    for entry in positions.values():
        entry["weight"] = entry["market_value"] / total_market_value

    benchmark_prices = adjusted_close_df[selected_benchmark_ticker].dropna()
    if len(benchmark_prices) < RISK_MIN_OBSERVATIONS + 1:
        result["status"] = "DATA_NG:BENCHMARK_DATA_SHORT"
        return result

    benchmark_index = benchmark_prices.index
    benchmark_returns = benchmark_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    ticker_returns = {}
    for ticker, entry in positions.items():
        if ticker not in adjusted_close_df.columns:
            continue

        prices = adjusted_close_df[ticker].reindex(benchmark_index)
        first_valid = prices.first_valid_index()
        last_valid = prices.last_valid_index()
        if first_valid is None or last_valid is None:
            continue

        filled = prices.copy()
        filled.loc[first_valid:last_valid] = filled.loc[first_valid:last_valid].ffill(limit=RISK_FFILL_LIMIT)
        returns = filled.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

        if int(returns.notna().sum()) >= RISK_MIN_OBSERVATIONS:
            ticker_returns[ticker] = returns

    eligible_tickers = sorted(ticker_returns.keys())
    coverage = sum(positions[ticker]["weight"] for ticker in eligible_tickers)
    result["coverage"] = float(coverage)
    result["calc_symbol_count"] = len(eligible_tickers)

    if coverage < RISK_MIN_COVERAGE:
        result["status"] = "DATA_NG:COVERAGE_LT_80"
        return result
    if len(eligible_tickers) < 2:
        result["status"] = "DATA_NG:SYMBOL_LT_2"
        return result

    aligned = pd.DataFrame({selected_benchmark_ticker: benchmark_returns})
    for ticker in eligible_tickers:
        aligned[ticker] = ticker_returns[ticker]
    aligned = aligned.dropna(how="any")

    if len(aligned) < RISK_MIN_OBSERVATIONS:
        result["status"] = "DATA_NG:COMMON_DATA_SHORT"
        return result

    raw_weights = np.array([positions[ticker]["weight"] for ticker in eligible_tickers], dtype=float)
    raw_weight_sum = float(raw_weights.sum())
    if raw_weight_sum <= 0:
        result["status"] = "DATA_NG:WEIGHT_ZERO"
        return result
    weights = raw_weights / raw_weight_sum

    stock_returns = aligned[eligible_tickers].astype(float)
    benchmark_daily = aligned[selected_benchmark_ticker].astype(float)
    covariance = stock_returns.cov().to_numpy(dtype=float)

    if covariance.shape != (len(eligible_tickers), len(eligible_tickers)) or not np.isfinite(covariance).all():
        result["status"] = "DATA_NG:COVARIANCE_INVALID"
        return result

    portfolio_variance_daily = float(weights.T @ covariance @ weights)
    if not math.isfinite(portfolio_variance_daily) or portfolio_variance_daily <= 0:
        result["status"] = "DATA_NG:PORTFOLIO_VARIANCE"
        return result

    portfolio_vol_daily = math.sqrt(portfolio_variance_daily)
    portfolio_vol_annual = portfolio_vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    covariance_weight = covariance @ weights
    component_risk = weights * covariance_weight / portfolio_vol_daily
    absolute_component = np.abs(component_risk)
    absolute_component_sum = float(absolute_component.sum())
    if absolute_component_sum <= 0 or not math.isfinite(absolute_component_sum):
        result["status"] = "DATA_NG:RISK_CONTRIBUTION"
        return result
    risk_share = absolute_component / absolute_component_sum

    annual_vols = stock_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    top_entries = []
    for idx, ticker in enumerate(eligible_tickers):
        top_entries.append(
            {
                "ticker": ticker,
                "name": display_name(positions[ticker].get("name"), ticker),
                "weight": float(positions[ticker]["weight"]),
                "annual_vol": float(annual_vols[ticker]),
                "risk_contribution_pct": float(risk_share[idx]),
            }
        )
    top_entries.sort(key=lambda x: (-x["risk_contribution_pct"], x["ticker"]))

    risk_hhi = float(np.square(risk_share).sum())
    effective_risk_symbol_count = (1.0 / risk_hhi) if risk_hhi > 0 else ""
    top1 = top_entries[0]
    top3_risk_contribution = sum(x["risk_contribution_pct"] for x in top_entries[:3])

    money_top_ticker = max(positions.values(), key=lambda x: (x["weight"], x["ticker"]))["ticker"]
    money_top_difference = "一致" if money_top_ticker == top1["ticker"] else "不一致"

    correlation = stock_returns.corr().to_numpy(dtype=float)
    pair_weight_sum = 0.0
    weighted_correlation_sum = 0.0
    for i in range(len(eligible_tickers)):
        for j in range(i + 1, len(eligible_tickers)):
            pair_weight = float(weights[i] * weights[j])
            corr_value = float(correlation[i, j])
            if math.isfinite(corr_value):
                pair_weight_sum += pair_weight
                weighted_correlation_sum += pair_weight * corr_value
    weighted_average_correlation = (
        weighted_correlation_sum / pair_weight_sum if pair_weight_sum > 0 else 0.0
    )
    if weighted_average_correlation >= 0.60:
        correlation_memo = "高"
    elif weighted_average_correlation >= 0.30:
        correlation_memo = "中"
    else:
        correlation_memo = "低"

    portfolio_daily = stock_returns.to_numpy(dtype=float) @ weights
    portfolio_daily = pd.Series(portfolio_daily, index=aligned.index, dtype=float)

    benchmark_variance = float(benchmark_daily.var(ddof=1))
    beta_to_topix = ""
    if math.isfinite(benchmark_variance) and benchmark_variance > 0:
        beta_to_topix = float(portfolio_daily.cov(benchmark_daily) / benchmark_variance)

    active_daily = portfolio_daily - benchmark_daily
    tracking_error_1y = float(active_daily.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    information_ratio_1y = ""
    if math.isfinite(tracking_error_1y) and tracking_error_1y > 0:
        information_ratio_1y = float(
            active_daily.mean() * TRADING_DAYS_PER_YEAR / tracking_error_1y
        )

    downside_capture_1y = ""
    downside_mask = benchmark_daily < 0
    if bool(downside_mask.any()):
        portfolio_downside_return = float((1.0 + portfolio_daily[downside_mask]).prod() - 1.0)
        benchmark_downside_return = float((1.0 + benchmark_daily[downside_mask]).prod() - 1.0)
        if benchmark_downside_return < 0:
            downside_capture_1y = float(portfolio_downside_return / benchmark_downside_return)

    portfolio_index = (1.0 + portfolio_daily).cumprod()
    running_max = portfolio_index.cummax()
    drawdown = portfolio_index / running_max - 1.0
    max_drawdown_1y = float(drawdown.min()) if not drawdown.empty else ""

    top1_weight = float(top1["weight"])
    top3_weight = sum(float(x["weight"]) for x in top_entries[:3])

    stress_topix_minus10 = "" if beta_to_topix == "" else clip(float(beta_to_topix) * -0.10, -1.0, 1.0)
    stress_topix_minus20 = "" if beta_to_topix == "" else clip(float(beta_to_topix) * -0.20, -1.0, 1.0)
    stress_top1_minus20 = clip(top1_weight * -0.20, -1.0, 1.0)
    stress_top3_minus20 = clip(top3_weight * -0.20, -1.0, 1.0)
    stress_combined = ""
    if beta_to_topix != "":
        stress_combined = clip(float(beta_to_topix) * -0.10 + top3_weight * -0.15, -1.0, 1.0)

    result.update(
        {
            "status": f"OK:{len(eligible_tickers)}銘柄/{coverage:.1%}",
            "portfolio_vol_annual": float(portfolio_vol_annual),
            "top1_risk_contributor_ticker": top1["ticker"],
            "top1_risk_contributor_name": top1["name"],
            "top1_risk_contribution_pct": float(top1["risk_contribution_pct"]),
            "top3_risk_contribution_pct": float(top3_risk_contribution),
            "risk_hhi": float(risk_hhi),
            "effective_risk_symbol_count": float(effective_risk_symbol_count),
            "money_top1_difference": money_top_difference,
            "correlation_concentration_memo": correlation_memo,
            "beta_to_topix": beta_to_topix,
            "tracking_error_1y": tracking_error_1y,
            "information_ratio_1y": information_ratio_1y,
            "downside_capture_1y": downside_capture_1y,
            "max_drawdown_1y": max_drawdown_1y,
            "stress_topix_minus10_loss_pct": stress_topix_minus10,
            "stress_topix_minus20_loss_pct": stress_topix_minus20,
            "stress_top1_risk_minus20_loss_pct": stress_top1_minus20,
            "stress_top3_risk_minus20_loss_pct": stress_top3_minus20,
            "stress_topix_minus10_top3_minus15_loss_pct": stress_combined,
            "top_entries": top_entries[:5],
        }
    )
    return result


def build_history_payload(ws, snapshot_date: str, history_row: list):
    """
    履歴は AU:CG にだけ持つ。
    - snapshot_date は実行日を使う
    - 同じ日付があれば上書き
    - なければ追加
    - 最新日付を上、古い日付を下に並べる
    """
    snapshot_date = str(snapshot_date).strip()

    # NO_DATA や不正日付のときは履歴保存しない
    if snapshot_date == "" or snapshot_date == "NO_DATA":
        return None
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", snapshot_date):
        return None

    new_row = normalize_hist_row(history_row, HIST_COLS)
    new_row[0] = snapshot_date  # 念のため固定

    existing = ws.get(HIST_DATA_RANGE)
    history_rows = []
    replaced = False

    for r in existing:
        rr = normalize_hist_row(r, HIST_COLS)
        if all(str(x).strip() == "" for x in rr):
            continue

        key = str(rr[0]).strip()
        if key == snapshot_date:
            history_rows.append(new_row)
            replaced = True
        else:
            history_rows.append(rr)

    if not replaced:
        history_rows.append(new_row)

    # yyyy-mm-dd なので文字列降順でOK
    history_rows.sort(key=lambda x: str(x[0]).strip(), reverse=True)

    payload = []
    if history_rows:
        payload.append(
            {
                "range": f"{HIST_START_COL}2:{HIST_END_COL}{len(history_rows) + 1}",
                "values": history_rows,
            }
        )
    return payload



def normalize_obs_db_row(row, size=OBS_DB_COLS):
    row = list(row[:size])
    if len(row) < size:
        row += [""] * (size - len(row))
    return row


def normalize_bool(value):
    s = "" if value is None else str(value).strip().upper()
    return s in ("TRUE", "1", "YES", "Y")


def quote_sheet_name_for_a1(sheet_name: str):
    return "'" + str(sheet_name).replace("'", "''") + "'"


def read_observation_db(ws):
    """
    CH:CU の観測DBを position_key をキーに読み込む。
    既存DBに重複キーがある場合は、上書き事故防止のため停止する。
    """
    values = ws.get(OBS_DB_DATA_RANGE)
    obs_db = {}

    for r in values:
        rr = normalize_obs_db_row(r)
        position_key = str(rr[0]).strip()
        if position_key == "":
            continue

        if position_key in obs_db:
            raise ValueError(f"FATAL:DUPLICATE_OBS_DB_KEY:{position_key}")

        obs_db[position_key] = {
            "position_key": position_key,
            "ticker": str(rr[1]).strip(),
            "custody_type": str(rr[2]).strip(),
            "code": str(rr[3]).strip(),
            "name": str(rr[4]).strip(),
            "first_seen_date": normalize_yyyymmdd(rr[5]),
            "first_seen_qty": str(rr[6]).strip(),
            "first_seen_cost": str(rr[7]).strip(),
            "last_seen_date": normalize_yyyymmdd(rr[8]),
            "last_seen_qty": str(rr[9]).strip(),
            "last_seen_cost": str(rr[10]).strip(),
            "active": normalize_bool(rr[11]),
            "inactive_since": normalize_yyyymmdd(rr[12]),
            "last_updated_at_jst": str(rr[13]).strip(),
        }

    return obs_db


def normalize_dividend_cache_row(row, size=DIVIDEND_CACHE_COLS):
    row = list(row[:size])
    if len(row) < size:
        row += [""] * (size - len(row))
    return row


def parse_dividend_cache_datetime(value, jst):
    s = "" if value is None else str(value).strip()
    if s == "":
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=jst)
    except Exception:
        return None


def is_dividend_cache_fresh(cache_row, now_jst):
    fetched_at = parse_dividend_cache_datetime(
        cache_row.get("fetched_at_jst", ""),
        now_jst.tzinfo,
    )
    if fetched_at is None:
        return False
    return (now_jst - fetched_at).total_seconds() <= DIVIDEND_CACHE_TTL_DAYS * 24 * 60 * 60


def read_dividend_yield_cache(ws):
    """
    CV:CY の配当利回りキャッシュを ticker キーで読み込む。
    重複tickerは後勝ちにして、次回書き戻し時に正規化する。
    """
    values = ws.get(DIVIDEND_CACHE_DATA_RANGE)
    cache = {}

    for r in values:
        rr = normalize_dividend_cache_row(r)
        ticker = str(rr[0]).strip()
        if ticker == "":
            continue

        cache[ticker] = {
            "ticker": ticker,
            "dividend_yield": str(rr[1]).strip(),
            "fetched_at_jst": str(rr[2]).strip(),
            "fetch_status": str(rr[3]).strip(),
        }

    return cache


def apply_dividend_cache_value(ticker, cache_row, div_yield_map, div_yield_display_map):
    dy = parse_float(cache_row.get("dividend_yield", ""))
    if dy is None:
        div_yield_map[ticker] = 0.0
        div_yield_display_map[ticker] = 0.0
    else:
        div_yield_map[ticker] = float(dy)
        div_yield_display_map[ticker] = float(dy)


def build_dividend_cache_matrix(updated_cache, tickers):
    out = []
    for tk in sorted(set(tickers)):
        cache_row = updated_cache.get(tk, {})
        out.append([
            tk,
            cache_row.get("dividend_yield", ""),
            cache_row.get("fetched_at_jst", ""),
            cache_row.get("fetch_status", ""),
        ])
    return out


def fetch_sheet_metadata_safe(sh, params):
    """
    gspread のバージョン差異に備え、params の渡し方を両対応にする。
    """
    try:
        return sh.fetch_sheet_metadata(params=params)
    except TypeError:
        return sh.fetch_sheet_metadata(params)


def detect_custody_boundary_row_or_raise(sh, sheet_name, raw_data, start_row=START_ROW, max_row=MAX_ROW):
    """
    B列の太字セルから、特定預かりの最終行を検出する。
    ただし、A列が有効な銘柄コードである行だけを対象にする。

    仕様：
      - 太字の有効銘柄行が1件ちょうどの場合のみ正常
      - 0件 / 2件以上 / 書式取得失敗は FATAL で停止
    """
    total_rows = max_row - start_row + 1
    valid_code_rows = set()

    for i in range(total_rows):
        row_data = raw_data[i] if i < len(raw_data) else []
        code = str(row_data[0]).strip() if len(row_data) > 0 else ""
        if parse_ticker(code) is not None:
            valid_code_rows.add(start_row + i)

    if not valid_code_rows:
        raise ValueError("FATAL:NO_VALID_HOLDING_ROWS")

    quoted_sheet_name = quote_sheet_name_for_a1(sheet_name)
    params = {
        "includeGridData": True,
        "ranges": [f"{quoted_sheet_name}!B{start_row}:B{max_row}"],
        "fields": "sheets(properties(title),data(rowData(values(userEnteredFormat(textFormat(bold)),effectiveFormat(textFormat(bold))))))",
    }

    try:
        metadata = fetch_sheet_metadata_safe(sh, params)
    except Exception as e:
        raise ValueError(f"FATAL:CUSTODY_BOUNDARY_FORMAT_READ_FAILED:{e}")

    target_sheet = None
    for sheet in metadata.get("sheets", []):
        props = sheet.get("properties", {})
        if props.get("title") == sheet_name:
            target_sheet = sheet
            break

    if target_sheet is None:
        raise ValueError("FATAL:CUSTODY_BOUNDARY_FORMAT_READ_FAILED:SHEET_NOT_FOUND")

    data = target_sheet.get("data", [])
    row_data_list = data[0].get("rowData", []) if data else []
    bold_valid_rows = []

    for i, fmt_row in enumerate(row_data_list):
        sheet_row = start_row + i
        if sheet_row not in valid_code_rows:
            continue

        values = fmt_row.get("values", [])
        if not values:
            continue

        cell = values[0]
        user_bold = (
            cell.get("userEnteredFormat", {})
                .get("textFormat", {})
                .get("bold", False)
        )
        effective_bold = (
            cell.get("effectiveFormat", {})
                .get("textFormat", {})
                .get("bold", False)
        )

        if bool(user_bold or effective_bold):
            bold_valid_rows.append(sheet_row)

    if len(bold_valid_rows) == 0:
        raise ValueError("FATAL:CUSTODY_BOUNDARY_BOLD_ROW_NOT_FOUND")

    if len(bold_valid_rows) > 1:
        rows_str = ",".join(str(x) for x in bold_valid_rows)
        raise ValueError(f"FATAL:MULTIPLE_CUSTODY_BOUNDARY_BOLD_ROWS:{rows_str}")

    return bold_valid_rows[0]


def infer_custody_type(sheet_row, custody_boundary_row):
    if sheet_row <= custody_boundary_row:
        return CUSTODY_TOKUTEI
    return CUSTODY_NISA


def build_position_key(ticker, custody_type):
    return f"{ticker}|{custody_type}"


def build_observation_db_matrix(obs_db, rows, now_jst):
    """
    既存DBと現在の有効保有行から、CH:CUへ書き戻すDB行を作る。
    価格取得失敗は売却ではないため、holding_input_ok の行は active=True として扱う。
    """
    today = now_jst.strftime("%Y%m%d")
    updated_at = now_jst.strftime("%Y-%m-%d %H:%M:%S")

    current_rows = [r for r in rows if r.get("holding_input_ok")]
    current_keys = {r.get("position_key") for r in current_rows if r.get("position_key")}

    new_db = {pk: dict(obs) for pk, obs in obs_db.items()}

    # 今回A:Dに存在しない既存DB行は inactive にする。
    for pk, obs in list(new_db.items()):
        if pk not in current_keys:
            if obs.get("active"):
                obs["active"] = False
                obs["inactive_since"] = today
                obs["last_updated_at_jst"] = updated_at
            new_db[pk] = obs

    # 今回A:Dに存在する有効保有行を active として更新する。
    for row in current_rows:
        pk = row.get("position_key", "")
        if pk == "":
            continue

        new_db[pk] = {
            "position_key": pk,
            "ticker": row.get("ticker", ""),
            "custody_type": row.get("custody_type", ""),
            "code": row.get("code", ""),
            "name": row.get("name", ""),
            "first_seen_date": row.get("first_seen_date", ""),
            "first_seen_qty": row.get("first_seen_qty", ""),
            "first_seen_cost": row.get("first_seen_cost", ""),
            "last_seen_date": today,
            "last_seen_qty": format_qty(row.get("shares", "")),
            "last_seen_cost": format_qty(row.get("cost", "")),
            "active": True,
            "inactive_since": "",
            "last_updated_at_jst": updated_at,
        }

    custody_rank = {CUSTODY_TOKUTEI: 0, CUSTODY_NISA: 1}
    sorted_obs = sorted(
        new_db.values(),
        key=lambda x: (
            str(x.get("ticker", "")),
            custody_rank.get(str(x.get("custody_type", "")), 99),
            str(x.get("position_key", "")),
        )
    )

    out = []
    for obs in sorted_obs:
        out.append([
            obs.get("position_key", ""),
            obs.get("ticker", ""),
            obs.get("custody_type", ""),
            obs.get("code", ""),
            obs.get("name", ""),
            obs.get("first_seen_date", ""),
            obs.get("first_seen_qty", ""),
            obs.get("first_seen_cost", ""),
            obs.get("last_seen_date", ""),
            obs.get("last_seen_qty", ""),
            obs.get("last_seen_cost", ""),
            "TRUE" if obs.get("active") else "FALSE",
            obs.get("inactive_since", ""),
            obs.get("last_updated_at_jst", ""),
        ])

    return out


# =========================
# Main
# =========================
def main():
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst)

    # -------------------------
    # 1) Sheet read
    # -------------------------
    client = get_gspread_client()
    
    app_config_str = os.environ.get("APP_CONFIG")
    if not app_config_str:
        raise ValueError("APP_CONFIG is not set.")
    app_config = json.loads(app_config_str)
    
    sheet_id = app_config.get("spreadsheet_id")
    if not sheet_id:
        raise ValueError("spreadsheet_id is missing in APP_CONFIG.")
        
    sheet_name = app_config.get("sheet_name")
    if not sheet_name:
        raise ValueError("sheet_name is missing in APP_CONFIG.")

    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(sheet_name)

    raw_data = ws.get(f"A{START_ROW}:T{MAX_ROW}")  # list[list[str]]

    # 太字境界は必須。ここで失敗した場合は、以降の書き込み処理に進ませない。
    custody_boundary_row = detect_custody_boundary_row_or_raise(
        sh=sh,
        sheet_name=sheet_name,
        raw_data=raw_data,
        start_row=START_ROW,
        max_row=MAX_ROW,
    )

    # 観測DB（CH:CU）を読み込む。空の場合だけ、初回移行としてQ:Rを参照する。
    obs_db = read_observation_db(ws)
    obs_db_is_empty = len(obs_db) == 0

    # 配当利回りキャッシュ（CV:CY）を読み込む。
    dividend_cache = read_dividend_yield_cache(ws)

    # rows: 14..2000 を全て保持（出力行数固定のため）
    total_rows = MAX_ROW - START_ROW + 1
    rows = []
    tickers = []
    input_invalid_count = 0

    # バリデーション
    for i in range(total_rows):
        sheet_row = START_ROW + i
        row_data = raw_data[i] if i < len(raw_data) else []
        a = row_data[0] if len(row_data) > 0 else ""
        code = str(a).strip()

        if code == "":
            rows.append({"is_empty": True})
            continue

        info = {
            "is_empty": False,
            "status": "OK",            # OK / DATA_NG
            "health_level": "",        # OK/CAUTION/EXIT/DATA_NG
            "health_reason": "",       # reasons str (max 2 by rule)
            "sheet_row": sheet_row,
            "code": code,
            "name": row_data[1] if len(row_data) > 1 else "",
            "shares": None,
            "cost": None,
            "ticker": None,
            "custody_type": "",
            "position_key": "",
            "holding_input_ok": False,
            "first_seen_date": "",
            "first_seen_qty": "",
            "first_seen_cost": "",
            "observed_months": None,
            "observed_holding_ym": "",
            "q3_judgement": "",
        }

        # 初回移行時だけ、現在のQ:Rを移行元として使う。
        existing_first_seen_date_from_q = row_data[16] if len(row_data) > 16 else ""
        existing_first_seen_qty_from_r = row_data[17] if len(row_data) > 17 else ""

        tk = parse_ticker(code)
        if tk is None:
            info["status"] = "DATA_NG"
            info["health_level"] = "DATA_NG"
            info["health_reason"] = "INPUT_NG:CODE_FORMAT"
            input_invalid_count += 1
            rows.append(info)
            continue

        custody_type = infer_custody_type(sheet_row, custody_boundary_row)
        position_key = build_position_key(tk, custody_type)

        info["ticker"] = tk
        info["custody_type"] = custody_type
        info["position_key"] = position_key

        shares = parse_float(row_data[2] if len(row_data) > 2 else None)
        cost = parse_float(row_data[3] if len(row_data) > 3 else None)

        if shares is None or cost is None:
            info["status"] = "DATA_NG"
            info["health_level"] = "DATA_NG"
            info["health_reason"] = "INPUT_NG:NUM_PARSE"
            input_invalid_count += 1
            rows.append(info)
            continue

        if shares <= 0:
            info["status"] = "DATA_NG"
            info["health_level"] = "DATA_NG"
            info["health_reason"] = "INPUT_NG:SHARES_NONPOSITIVE"
            input_invalid_count += 1
            rows.append(info)
            continue

        if cost < 0:
            info["status"] = "DATA_NG"
            info["health_level"] = "DATA_NG"
            info["health_reason"] = "INPUT_NG:COST_NEGATIVE"
            input_invalid_count += 1
            rows.append(info)
            continue

        obs = obs_db.get(position_key)

        if obs_db_is_empty:
            first_seen_date = normalize_yyyymmdd(existing_first_seen_date_from_q)
            if first_seen_date == "":
                first_seen_date = now_jst.strftime("%Y%m%d")

            first_seen_qty = str(existing_first_seen_qty_from_r).strip()
            if first_seen_qty == "":
                first_seen_qty = format_qty(shares)

            first_seen_cost = format_qty(cost)
        elif obs is None:
            first_seen_date = now_jst.strftime("%Y%m%d")
            first_seen_qty = format_qty(shares)
            first_seen_cost = format_qty(cost)
        elif obs.get("active") is False:
            # 全売却後の再購入は、今回の保有サイクルとして初回観測をリセットする。
            first_seen_date = now_jst.strftime("%Y%m%d")
            first_seen_qty = format_qty(shares)
            first_seen_cost = format_qty(cost)
        else:
            first_seen_date = normalize_yyyymmdd(obs.get("first_seen_date"))
            if first_seen_date == "":
                first_seen_date = now_jst.strftime("%Y%m%d")

            first_seen_qty = str(obs.get("first_seen_qty", "")).strip()
            if first_seen_qty == "":
                first_seen_qty = format_qty(shares)

            first_seen_cost = str(obs.get("first_seen_cost", "")).strip()
            if first_seen_cost == "":
                first_seen_cost = format_qty(cost)

        observed_months = calc_observed_months(first_seen_date, now_jst)

        info["shares"] = float(shares)
        info["cost"] = float(cost)
        info["holding_input_ok"] = True
        info["first_seen_date"] = first_seen_date
        info["first_seen_qty"] = first_seen_qty
        info["first_seen_cost"] = first_seen_cost
        info["observed_months"] = observed_months
        info["observed_holding_ym"] = format_observed_ym(observed_months)
        info["q3_judgement"] = "判定不能"

        tickers.append(tk)
        rows.append(info)

    # 有効銘柄（入力がOK）
    unique_tickers = sorted(set(tickers))
    valid_input_count = len(unique_tickers)

    if valid_input_count == 0:
        raise ValueError("FATAL:NO_VALID_INPUT_ROWS")

    current_position_keys = [r.get("position_key") for r in rows if r.get("holding_input_ok")]
    duplicate_position_keys = sorted([k for k, v in Counter(current_position_keys).items() if k and v > 1])
    if duplicate_position_keys:
        raise ValueError("FATAL:DUPLICATE_POSITION_KEY:" + ",".join(duplicate_position_keys))

    # -------------------------
    # 2) yfinance fetch (batch)
    # -------------------------
    # 仕様：current_price は fast_info.last_price を優先し、失敗時のみ close_5d 最新Close にfallback
    # 仕様：health/forecast は Close、1年PF/TOPIXリターンは分割調整後価格を優先

    # ベンチ候補も含めて一括で落とす
    # TOPIX系ベンチマークは、優先順に正常な1年リターンが取れた最初の銘柄を採用する。
    all_tickers_for_price = sorted(set(unique_tickers + BENCHMARK_TICKERS))

    # yfinance download
    close_5d = pd.DataFrame()
    close_1y = pd.DataFrame()
    close_1y_adjusted = pd.DataFrame()
    close_1y_return = pd.DataFrame()
    close_3y = pd.DataFrame()

    try:
        dl_5d = yf.download(
            tickers=all_tickers_for_price,
            period="5d",
            interval="1d",
            group_by="column",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        close_5d = extract_close_df(dl_5d, [all_tickers_for_price[0]] if len(all_tickers_for_price) == 1 else all_tickers_for_price)
    except Exception:
        close_5d = pd.DataFrame()

    try:
        dl_1y = yf.download(
            tickers=all_tickers_for_price,
            period="1y",
            interval="1d",
            group_by="column",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        tickers_for_df = [all_tickers_for_price[0]] if len(all_tickers_for_price) == 1 else all_tickers_for_price
        close_1y = extract_close_df(dl_1y, tickers_for_df)
        close_1y_adjusted = extract_price_df(dl_1y, tickers_for_df, "Adj Close", "Close")
    except Exception:
        close_1y = pd.DataFrame()
        close_1y_adjusted = pd.DataFrame()

    try:
        # 1年PF/TOPIXリターン用。auto_adjust=True の Close は分割・配当調整後の時系列になる。
        # 1306.Tの受益権分割のようなケースで、通常Close比較による異常値を避ける。
        dl_1y_return = yf.download(
            tickers=all_tickers_for_price,
            period="1y",
            interval="1d",
            group_by="column",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        tickers_for_df = [all_tickers_for_price[0]] if len(all_tickers_for_price) == 1 else all_tickers_for_price
        close_1y_return = extract_close_df(dl_1y_return, tickers_for_df)
    except Exception:
        close_1y_return = pd.DataFrame()

    try:
        dl_3y = yf.download(
            tickers=all_tickers_for_price,
            period="3y",
            interval="1d",
            group_by="column",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        close_3y = extract_close_df(dl_3y, [all_tickers_for_price[0]] if len(all_tickers_for_price) == 1 else all_tickers_for_price)
    except Exception:
        close_3y = pd.DataFrame()

    # 価格データ時刻（仕様：取得した最新バーの日時文字列）
    latest_bar = ""
    try:
        idx = close_5d.dropna(how="all").index
        if len(idx) > 0:
            latest_bar = pd.Timestamp(idx.max()).strftime("%Y-%m-%d")
        else:
            latest_bar = "NO_DATA"
    except Exception:
        latest_bar = "NO_DATA"

    # -------------------------
    # 2.5) current price fetch (fast_info)
    # -------------------------
    # E列 current_price は fast_info.last_price を優先する。
    # 同一tickerを複数保有行で持つ場合でも、API取得はtickerごとに1回だけ行う。
    current_price_fast_map = {}

    for tk in unique_tickers:
        current_price_fast_map[tk] = fetch_current_price_fast(tk)

    # benchmark availability（1年リターンは auto_adjust=True の調整後Closeを最優先）
    if not close_1y_return.empty:
        return_1y_df = close_1y_return
    elif not close_1y_adjusted.empty:
        return_1y_df = close_1y_adjusted
    else:
        return_1y_df = close_1y

    selected_benchmark_ticker, topix_val = pick_benchmark_return(return_1y_df, BENCHMARK_TICKERS)

    # 一括取得で正常な値が取れない場合は、候補を優先順に個別取得する。
    if topix_val is None:
        for bm_ticker in BENCHMARK_TICKERS:
            ret = fetch_single_benchmark_return(bm_ticker)
            if ret is not None:
                selected_benchmark_ticker = bm_ticker
                topix_val = ret
                break

    benchmark_fail_count = 0 if topix_val is not None else 1

    # -------------------------
    # 3) info fetch (dividendYield) - 7日キャッシュ / fail-safe
    # -------------------------
    # 仕様：O列表示は配当利回り。取得不能時の表示は0、期待年率計算でも0扱い。
    # 7日以内のキャッシュがあれば yfinance の info 取得をスキップする。
    div_yield_map = {tk: 0.0 for tk in unique_tickers}
    div_yield_display_map = {tk: 0.0 for tk in unique_tickers}
    updated_dividend_cache = {}
    fetched_at_now = now_jst.strftime("%Y-%m-%d %H:%M:%S")

    for tk in unique_tickers:
        cache_row = dividend_cache.get(tk)

        if cache_row is not None and is_dividend_cache_fresh(cache_row, now_jst):
            apply_dividend_cache_value(tk, cache_row, div_yield_map, div_yield_display_map)
            updated_dividend_cache[tk] = cache_row
            continue

        try:
            info = yf.Ticker(tk).info  # ここが落ちても価格は巻き添えにしない
            dy = info.get("dividendYield", None)

            if isinstance(dy, (int, float)) and not (isinstance(dy, float) and math.isnan(dy)):
                dy_value = float(dy)
                div_yield_map[tk] = dy_value
                div_yield_display_map[tk] = dy_value
                updated_dividend_cache[tk] = {
                    "ticker": tk,
                    "dividend_yield": dy_value,
                    "fetched_at_jst": fetched_at_now,
                    "fetch_status": "OK",
                }
            else:
                # infoは取得できたが dividendYield がない場合。表示・計算ともに0扱い。
                div_yield_map[tk] = 0.0
                div_yield_display_map[tk] = 0.0
                updated_dividend_cache[tk] = {
                    "ticker": tk,
                    "dividend_yield": 0.0,
                    "fetched_at_jst": fetched_at_now,
                    "fetch_status": "NO_DIVIDEND_YIELD",
                }
        except Exception:
            # info取得自体が失敗した場合は、前回値があれば維持し、取得日は更新しない。
            if cache_row is not None:
                apply_dividend_cache_value(tk, cache_row, div_yield_map, div_yield_display_map)
                updated_dividend_cache[tk] = cache_row
            else:
                div_yield_map[tk] = 0.0
                div_yield_display_map[tk] = 0.0
                updated_dividend_cache[tk] = {
                    "ticker": tk,
                    "dividend_yield": 0.0,
                    "fetched_at_jst": "",
                    "fetch_status": "FETCH_FAILED",
                }

    # -------------------------
    # 4) per-row compute
    # -------------------------
    price_fail_count = 0

    # row-level computed fields will be attached to row dict
    total_market_value = 0
    valid_price_rows = []  # 入力OKかつ価格OK（total計算の母集団）

    for row in rows:
        if row.get("is_empty"):
            continue
        if row.get("status") != "OK":
            continue

        tk = row["ticker"]

        # current_price: fast_info.last_price -> existing close_5d fallback
        cur = current_price_fast_map.get(tk)

        if cur is None:
            try:
                if tk in close_5d.columns:
                    s = close_5d[tk].dropna()
                    if not s.empty:
                        cur = float(s.iloc[-1])
            except Exception:
                cur = None

        if cur is None or (isinstance(cur, float) and math.isnan(cur)):
            # 価格取得失敗 → 行はDATA_NG（仕様：E〜Pは空、M=DATA_NG、N=理由）
            row["status"] = "DATA_NG"
            row["health_level"] = "DATA_NG"
            row["health_reason"] = "DATA_NG:NO_PRICE"
            price_fail_count += 1
            continue

        # snapshot
        shares = float(row["shares"])
        cost = float(row["cost"])

        market_value = int(round(cur * shares))
        cost_value = int(round(cost * shares))
        pnl_jpy = market_value - cost_value
        pnl_pct = "" if cost_value == 0 else (market_value / cost_value) - 1

        row["current_price"] = cur
        row["market_value"] = market_value
        row["cost_value"] = cost_value
        row["pnl_jpy"] = pnl_jpy
        row["pnl_pct"] = pnl_pct
        row["q3_judgement"] = calc_3mo_judgement(row.get("observed_months"), pnl_pct)

        total_market_value += market_value

        # health
        # 必要データ長：SMA200=200本、52w=252本。足りなければ DATA_NG / HIST_SHORT
        health_level = "DATA_NG"
        reasons = []

        series_3y = pd.Series(dtype=float)
        if tk in close_3y.columns:
            series_3y = close_3y[tk].dropna()

        if len(series_3y) < 252:
            health_level = "DATA_NG"
            reasons = ["HIST_SHORT"]
        else:
            sma200 = float(series_3y.tail(200).mean())
            high_52w = float(series_3y.tail(252).max())
            dd_52w = (cur / high_52w) - 1 if high_52w > 0 else 0.0

            # 理由は最大2、順序固定：SMA200 → DD
            if cur < sma200:
                reasons.append("SMA200_DOWN")

            if dd_52w <= DD_EXIT:
                health_level = "EXIT"
                reasons.append("DD52W_<=-30%")
            elif dd_52w <= DD_CAUTION:
                # CAUTION（DD条件）
                health_level = "CAUTION"
                reasons.append("DD52W_<=-20%")
            else:
                # DDは問題なし。SMA200でCAUTION/OKを確定
                health_level = "OK" if cur >= sma200 else "CAUTION"

            # 2理由制限
            reasons = reasons[:2]

        row["health_level"] = health_level
        row["health_reason"] = ";".join(reasons)

        # forecast (ret_base)
        series_1y = pd.Series(dtype=float)
        if tk in close_1y.columns:
            series_1y = close_1y[tk].dropna()

        cagr_3y = calc_cagr(series_3y)
        cagr_1y = calc_cagr(series_1y)
        cagr = cagr_3y if cagr_3y is not None else (cagr_1y if cagr_1y is not None else 0.0)

        dy = float(div_yield_map.get(tk, 0.0))
        ret_base_raw = dy + float(cagr)
        ret_base = clip(ret_base_raw, RET_CLIP_MIN, RET_CLIP_MAX)
        if ret_base is None:
            ret_base = 0.0

        row["ret_base"] = float(ret_base)
        row["dividend_yield"] = div_yield_display_map.get(tk, "")

        valid_price_rows.append(row)

    # -------------------------
    # 5) exec_status (仕様厳守)
    # -------------------------
    # 仕様：
    # FAILED：有効銘柄が0件、または total_market_value=0
    # PARTIAL：有効銘柄の価格取得失敗が1件以上（or benchmark fail）
    # OK：上記以外
    if valid_input_count == 0 or total_market_value == 0:
        exec_status = "FAILED"
    else:
        if price_fail_count > 0 or benchmark_fail_count > 0:
            exec_status = "PARTIAL"
        else:
            exec_status = "OK"

    # error_count（仕様：入力不正 + 価格取得失敗 + ベンチ失敗）
    error_count = int(input_invalid_count + price_fail_count + benchmark_fail_count)

    # -------------------------
    # 6) weights & analyze
    # -------------------------
    if total_market_value > 0:
        for r in valid_price_rows:
            r["weight"] = r["market_value"] / total_market_value
    else:
        for r in valid_price_rows:
            r["weight"] = ""

    weights = sorted([r["weight"] for r in valid_price_rows if isinstance(r.get("weight"), float)], reverse=True)
    top1 = weights[0] if len(weights) >= 1 else 0
    top3 = sum(weights[:3]) if len(weights) >= 1 else 0
    hhi = sum(w * w for w in weights) if len(weights) else 0

    # health counts（曖昧排除：DATA_NGはCAUTIONにカウントして「注意が必要」に寄せる）
    cnt_exit = 0
    cnt_caution = 0
    cnt_ok = 0
    for r in valid_price_rows:
        hl = r.get("health_level")
        if hl == "EXIT":
            cnt_exit += 1
        elif hl == "OK":
            cnt_ok += 1
        else:
            # CAUTION or DATA_NG
            cnt_caution += 1

    # 要注意Top（EXIT > CAUTION > DATA_NG、同ランクはweight降順）
    sev_rank = {"EXIT": 3, "CAUTION": 2, "DATA_NG": 1, "OK": 0}
    attention = []
    for r in valid_price_rows:
        hl = r.get("health_level", "DATA_NG")
        if hl in ("EXIT", "CAUTION", "DATA_NG"):
            attention.append(r)

    attention.sort(
        key=lambda x: (
            -sev_rank.get(x.get("health_level", "DATA_NG"), 0),
            -float(x.get("weight", 0.0)) if isinstance(x.get("weight"), float) else 0.0,
            x.get("ticker", "")
        )
    )
    def fmt_attention(rr):
        return display_name(rr.get("name", ""), rr.get("ticker", ""))
    c_top1 = fmt_attention(attention[0]) if len(attention) > 0 else ""
    c_top2 = fmt_attention(attention[1]) if len(attention) > 1 else ""
    c_top3 = fmt_attention(attention[2]) if len(attention) > 2 else ""

    # -------------------------
    # 7) rebalance（仕様固定）
    # -------------------------
    # available_cash = 売り（SELL/REDUCE）の合計（正の円）
    available_cash = 0
    buy_candidates = []
    needs_sum = 0.0

    if exec_status != "FAILED":
        for r in valid_price_rows:
            w = r.get("weight")
            hl = r.get("health_level")

            # デフォルト
            r["action"] = "HOLD"
            r["trade_jpy"] = 0

            if hl == "EXIT":
                r["action"] = "SELL"
                r["trade_jpy"] = -int(r["market_value"])
                available_cash += int(-r["trade_jpy"])
            elif isinstance(w, float) and w > MAX_SINGLE_WEIGHT:
                r["action"] = "REDUCE"
                desired = -(w - TARGET_SINGLE_WEIGHT) * total_market_value  # 負
                trade = int(round(max(desired, -r["market_value"])))       # 負（売り過ぎ防止）
                r["trade_jpy"] = trade
                available_cash += int(-r["trade_jpy"])
            elif hl == "OK" and isinstance(w, float) and w < TARGET_SINGLE_WEIGHT:
                r["action"] = "BUY"
                need = (TARGET_SINGLE_WEIGHT - w) * total_market_value  # 正
                r["need_jpy"] = float(need)
                needs_sum += float(need)
                buy_candidates.append(r)
            else:
                # HOLD
                pass

        allocated_buy_total = 0
        if needs_sum <= 0 or available_cash <= 0:
            # 仕様：needs_sum==0 or available_cash==0 のとき BUYはHOLDに変更しQ=0
            for r in buy_candidates:
                r["action"] = "HOLD"
                r["trade_jpy"] = 0
        else:
            for r in buy_candidates:
                allocated = int(round(available_cash * r["need_jpy"] / needs_sum))
                r["trade_jpy"] = allocated  # BUYは正
                allocated_buy_total += allocated

        unallocated_cash = max(0, int(available_cash - allocated_buy_total))
    else:
        unallocated_cash = 0

    # -------------------------
    # 8) portfolio forecast & simulate（仕様固定）
    # -------------------------
    if exec_status != "FAILED" and len(valid_price_rows) > 0:
        pf_ret_base = sum(float(r["weight"]) * float(r["ret_base"]) for r in valid_price_rows if isinstance(r.get("weight"), float))
        pf_ret_opt = pf_ret_base + 0.05
        pf_ret_pess = pf_ret_base - 0.05

        value_10y_base = int(round(total_market_value * (1 + pf_ret_base) ** 10))
        value_10y_opt = int(round(total_market_value * (1 + pf_ret_opt) ** 10))
        value_10y_pess = int(round(total_market_value * (1 + pf_ret_pess) ** 10))
    else:
        pf_ret_base = pf_ret_opt = pf_ret_pess = 0
        value_10y_base = value_10y_opt = value_10y_pess = 0

    # -------------------------
    # 9) backtest（仕様固定：1y、2点間リターンの加重平均）
    # -------------------------
    pf_cum = ""
    topix_cum = ""
    alpha = ""

    if exec_status != "FAILED" and len(valid_price_rows) > 0:
        try:
            # TOPIX系ベンチマークの1年リターンは事前にフォールバック選定済み。

            # PFリターンの計算（各銘柄の1年リターン × ウェイト）
            pf_val = 0.0
            valid_pf_weight = 0.0
            
            for r in valid_price_rows:
                tk = r["ticker"]
                w = float(r.get("weight", 0.0))
                
                if tk in return_1y_df.columns:
                    s = return_1y_df[tk].dropna()
                    if len(s) >= 2:
                        ret_1y = (s.iloc[-1] / s.iloc[0]) - 1
                        pf_val += ret_1y * w
                        valid_pf_weight += w

            # 全く計算できなかった場合を除外
            if valid_pf_weight > 0:
                pf_cum = float(pf_val)
                
                if topix_val is not None:
                    topix_cum = float(topix_val)
                    alpha = float(pf_val - topix_val)

        except Exception:
            # 例外時は空欄（実行ステータスには影響させない）
            pf_cum = topix_cum = alpha = ""

    # -------------------------
    # 9.5) portfolio risk / benchmark analytics
    # -------------------------
    # 新規分析はauto_adjust=Trueの1年日足だけを使用し、失敗しても既存処理は継続する。
    portfolio_analytics = calculate_portfolio_analytics(
        valid_price_rows=valid_price_rows,
        adjusted_close_df=close_1y_return,
        selected_benchmark_ticker=selected_benchmark_ticker,
    )

    # -------------------------
    # 10) Build output matrices
    # -------------------------
    # E..T = 16 columns（E..P 既存出力 + Q..T 初回観測系4列）
    output_matrix = []
    for row in rows:
        if row.get("is_empty"):
            output_matrix.append([""] * 16)
            continue

        if row.get("status") != "OK":
            # 無効行の場合も列の並び順に合わせる
            output_matrix.append([
                "", "", "", "", "", "",            # E..J
                "DATA_NG",                         # K (health_level)
                str(row.get("health_reason", "")), # L (health_reason)
                "", "", "", "",                    # M..P
                row.get("first_seen_date", ""),     # Q: 初回観測日
                row.get("first_seen_qty", ""),      # R: 初回観測数量
                row.get("observed_holding_ym", ""), # S: 観測保有年月数
                row.get("q3_judgement", ""),        # T: 3か月判定
            ])
            continue

        # OK
        cur = row.get("current_price", "")
        mv = row.get("market_value", "")
        cv = row.get("cost_value", "")
        pnl = row.get("pnl_jpy", "")
        pnlp = row.get("pnl_pct", "")
        w = row.get("weight", "")
        
        hl = row.get("health_level", "DATA_NG")
        hr = row.get("health_reason", "")
        
        act = row.get("action", "HOLD")
        trade = row.get("trade_jpy", 0)
        
        dy_out = row.get("dividend_yield", "")
        rb = row.get("ret_base", "")

        output_matrix.append([
            cur,        # E: 現在値
            mv,         # F: 評価額
            cv,         # G: 取得額
            pnl,        # H: 損益(円)
            pnlp,       # I: 損益(%)
            w,          # J: ウェイト
            hl,         # K: ヘルスチェック (移動)
            hr,         # L: 判定理由 (移動)
            act,        # M: 売買アクション (移動)
            trade,      # N: 売買金額 (移動)
            dy_out,                             # O: 配当利回り
            rb,                                 # P: 期待年率 (後ろへ)
            row.get("first_seen_date", ""),     # Q: 初回観測日
            row.get("first_seen_qty", ""),      # R: 初回観測数量
            row.get("observed_holding_ym", ""), # S: 観測保有年月数
            row.get("q3_judgement", ""),        # T: 3か月判定
        ])

    # dashboard matrix（A1:X11）
    dash = build_dashboard_matrix()

    labels = {
        "A1": "更新日時(JST)",
        "A2": "総評価額(JPY)",
        "A3": "総取得額(JPY)",
        "A4": "総損益(JPY)",
        "A5": "総損益(%)",
        "A6": "銘柄数",
        "D2": "上位1銘柄ウェイト",
        "D3": "上位3銘柄ウェイト合計",
        "D4": "HHI(銘柄集中度)",
        "D5": "EXIT銘柄数",
        "D6": "CAUTION銘柄数",
        "D7": "OK銘柄数",
        "D8": "要注意Top1",
        "D9": "要注意Top2",
        "D10": "要注意Top3",
        "G2": "期待年率(base)",
        "G3": "期待年率(opt)",
        "G4": "期待年率(pess)",
        "G5": "リバランス余剰現金",
        "G6": "好調銘柄数",
        "G7": "不調銘柄数",
        "G8": "好調ウェイト合計",
        "G9": "不調ウェイト合計",
        "G10": "24か月以上不調数",
        "G11": "株数増加銘柄数",
        "J1": "リスク寄与度",
        "J2": "PF年率ボラ",
        "J3": "最大リスク寄与",
        "J4": "リスク寄与Top3合計",
        "J5": "リスクHHI",
        "J6": "実効リスク銘柄数",
        "J7": "金額Top1との差",
        "J8": "相関集中メモ",
        "J9": "リスク寄与Top2",
        "J10": "リスク寄与Top3",
        "J11": "リスク計算状態",
        "M1": "TOPIX比較・実行状態",
        "M2": "過去1年PFリターン",
        "M3": "過去1年TOPIX系",
        "M4": "α(PF-TOPIX系)",
        "M5": "価格データ時刻",
        "M6": "実行ステータス",
        "M7": "エラー件数",
        "M8": "β(TOPIX)",
        "M9": "トラッキングエラー",
        "M10": "情報レシオ",
        "M11": "下落捕捉率",
        "P1": "ストレステスト",
        "P2": "シナリオ",
        "Q2": "損失率",
        "R2": "概算損失",
        "P3": "TOPIX -10%",
        "P4": "TOPIX -20%",
        "P5": "最大リスク銘柄 -20%",
        "P6": "上位3リスク銘柄 -20%",
        "P7": "TOPIX -10% + RiskTop3追加 -15%",
        "P8": "想定最大DD",
        "T1": "リスク寄与Top5",
        "T2": "順位",
        "U2": "企業名",
        "V2": "W",
        "W2": "年率ボラ",
        "X2": "リスク寄与",
    }
    for cell, label in labels.items():
        set_cell(dash, cell, label)

    # values（既存 + 新規）
    total_cost = sum(int(r.get("cost_value", 0)) for r in valid_price_rows) if valid_price_rows else 0
    total_pnl = sum(int(r.get("pnl_jpy", 0)) for r in valid_price_rows) if valid_price_rows else 0
    total_pnl_pct = (total_market_value / total_cost - 1) if total_cost > 0 else 0
    symbol_count = len(unique_tickers)

    good_rows = [r for r in valid_price_rows if "：好調" in str(r.get("q3_judgement", ""))]
    poor_rows = [r for r in valid_price_rows if "：不調" in str(r.get("q3_judgement", ""))]
    good_count = len(good_rows)
    poor_count = len(poor_rows)
    good_weight_sum = sum(float(r.get("weight", 0.0)) for r in good_rows if isinstance(r.get("weight"), float))
    poor_weight_sum = sum(float(r.get("weight", 0.0)) for r in poor_rows if isinstance(r.get("weight"), float))
    poor_24m_count = sum(
        1 for r in poor_rows
        if isinstance(r.get("observed_months"), int) and r.get("observed_months") >= 24
    )
    increased_qty_count = sum(
        1 for r in valid_price_rows
        if parse_float(r.get("first_seen_qty")) is not None
        and parse_float(r.get("shares")) is not None
        and float(parse_float(r.get("shares"))) > float(parse_float(r.get("first_seen_qty")))
    )

    set_cell(dash, "B1", now_jst.strftime("%Y-%m-%d %H:%M:%S"))
    set_cell(dash, "B2", int(total_market_value))
    set_cell(dash, "B3", int(total_cost))
    set_cell(dash, "B4", int(total_pnl))
    set_cell(dash, "B5", float(total_pnl_pct))
    set_cell(dash, "B6", int(symbol_count))

    set_cell(dash, "E2", float(top1))
    set_cell(dash, "E3", float(top3))
    set_cell(dash, "E4", float(hhi))
    set_cell(dash, "E5", int(cnt_exit))
    set_cell(dash, "E6", int(cnt_caution))
    set_cell(dash, "E7", int(cnt_ok))
    set_cell(dash, "E8", c_top1)
    set_cell(dash, "E9", c_top2)
    set_cell(dash, "E10", c_top3)

    set_cell(dash, "H2", float(pf_ret_base))
    set_cell(dash, "H3", float(pf_ret_opt))
    set_cell(dash, "H4", float(pf_ret_pess))
    set_cell(dash, "H5", int(unallocated_cash))
    set_cell(dash, "H6", int(good_count))
    set_cell(dash, "H7", int(poor_count))
    set_cell(dash, "H8", float(good_weight_sum))
    set_cell(dash, "H9", float(poor_weight_sum))
    set_cell(dash, "H10", int(poor_24m_count))
    set_cell(dash, "H11", int(increased_qty_count))

    analytics_top_entries = portfolio_analytics.get("top_entries", [])
    top1_risk_text = ""
    top2_risk_text = ""
    top3_risk_text = ""
    if len(analytics_top_entries) >= 1:
        top1_risk_text = f'{analytics_top_entries[0]["name"]} / {analytics_top_entries[0]["risk_contribution_pct"]:.1%}'
    if len(analytics_top_entries) >= 2:
        top2_risk_text = f'{analytics_top_entries[1]["name"]} / {analytics_top_entries[1]["risk_contribution_pct"]:.1%}'
    if len(analytics_top_entries) >= 3:
        top3_risk_text = f'{analytics_top_entries[2]["name"]} / {analytics_top_entries[2]["risk_contribution_pct"]:.1%}'

    set_cell(dash, "K2", portfolio_analytics.get("portfolio_vol_annual", ""))
    set_cell(dash, "K3", top1_risk_text)
    set_cell(dash, "K4", portfolio_analytics.get("top3_risk_contribution_pct", ""))
    set_cell(dash, "K5", portfolio_analytics.get("risk_hhi", ""))
    set_cell(dash, "K6", portfolio_analytics.get("effective_risk_symbol_count", ""))
    set_cell(dash, "K7", portfolio_analytics.get("money_top1_difference", ""))
    set_cell(dash, "K8", portfolio_analytics.get("correlation_concentration_memo", ""))
    set_cell(dash, "K9", top2_risk_text)
    set_cell(dash, "K10", top3_risk_text)
    set_cell(dash, "K11", portfolio_analytics.get("status", ""))

    set_cell(dash, "N2", pf_cum)
    set_cell(dash, "N3", topix_cum)
    set_cell(dash, "N4", alpha)
    set_cell(dash, "N5", latest_bar)
    set_cell(dash, "N6", exec_status)
    set_cell(dash, "N7", int(error_count))
    set_cell(dash, "N8", portfolio_analytics.get("beta_to_topix", ""))
    set_cell(dash, "N9", portfolio_analytics.get("tracking_error_1y", ""))
    set_cell(dash, "N10", portfolio_analytics.get("information_ratio_1y", ""))
    set_cell(dash, "N11", portfolio_analytics.get("downside_capture_1y", ""))

    stress_keys = [
        (3, "stress_topix_minus10_loss_pct"),
        (4, "stress_topix_minus20_loss_pct"),
        (5, "stress_top1_risk_minus20_loss_pct"),
        (6, "stress_top3_risk_minus20_loss_pct"),
        (7, "stress_topix_minus10_top3_minus15_loss_pct"),
        (8, "max_drawdown_1y"),
    ]
    for row_num, key in stress_keys:
        loss_pct = portfolio_analytics.get(key, "")
        set_cell(dash, f"Q{row_num}", loss_pct)
        loss_jpy = ""
        if loss_pct != "" and loss_pct is not None:
            loss_jpy = int(round(total_market_value * float(loss_pct)))
        set_cell(dash, f"R{row_num}", loss_jpy)

    for idx, entry in enumerate(analytics_top_entries[:5], start=1):
        row_num = idx + 2
        set_cell(dash, f"T{row_num}", idx)
        set_cell(dash, f"U{row_num}", entry.get("name", ""))
        set_cell(dash, f"V{row_num}", entry.get("weight", ""))
        set_cell(dash, f"W{row_num}", entry.get("annual_vol", ""))
        set_cell(dash, f"X{row_num}", entry.get("risk_contribution_pct", ""))

    # -------------------------
    # 10.5) Build history row (AU:CG)
    # -------------------------
    snapshot_date = now_jst.strftime("%Y-%m-%d")
    history_row = [
        snapshot_date,
        now_jst.strftime("%Y-%m-%d %H:%M:%S"),
        int(total_market_value),
        int(total_cost),
        int(total_pnl),
        float(total_pnl_pct),
        int(symbol_count),
        float(top1),
        float(top3),
        float(hhi),
        int(cnt_exit),
        int(cnt_caution),
        int(cnt_ok),
        float(pf_ret_base),
        float(pf_ret_opt),
        float(pf_ret_pess),
        int(value_10y_base),
        int(value_10y_opt),
        int(value_10y_pess),
        int(unallocated_cash),
        pf_cum,
        topix_cum,
        alpha,
        portfolio_analytics.get("portfolio_vol_annual", ""),
        portfolio_analytics.get("top1_risk_contributor_ticker", ""),
        portfolio_analytics.get("top1_risk_contribution_pct", ""),
        portfolio_analytics.get("top3_risk_contribution_pct", ""),
        portfolio_analytics.get("risk_hhi", ""),
        portfolio_analytics.get("effective_risk_symbol_count", ""),
        portfolio_analytics.get("beta_to_topix", ""),
        portfolio_analytics.get("tracking_error_1y", ""),
        portfolio_analytics.get("information_ratio_1y", ""),
        portfolio_analytics.get("downside_capture_1y", ""),
        portfolio_analytics.get("max_drawdown_1y", ""),
        portfolio_analytics.get("stress_topix_minus10_loss_pct", ""),
        portfolio_analytics.get("stress_topix_minus20_loss_pct", ""),
        portfolio_analytics.get("stress_top1_risk_minus20_loss_pct", ""),
        portfolio_analytics.get("stress_top3_risk_minus20_loss_pct", ""),
        portfolio_analytics.get("stress_topix_minus10_top3_minus15_loss_pct", ""),
    ]

    history_payload = build_history_payload(ws, snapshot_date, history_row)
    obs_db_matrix = build_observation_db_matrix(obs_db, rows, now_jst)
    dividend_cache_matrix = build_dividend_cache_matrix(updated_dividend_cache, unique_tickers)

    fixed_row_count = MAX_ROW - START_ROW + 1
    obs_db_matrix_padded = pad_matrix_rows(obs_db_matrix, fixed_row_count, OBS_DB_COLS)
    dividend_cache_matrix_padded = pad_matrix_rows(
        dividend_cache_matrix,
        fixed_row_count,
        DIVIDEND_CACHE_COLS,
    )

    # -------------------------
    # 11) Write to sheet (single batch update / no clear)
    # -------------------------
    update_payload = [
        {"range": DASH_RANGE, "values": dash},
        {"range": DIVIDEND_OUTPUT_HEADER_RANGE, "values": [["配当利回り"]]},
        {"range": OBS_HEADER_RANGE, "values": [OBS_HEADERS]},
        {"range": OUT_RANGE, "values": output_matrix},
        {"range": HIST_HEADER_RANGE, "values": [HIST_HEADERS]},
        {"range": OBS_DB_HEADER_RANGE, "values": [OBS_DB_HEADERS]},
        {"range": OBS_DB_DATA_RANGE, "values": obs_db_matrix_padded},
        {"range": DIVIDEND_CACHE_HEADER_RANGE, "values": [DIVIDEND_CACHE_HEADERS]},
        {"range": DIVIDEND_CACHE_DATA_RANGE, "values": dividend_cache_matrix_padded},
    ]

    if history_payload is not None:
        update_payload.extend(history_payload)

    ws.batch_update(
        update_payload,
        value_input_option="RAW",
    )


if __name__ == "__main__":
    main()
