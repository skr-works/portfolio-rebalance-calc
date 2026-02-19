import os
import json
import re
import math
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

BENCHMARK_TICKER = "^TOPX"

# rebalance固定パラメータ
MAX_SINGLE_WEIGHT = 0.15
TARGET_SINGLE_WEIGHT = 0.08

# forecast clip
RET_CLIP_MIN = -0.30
RET_CLIP_MAX = 0.30

# health閾値
DD_EXIT = -0.30
DD_CAUTION = -0.20

# output ranges
DASH_RANGE = "A1:N10"
OUT_RANGE = f"E{START_ROW}:Q{MAX_ROW}"

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
    code = str(code).strip()
    if re.match(r"^\d{4}$", code):
        return f"{code}.T"
    if re.match(r"^\d{4}\.T$", code):
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


def build_dashboard_matrix():
    # 10 rows x 14 cols (A..N)
    return [["" for _ in range(14)] for __ in range(10)]


def set_cell(mat, a1: str, value):
    """
    A1形式（A1〜N10）だけを対象にセットする。
    """
    col_letters = re.match(r"^([A-N]+)", a1).group(1)
    row_num = int(re.match(r"^[A-N]+(\d+)$", a1).group(1))

    # A=0 ... N=13
    col_map = {chr(ord("A") + i): i for i in range(14)}
    if col_letters not in col_map:
        raise ValueError(f"Invalid column: {col_letters}")
    c = col_map[col_letters]
    r = row_num - 1
    if not (0 <= r < 10 and 0 <= c < 14):
        raise ValueError(f"Cell out of dashboard range: {a1}")

    mat[r][c] = value


def clip(x, lo, hi):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(lo, min(hi, float(x)))


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

    raw_data = ws.get(f"A{START_ROW}:D{MAX_ROW}")  # list[list[str]]

    # rows: 14..2000 を全て保持（出力行数固定のため）
    total_rows = MAX_ROW - START_ROW + 1
    rows = []
    tickers = []
    input_invalid_count = 0

    # バリデーション
    for i in range(total_rows):
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
            "code": code,
            "name": row_data[1] if len(row_data) > 1 else "",
            "shares": None,
            "cost": None,
            "ticker": None,
        }

        tk = parse_ticker(code)
        if tk is None:
            info["status"] = "DATA_NG"
            info["health_level"] = "DATA_NG"
            info["health_reason"] = "INPUT_NG:CODE_FORMAT"
            input_invalid_count += 1
            rows.append(info)
            continue

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

        info["ticker"] = tk
        info["shares"] = float(shares)
        info["cost"] = float(cost)

        tickers.append(tk)
        rows.append(info)

    # 有効銘柄（入力がOK）
    unique_tickers = sorted(set(tickers))
    valid_input_count = len(unique_tickers)

    # -------------------------
    # 2) yfinance fetch (batch)
    # -------------------------
    # 仕様：current_price は history(period="5d", interval="1d") 最新Close
    # 仕様：health/backtest/forecast は Close のみ利用

    all_tickers_for_price = unique_tickers[:]  # copy
    # ベンチも含めて一括で落とす
    all_tickers_for_price.append(BENCHMARK_TICKER)

    # yfinance download
    close_5d = pd.DataFrame()
    close_1y = pd.DataFrame()
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
        close_1y = extract_close_df(dl_1y, [all_tickers_for_price[0]] if len(all_tickers_for_price) == 1 else all_tickers_for_price)
    except Exception:
        close_1y = pd.DataFrame()

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

    # benchmark availability
    benchmark_fail_count = 0
    if BENCHMARK_TICKER not in close_1y.columns or close_1y[BENCHMARK_TICKER].dropna().empty:
        benchmark_fail_count = 1

    # -------------------------
    # 3) info fetch (dividendYield, sector) - fail-safe
    # -------------------------
    # 仕様：dividendYield は info.get("dividendYield") のみ、取れなければ0
    # sector も取れなければ空（※仕様に従い「取れれば埋める」）
    div_yield_map = {tk: 0.0 for tk in unique_tickers}
    sector_map = {tk: "" for tk in unique_tickers}

    for tk in unique_tickers:
        try:
            info = yf.Ticker(tk).info  # ここが落ちても価格は巻き添えにしない
            dy = info.get("dividendYield", None)
            if isinstance(dy, (int, float)) and not (isinstance(dy, float) and math.isnan(dy)):
                div_yield_map[tk] = float(dy)
            sec = info.get("sector", "")
            if isinstance(sec, str):
                sector_map[tk] = sec
        except Exception:
            # 取れない場合は仕様通り 0 / "" のまま
            pass

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

        # current_price: close_5d latest non-NaN
        cur = None
        try:
            if tk in close_5d.columns:
                s = close_5d[tk].dropna()
                if not s.empty:
                    cur = float(s.iloc[-1])
        except Exception:
            cur = None

        if cur is None or (isinstance(cur, float) and math.isnan(cur)):
            # 価格取得失敗 → 行はDATA_NG（仕様：E〜Qは空、M=DATA_NG、N=理由）
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

        # sector
        row["sector"] = sector_map.get(tk, "")

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
        tk = rr.get("ticker", "")
        reason = rr.get("health_reason", "")
        return f"{tk}:{reason}" if tk else ""
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
    # 9) backtest（仕様固定：1y、固定ウェイト近似、TOPIX）
    # -------------------------
    pf_cum = ""
    topix_cum = ""
    alpha = ""

    if exec_status != "FAILED" and benchmark_fail_count == 0 and len(valid_price_rows) > 0:
        # 日次リターンを作る
        cols = [BENCHMARK_TICKER] + [r["ticker"] for r in valid_price_rows]
        # close_1y から必要列を取る（無い列がある場合は欠損→dropnaで落ちる）
        try:
            df_close = close_1y[cols].copy()
            df_ret = df_close.pct_change()

            # 仕様：いずれか欠損がある日は除外
            df_ret = df_ret.dropna()

            if not df_ret.empty:
                weights_dict = {r["ticker"]: float(r["weight"]) for r in valid_price_rows if isinstance(r.get("weight"), float)}
                # PF日次リターン
                pf_daily = None
                for tk, w in weights_dict.items():
                    if tk in df_ret.columns:
                        s = df_ret[tk] * w
                        pf_daily = s if pf_daily is None else (pf_daily + s)

                if pf_daily is not None:
                    pf_cum = (1 + pf_daily).prod() - 1
                    topix_cum = (1 + df_ret[BENCHMARK_TICKER]).prod() - 1
                    alpha = pf_cum - topix_cum
        except Exception:
            # backtestが壊れても exec_status は変えない（仕様にないため）
            pf_cum = topix_cum = alpha = ""

    # -------------------------
    # 10) Build output matrices
    # -------------------------
    # E..Q = 13 columns
    output_matrix = []
    for row in rows:
        if row.get("is_empty"):
            output_matrix.append([""] * 13)
            continue

        if row.get("status") != "OK":
            # 仕様：E〜Qは空、M=DATA_NG、N=理由
            output_matrix.append([
                "", "", "", "", "", "", "", "",   # E..L
                "DATA_NG",                         # M
                str(row.get("health_reason", "")), # N
                "", "", ""                         # O..Q
            ])
            continue

        # OK（価格もOKなら valid_price_rows に入っているはず。念のため）
        # sector / health / ret_base / rebalance を含めて出す
        ticker = row.get("ticker", "")
        cur = row.get("current_price", "")
        mv = row.get("market_value", "")
        cv = row.get("cost_value", "")
        pnl = row.get("pnl_jpy", "")
        pnlp = row.get("pnl_pct", "")
        w = row.get("weight", "")
        sec = row.get("sector", "")

        hl = row.get("health_level", "DATA_NG")
        hr = row.get("health_reason", "")

        rb = row.get("ret_base", "")

        act = row.get("action", "HOLD")
        trade = row.get("trade_jpy", 0)

        output_matrix.append([
            ticker,     # E
            cur,        # F
            mv,         # G
            cv,         # H
            pnl,        # I
            pnlp,       # J
            w,          # K
            sec,        # L
            hl,         # M
            hr,         # N
            rb,         # O
            act,        # P
            trade,      # Q
        ])

    # dashboard matrix（A1:N10）
    dash = build_dashboard_matrix()

    # labels（固定）
    labels = {
        "A1": "更新日時(JST)", "A2": "総評価額(JPY)", "A3": "総取得額(JPY)", "A4": "総損益(JPY)", "A5": "総損益(%)", "A6": "銘柄数",
        "D2": "上位1銘柄ウェイト(%)", "D3": "上位3銘柄ウェイト合計(%)", "D4": "HHI(銘柄集中度)", "D5": "EXIT銘柄数", "D6": "CAUTION銘柄数", "D7": "OK銘柄数",
        "D8": "要注意Top1", "D9": "要注意Top2", "D10": "要注意Top3",
        "H2": "期待年率(base)", "H3": "期待年率(opt)", "H4": "期待年率(pess)",
        "H5": "10年後価値(base)", "H6": "10年後価値(opt)", "H7": "10年後価値(pess)", "H8": "リバランス余剰現金(概算)",
        "L2": "過去1年PFリターン", "L3": "過去1年TOPIXリターン", "L4": "α(PF-TOPIX)",
        "L5": "価格データ時刻", "L6": "実行ステータス", "L7": "エラー件数",
    }
    for k, v in labels.items():
        set_cell(dash, k, v)

    # values（固定セル）
    total_cost = sum(int(r.get("cost_value", 0)) for r in valid_price_rows) if valid_price_rows else 0
    total_pnl = sum(int(r.get("pnl_jpy", 0)) for r in valid_price_rows) if valid_price_rows else 0
    total_pnl_pct = (total_market_value / total_cost - 1) if total_cost > 0 else 0
    symbol_count = len(unique_tickers)

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

    set_cell(dash, "I2", float(pf_ret_base))
    set_cell(dash, "I3", float(pf_ret_opt))
    set_cell(dash, "I4", float(pf_ret_pess))
    set_cell(dash, "I5", int(value_10y_base))
    set_cell(dash, "I6", int(value_10y_opt))
    set_cell(dash, "I7", int(value_10y_pess))
    set_cell(dash, "I8", int(unallocated_cash))

    set_cell(dash, "M2", pf_cum)
    set_cell(dash, "M3", topix_cum)
    set_cell(dash, "M4", alpha)
    set_cell(dash, "M5", latest_bar)
    set_cell(dash, "M6", exec_status)
    set_cell(dash, "M7", int(error_count))

    # -------------------------
    # 11) Write to sheet (clear -> write)
    # -------------------------
    ws.batch_clear([DASH_RANGE, OUT_RANGE])

    ws.batch_update(
        [
            {"range": DASH_RANGE, "values": dash},
            {"range": OUT_RANGE, "values": output_matrix},
        ],
        value_input_option="RAW",
    )


if __name__ == "__main__":
    main()
