#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ================== 配置区 ==================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = os.environ.get("DINGTALK_SECRET", "")  # 如果无加签则留空

# 要用到的 yfinance 代码
EURUSD_TICKER = "EURUSD=X"
DXY_TICKER = "DX-Y.NYB"
EURGBP_TICKER = "EURGBP=X"
EURJPY_TICKER = "EURJPY=X"
DAX_TICKER = "^GDAXI"
US10Y_TICKER = "^TNX"        # 10年期美债收益率
DE10Y_TICKER = "DE10Y.DE"    # 德国10年国债收益率 (Yahoo可能有)

# 数据获取周期：日线用于趋势，4小时用于入场细化
PERIOD_DAILY = "3mo"    # 3个月日线
PERIOD_4H = "30d"       # 30天4小时K线 (实际Yahoo 4h数据需用 interval="60m" 后重采样，这里简化用1h数据)
INTERVAL_4H = "60m"     # 获取1小时数据，后续可重采样为4h，也可直接用1h分析

# ================== 数据获取函数 ==================
def fetch_historical(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """使用 yfinance 获取历史数据，返回 DataFrame"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"⚠️ 获取 {ticker} 数据为空")
        return data
    except Exception as e:
        print(f"❌ 获取 {ticker} 数据失败: {e}")
        return pd.DataFrame()


# ================== 技术指标计算 ==================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})

def compute_bollinger(series: pd.Series, period: int = 20, std: int = 2) -> pd.DataFrame:
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return pd.DataFrame({"Mid": sma, "Upper": upper, "Lower": lower})

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ================== 交易信号评分 ==================
def generate_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    基于日线数据生成综合信号：
    返回：方向倾向（偏多/偏空/震荡）、总体评分、各指标信号摘要
    """
    if df.empty or len(df) < 30:
        return {"direction": "数据不足", "score": 0, "details": {}}

    close = df["Close"].squeeze()
    rsi = compute_rsi(close).iloc[-1]
    macd_df = compute_macd(close)
    macd_val = macd_df["MACD"].iloc[-1]
    signal_val = macd_df["Signal"].iloc[-1]
    hist = macd_df["Histogram"].iloc[-1]
    bb = compute_bollinger(close)
    bb_upper = bb["Upper"].iloc[-1]
    bb_lower = bb["Lower"].iloc[-1]
    bb_mid = bb["Mid"].iloc[-1]
    sma20 = compute_sma(close, 20).iloc[-1]
    sma50 = compute_sma(close, 50).iloc[-1] if len(close) >= 50 else None
    price = close.iloc[-1]

    score = 0
    details = {}

    # 1. RSI 信号
    if rsi < 30:
        details["RSI"] = f"超卖({rsi:.1f})，利多"
        score += 2
    elif rsi > 70:
        details["RSI"] = f"超买({rsi:.1f})，利空"
        score -= 2
    else:
        details["RSI"] = f"中性({rsi:.1f})"

    # 2. MACD 信号
    if macd_val > signal_val and hist > 0:
        details["MACD"] = "金叉且柱状图为正，利多"
        score += 1
    elif macd_val < signal_val and hist < 0:
        details["MACD"] = "死叉且柱状图为负，利空"
        score -= 1
    else:
        details["MACD"] = "方向不明"

    # 3. 价格与布林带关系
    if price >= bb_upper:
        details["Bollinger"] = f"触及上轨({bb_upper:.4f})，超买，利空"
        score -= 1
    elif price <= bb_lower:
        details["Bollinger"] = f"触及下轨({bb_lower:.4f})，超卖，利多"
        score += 1
    else:
        details["Bollinger"] = f"在通道内，中轨{bb_mid:.4f}"

    # 4. 均线关系
    if sma50 is not None:
        if price > sma20 > sma50:
            details["MA"] = f"多头排列(MA20:{sma20:.4f} > MA50:{sma50:.4f})，利多"
            score += 1
        elif price < sma20 < sma50:
            details["MA"] = f"空头排列(MA20:{sma20:.4f} < MA50:{sma50:.4f})，利空"
            score -= 1
        else:
            details["MA"] = "均线缠绕，趋势不明"
    else:
        details["MA"] = f"价格相对MA20({sma20:.4f})，{'上方' if price > sma20 else '下方'}"

    # 总结方向
    if score >= 3:
        direction = "偏多"
    elif score <= -3:
        direction = "偏空"
    else:
        direction = "震荡"

    return {
        "direction": direction,
        "score": score,
        "price": float(price),
        "rsi": float(round(rsi, 1)),
        "macd_hist": float(round(hist, 5)),
        "bb_upper": float(round(bb_upper, 4)),
        "bb_lower": float(round(bb_lower, 4)),
        "sma20": float(round(sma20, 4)),
        "sma50": float(round(sma50, 4)) if sma50 is not None else None,
        "details": details
    }


# ================== 相关品种数据获取 ==================
def get_related_prices() -> Dict[str, Optional[float]]:
    """获取DXY、EURGBP、EURJPY等最新价格"""
    tickers = {
        "DXY": DXY_TICKER,
        "EURGBP": EURGBP_TICKER,
        "EURJPY": EURJPY_TICKER,
        "DAX": DAX_TICKER,
        "US10Y": US10Y_TICKER,
        "DE10Y": DE10Y_TICKER,
    }
    results = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not data.empty:
                price = data["Close"].iloc[-1].item()  # 取标量值
                results[name] = round(price, 4)
            else:
                results[name] = None
        except Exception as e:
            print(f"获取 {name} 失败: {e}")
            results[name] = None
    return results


# ================== AI 分析增强 Prompt ==================
def generate_ai_analysis(signal_data: Dict, related: Dict) -> str:
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置 DEEPSEEK_API_KEY"

    price = signal_data.get("price", "N/A")
    direction = signal_data.get("direction", "N/A")
    score = signal_data.get("score", 0)
    details = json.dumps(signal_data.get("details", {}), ensure_ascii=False)
    related_str = json.dumps(related, ensure_ascii=False)

    prompt = f"""
你是一位拥有10年经验的外汇交易策略师，专精于EUR/USD。

【当前市场状态】
- 欧元/美元现价：{price}
- 技术信号评级：{direction} (综合评分{score})
- 详细指标：
{details}

【相关市场最新报价】
{related_str}

请基于以上全部信息撰写一份**可执行的短线交易分析报告**（持仓时间：日内至2天），必须包含以下部分：

1. **多空结论**：明确给出今日倾向（做多/做空/观望），并说明核心理由（至少2条）。
2. **关键价位**：列出日内有效支撑与阻力位（各2个），并解释其来源（前高、前低、均线、布林带等）。
3. **交易信号解读**：重点解释RSI、MACD、布林带、均线排列中最重要的1-2个信号的交易含义。
4. **关联市场佐证**：结合DXY、德美利差（DE10Y-US10Y）或风险情绪（DAX）等判断对欧元的潜在影响。
5. **建议策略**：
   - 若适宜交易，给出具体入场区域、止损设置（点数或价位）、第一目标位和风险报酬比。
   - 若建议观望，说明需要等待哪种条件（例如突破某价位、MACD金叉等）。
6. **风险提示**：今日需关注的经济数据/事件（可参考常规日历，如美国CPI、欧元区PMI等），及黑天鹅风险。

要求：语言专业、简洁，直接给出观点而非模棱两可。结论部分用**加粗**标注。
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一名专业的外汇策略师，回答直接、具体，给出明确的交易方向。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1200
    }
    try:
        resp = requests.post("https://api.deepseek.com/v1/chat/completions",
                             json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ AI API 错误 {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"❌ AI 请求异常: {e}"


# ================== 钉钉推送 ==================
def send_dingtalk(webhook: str, secret: str, content: str) -> bool:
    if not webhook:
        return False
    headers = {"Content-Type": "application/json"}
    url = webhook
    if secret:
        timestamp = str(round(time.time() * 1000))
        sign_str = f"{timestamp}\n{secret}"
        sign = base64.b64encode(
            hmac.new(secret.encode(), sign_str.encode(), hashlib.sha256).digest()
        ).decode()
        url = f"{webhook}&timestamp={timestamp}&sign={sign}"

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": "📊 欧元/美元智能分析",
            "text": content[:4000]
        }
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        result = resp.json()
        if result.get("errcode") == 0:
            print("✅ 钉钉推送成功")
            return True
        else:
            print(f"❌ 钉钉返回错误: {result}")
            return False
    except Exception as e:
        print(f"❌ 钉钉异常: {e}")
        return False


# ================== 主逻辑 ==================
def main():
    print("📡 获取欧元/美元日线数据...")
    df_daily = fetch_historical(EURUSD_TICKER, period="3mo", interval="1d")
    if df_daily.empty:
        print("无法获取EUR/USD日线数据，程序终止。")
        sys.exit(1)

    print("🔍 计算技术指标与信号...")
    signals = generate_signals(df_daily)

    print("🌐 获取相关品种数据...")
    related = get_related_prices()

    print("🤖 调用 DeepSeek 生成分析报告...")
    ai_report = generate_ai_analysis(signals, related)

    # 构造最终报告
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    price = signals.get("price", "N/A")
    direction = signals.get("direction", "N/A")
    score = signals.get("score", 0)

    report_lines = [
        f"# 📊 欧元/美元智能分析报告",
        f"**生成时间**：{today}",
        f"**当前汇率**：{price}",
        f"**技术信号**：{direction} (评分: {score})",
        "",
        "---",
        "",
        "## 🤖 AI 策略面分析",
        ai_report,
        "",
        "---",
        "*本报告由 AI 自动生成，仅供参考，不构成投资建议。*"
    ]
    full_report = "\n".join(report_lines)

    # 控制台输出
    print("\n" + full_report)

    # 推送至钉钉
    if DINGTALK_WEBHOOK:
        send_dingtalk(DINGTALK_WEBHOOK, DINGTALK_SECRET, full_report)
    else:
        print("⚠️ 未配置钉钉 Webhook，跳过推送。")


if __name__ == "__main__":
    main()
