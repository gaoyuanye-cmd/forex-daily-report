#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全自动 EUR/USD 专业分析脚本
包含：技术指标信号、CFTC COT机构持仓、IG散户情绪、关联市场、DeepSeek AI策略、钉钉推送
"""

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

# 尝试导入 AKShare (用于COT数据)
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️ 未安装 akshare，COT持仓数据将无法获取。请执行: pip install akshare")

# ================== 配置区 (从环境变量读取) ==================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DINGTALK_WEBHOOK = os.environ.get("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET = os.environ.get("DINGTALK_SECRET", "")  # 如果机器人无加签则留空

# 符号定义
EURUSD_TICKER = "EURUSD=X"
DXY_TICKER = "DX-Y.NYB"
EURGBP_TICKER = "EURGBP=X"
EURJPY_TICKER = "EURJPY=X"
DAX_TICKER = "^GDAXI"
US10Y_TICKER = "^TNX"
DE10Y_TICKER = "DE10Y.DE"  # 德国10年国债（可能无数据，忽略即可）

# ================== 数据获取工具 ==================
def fetch_historical(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """使用 yfinance 获取历史K线"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"⚠️ {ticker} 数据为空")
        return data
    except Exception as e:
        print(f"❌ 获取 {ticker} 失败: {e}")
        return pd.DataFrame()

def get_related_prices() -> Dict[str, Optional[float]]:
    """获取相关品种最新收盘价"""
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
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                price = float(df["Close"].iloc[-1].iloc[0])
                results[name] = round(price, 4)
            else:
                results[name] = None
        except Exception as e:
            results[name] = None
    return results


# ================== 技术指标计算 ==================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})

def compute_bollinger(series: pd.Series, period=20, std=2) -> pd.DataFrame:
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    return pd.DataFrame({
        "Mid": sma,
        "Upper": sma + std * std_dev,
        "Lower": sma - std * std_dev
    })

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ================== 综合交易信号 ==================
def generate_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """基于日线数据计算技术指标信号和方向评分"""
    if df.empty or len(df) < 50:
        return {"direction": "数据不足", "score": 0, "details": {}}

    close = df["Close"].squeeze()
    price = float(close.iloc[-1])

    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    macd_df = compute_macd(close)
    macd_val = float(macd_df["MACD"].iloc[-1])
    signal_val = float(macd_df["Signal"].iloc[-1])
    hist = float(macd_df["Histogram"].iloc[-1])

    bb = compute_bollinger(close)
    bb_upper = float(bb["Upper"].iloc[-1])
    bb_lower = float(bb["Lower"].iloc[-1])
    bb_mid = float(bb["Mid"].iloc[-1])

    sma20 = float(compute_sma(close, 20).iloc[-1])
    sma50 = float(compute_sma(close, 50).iloc[-1]) if len(close) >= 50 else None

    score = 0
    details = {}

    # RSI 信号
    if rsi < 30:
        details["RSI"] = f"超卖({rsi:.1f})，利多"
        score += 2
    elif rsi > 70:
        details["RSI"] = f"超买({rsi:.1f})，利空"
        score -= 2
    else:
        details["RSI"] = f"中性({rsi:.1f})"

    # MACD 信号
    if macd_val > signal_val and hist > 0:
        details["MACD"] = "金叉且柱状图为正，利多"
        score += 1
    elif macd_val < signal_val and hist < 0:
        details["MACD"] = "死叉且柱状图为负，利空"
        score -= 1
    else:
        details["MACD"] = "方向不明"

    # 布林带
    if price >= bb_upper:
        details["Bollinger"] = f"触及上轨({bb_upper:.4f})，超买"
        score -= 1
    elif price <= bb_lower:
        details["Bollinger"] = f"触及下轨({bb_lower:.4f})，超卖"
        score += 1
    else:
        details["Bollinger"] = f"通道内，中轨{bb_mid:.4f}"

    # 均线排列
    if sma50 is not None:
        if price > sma20 > sma50:
            details["MA"] = f"多头排列 (MA20:{sma20:.4f} > MA50:{sma50:.4f})"
            score += 1
        elif price < sma20 < sma50:
            details["MA"] = f"空头排列 (MA20:{sma20:.4f} < MA50:{sma50:.4f})"
            score -= 1
        else:
            details["MA"] = "均线缠绕"
    else:
        details["MA"] = f"价格相对MA20 ({sma20:.4f}) {'上方' if price > sma20 else '下方'}"

    # 最终方向
    if score >= 3:
        direction = "偏多"
    elif score <= -3:
        direction = "偏空"
    else:
        direction = "震荡"

    return {
        "direction": direction,
        "score": score,
        "price": price,
        "rsi": rsi,
        "macd_hist": hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "sma20": sma20,
        "sma50": sma50,
        "details": details
    }


# ================== COT 机构持仓数据（AKShare） ==================
def fetch_cot_akshare() -> Dict[str, Any]:
    """通过AKShare获取欧元期货COT非商业净持仓"""
    if not AKSHARE_AVAILABLE:
        return {}
    try:
        # AKShare 接口：CFTC持仓报告
        cot_df = ak.cftc_commodity_net_position(symbol="欧元/美元")
        if cot_df is None or cot_df.empty:
            print("⚠️ AKShare COT数据为空")
            return {}

        # 取最近两行计算周变动
        latest = cot_df.iloc[-1]
        prev = cot_df.iloc[-2] if len(cot_df) > 1 else latest
        net_now = int(latest["非商业净持仓"])
        net_prev = int(prev["非商业净持仓"])
        return {
            "date": str(latest.name)[:10],
            "net_noncommercial": net_now,
            "weekly_change": net_now - net_prev,
            "long": int(latest["非商业多头持仓"]),
            "short": int(latest["非商业空头持仓"]),
        }
    except Exception as e:
        print(f"❌ AKShare COT获取异常: {e}")
    return {}


# ================== IG 散户情绪（公开页面抓取） ==================
def fetch_ig_sentiment() -> Dict[str, Optional[float]]:
    """抓取IG官网EUR/USD散户多空比例（可能因页面结构改变而失效）"""
    result = {"long_pct": None, "short_pct": None}
    try:
        url = "https://www.ig.com/en/forex/markets-forex/eur-usd"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return result
        # 简单文本搜索（实际使用时可能需要更健壮的解析）
        if "sentiment" in resp.text.lower():
            # 示例提取逻辑（具体需根据实际DOM编写，此处仅做示意）
            # 您可以在此补充HTML解析代码
            pass
    except Exception as e:
        pass
    return result


# ================== AI 分析生成 ==================
def generate_ai_analysis(signals: Dict, related: Dict, cot: Dict) -> str:
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置 DEEPSEEK_API_KEY"

    price = signals.get("price", "N/A")
    direction = signals.get("direction", "N/A")
    score = signals.get("score", 0)
    details = json.dumps(signals.get("details", {}), ensure_ascii=False)
    related_str = json.dumps(related, ensure_ascii=False)
    cot_str = "无数据" if not cot else f"非商业净持仓：{cot.get('net_noncommercial','N/A')}手，周变动：{cot.get('weekly_change','N/A')}手"

    prompt = f"""
你是一位拥有10年经验的外汇策略师，专精EUR/USD。

【当前市场状态】
- 欧元/美元现价：{price}
- 技术信号方向：{direction} (综合评分：{score})
- 技术指标详情：
{details}

【机构持仓】
最新的CFTC持仓报告显示：
{cot_str}
（净持仓正值为多头占优，负值为空头占优，周变动反映机构情绪边际变化）

【关联市场】
{related_str}

请结合以上所有信息，撰写一份**可执行的短线交易策略报告**（持仓周期：日内至2天），需包含以下部分：
1. **多空结论**：明确今日倾向（做多/做空/观望），并给出技术+持仓双重依据。
2. **关键支撑/阻力**：给出2个有效支撑和2个有效阻力位，解释其来源。
3. **机构行为解读**：结合COT净持仓及周变动，判断当前机构方向及是否支持你的交易方向。
4. **关联市场交叉验证**：例如DXY、利差、DAX与欧元的关系是否一致。
5. **具体策略**：
   - 若适宜交易：入场区域、止损点数/价位、第一目标、风险报酬比。
   - 若建议观望：明确等待哪种条件（如突破XX关口或MACD金叉等）。
6. **风险提示**：今天需重点关注的经济数据/事件，及突发风险。

要求：专业、具体，避免模糊。直接输出结论，用加粗标注方向。
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是专业外汇策略师，回答直接、具体，包含明确交易建议。"},
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
            return f"❌ AI API错误 {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"❌ AI请求异常: {e}"


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
            "title": "📊 EUR/USD 智能分析",
            "text": content[:4000]
        }
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.json().get("errcode") == 0:
            print("✅ 钉钉推送成功")
            return True
        else:
            print(f"❌ 钉钉返回错误: {resp.json()}")
            return False
    except Exception as e:
        print(f"❌ 推送异常: {e}")
        return False


# ================== 主流程 ==================
def main():
    print("=" * 50)
    print("📡 获取 EUR/USD 日线数据...")
    df = fetch_historical(EURUSD_TICKER, period="3mo", interval="1d")
    if df.empty:
        print("❌ 无法获取EUR/USD数据，终止。")
        sys.exit(1)

    print("🔍 计算技术指标与交易信号...")
    signals = generate_signals(df)

    print("🌐 获取相关品种价格...")
    related = get_related_prices()

    print("🏦 获取 CFTC COT 机构持仓...")
    cot_data = fetch_cot_akshare()

    print("🤖 调用 DeepSeek 生成策略分析...")
    ai_report = generate_ai_analysis(signals, related, cot_data)

    # 组装最终报告
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    price = signals.get("price", "N/A")
    direction = signals.get("direction", "N/A")
    score = signals.get("score", 0)

    report = f"""# 📊 欧元/美元智能分析报告

**生成时间**：{now_str}
**当前汇率**：{price}
**技术信号**：{direction} (评分：{score})

---

## 🏦 机构持仓 (COT)
- 非商业净持仓：{cot_data.get('net_noncommercial','N/A')} 手
- 周变动：{cot_data.get('weekly_change','N/A')} 手
- 解读：{'机构净多头' if cot_data.get('net_noncommercial',0)>0 else '机构净空头'}  {'增加' if cot_data.get('weekly_change',0)>0 else '减少'}仓位

---

## 🤖 AI 策略分析
{ai_report}

---

*免责声明：本报告由AI自动生成，仅供研究参考，不构成投资建议。*
"""
    print("\n" + report)

    # 推送
    if DINGTALK_WEBHOOK:
        send_dingtalk(DINGTALK_WEBHOOK, DINGTALK_SECRET, report)
    else:
        print("⚠️ 未配置钉钉Webhook，跳过推送。")


if __name__ == "__main__":
    main()
