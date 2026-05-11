#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hmac
import base64
import hashlib
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas_ta as ta

# ========== 配置部分 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 监控的外汇标的
FOREX_PAIRS = {
    "EURUSD=X": "欧元/美元",
    "GBPUSD=X": "英镑/美元",
    "DX-Y.NYB": "美元指数",
}

# ========== 技术指标计算函数 ==========
def calculate_rsi(data_series: pd.Series, window: int = 14) -> float:
    """计算 RSI (相对强弱指数)"""
    if len(data_series) < window + 1:
        return 50.0
    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_ema(data_series: pd.Series, window: int = 20) -> float:
    """计算 EMA (指数移动平均线)"""
    if len(data_series) < window:
        return float(data_series.iloc[-1])
    ema = data_series.ewm(span=window, adjust=False).mean()
    return float(ema.iloc[-1])

def calculate_sma(data_series: pd.Series, window: int = 50) -> float:
    """计算 SMA (简单移动平均线)"""
    if len(data_series) < window:
        return float(data_series.iloc[-1])
    sma = data_series.rolling(window=window).mean()
    return float(sma.iloc[-1])

def calculate_macd(data_series: pd.Series) -> Tuple[float, float, float]:
    """
    计算 MACD
    返回: (macd_line, signal_line, histogram)
    """
    if len(data_series) < 34:
        return 0.0, 0.0, 0.0
    short_ema = data_series.ewm(span=12, adjust=False).mean()
    long_ema = data_series.ewm(span=26, adjust=False).mean()
    macd_line = (short_ema - long_ema).iloc[-1]
    signal_line = macd_line.ewm(span=9, adjust=False).mean() if len(macd_line) > 9 else macd_line
    histogram = macd_line - signal_line
    return float(macd_line), float(signal_line), float(histogram)

def calculate_bollinger_bands(data_series: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[float, float, float]:
    """计算布林带"""
    if len(data_series) < window:
        return float(data_series.iloc[-1]), float(data_series.iloc[-1]), float(data_series.iloc[-1])
    sma = data_series.rolling(window=window).mean().iloc[-1]
    std = data_series.rolling(window=window).std().iloc[-1]
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return float(upper), float(sma), float(lower)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """计算 ATR (平均真实波幅)"""
    if len(high) < window + 1 or len(low) < window + 1 or len(close) < window + 1:
        return 0.0
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = ranges.rolling(window=window).mean()
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

# ========== 外汇数据获取（含基础技术指标）==========
def fetch_forex_data(ticker: str, name: str, days_back: int = 60) -> Dict[str, Any]:
    """获取外汇价格数据并计算技术指标"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            return {"name": name, "error": "No data available"}
        
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
        change_pct = (current - prev) / prev * 100
        
        # 获取20日和50日价格数据用于技术指标
        hist_20 = stock.history(period="1mo")
        hist_50 = stock.history(period="3mo")
        close_series_20 = hist_20['Close'] if not hist_20.empty else pd.Series()
        close_series_50 = hist_50['Close'] if not hist_50.empty else pd.Series()
        close_series_full = hist['Close']
        
        # 计算技术指标
        rsi = calculate_rsi(close_series_full, 14) if len(close_series_full) >= 15 else 50.0
        ema_20 = calculate_ema(close_series_full, 20) if len(close_series_full) >= 20 else current
        sma_50 = calculate_sma(close_series_full, 50) if len(close_series_full) >= 50 else current
        macd_line, signal_line, macd_hist = calculate_macd(close_series_full) if len(close_series_full) >= 34 else (0.0, 0.0, 0.0)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close_series_full, 20, 2) if len(close_series_full) >= 20 else (current, current, current)
        
        # 计算 ATR (波动率)
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'], 14) if len(hist) >= 15 else 0.0
        
        # 计算成交量比
        avg_volume = hist['Volume'].iloc[-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
        volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # 判断趋势方向
        if ema_20 > sma_50:
            trend = "多头排列（看涨）"
        elif ema_20 < sma_50:
            trend = "空头排列（看跌）"
        else:
            trend = "均线粘合（震荡）"
        
        # 判断RSI状态
        if rsi > 70:
            rsi_status = "超买区域（RSI > 70），注意回调风险"
        elif rsi < 30:
            rsi_status = "超卖区域（RSI < 30），可能反弹"
        else:
            rsi_status = f"中性区域（RSI = {rsi:.1f}）"
        
        # 判断布林带位置
        if current > upper_bb:
            bb_status = "突破上轨，强势超买"
        elif current < lower_bb:
            bb_status = "跌破下轨，弱势超卖"
        else:
            bb_status = "运行在布林带中轨与上轨/下轨之间"
        
        # 判断主要支撑阻力位（简单取前期高低点）
        recent_high = hist['High'].iloc[-20:].max() if len(hist) >= 20 else current * 1.02
        recent_low = hist['Low'].iloc[-20:].min() if len(hist) >= 20 else current * 0.98
        
        return {
            "name": name,
            "ticker": ticker,
            "current_price": round(current, 4),
            "prev_price": round(prev, 4),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "rsi_status": rsi_status,
            "ema_20": round(ema_20, 4),
            "sma_50": round(sma_50, 4),
            "trend": trend,
            "macd_line": round(macd_line, 5),
            "signal_line": round(signal_line, 5),
            "macd_histogram": round(macd_hist, 5),
            "upper_bb": round(upper_bb, 4),
            "middle_bb": round(middle_bb, 4),
            "lower_bb": round(lower_bb, 4),
            "bb_status": bb_status,
            "atr": round(atr, 5),
            "volume_ratio": round(volume_ratio, 2),
            "resistance": round(recent_high, 4),
            "support": round(recent_low, 4),
        }
    except Exception as e:
        return {"name": name, "error": str(e)}

def fetch_all_data() -> Dict[str, Any]:
    """批量获取所有外汇对的数据"""
    results = {}
    for ticker, name in FOREX_PAIRS.items():
        data = fetch_forex_data(ticker, name, days_back=60)
        results[name] = data
    return results

# ========== 经济日历获取（可选）==========
def fetch_economic_calendar() -> List[Dict[str, Any]]:
    """获取今日重要经济事件（简化版，返回模拟数据用于演示）"""
    today = datetime.now().strftime("%Y-%m-%d")
    events = []
    try:
        from ecocal import Calendar
        try:
            ec = Calendar(startHorizon=today, endHorizon=today, withDetails=True)
            if ec.getCalendar() and len(ec.getCalendar()) > 0:
                for date, date_events in ec.getCalendar().items():
                    for evt in date_events:
                        if any(keyword in evt.get('title', '') for keyword in ['EUR', 'USD', 'ECB', 'FOMC', 'Fed']):
                            events.append({
                                'time': evt.get('date', ''),
                                'country': evt.get('country', ''),
                                'event': evt.get('title', ''),
                                'impact': evt.get('impact', ''),
                                'expected': evt.get('forecast', ''),
                                'previous': evt.get('previous', ''),
                            })
        except Exception as e:
            print(f"加载经济日历数据失败: {e}")
    except ImportError:
        print("ecocal 库不可用，在 GitHub Actions 运行时会自动安装")
    return events

# ========== DeepSeek AI 分析 ==========
def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API 进行市场分析"""
    if not DEEPSEEK_API_KEY:
        return "❌ 请配置 DEEPSEEK_API_KEY 环境变量"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业的外汇策略分析师，擅长解读汇率市场数据并撰写客观、专业的日报。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ API 调用失败，状态码：{response.status_code}，详情：{response.text}"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

def generate_ai_analysis(data: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    """生成专业的外汇市场分析简报"""
    # 提取欧元/美元数据作为主要分析对象
    eurusd = data.get("欧元/美元", {})
    if "error" in eurusd:
        return f"数据获取失败: {eurusd.get('error')}"
    
    # 提取其他相关数据
    gbpusd = data.get("英镑/美元", {})
    dxy = data.get("美元指数", {})
    
    # 格式化经济事件
    events_text = ""
    if events:
        for evt in events[:5]:
            events_text += f"- {evt['time']} | {evt['country']} | {evt['event']}"
            if evt.get('impact'):
                events_text += f" | 影响: {evt['impact']}"
            events_text += "\n"
    else:
        events_text = "今日暂无重大经济数据发布"
    
    prompt = f"""
你是一位资深的外汇市场分析师。请基于以下市场数据，撰写一份专业的欧元/美元（EUR/USD）每日分析报告。

数据日期：{datetime.now().strftime('%Y-%m-%d')}

【核心货币对数据】
欧元/美元（EUR/USD）：
- 当前价格：{eurusd.get('current_price')}
- 昨日收盘：{eurusd.get('prev_price')}
- 日涨跌幅：{eurusd.get('change_pct')}%

【技术分析指标】
- RSI(14): {eurusd.get('rsi')} → {eurusd.get('rsi_status')}
- EMA(20): {eurusd.get('ema_20')}
- SMA(50): {eurusd.get('sma_50')}
- 均线趋势判断：{eurusd.get('trend')}
- MACD: 线 {eurusd.get('macd_line')} | 信号线 {eurusd.get('signal_line')} | 柱状图 {eurusd.get('macd_histogram')}
- 布林带: 上轨 {eurusd.get('upper_bb')} | 中轨 {eurusd.get('middle_bb')} | 下轨 {eurusd.get('lower_bb')}
- 布林带判断：{eurusd.get('bb_status')}
- ATR波动率：{eurusd.get('atr')}
- 成交量比：{eurusd.get('volume_ratio')}

【关键技术位】
- 关键阻力位：{eurusd.get('resistance')}
- 关键支撑位：{eurusd.get('support')}

【相关市场参考】
- 美元指数：当前 {dxy.get('current_price', 'N/A')}，涨跌幅 {dxy.get('change_pct', 'N/A')}%
- 英镑/美元：当前 {gbpusd.get('current_price', 'N/A')}，涨跌幅 {gbpusd.get('change_pct', 'N/A')}%

【今日经济日历】
{events_text}

请按照以下框架输出分析报告：

### 一、市场综述
总结当日欧元/美元走势特征，判断市场情绪（看涨/看跌/震荡）。

### 二、技术分析详解
- 趋势分析（基于EMA/SMA均线排列）
- 动量分析（基于RSI和MACD）
- 波动区间（基于布林带和ATR）
- 关键支撑阻力位判断

### 三、资金面与跨市场验证
- 美元指数与欧元的相关性分析
- 英镑/美元提供的参考信号

### 四、宏观事件展望
- 结合今日经济日历，判断数据发布对汇价的潜在影响
- 美欧央行政策预期对比

### 五、综合判断与操作建议
- 短期方向预测（1-2个交易日）
- 核心风险提示

请用语专业、客观，每条判断都要有数据支撑。最后输出一句核心结论。
"""
    return call_deepseek(prompt)

# ========== 推送模块 ==========
def send_to_dingtalk(webhook_url: str, content: str) -> bool:
    """发送 Markdown 消息到钉钉群"""
    headers = {"Content-Type": "application/json"}
    truncated = content[:4000] + "..." if len(content) > 4000 else content
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "📊 欧元/美元每日分析", "text": truncated}
    }
    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"钉钉发送失败：{e}")
        return False

def send_to_feishu(webhook_url: str, secret: str, content: str) -> bool:
    """发送消息到飞书群（支持签名校验）"""
    if not secret:
        return False
    timestamp = str(int(time.time()))
    sign_str = f"{timestamp}\n{secret}"
    sign = base64.b64encode(hmac.new(secret.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    headers = {"Content-Type": "application/json"}
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": "📊 欧元/美元每日分析",
                    "content": [[{"tag": "text", "text": content[:3000]}]]
                }
            }
        }
    }
    full_url = f"{webhook_url}?timestamp={timestamp}&sign={sign}"
    try:
        resp = requests.post(full_url, json=payload, headers=headers, timeout=10)
        return resp.status_code == 200 and resp.json().get("code") == 0
    except Exception as e:
        print(f"飞书发送失败：{e}")
        return False

# ========== 主函数 ==========
def main():
    print(f"开始生成欧元/美元每日分析报告：{datetime.now()}")
    
    print("正在获取市场数据并计算技术指标...")
    data = fetch_all_data()
    if not data or "error" in data.get("欧元/美元", {}):
        print("数据获取失败")
        return
    
    print("正在获取经济日历...")
    events = fetch_economic_calendar()
    
    print("正在调用 DeepSeek AI 进行专业分析...")
    analysis = generate_ai_analysis(data, events)
    
    print("正在生成报告文本...")
    today = datetime.now().strftime("%Y-%m-%d")
    eurusd = data.get("欧元/美元", {})
    
    report = f"""# 📊 欧元/美元每日分析报告

**分析日期**：{today}

---

## 📈 市场行情概览

| 指标 | 数值 |
|------|------|
| **当前价格** | {eurusd.get('current_price', 'N/A')} |
| **昨日收盘** | {eurusd.get('prev_price', 'N/A')} |
| **日涨跌幅** | {eurusd.get('change_pct', 'N/A'):+.2f}% |
| **关键阻力位** | {eurusd.get('resistance', 'N/A')} |
| **关键支撑位** | {eurusd.get('support', 'N/A')} |

---

## 📊 技术指标速览

| 指标 | 数值 | 信号解读 |
|------|------|----------|
| RSI(14) | {eurusd.get('rsi', 'N/A')} | {eurusd.get('rsi_status', '')} |
| EMA(20) | {eurusd.get('ema_20', 'N/A')} | — |
| SMA(50) | {eurusd.get('sma_50', 'N/A')} | — |
| 均线趋势 | — | {eurusd.get('trend', 'N/A')} |
| MACD柱状图 | {eurusd.get('macd_histogram', 'N/A')} | {'正值（多头动能）' if eurusd.get('macd_histogram', 0) > 0 else '负值（空头动能）' if eurusd.get('macd_histogram', 0) < 0 else '中性'} |
| 布林带位置 | — | {eurusd.get('bb_status', 'N/A')} |
| ATR波动率 | {eurusd.get('atr', 'N/A')} | {'高（波动加剧）' if eurusd.get('atr', 0) > 0.005 else '正常'} |
| 成交量比 | {eurusd.get('volume_ratio', 'N/A')} | {'异常放量' if eurusd.get('volume_ratio', 0) > 1.5 else '正常'} |

---

## 🤖 AI 专业分析

{analysis}

---

*报告由 DeepSeek AI 生成，仅供参考，不构成任何投资建议。*
"""
    
    print("正在推送至消息渠道...")
    dingtalk_url = os.environ.get("DINGTALK_WEBHOOK", "")
    feishu_url = os.environ.get("FEISHU_WEBHOOK", "")
    feishu_secret = os.environ.get("FEISHU_SECRET", "")
    
    if dingtalk_url:
        if send_to_dingtalk(dingtalk_url, report):
            print("✅ 钉钉推送成功")
        else:
            print("❌ 钉钉推送失败")
    
    if feishu_url and feishu_secret:
        if send_to_feishu(feishu_url, feishu_secret, report):
            print("✅ 飞书推送成功")
        else:
            print("❌ 飞书推送失败")
    
    print("✅ 报告生成完毕")

if __name__ == "__main__":
    main()
