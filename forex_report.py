#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import json
from datetime import datetime
from typing import Dict, Any

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== 使用更稳定的免费 API 获取 EUR/USD ==========
def fetch_eurusd() -> Dict[str, Any]:
    """尝试多个免费 API 获取 EUR/USD 汇率，返回当前价格"""
    apis = [
        "https://api.exchangerate-api.com/v4/latest/EUR",   # 返回 rates.USD
        "https://open.er-api.com/v6/latest/EUR",            # 返回 rates.USD
        "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/eur.json"  # 返回 eur.usd
    ]
    
    for url in apis:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # 尝试不同 API 的解析逻辑
                if "rates" in data and "USD" in data["rates"]:
                    rate = data["rates"]["USD"]
                    return {"current_price": round(rate, 4), "error": None}
                elif "usd" in data.get("eur", {}):
                    rate = data["eur"]["usd"]
                    return {"current_price": round(rate, 4), "error": None}
                elif "rate" in data and "USD" in data["rate"]:
                    # 兼容某些简化的返回格式
                    rate = data["rate"]["USD"]
                    return {"current_price": round(rate, 4), "error": None}
            # 如果不成功，继续尝试下一个 API
        except Exception as e:
            continue
    
    return {"error": "所有 API 均失败，无法获取汇率"}

# ========== 简单的支撑阻力（基于当前价格估算）==========
def calculate_support_resistance(price: float) -> tuple:
    """根据当前价格估算支撑阻力（使用 0.5% 波动范围）"""
    atr_est = price * 0.005
    support = round(price - atr_est, 4)
    resistance = round(price + atr_est, 4)
    return support, resistance

# ========== DeepSeek AI 分析 ==========
def call_deepseek(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置 DEEPSEEK_API_KEY"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业的外汇策略分析师。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 错误 {resp.status_code}"
    except Exception as e:
        return f"❌ 请求异常: {e}"

def generate_analysis(price: float, support: float, resistance: float) -> str:
    prompt = f"""
欧元/美元当前汇率：{price}
估算支撑位：{support}
估算阻力位：{resistance}

请根据以上数据，撰写一份简短的外汇日报，内容包括：
1. 当前价格位置判断（是否接近支撑/阻力）
2. 短期可能方向（看涨/看跌/震荡）
3. 关键支撑阻力的交易意义
4. 核心风险提示（如央行政策、经济数据）

要求：专业、简洁。
"""
    return call_deepseek(prompt)

# ========== 钉钉推送（支持加签）==========
import time
import hmac
import hashlib
import base64

def send_to_dingtalk(webhook_url: str, secret: str, content: str) -> bool:
    timestamp = str(round(time.time() * 1000))
    sign_str = f"{timestamp}\n{secret}"
    sign = base64.b64encode(hmac.new(secret.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')
    full_url = f"{webhook_url}&timestamp={timestamp}&sign={sign}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "📊 欧元/美元日报", "text": content[:4000]}
    }
    try:
        resp = requests.post(full_url, json=payload, headers=headers, timeout=10)
        result = resp.json()
        if result.get("errcode") == 0:
            return True
        else:
            print(f"钉钉返回错误：{result}")
            return False
    except Exception as e:
        print(f"钉钉发送异常：{e}")
        return False
def send_to_dingtalk_no_sign(webhook_url: str, content: str) -> bool:
    """无加签版本，用于关闭了加签的钉钉机器人"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "📊 欧元/美元日报", "text": content[:4000]}
    }
    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        result = resp.json()
        if result.get("errcode") == 0:
            return True
        else:
            print(f"钉钉返回错误：{result}")
            return False
    except Exception as e:
        print(f"钉钉发送异常：{e}")
        return False
# ========== 主函数 ==========
def main():
    print("获取欧元/美元汇率...")
    eur_data = fetch_eurusd()
    if eur_data.get("error"):
        print(f"数据获取失败：{eur_data['error']}")
        return
    
    price = eur_data["current_price"]
    support, resistance = calculate_support_resistance(price)
    
    print(f"当前汇率：{price}，支撑：{support}，阻力：{resistance}")
    print("调用 AI 分析...")
    analysis = generate_analysis(price, support, resistance)
    
    today = datetime.now().strftime("%Y-%m-%d")
    report = f"""# 📊 欧元/美元每日分析

**日期**：{today}
**当前汇率**：{price}
**估算支撑**：{support}
**估算阻力**：{resistance}

---

## 🤖 AI 分析

{analysis}

---
*报告由 AI 生成，仅供参考。*
"""
    
    # 推送
       # 推送（无加签版本）
    webhook = os.environ.get("DINGTALK_WEBHOOK", "")
    if webhook:
        if send_to_dingtalk_no_sign(webhook, report):
            print("钉钉推送成功")
        else:
            print("钉钉推送失败")
    else:
        print("未配置钉钉 Webhook，跳过推送")

if __name__ == "__main__":
    main()
