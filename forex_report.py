#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import json
from datetime import datetime
from typing import Dict, Any

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== 使用免费 API 获取 EUR/USD 汇率 ==========
def fetch_eurusd() -> Dict[str, Any]:
    """从 exchangerate.host 获取 EUR/USD 实时汇率"""
    try:
        # 免费接口，无需 Key
        url = "https://api.exchangerate.host/latest?base=EUR&symbols=USD"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            rate = data["rates"]["USD"]
            return {
                "current_price": round(rate, 4),
                "change_pct": 0.0,  # 该 API 不提供日内涨跌，可以忽略或估算
                "error": None
            }
        else:
            return {"error": f"API 返回 {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ========== 简单的支撑阻力（基于近期波动估算）==========
def calculate_support_resistance(price: float) -> tuple:
    """根据当前价格估算支撑阻力（简单方法）"""
    # 使用 ATR 概念，假设每日波动 0.5% ~ 1%
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
1. 当前价格位置判断
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
    webhook = os.environ.get("DINGTALK_WEBHOOK", "https://oapi.dingtalk.com/robot/send?access_token=06b76a64f6aedfad1f76478af6fe749fe43d33b1417d6b2cff10dad3b45b997a")
    secret = os.environ.get("DINGTALK_SECRET", "报告")  # 钉钉加签密钥，如果没有加签可以留空
    if webhook and secret:
        if send_to_dingtalk(webhook, secret, report):
            print("钉钉推送成功")
        else:
            print("钉钉推送失败")
    else:
        print("未配置钉钉 Webhook 或 Secret，跳过推送")

if __name__ == "__main__":
    main()
