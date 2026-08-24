import io
import json
import re
import sqlite3
import datetime
from pathlib import Path
from urllib.parse import quote
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from fastapi import FastAPI, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import sys
import os
from pathlib import Path

def resource_path(relative_path):
    """获取只读资源路径（如模板文件），适用于PyInstaller打包"""
    try:
        base_path = sys._MEIPASS  # PyInstaller解压临时目录
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_writable_path(filename):
    """获取可写文件路径（数据库、配置文件），放在可执行文件同目录"""
    if getattr(sys, 'frozen', False):
        # 打包后：exe所在目录
        app_dir = os.path.dirname(sys.executable)
    else:
        # 开发环境：脚本所在目录
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, filename)

# 路径设置
TEMPLATE_PATH = Path(resource_path("templates/index.html"))   # 只读模板
DB_PATH = Path(get_writable_path("a_share_data.db"))          # 可写数据库
CONFIG_PATH = Path(get_writable_path("config.json"))          # 可写配置文件

app = FastAPI(title="A股分析Agent（东方财富数据版）")

# ---------- 配置文件路径（用于保存API Key） ----------


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# ---------- 数据库 ----------


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (date TEXT PRIMARY KEY,
                  raw_data TEXT,
                  analysis_json TEXT,
                  created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ---------- 提示词模板 ----------
DEFAULT_PROMPT_TEMPLATE = """你是一个金融数据分析助手。以下是{date}的A股市场数据摘要（已由程序整理，数据真实可靠）：

请根据这些数据，输出JSON格式的分析结果，包含以下字段（所有值均为字符串或数字，不要嵌套其他对象）：
{{
    "date": "{date}",
    "market_total_volume": "沪深京三市总成交量（或成交额，注明单位）",
    "market_total_volume_change": "三市总成交量缩放比例（如+5.2%或-3.1%）",
    "indices": [
        {{"name": "沪指", "volume": "成交量", "volume_change": "缩放比例", "close": "收盘价", "yellow_white_line": "黄白线描述"}},
        {{"name": "深成指", "volume": "成交量", "volume_change": "缩放比例", "close": "收盘价", "yellow_white_line": "黄白线描述"}},
        {{"name": "创业板指", "volume": "成交量", "volume_change": "缩放比例", "close": "收盘价", "yellow_white_line": "黄白线描述"}},
        {{"name": "科创50", "volume": "成交量", "volume_change": "缩放比例", "close": "收盘价", "yellow_white_line": "黄白线描述"}},
        {{"name": "北证50", "volume": "成交量", "volume_change": "缩放比例", "close": "收盘价", "yellow_white_line": "黄白线描述"}}
    ],
    "stock_stats": {{
        "limit_up": 涨停家数,
        "limit_down": 跌停家数,
        "flat": 平盘家数,
        "up": 上涨家数,
        "down": 下跌家数
    }},
    "institutional_view": "机构对次日开盘的看法（请基于成交量变化、指数涨跌、市场涨跌分布等给出具体分析，不要使用'数据不足'或模糊表达）"
}}

注意：
- 对于缺失的数据，请根据已有信息进行合理推断，并给出数值或文字描述，不要写“数据不足”或留空。
- 机构观点务必给出，可以结合技术面、资金面、情绪面等。
- 确保输出是有效的JSON，不要包含任何额外文本或代码块标记。"""

# ---------- 东方财富 API 数据获取 ----------
def fetch_index_daily_em(secid: str) -> pd.DataFrame:
    """获取指数日K线数据，返回DataFrame包含date, close, volume等列"""
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": "101",
        "fqt": "1",
        "beg": "20200101",
        "end": "20500101"
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("data") is None or data["data"].get("klines") is None:
        return pd.DataFrame()
    klines = data["data"]["klines"]
    rows = []
    for line in klines:
        parts = line.split(",")
        rows.append({
            "date": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": float(parts[5]),
            "amount": float(parts[6]),
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df

def fetch_spot_stats() -> Dict[str, Any]:
    """获取实时涨跌家数统计（仅当日有效）"""
    url = "http://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "5000",
        "po": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f2,f3"
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("data") is None or data["data"].get("diff") is None:
        return {}
    diff = data["data"]["diff"]
    up = down = flat = limit_up = limit_down = 0
    for item in diff:
        pct = item.get("f3", 0)
        if pct is None or pct == "-":
            continue
        pct = float(pct)
        if pct > 0:
            up += 1
            if pct >= 9.9:
                limit_up += 1
        elif pct < 0:
            down += 1
            if pct <= -9.9:
                limit_down += 1
        else:
            flat += 1
    return {
        "up": up,
        "down": down,
        "flat": flat,
        "limit_up": limit_up,
        "limit_down": limit_down
    }

def fetch_minute_data(secid: str = "1.000001") -> List[Dict]:
    """获取当日分时数据"""
    url = "http://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ndays": "1",
        "iscr": "0"
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("data") is None or data["data"].get("trends") is None:
        return []
    trends = data["data"]["trends"]
    result = []
    for line in trends:
        parts = line.split(",")
        result.append({
            "time": parts[0],
            "price": float(parts[2]),
            "avg_price": float(parts[7])
        })
    return result

def fetch_market_data(date_str: str) -> str:
    """
    获取指定日期的市场数据，返回文本摘要。
    使用东方财富API，稳定可靠。
    """
    indices = {
        "沪指": "1.000001",
        "深成指": "0.399001",
        "创业板指": "0.399006",
        "科创50": "1.000688",
        "北证50": "0.899050"
    }
    parts = []

    # 1. 指数数据
    index_lines = []
    total_volume = 0
    total_prev_volume = 0
    for name, secid in indices.items():
        try:
            df = fetch_index_daily_em(secid)
            if df.empty:
                index_lines.append(f"{name}: 无数据")
                continue
            today_df = df[df['date'] == date_str]
            if today_df.empty:
                index_lines.append(f"{name}: 当日无数据")
                continue
            today = today_df.iloc[0]
            volume = today['volume']
            close = today['close']
            prev_df = df[df['date'] < date_str].tail(1)
            if not prev_df.empty:
                prev_volume = prev_df.iloc[0]['volume']
                if prev_volume and prev_volume != 0:
                    change_pct = (volume - prev_volume) / prev_volume * 100
                    change_str = f"{change_pct:+.2f}%"
                else:
                    change_str = "无法计算"
                total_prev_volume += prev_volume
            else:
                change_str = "无法计算"
            total_volume += volume
            index_lines.append(f"{name}: 成交量={volume}手, 收盘价={close}, 缩放量={change_str}")
        except Exception as e:
            index_lines.append(f"{name}: 获取失败 ({str(e)})")
    parts.append("【指数数据】\n" + "\n".join(index_lines))

    # 三市总成交量
    if total_prev_volume > 0:
        market_change = (total_volume - total_prev_volume) / total_prev_volume * 100
        parts.append(f"沪深京三市总成交量估算值: {total_volume}手, 缩放量: {market_change:+.2f}%")
    else:
        parts.append("沪深京三市总成交量: 无法计算（缺少前一日数据）")

    # 2. 涨跌家数统计
    if date_str == datetime.date.today().strftime("%Y-%m-%d"):
        stats = fetch_spot_stats()
        if stats:
            parts.append(f"【涨跌统计】上涨:{stats['up']}家, 下跌:{stats['down']}家, 平盘:{stats['flat']}家, 涨停约:{stats['limit_up']}家, 跌停约:{stats['limit_down']}家")
        else:
            parts.append("【涨跌统计】获取失败，请根据指数表现推断")
    else:
        parts.append("【涨跌统计】历史日期涨跌家数未提供，请根据指数涨跌和成交量变化推断市场情绪，并给出大致数字或描述。")

    # 3. 分时数据（仅当日）
    if date_str == datetime.date.today().strftime("%Y-%m-%d"):
        minute_data = fetch_minute_data("1.000001")
        if minute_data:
            white_line = [d['price'] for d in minute_data]
            yellow_line = [d['avg_price'] for d in minute_data]
            parts.append(f"【沪指分时数据】白线(实时): {white_line[:10]}... (共{len(white_line)}点)；黄线(均价): {yellow_line[:10]}...")
        else:
            parts.append("【沪指分时数据】无数据")
    else:
        parts.append("【沪指分时数据】历史分时数据不获取（无法分析黄白线，请填写'无分时数据'）")

    return "\n\n".join(parts)

# ---------- 数据库操作 ----------
def save_report(date_str: str, raw_data: str, analysis_json: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT OR REPLACE INTO reports (date, raw_data, analysis_json, created_at) VALUES (?,?,?,?)",
              (date_str, raw_data, analysis_json, created_at))
    conn.commit()
    conn.close()

def get_report(date_str: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT raw_data, analysis_json FROM reports WHERE date=?", (date_str,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"raw_data": row[0], "analysis_json": row[1]}
    return None

def get_all_dates() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date FROM reports ORDER BY date")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_reports_range(start_date: str, end_date: str) -> List[tuple]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, analysis_json FROM reports WHERE date BETWEEN ? AND ? ORDER BY date",
              (start_date, end_date))
    rows = c.fetchall()
    conn.close()
    return rows

# ---------- 调用 DeepSeek ----------
def call_deepseek(prompt: str, api_url: str, api_key: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业的A股市场数据分析助手，输出严格的JSON格式。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
        raise ValueError("无法解析API返回的JSON")

# ---------- 生成 Excel（区间汇总） ----------
def generate_excel_range(start_date: str, end_date: str) -> io.BytesIO:
    rows = get_reports_range(start_date, end_date)
    if not rows:
        raise ValueError("该区间内没有报告数据")

    summary = []
    for date_str, analysis_json in rows:
        analysis = json.loads(analysis_json)
        row = {"日期": date_str}
        row["三市总成交量"] = analysis.get("market_total_volume", "")
        row["三市总成交量缩放"] = analysis.get("market_total_volume_change", "")

        indices = analysis.get("indices", [])
        for idx in indices:
            name = idx.get("name", "")
            row[f"{name}成交量"] = idx.get("volume", "")
            row[f"{name}缩放"] = idx.get("volume_change", "")
            row[f"{name}收盘"] = idx.get("close", "")
            row[f"{name}黄白线"] = idx.get("yellow_white_line", "")

        stats = analysis.get("stock_stats", {})
        row["涨停"] = stats.get("limit_up", "")
        row["跌停"] = stats.get("limit_down", "")
        row["平盘"] = stats.get("flat", "")
        row["上涨"] = stats.get("up", "")
        row["下跌"] = stats.get("down", "")

        row["机构观点"] = analysis.get("institutional_view", "")
        summary.append(row)

    df = pd.DataFrame(summary)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="区间汇总", index=False)
    output.seek(0)
    return output

# ---------- FastAPI 路由 ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATE_PATH
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("{{ default_prompt }}", DEFAULT_PROMPT_TEMPLATE)

    # 注入已保存的API Key（如果有）
    config = load_config()
    saved_api_key = config.get("api_key", "")
    html = html.replace("{{ saved_api_key }}", saved_api_key)

    # 注入日期信息
    dates = get_all_dates()
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        range_info = f"数据库已有报告日期范围：{min_date} 至 {max_date}（共{len(dates)}天）"
    else:
        range_info = "暂无历史报告"
    html = html.replace("{{ date_range_info }}", range_info)
    options = "".join([f'<option value="{d}">{d}</option>' for d in dates])
    html = html.replace("{{ date_options }}", options)
    return HTMLResponse(content=html)

@app.post("/save_api_key")
async def save_api_key(api_key: str = Form(...)):
    config = load_config()
    config["api_key"] = api_key
    save_config(config)
    return {"status": "success"}

@app.post("/generate_range")
async def generate_range(
    start_date: str = Form(...),
    end_date: str = Form(...),
    api_url: str = Form("https://api.deepseek.com/v1/chat/completions"),
    api_key: str = Form(...),
    model: str = Form("deepseek-chat"),
    prompt_template: str = Form(DEFAULT_PROMPT_TEMPLATE)
):
    date_list = []
    cur = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    if cur > end:
        raise HTTPException(status_code=400, detail="开始日期不能晚于结束日期")
    while cur <= end:
        date_list.append(cur.strftime("%Y-%m-%d"))
        cur += datetime.timedelta(days=1)

    results = []
    for date_str in date_list:
        try:
            if get_report(date_str):
                results.append({"date": date_str, "status": "已存在，跳过"})
                continue

            raw_data = fetch_market_data(date_str)
            prompt = prompt_template.replace("{date}", date_str).replace("{data}", raw_data)
            analysis = call_deepseek(prompt, api_url, api_key, model)
            save_report(date_str, raw_data, json.dumps(analysis, ensure_ascii=False))
            results.append({"date": date_str, "status": "成功"})
        except Exception as e:
            results.append({"date": date_str, "status": f"失败: {str(e)}"})
    return {"results": results}

@app.get("/download_range")
async def download_range(
    start_date: str = Query(...),
    end_date: str = Query(...)
):
    try:
        excel_bytes = generate_excel_range(start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    filename = f"A股分析区间汇总_{start_date}_至_{end_date}.xlsx"
    encoded = quote(filename)
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"}
    return StreamingResponse(excel_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

@app.get("/dates")
async def list_dates():
    return {"dates": get_all_dates()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)