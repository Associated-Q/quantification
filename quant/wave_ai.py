"""浪型候选的 AI 复核：模型只在规则枚举出的候选里选择主数法。

约束（与评分链路隔离）：
- 候选的点位、日期、比例全部来自 wave.wave_groups，模型不生成任何价格；
- 返回的编号必须存在于候选集合，否则视为失败，不静默采用；
- 结果只影响图上高亮的主数法与斐波那契锚点，不进入评分。
"""
import json

import wave

MAX_CANDIDATES = 24
MAX_ALTERNATIVES = 3
MAX_REASON = 300


def build_payload(df, result, tcol, symbol, period):
    """把规则枚举出的浪组整理成模型可复核的结构化候选。"""
    times = [str(x) for x in df[tcol]]
    n = len(df)
    cur = float(df["close"].iloc[-1])
    current_key = [int(p[0]) for p in (result.get("pivots") or [])]

    candidates = []
    groups = wave.wave_groups(df)
    if len(groups) > MAX_CANDIDATES:
        # 优先保留最近的浪组，再按起点排序编号，避免最新的结构被截掉
        groups = sorted(groups, key=lambda g: g["pivots"][-1][0], reverse=True)[:MAX_CANDIDATES]
        groups.sort(key=lambda g: (g["pivots"][0][0], g["degree"]))
    for idx, g in enumerate(groups):
        p = g["pivots"]
        r2, r4, _, legs = wave._ratios(p, g["up"])
        prices = [float(x[1]) for x in p]
        span = max(prices) - min(prices)
        longest = max(legs[0], legs[2]) if max(legs[0], legs[2]) else None
        candidates.append({
            "id": idx,
            "degree": g["degree_name"],
            "direction": "上升" if g["up"] else "下降",
            "pattern": g["pattern"],
            "failed": bool(g["failed"]),
            "start": times[p[0][0]],
            "end": times[p[-1][0]],
            "bars_since_end": n - 1 - p[-1][0],
            "span_pct": round(span / cur * 100, 1) if cur else None,
            "current": [int(x[0]) for x in p] == current_key,
            "wave2_retrace": round(r2, 3) if r2 is not None else None,
            "wave4_retrace": round(r4, 3) if r4 is not None else None,
            "wave3_vs_longest": round(legs[1] / longest, 2) if longest else None,
            "points": [{"date": times[int(i)], "price": round(float(v), 3)}
                       for i, v, _ in p],
        })

    return {
        "symbol": symbol,
        "period": period,
        "asof": times[-1],
        "price": round(cur, 3),
        "current": {"valid": bool(result.get("valid")),
                    "why": result.get("why") or "",
                    "direction": result.get("direction"),
                    "pattern": result.get("pattern")},
        "candidates": candidates,
    }


PROMPT = """你在复核一张 {period} K 线图的艾略特浪型划分。

下面所有候选都是确定性规则枚举出来的，点位、日期、比例都是真实数据。
你的任务：只在这些候选里选出「当前最适合作为主数法」的一组，或判断没有可采用的结构。

判断要求：
1. 只能返回候选里的编号，禁止发明新的点位、日期或候选；
2. 优先考虑级别与近期行情幅度匹配、起点落在显著波段极值、浪2/浪4 回撤与浪3 比例合规的结构；
3. 结构已陈旧但仍是最新可用结构时可以选，但必须在 reason 里说明；
4. 当前规则已经判为有效的那组（current=true）不是必须选，可以选更合理的其他候选；
5. alternatives 最多 {max_alts} 个，按优先级排序；
6. 只输出严格 JSON，不要 markdown 代码块，不要任何多余字段。

输出结构：
{{"chosen": 编号或 null, "alternatives": [编号...], "confidence": 0到1的小数, "reason": "不超过80字的中文理由"}}

候选与当前状态：
{candidates}"""


def build_prompt(payload):
    return PROMPT.format(period=payload.get("period"),
                         max_alts=MAX_ALTERNATIVES,
                         candidates=json.dumps(payload, ensure_ascii=False, indent=1))


def _as_int(value):
    if isinstance(value, bool):
        raise ValueError("布尔值不是合法编号")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"不是合法编号：{value!r}")


def validate_choice(data, ids):
    """校验模型输出；编号不在候选内时硬报错，不做猜测性兜底。"""
    if not isinstance(data, dict):
        raise ValueError("模型返回的不是 JSON 对象")
    raw = data.get("chosen")
    chosen = None if raw is None else _as_int(raw)
    if chosen is not None and chosen not in ids:
        raise ValueError(f"模型选择了不存在的候选编号 {chosen}（可选：{ids}）")

    alternatives, seen = [], set()
    for value in data.get("alternatives") or []:
        try:
            n = _as_int(value)
        except ValueError:
            continue
        if n in ids and n != chosen and n not in seen:
            seen.add(n)
            alternatives.append(n)

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None

    reason = data.get("reason")
    reason = reason.strip()[:MAX_REASON] if isinstance(reason, str) else ""
    return {"chosen": chosen, "alternatives": alternatives[:MAX_ALTERNATIVES],
            "confidence": confidence, "reason": reason}


def choose(payload):
    """调用模型并在候选集合内校验；模型不可用时抛 AIAnalysisError。"""
    import ai_analyst
    data = ai_analyst.chat_json(build_prompt(payload), "wave")
    return validate_choice(data, [c["id"] for c in payload["candidates"]])


def assemble(payload, choice, model=None):
    """把已校验的选择整理成接口返回；选中候选必须仍在候选集合内。"""
    chosen = None
    if choice.get("chosen") is not None:
        cand = next((c for c in payload["candidates"] if c["id"] == choice["chosen"]), None)
        if cand is None:
            raise ValueError(f"选中编号 {choice['chosen']} 不在候选集合内")
        chosen = dict(cand)
        chosen["fib"] = {
            "source": "wave", "degree": cand["degree"],
            "label": f"AI 主数法（{cand['degree']}级别）",
            "start": cand["points"][0], "end": cand["points"][-1],
            "levels": wave.fib_levels(cand["points"][0]["price"],
                                      cand["points"][-1]["price"]),
        }
    return {"symbol": payload["symbol"], "period": payload["period"],
            "model": model, "cached": False, "chosen": chosen,
            "alternatives": choice.get("alternatives") or [],
            "confidence": choice.get("confidence"),
            "reason": choice.get("reason") or "",
            "candidates": payload["candidates"]}
