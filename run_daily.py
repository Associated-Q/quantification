"""每日输出：只打印规定栏目，同时保存到 kc50/data/daily_log/。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kc50.core import (add_indicators, find_pivots, load,         # noqa: E402
                       pivots_asof)
from kc50.config import GANN_ANCHOR                            # noqa: E402
from kc50.model import FEATURES, _make_model, build_features, GAP  # noqa: E402
from kc50.structs import (CYCLES, RETRACE, analyze_wave, fib_levels,  # noqa: E402
                          fib_structural, fib_swing, fib_time_windows,
                          gann_angles, gann_stats, hexagram, reversal_base_rate)
from scipy.stats import binomtest                              # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "kc50", "data")
LOG = os.path.join(OUT, "daily_log")
DESIGNATED = GANN_ANCHOR


def predict_now(feat: pd.DataFrame, horizon: int, decision_date: pd.Timestamp) -> float:
    ycol = f"y{horizon}"
    data = feat.copy()
    cur = data[data["date"] == decision_date]
    if cur.empty:
        return float("nan")
    t_idx = int(cur["idx"].iat[0])
    ok = (data["idx"] + horizon <= t_idx - GAP) & data[ycol].notna()
    tr = data[ok]
    if len(tr) < 120 or tr[ycol].nunique() < 2:
        return float("nan")
    m = _make_model()
    m.fit(tr[FEATURES].to_numpy(float), tr[ycol].to_numpy(int))
    return float(m.predict_proba(cur[FEATURES].to_numpy(float))[0, 1])


def gann_section(df, piv, t, anchor_date):
    idx_of = {d: i for i, d in enumerate(df["date"])}
    rows = []
    a_idx = idx_of.get(anchor_date)
    if a_idx is not None:
        a_price = float(df["close"].iat[a_idx])
        a_atr = float(df["atr14"].iat[a_idx])
        days = t - a_idx
        for c in CYCLES:
            ctr = a_idx + c
            lo, hi = ctr - 1, ctr + 1
            status = ("已实现" if hi < t else "进行中" if lo <= t <= hi
                      else "待验证")
            rows.append({"anchor": anchor_date, "anchor_idx": a_idx, "cycle": c,
                         "center": ctr, "lo": lo, "hi": hi, "status": status})
    for _, r in piv.tail(3).iterrows():
        ai = int(r["idx"])
        for c in CYCLES:
            ctr = ai + c
            lo, hi = ctr - 1, ctr + 1
            if ctr < t - 3:
                continue
            status = ("已实现" if hi < t else "进行中" if lo <= t <= hi
                      else "待验证")
            rows.append({"anchor": r["date"], "anchor_idx": ai, "cycle": c,
                         "center": ctr, "lo": lo, "hi": hi, "status": status})
    return pd.DataFrame(rows).drop_duplicates(subset=["anchor", "cycle"])


def main():
    df, rep = load("2023-01-01")
    df = add_indicators(df)
    piv = find_pivots(df)
    feat = build_features(df, piv)
    t = len(df) - 1
    d = df.iloc[t]
    p = pivots_asof(piv, t)
    date = d["date"]

    n = len(df)

    def date_at(i: int) -> str:
        """i 超出已收盘数据时，按工作日粗略外推并标 ≈。"""
        if i < n:
            return f"{df['date'].iat[i]:%m-%d}"
        fut = pd.bdate_range(df["date"].iat[-1] + pd.Timedelta(days=1),
                             periods=(i - n + 1))
        return f"≈{fut[-1]:%m-%d}"

    backtest = json.load(open(os.path.join(OUT, "backtest.json"), encoding="utf-8"))

    probs, valid = {}, {}
    for h in (3, 5):
        bt = backtest[str(h)]
        base = float(bt["base_rate"])
        raw = predict_now(feat, h, date)
        ok = (bt["balanced_ci"][0] > 0.5) and (bt["brier"] < bt["base_brier"])
        valid[h] = ok
        probs[h] = {"raw": raw, "used": raw if ok else base, "base": base, "ok": ok}

    print(f"数据日 {date:%Y-%m-%d}｜收盘 {d['close']:.2f}"
          f"（高 {d['high']:.2f} 低 {d['low']:.2f}）｜ATR14 {d['atr14']:.2f}"
          f"｜ma20 {d['ma20']:.2f}")
    print(f"（今日 {pd.Timestamp.today():%Y-%m-%d} 尚未收盘，以下基于最近一个完整交易日）")
    if rep:
        for k, v in rep.items():
            print(f"数据核验｜{k}：{v if len(str(v)) < 160 else str(v)[:160] + ' ...'}")
    else:
        print("数据核验：无重复、无缺失交易日、无OHLC矛盾、双源收盘价一致")

    # 1 趋势
    trend_down = d["close"] < d["ma20"]
    print("\n一、当前趋势")
    print(f"{'下跌' if trend_down else '上涨'}。收盘位于 ma20 "
          f"{'下' if trend_down else '上'}方 {abs(d['close'] / d['ma20'] - 1):.2%}，"
          f"ma20 五日斜率 {d['ma20_slope5']:+.2%}。近 5 日量比 {d['vol_ratio']:.2f}。")

    # 2 波浪
    w = analyze_wave(p, float(d["close"]))
    print("\n二、当前波浪结构及失效点")
    print(f"{w['state']}（{w['detail']}）")
    if len(p) >= 2:
        a, b = p.iloc[-2], p.iloc[-1]
        print(f"\n最近两个已确认转折点：{a['date']:%m-%d} {a['kind']} @ {a['price']:.2f}"
              f" → {b['date']:%m-%d} {b['kind']} @ {b['price']:.2f}")
    print(f"已确认转折点 {len(p)} 个（最后一个 idx={int(p.iloc[-1]['confirm_idx'])}，"
          f"收盘后才可用）")
    if w["invalid"] and w["state"] != "结构不明":
        print(f"失效点：收盘 {'跌破' if w['label_code'] >= 0 else '升破'} "
              f"{w['invalid']:.2f} → 当前结构作废")
    else:
        print(f"失效点：结构不明，无浪级失效点。结构观察位 = 低点 "
              f"{p[p['kind'] == 'L']['price'].iloc[-1]:.2f} / 高点 "
              f"{p[p['kind'] == 'H']['price'].iloc[-1]:.2f}")

    # 3 江恩
    gw = gann_section(df, piv, t, DESIGNATED)
    print("\n三、最近江恩变盘窗口")
    idx_of = {dd: i for i, dd in enumerate(df["date"])}
    if DESIGNATED not in idx_of:
        raise SystemExit(f"江恩锚点 {DESIGNATED:%Y-%m-%d} 不是交易日或超出数据范围，"
                         f"请检查 kc50/config.py")
    if True:
        ai = idx_of[DESIGNATED]
        a_px = float(df["close"].iat[ai])
        a_atr = float(df["atr14"].iat[ai])
        days = t - ai
        print(f"锚点 {DESIGNATED:%Y-%m-%d}（固定起点）｜当前为锚点后第 {days} 个交易日"
              f"｜锚点收 {a_px:.2f}｜锚点 ATR14 {a_atr:.2f}")
        print("\n周期   窗口中心   ±1 窗口        状态")
        sub = gw[gw["anchor"] == DESIGNATED].sort_values("cycle")
        for _, r in sub.iterrows():
            print(f"{r['cycle']:>3}日  {date_at(int(r['center']))}   "
                  f"{date_at(int(r['lo']))} ~ {date_at(int(r['hi']))}   {r['status']}")

        up_l = gann_angles(a_px, a_atr, days, direction=1)
        print(f"\n1×1 上升线（ATR 标准化）当前值 {up_l['1x1']:.2f}，"
              f"收盘 {d['close']:.2f} 在其下方 "
              f"{(up_l['1x1'] - d['close']) / a_atr:.1f} ATR（该定义下已失真，见说明）")
    print("\n其它转折点窗口（次要，仅供参考；★=两个以上周期前后1日内重合）：")
    up = gw[(gw["center"] >= t) & (gw["anchor"] != DESIGNATED)].sort_values("center").head(3)
    for _, r in up.iterrows():
        same = gw[(gw["cycle"] != r["cycle"])
                  & ((gw["center"] - r["center"]).abs() <= 2)]
        mark = "★" if len(same) else " "
        print(f" {mark}{pd.Timestamp(r['anchor']):%m-%d} 起 +{r['cycle']:>2}日 → "
              f"中心 {date_at(int(r['center']))}（"
              f"{date_at(int(r['lo']))} ~ {date_at(int(r['hi']))}）{r['status']}")

    base_r = reversal_base_rate(df)
    print("\n统计地位（命中=窗口前后3日方向相反；基率=任意一天前后3日方向相反的比例）")
    for label, ln in (("最近10个阶段转折点", 10), ("全样本", None)):
        st = gann_stats(df, piv, last_n=ln)
        if not len(st):
            continue
        k, nst = int(st["reversal"].sum()), len(st)
        pv = binomtest(k, nst, base_r).pvalue
        print(f"  {label}：{nst} 个已实现窗口，命中 {k}/{nst} = {k / nst:.2%}，"
              f"基率 {base_r:.2%}，p = {pv:.4g}，"
              f"{'不能拒绝随机' if pv > 0.05 else '偏离随机（仍需样本外复核）'}")
    nexts = gw[gw["center"] >= t].sort_values("center")
    if len(nexts):
        r = nexts.iloc[0]
        print(f"最近一个待验证窗口：{date_at(int(r['center']))}"
              f"（{pd.Timestamp(r['anchor']):%m-%d} 锚点 +{r['cycle']}日），"
              f"窗口 {date_at(int(r['lo']))} ~ {date_at(int(r['hi']))}")

    # 4 斐波那契
    sw = fib_swing(p, min_atr_mult=3.0)
    print("\n四、斐波那契关键支撑、压力")
    if sw:
        print(f"基准波段 {sw['low']:.2f} – {sw['high']:.2f}"
              f"（{sw['start_date']:%m-%d} → {sw['end_date']:%m-%d}，"
              f"振幅 {sw['range']:.2f}，{sw['bars']} 个交易日）")
        lv = fib_levels(sw)
        print("\n回撤位    价位     相对当前")
        for _, r in lv[lv["kind"] == "retrace"].sort_values("ratio").iterrows():
            side = "压力" if r["price"] > d["close"] else "支撑"
            print(f"{r['ratio'] * 100:>5.1f}%  {r['price']:>8.2f}   {side}")
        print("扩展位：")
        for _, r in lv[lv["kind"] == "extend"].sort_values("ratio").iterrows():
            print(f"{r['ratio'] * 100:>5.1f}%  {r['price']:>8.2f}")
        tw = fib_time_windows(sw)
        print("时间窗口：" + "｜".join(
            f"{r['ratio']:g}×→{date_at(int(r['center_idx']))}"
            for _, r in tw.iterrows()))
        print(f"结构位：{p.iloc[-1]['date']:%m-%d} 转折 {p.iloc[-1]['price']:.2f}、"
              f"本日低 {d['low']:.2f}")

        sf = fib_structural(p)
        if sf:
            sl = fib_levels(sf)
            print(f"\n全段结构 {sf['low']:.2f} – {sf['high']:.2f}"
                  f"（{sf['start_date']:%Y-%m-%d} → {sf['end_date']:%Y-%m-%d}，"
                  f"振幅 {sf['range']:.2f}）")
            cells = []
            for _, r in sl[sl["kind"] == "retrace"].sort_values("ratio").iterrows():
                side = "压力" if r["price"] > d["close"] else "支撑"
                cells.append(f"{r['ratio'] * 100:g}% = {r['price']:.2f}（{side}）")
            print("｜".join(cells))
    else:
        print("无满足振幅阈值的已确认波段，斐波那契位暂不计算")

    # 5/6 概率
    for h in (3, 5):
        pr = probs[h]
        up = pr["used"] * 100
        print(f"\n{'五' if h == 3 else '六'}、未来 {h} 日上涨/下跌概率")
        print(f"上涨 {up:.2f}% ／ 下跌 {100 - up:.2f}%")
        if pr["ok"]:
            print("口径：模型校准概率（通过样本外验证）")
        else:
            print(f"口径：无条件基率（样本外区间“{h} 日后收涨”的实际占比），"
                  f"即完全没有信息时的最优估计。模型原始输出 {pr['raw']:.1%}——未通过验证"
                  f"（样本外 {backtest[str(h)]['acc']:.2%}，"
                  f"CI {backtest[str(h)]['acc_ci'][0]:.2%}–"
                  f"{backtest[str(h)]['acc_ci'][1]:.2%}），禁止使用。")

    # 7 方向
    print("\n七、当前方向")
    if not any(valid.values()):
        print("观望")
    else:
        sig = probs[3]["used"] > 0.5
        print("看多" if sig else "看空")

    # 8 确认条件
    print("\n八、最重要的入场确认条件")
    print(f"收盘站上 {d['ma20']:.2f}（ma20）且 ma20 斜率转正并维持 3 个交易日。"
          "在此之前，任何反弹都只是下跌结构中的反弹。")
    print(f"次级确认：收盘 > {d['high']:.2f}（本日高点）。")
    bt3 = backtest["3"]
    bt5 = backtest["5"]
    print(f"\n这是 20 日均线基准的规则，不是本模型信号。该基准样本外 3 日 "
          f"{bt3['ma20_acc']:.2%}（CI {bt3['ma20_ci'][0]:.2%}–{bt3['ma20_ci'][1]:.2%}）、"
          f"5 日 {bt5['ma20_acc']:.2%}（CI {bt5['ma20_ci'][0]:.2%}–"
          f"{bt5['ma20_ci'][1]:.2%}）。")

    # 9 失效
    print("\n九、预测失效条件")
    print(f"收盘跌破 {d['low']:.2f}（本日低点）→ 反弹受阻确认")
    if sw:
        print(f"收盘跌破 {sw['low']:.2f}（波段低点）→ 下跌结构延续")
    print(f"收盘站上 {d['ma20']:.2f} 且斜率转正 → “观望”作废，转趋势跟随")

    # 易经
    hx = hexagram(date)
    print(f"\n附：易经六爻（权重 0）")
    print(f"本卦 {hx['name']}（{hx['upper']}{hx['upper_sym']}上 "
          f"{hx['lower']}{hx['lower_sym']}下）→ {hx['moving']}爻动 → "
          f"变卦 {hx['changed_name']}。未纳入模型。起卦：{hx['seed']}")

    os.makedirs(LOG, exist_ok=True)
    rec = {
        "date": date.strftime("%Y-%m-%d"), "close": float(d["close"]),
        "high": float(d["high"]), "low": float(d["low"]),
        "atr14": float(d["atr14"]), "ma20": float(d["ma20"]),
        "wave": w["state"], "wave_detail": w["detail"], "wave_invalid": w["invalid"],
        "pivots_last": [{"date": x["date"].strftime("%Y-%m-%d"), "kind": x["kind"],
                         "price": float(x["price"])} for x in p.tail(3).to_dict("records")],
        "fib_swing": {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                      for k, v in (sw or {}).items()},
        "prob3": probs[3], "prob5": probs[5],
        "model_valid": valid, "direction": "观望" if not any(valid.values())
        else ("看多" if probs[3]["used"] > 0.5 else "看空"),
        "yijing": hx,
    }
    path = os.path.join(LOG, f"{date:%Y-%m-%d}.json")
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2, default=float)
    print(f"\n已保存 {path}")


if __name__ == "__main__":
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    os.makedirs(LOG, exist_ok=True)
    with open(os.path.join(LOG, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
