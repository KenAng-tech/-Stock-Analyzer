#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""23:10 决策复盘链 (2026-09-12 整合 P1 决策链调度化, crontab 形)

每日 23:10 触发 5002 主进程重算 (urllib POST /api/decision/recalibrate):
  IC 闭环重算 (90日K线 → 角色 hit/miss) + Brier/ECE 刷新 + 角色门控
  (role Brier>0.55 & n≥20 → weight×0.7) — 不依赖「每50次决策」触发
  (日常 decide 少, 闭环 09-02 上线以来从未自然跑满 50 — 本链补触发)。

形制先例: scripts/daily_sota_train.py (launchd 直调) / crontab daily_review
(21:00 vnpy 复盘) 同族; 失败只记日志 (链坏 ≠ 链死 — 主链 decide 不依赖它)。

crontab:
  10 23 * * * cd /Users/claw/stock_analyzer && ./venv/bin/python scripts/decision_replay.py >> /tmp/decision_replay.log 2>&1
"""

import json
import os
import sys
import urllib.request

BASE = os.environ.get("STOCK_ANALYZER_URL", "http://127.0.0.1:5002")
LOG_FILE = "/tmp/decision_replay.log"


def _rotate(path, max_bytes=10 * 1024 * 1024, keep=5):
    """日志轮转 (self_implemented 先例: 09-04 同形)"""
    try:
        if os.path.exists(path) and os.path.getsize(path) > max_bytes:
            for i in range(keep, 1, -1):
                src, dst = f"{path}.{i-1}", f"{path}.{i}"
                if os.path.exists(src):
                    os.replace(src, dst)
            os.replace(path, path + ".1")
    except OSError:
        pass


def post(path, timeout=120):
    req = urllib.request.Request(BASE + path, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
        ok = r.status == 200 and '"success":true' in body.replace(" ", "")
        return ok, body[:300]
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def get(path, timeout=25):
    """GET 主进程链 (预热用, 跨进程非 self-HTTP — 09-02 红线只禁主进程自调)"""
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
        return r.status == 200, body[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


# 2026-09-13 #18 链 C: 四票热读清单 (08:00 晨报/晨间 analyze 消费; 23:10 预热)
WARM_STOCKS = ("sz300620", "sh688981", "sz300502", "sh688800", "sh603929")


def main():
    _rotate(LOG_FILE)
    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok, body = post("/api/decision/recalibrate", timeout=120)
    tag = "RECAL_OK" if ok else "RECAL_FAIL"
    try:
        cal = json.loads(body).get("data", {}).get("calibration", {}) \
            if ok and body.startswith("{") else {}
        line = (f"[{ts}] {tag} brier={cal.get('brier', cal.get('skipped', '-'))} "
                f"n_hist={cal.get('n_history', '-')} gated={cal.get('gated_roles', '-')}")
    except Exception:
        line = f"[{ts}] {tag} {body[:180]}"

    # 决策产生链 (2026-09-22 闭环提效, 用户拍板): 链尾把监控池 5 票跑
    # consensus.decide → 台账累积 (门控阈 n≥20; 此前无自动产源恒空转)。
    # 新决策需 ≥3 交易日才有 outcome 可重算, 故与 recalibrate 先后无碍。
    # 8080 辩论串行长尾 → timeout 240s 富余 (crontab 链无实时要求)。
    ok_s, body_s = post("/api/decision/scan", timeout=240)
    try:
        # scan 端点返回顶层形 (无 data 包裹, 与 eval/recalibrate 键位不同)
        _sc_raw = json.loads(body_s) if ok_s and body_s.startswith("{") else {}
        sc = _sc_raw.get("data", _sc_raw)
        line += (f" scan={'OK' if ok_s else 'FAIL'}"
                 f"(n={sc.get('scanned', '-')}/{len(sc.get('errors', []) or [])})"
                 if ok_s else f" scan=FAIL:{body_s[:80]}")
    except Exception:
        line += f" scan=FAIL:{body_s[:80]}"
    if not ok_s:
        ok = False

    # 评估产生链 (2026-09-22 接通): ≥3 交易日种子对账 (90d K 真实收益 →
    # record_evaluation) + 监控池 predict 新种子。Phase 3 面板三 item 此前恒空
    # = 评估回链从未建 (09-22 实锤), 本段接通它 (scan 链同宗)。
    # timeout 360s (09-22 实修: 重启后冷链首跑 predict×5 串行走完 >240s 撞超时;
    # crontab 链无实时要求, 慢≠断)
    # 派发形 (09-22 实修): 端点即回不等长链 (同步形曾 >240s 恒占 = 链坏形态)
    ok_e, body_e = post("/api/fusion/evaluate", timeout=15)
    _be = body_e.replace(" ", "")
    if ok_e and '"dispatched":true' in _be:
        line += " eval=SUBMIT"
    elif ok_e and '"skipped"' in _be:
        line += " eval=BUSY"
    else:
        line += f" eval=FAIL:{body_e[:60]}"
        ok = False

    # 因子衰减监控喂料 (2026-09-22 接通): 监控池 (factor_t, t→t+1 收益) 历史对
    # → factor_ic_monitor (此前 9 天唯一喂料 = analyze 链偶发触发, 重启清零 +
    # 面板键名错位 = 三面板恒空)。日增量 2 对/票, 衰减告警链 20 样本起判。
    ok_d, body_d = post("/api/factors/alpha/decay/record", timeout=240)
    try:
        dc = json.loads(body_d).get("data", {}) if ok_d else {}
        line += (f" decay={'OK' if ok_d else 'FAIL'}"
                 f"(s={dc.get('stocks', '-')}/p={dc.get('pairs', '-')}/"
                 f"a={dc.get('alerts_total', '-')})"
                 if ok_d else f" decay=FAIL:{body_d[:60]}")
    except Exception:
        line += f" decay=FAIL:{body_d[:60]}"
    if not ok_d:
        ok = False

    # P1-a performative 反馈链 (2026-09-22, arXiv 2412.10545): 成熟 model seed
    # (23:10 scan 链每晚自产) → ≥3d K线真收益对账 → UDE.update_accuracy 喂入
    # (每模型 EWMA acc → softmax 自动降档)。非实时自指 = replay 真 outcome 形。
    ok_f, body_f = post("/api/decision/accuracy", timeout=25)
    try:
        fa = json.loads(body_f).get("data", {}) if ok_f and body_f.startswith("{") else {}
        line += (f" feed={'OK' if ok_f else 'FAIL'}"
                 f"(pairs={fa.get('fed_pairs', '-')}|skip={fa.get('skipped', '-')})"
                 if ok_f else f" feed=FAIL:{body_f[:60]}")
    except Exception:
        line += f" feed=FAIL:{body_f[:60]}"
    if not ok_f:
        ok = False

    # P2-A BPS 合成层 (2026-09-22, arXiv 2510.07180): 成熟 seed → η EWMA
    # (式3 log q, δ0.05) → softmax(η) 模型权重 → bps_weights.jsonl 观察档。
    # 与 feed 段独立 (consumed_bps 独立第二锁, 同 seed 双观察); UDE_BPS_WEIGHT
    # 三档 (09-23 S5): '1'=即喂 / '2'=auto 门 (观察 ≥3 记录后 auto feed, 现='2'
    # 经 launchd 注入 = ~09-25/26 首观察, 首 feed ≈09-29) / 默认 0 = 纯观察
    ok_b, body_b = post("/api/decision/bps", timeout=30)
    try:
        bb = json.loads(body_b).get("data", {}) if ok_b and body_b.startswith("{") else {}
        line += (f" bps={'OK' if ok_b else 'FAIL'}"
                 f"(pairs={bb.get('fed_pairs', '-')}|skip={bb.get('skipped', '-')})"
                 if ok_b else f" bps=FAIL:{body_b[:60]}")
    except Exception:
        line += f" bps=FAIL:{body_b[:60]}"
    if not ok_b:
        ok = False

    # 链尾热读 (#18 链 C, 2026-09-13; 09-13 最新日期核查修注): 四票情报链
    # GET 主进程执行 → 跨进程链健康检查 (200 空 body/超时 = FAIL 记日志,
    # 链坏≠链死)。⚠ 缓存 TTL 全部 ≤6h < 23:10→08:00 间隔 (8h50m) — 预热
    # 缓存晨间必然过期, 主链 08:00 全部冷启动重取 = 最新日期形态 (非 2-4s
    # 省时的「读热」); 本链 = 健康检查 + 23:10 决策链闭环, 不背加速职责。
    warm = []
    for c in WARM_STOCKS:
        ok2, _b = get(f"/api/intel/{c}", timeout=25)
        warm.append(f"{c}:{'OK' if ok2 else 'FAIL'}")
    line += f" warm=[{','.join(warm)}]"

    # 单源写文件 (09-15 修: 原 print+显式写双输出, cron 又把 stdout 重定向
    # 到同一 LOG_FILE → 每条日志双写, routine replay_status 计数翻倍)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
