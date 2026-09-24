"""bps_synthesis.py — BPS 贝叶斯预测合成层 (2026-09-22, arXiv 2510.07180)

论文真身 = Bayesian Portfolio Optimization by Predictive Synthesis (Kato et
al., 非 belief polarization — 09-22 侦察更正): 多收益率预测器 → 式3 式
η_i,t=(1−δ)η_i,t−1+δ·log q_i,t, q_i,t=f_i,t(y_t) 预测密度 → 式4 式
w_t=softmax(η_t) 自适应合成权重。5002 化: per-model 二值 dir+conf 票
(2510.07180 预测器连续收益形, 5002 模型层离散化形) → q 离散形:
对 q=conf, 错 q=1-conf (0.6 对 = log(0.6) 重罚 ≠ 全错 — density 非 binary)。

链形 (accuracy_feed/P1-a 同宗, 09-22): model_seed.jsonl 23:10 自产 →
≥3d 成熟对 → 90d K 真收益对账 → η → softmax(η) 模型权重。consumed_bps
独立第二锁 (不与 accuracy_feed 的 seed_consumed 共享 = 双链独立喂, 观察
档双跑对比, 翻转待数据成熟 ~09-26); min_groups<20 诚实 skipped (Brier
n<20 同形); 链坏≠链死 (全路径吞异常)。默认 feed_weights=False = 只观察
不改 UDE — 激活 (flag 翻转) 待 bps_weights.jsonl 与 acc 链 accuracy 分叉
证据积累 (证据先行, 门控后动, 23:10 链同宗)。

消费方: UDE.decide 内联 (flag UDE_BPS_WEIGHT=1 → bps 权重覆盖 acc 链,
当前默认 0 = 不接线); 组合层 BL views 落点待 09-26 数据成熟拍板。
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

from modules.logger import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AB_DIR = os.path.join(_ROOT, 'runs', 'consensus_ab')


def load_bps_weights(weights: Dict, source) -> tuple:
    """bps 权重消费端 (2026-09-23 待拍板S2): 观察档/直传 dict → merge 现权重。

    merge ≠ 全替换: 出票模型取 bps 值, 未出票保原值 → 归一 (decide 萎缩锁)。
    source: 文件路径 str (读末行 'bps' = 观察档 append 形) 或 bps dict (测试直传)。
    坏/缺/空 → (原 weights, None) = 链坏≠链死 (全路径吞异常, recalibrate 同宗)。
    """
    try:
        if isinstance(source, dict):
            bps = source
        elif source and os.path.exists(source):
            with open(source, encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return dict(weights), None
            bps = json.loads(lines[-1]).get('bps') or {}
        else:
            return dict(weights), None
        if not bps or not all(isinstance(v, (int, float)) and v > 0
                              for v in bps.values()):
            return dict(weights), None
        merged = {m: float(bps.get(m, w)) for m, w in weights.items()}
        s = sum(merged.values())
        if s <= 0:
            return dict(weights), None
        return {m: v / s for m, v in merged.items()}, \
            f"bps={len(bps)} 模型 merge"
    except (OSError, ValueError, TypeError, KeyError):
        return dict(weights), None


def bps_obs_count(path: Optional[str]) -> int:
    """观察档行数 (S5 auto_flip 门源, 2026-09-23 残留②)。

    '2' = auto 档语义: 观察记录 ≥3 条才喂本轮权重 (第 4 次触发);
    <3 = 只观察不碰活链。缺/空/坏 = 0 = 诚实空 (链坏≠链死, 门安全向:
    记录攒不满 = 永不 auto feed, 非提前 flip)。
    """
    try:
        if not path or not os.path.exists(path):
            return 0
        with open(path, encoding='utf-8') as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def bps_calibrate(ude=None, seed_path: Optional[str] = None,
                  consumed_path: Optional[str] = None,
                  weights_path: Optional[str] = None,
                  min_groups: int = 20, delta: float = 0.05,
                  feed_weights: bool = False, fetcher=None) -> Dict:
    """23:10 replay 段入口: 成熟 seed → 式3 η → 式4 softmax → 权重 (+可选喂入)。

    Args:
        ude: UnifiedDecisionEngine (默认 None; 默认路径取实例四件套)
        seed_path: model_seed.jsonl 覆盖 (None→ude._seed_path/默认)
        consumed_path: consumed_bps.jsonl 第二锁 (防 seed 重喂稀释)
        weights_path: bps_weights.jsonl 观察档 (append 形, 非 replace —
            09-22 教训: 固定名多写方必崩, append+消费层防重双锁)
        min_groups: 成熟组门 (<此数诚实 skipped, Brier n<20 同形)
        delta: 式3 遗忘因子 (BPS 式3 δ)
        feed_weights: True 才写 ude.weights (默认 False = 观察档不触活链);
            'auto' = S5 门控 (观察记录 ≥3 条才喂, UDE_BPS_WEIGHT='2' 链尾)
        fetcher: 测试注入 K线源 (None→StockDataFetcher 真链)

    Returns:
        {'skipped': reason} 或 {'fed_pairs','groups','weights'}
    """
    try:
        sp = seed_path or getattr(ude, '_seed_path', None) or \
            os.path.join(_AB_DIR, 'model_seed.jsonl')
        cp = consumed_path or getattr(ude, '_bps_consumed', None) or \
            os.path.join(_AB_DIR, 'consumed_bps.jsonl')
        wp = weights_path or getattr(ude, '_bps_weights_path', None) or \
            os.path.join(_AB_DIR, 'bps_weights.jsonl')
        if not os.path.exists(sp):
            return {'skipped': f'seed 未建 ({sp})'}

        groups: Dict = {}   # (code, ts) → [(model, dir, conf), ...] 插序=文件行序
        with open(sp, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    groups.setdefault((s['code'], s['ts']), []).append(
                        (s['model'], s['dir'], float(s['conf'])))
                except (KeyError, ValueError, TypeError):
                    continue

        consumed = set()
        if os.path.exists(cp):
            with open(cp, encoding='utf-8') as f:
                for line in f:
                    try:
                        consumed.add(tuple(json.loads(line)))
                    except (ValueError, TypeError):
                        continue

        if fetcher is None:
            from modules.data_fetcher import StockDataFetcher
            fetcher = StockDataFetcher()

        # ── 式3: 成熟对扫描 (3d 成熟 + |ret|≥0.5% 噪声门, recalibrate 同形)
        eta: Dict = {}     # model → η (log q EWMA, δ)
        pairs = 0
        groups_fed = 0
        unripe = 0
        new_consumed: List[str] = []
        klines: Dict = {}
        for (code, ts), preds in groups.items():
            if (code, ts) in consumed:
                continue                       # 第二锁: 已喂组跳 (防稀释)
            try:
                ts_dt = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if (datetime.now() - ts_dt).days < 3:
                unripe += 1
                continue                       # 未成熟 (accuracy_feed 同门)
            if code not in klines:
                try:
                    klines[code] = fetcher.get_kline_data(code, 'daily', 90)
                except Exception:
                    klines[code] = None
            kl = klines.get(code) or []
            if len(kl) < 10:
                unripe += 1
                continue
            p0 = None
            for k in reversed(kl):             # 决策日收盘锚 (feed 形)
                d = str(k.get('date', ''))[:10]
                if d and d <= ts_dt.strftime('%Y-%m-%d'):
                    p0 = k.get('close')
                    break
            p1 = kl[-1].get('close')
            if not p0 or not p1 or p0 <= 0:
                unripe += 1
                continue
            ret = (p1 - p0) / p0
            if abs(ret) < 0.005:
                unripe += 1
                continue                       # 噪声区无方向语义 (同门)
            actual = 'up' if ret > 0 else 'down'
            n_pair = 0
            for m, d, c in preds:
                if d not in ('buy', 'sell'):
                    continue                   # 非方向票 (neutral 形) 非证据
                correct = (d == 'buy' and actual == 'up') or \
                          (d == 'sell' and actual == 'down')
                q = c if correct else (1.0 - c)     # 式3 q_i: density 非 binary
                q = max(min(q, 0.999), 1e-6)
                eta[m] = (1.0 - delta) * eta.get(m, 0.0) + delta * math.log(q)
                n_pair += 1
            if n_pair:
                new_consumed.append(json.dumps([code, ts]))
                pairs += n_pair
                groups_fed += 1

        if pairs == 0:
            return {'skipped': f'无成熟对 ({unripe} 组未成熟/坏线已跳)'}
        if groups_fed < min_groups:
            return {'skipped': f'成熟对 {groups_fed} 组<{min_groups} 门 '
                               f'(Brier n<20 同形, 跳=未消费下次仍可喂)'}

        # ── S5 auto 门 (残留② 09-23, flag='2' 链尾): 'auto' = 观察 ≥3 记录后
        # 第 4 次触发喂 (count 读 append 前旧数 = 不含本轮); <3 只观察不触活链
        feed = (feed_weights is True
                or (feed_weights == 'auto' and bps_obs_count(wp) >= 3))

        # ── 式4: softmax(η) 自适应权重 (BPS 式4, γ=1)
        vals = {m: math.exp(v) for m, v in eta.items()}
        s = sum(vals.values())
        weights = {m: (v / s if s > 0 else 1.0) for m, v in vals.items()}

        with open(cp, 'a', encoding='utf-8') as f:
            f.write('\n'.join(new_consumed) + '\n')
        # 观察记录 = 双链分叉证据自积累 (S1, 09-23): acc = 喂入时现链权重快照
        # + bps = 式4 新权重 → 09-27 拍板直接回读分叉, 无需人工盯双链
        acc_snap = {}
        if ude is not None:
            try:
                acc_snap = {m: round(float(v), 6) for m, v in sorted(ude.weights.items())}
            except (AttributeError, TypeError, ValueError):
                acc_snap = {}
        rec = {'ts': datetime.now().isoformat(timespec='seconds'),
               'pairs': pairs, 'groups': groups_fed,
               'eta': {m: round(v, 6) for m, v in sorted(eta.items())},
               'acc': acc_snap,
               'bps': {m: round(v, 6) for m, v in sorted(weights.items())}}
        with open(wp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logger.debug(f"[BPS] 合成更新 {pairs} 对/{groups_fed} 组 → "
                     f"{ {m: round(v, 3) for m, v in weights.items()} }")

        if feed and ude is not None:
            # 激活链 (23:10 段7, flag 门控): append 后读末行 = 含本条新记录 →
            # merge 入 UDE (S2 09-23: 与 init 持久载同源同语义, merge 非全替换
            # = 未出票 13+ 模型不被清零, decide 萎缩锁; 键 'bps' 形同观察档)
            w_now, info = load_bps_weights(ude.weights, wp)
            if info:
                ude.weights = w_now
                logger.info(f"[BPS] UDE weights merge 更新 ({info}, feed=True)")

        return {'fed_pairs': pairs, 'groups': groups_fed, 'weights': weights}
    except Exception as e:
        logger.debug(f"[BPS] 合成跳过: {e}")
        return {'skipped': f'bps 失败: {type(e).__name__}: {str(e)[:100]}'}
