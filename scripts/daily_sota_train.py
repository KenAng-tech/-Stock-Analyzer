#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
每日 SOTA 模型训练脚本 — 供 cron 调用

运行方式:
    cd /Users/claw/stock_analyzer
    source venv/bin/activate
    python scripts/daily_sota_train.py

日志输出到: logs/daily_sota_train.log
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 日志配置
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'daily_sota_train.log')


def _rotate_oversized(path: str, max_bytes: int = 10 * 1024 * 1024, keep: int = 4) -> None:
    """启动时轮转: 超过 max_bytes 的日志滚动为 .1~.4, 防止无限膨胀 (2026-09-04)。

    覆盖两类日志: daily_sota_train.log (FileHandler 写入) 与
    sota_train_error.log (LaunchAgent plist StandardErrorPath 重定向写入,
    Python 侧无法直接控制, 故在每次运行开头做尺寸检查)。
    """
    try:
        if not (os.path.exists(path) and os.path.getsize(path) > max_bytes):
            return
        oldest = f"{path}.{keep}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(keep - 1, 0, -1):
            src, dst = f"{path}.{i}", f"{path}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        os.rename(path, f"{path}.1")
    except OSError as e:
        print(f"[WARN] 日志轮转失败 {path}: {e}")


_rotate_oversized(LOG_FILE)
_rotate_oversized(os.path.join(LOG_DIR, 'sota_train_error.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('daily_sota_train')


def run_benchmark_chain(logger):
    """链尾 benchmark (2026-09-11 #31): 当日 DRL agent vs 4 经典 fixture vs buy&hold

    walk_forward 5×42 test 窗同尺子 (42 = 服务链 backtest_routes 同形, >30 非死窗);
    DRL adapter = agent._policy(_extract_state(rolling 30 bars)) — benchmark 专用形态,
    不接 /api/drl/decide 服务链 (其 PortfolioState 含市场 regime 输入, benchmark 用精简形)。

    固定 seed 42: policy 采样随机已锁 (可复现基准); 对比对象 = 昨日同 seed 基准行
    → 权重漂移肉眼可读 (09-08 教训「梯度验证需合成+噪声双测」延续)。
    合成 OHLC 是 benchmark 尺子 (非实盘回放, 不接 walkforward_routes 的 DB 链)。
    """
    import numpy as np
    import random as _random
    from modules.walkforward_backtester import WalkForwardBacktester
    from modules.models.drl_agent import DRLTradingAgent, PortfolioState
    from modules.strategies.classic_fixture import ALL_FIXTURES

    seed = 42
    np.random.seed(seed)
    _random.seed(seed)

    # 合成 OHLC 270 bars (60+42×5, 与 test_backdater/服务链同尺子)
    rng = np.random.RandomState(seed)
    klines, price = [], 100.0
    for i in range(270):
        ret = rng.randn() * 0.02
        o, c = price, price * (1 + ret)
        hi = max(o, c) * (1 + abs(rng.randn()) * 0.01)
        lo = min(o, c) * (1 - abs(rng.randn()) * 0.01)
        klines.append({'date': f'2026-0{(i % 9) + 1}-{(i % 28) + 1:02d}',
                       'open': o, 'high': hi, 'low': lo, 'close': c,
                       'volume': int(rng.randint(100000, 1000000)),
                       'prev_close': o, 'atr': price * 0.03})
        price = c

    # ── DRL adapter: rolling 30 bars + 精简 PortfolioState → _policy → 动作 ──
    agent = DRLTradingAgent.load()   # 当日 drl_agent.npy (缺文件自动新建 = 随机策略对照)
    from collections import deque
    hist = deque(maxlen=30)

    def drl_strategy(bar, position, capital):
        hist.append(bar)
        if len(hist) < 30:
            return 'hold'
        k = list(hist)
        pos = {}
        if position > 0:
            pos['benchmark'] = {'value': position * bar['close'],
                                'unrealized_pnl': 0.0}
        ps = PortfolioState(cash=capital, positions=pos, total_value=capital)
        action, _conf = agent._policy(agent._extract_state(k, ps))
        if position == 0 and action > 0.1:
            return 'buy'
        if position > 0 and action < -0.1:
            return 'sell'
        return 'hold'

    def make_buy_hold():
        state = {'bought': False}

        def bh(bar, position, capital):
            if position == 0 and not state['bought']:
                state['bought'] = True
                return 'buy'
            return 'hold'
        return bh

    logger.info("  [BENCHMARK] walk_forward 同尺子对比 (5×42, seed=42):")
    candidates = [('DRL-今日', drl_strategy),
                 ('buy&hold', make_buy_hold())]
    candidates += [(name, ALL_FIXTURES[name]()) for name in ALL_FIXTURES]
    drl_r = None
    for name, fn in candidates:
        bt = WalkForwardBacktester(fn, initial_capital=1_000_000)
        try:
            r = bt.run_walk_forward(klines, train_period=60, test_period=42,
                                    n_windows=5)
            if name == 'DRL-今日':
                drl_r = (bt, r)
            s = r.get('summary', {})
            n_tx = sum(len(w.get('transactions', [])) for w in r['windows'])
            logger.info(
                f"    {name:10s} ret={s.get('mean_return', 0) * 100:+6.2f}% "
                f"sharpe={s.get('mean_sharpe', 0):+5.2f} "
                f"win={s.get('mean_winrate', 0) * 100:4.1f}% tx={n_tx:2d}")
        except Exception as e:
            logger.warning(f"    {name:10s} benchmark 失败: {e}")
    # DRL 过拟合三件套 (DSR/PBO/t) — 对照 09-10 恒 skipped 形态 = 链活化复验
    try:
        if drl_r is None:
            raise RuntimeError('DRL benchmark 未跑通')
        bt, r = drl_r
        so = bt.statistical_overfit_check(r['windows'])
        if 'skipped' in so:
            logger.info(f"  [BENCHMARK] DRL 统计诊断: skipped ({so['skipped'][:48]})")
        else:
            pbo = so.get('pbo_cscv', {}).get('pbo')
            dsr = so.get('dsr', {}).get('dsr')
            logger.info(f"  [BENCHMARK] DRL 统计诊断: DSR={dsr} PBO={pbo} "
                        f"(对照昨日行: 权重漂移 = 比值突变)")
    except Exception as e:
        logger.warning(f"  [BENCHMARK] DRL 统计诊断失败 (不影响训练结果): {e}")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练产物 → 模型文件映射 (版本流水线注册用, 2026-09-14)
_MODEL_FILES = {
    'patchtst': 'patchtst_best.pth',
    'mamba': 'mamba_hft.pth',
    'diffusion': 'diffusion_model.pth',
    'self_supervised': 'self_supervised.pth',
    'drl': 'drl_agent.npy',
}


def _observe_rolling_policy(logger):
    """链首观察 (P1-6, qlib rolling 重训策略): 距上次训练 < 间隔则只记日志不拦截。

    拦截开关留待观察期结束: 环境变量 ROLLING_MIN_INTERVAL_DAYS 默认 1 (不拦截)。
    """
    try:
        from modules.model_registry import get_version_registry
        min_interval = float(os.environ.get('ROLLING_MIN_INTERVAL_DAYS', '1'))
        days = get_version_registry().days_since_last_train('patchtst')
        if days is None:
            logger.info("[OBSERVE] rolling 策略: 无历史版本记录, 正常重训")
        elif days < min_interval:
            logger.info(f"[OBSERVE] rolling 策略本应跳过重训 (距上次 {days:.1f} 天 "
                        f"< {min_interval}) — 观察模式, 照常训练")
        else:
            logger.info(f"[OBSERVE] rolling 策略: 距上次 {days:.1f} 天, 正常重训")
    except Exception as e:
        logger.error(f"[OBSERVE] rolling 策略观察失败 (不影响训练): {e}")


def _train_range(result):
    """从训练结果提取数据日期范围 [start, end] (取不到返回 [])"""
    try:
        klines = (result.get('data') or {}).get('klines') or []
        dates = [k.get('date') for k in klines if k.get('date')]
        if dates:
            return [dates[0], dates[-1]]
    except Exception as e:
        logger.error(f"[VersionRegistry] train_range 提取失败: {e}")
    return []


def _register_model_versions(result, run_ctx, logger):
    """训练完成后注册版本 + 原子切 online (qlib online tag 思想, P1-6)。

    旧版本自动转 offline 保留可回滚; run_ctx 存在时同步落 model 工件。
    任何失败只 logger.error, 不影响训练主链结果。
    """
    try:
        from modules.model_registry import get_version_registry
        reg = get_version_registry()
        train_range = _train_range(result)
        models = result.get('models') or {}
        registered = []
        for name, fname in _MODEL_FILES.items():
            entry = models.get(name) or {}
            if not (entry.get('trained') or entry.get('drl_trained')):
                continue
            path = os.path.join(PROJECT_ROOT, 'modules', 'dl_models', fname)
            if not os.path.exists(path):
                logger.warning(f"[VersionRegistry] {name} 模型文件缺失, 跳过注册: {path}")
                continue
            metrics = {k: float(v) for k, v in entry.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            try:
                vid = reg.register_version(
                    name, path, metrics, train_range,
                    extra={'stock_code': result.get('stock_code', ''),
                           'elapsed_seconds': result.get('elapsed_seconds')})
                if reg.set_online(name, vid):
                    registered.append(vid)
                if run_ctx is not None:
                    run_ctx.save_model(path)
            except Exception as e:
                logger.error(f"[VersionRegistry] {name} 版本注册失败 (不影响训练): {e}")
        logger.info(f"[VersionRegistry] 本轮注册并上线 {len(registered)} 个版本: {registered}")
    except Exception as e:
        logger.error(f"[VersionRegistry] 版本注册链失败 (不影响训练结果): {e}")


def _observe_regime_weighting(result, logger):
    """regime 样本加权 (P2-13): REGIME_SAMPLE_WEIGHTING=1 时启用, 默认关只记分布摘要。

    训练入口 train_all_models 暂无 sample_weight 注入点 (scheduler 侧未接线),
    故启用态也只计算+记录权重, 不改变训练 — 诚实观察模式。
    历史后验用 segment_regimes 硬标签 one-hot 近似 (纯本地, 不发起网络请求)。
    """
    try:
        import numpy as np
        from modules.ml_processors import regime_sample_weights
        from modules.hyperparam_optimizer import segment_regimes
        klines = (result.get('data') or {}).get('klines') or []
        if len(klines) < 60:
            logger.info(f"[REGIME_WEIGHT] klines 不足 ({len(klines)}), 跳过权重摘要")
            return
        labels = segment_regimes(klines)
        regimes = sorted(set(labels))
        order = {r: i for i, r in enumerate(regimes)}
        posteriors = np.zeros((len(labels), len(regimes)))
        for i, lab in enumerate(labels):
            posteriors[i, order[lab]] = 1.0
        current = {labels[-1]: 1.0}
        w = regime_sample_weights(current, posteriors, regime_order=regimes)
        summary = (f"n={len(w)} 当前regime={labels[-1]} mean={w.mean():.3f} "
                   f"min={w.min():.3f} max={w.max():.3f} "
                   f"零权重样本占比={(w == 0).mean() * 100:.1f}%")
        if os.environ.get('REGIME_SAMPLE_WEIGHTING') == '1':
            logger.info(f"[REGIME_WEIGHT] 已启用 (REGIME_SAMPLE_WEIGHTING=1): {summary} "
                        f"— 注: 训练入口暂无权重注入点, 当前仅计算不生效")
        else:
            logger.info(f"[REGIME_WEIGHT] 未启用, 权重分布摘要: {summary}")
    except Exception as e:
        logger.error(f"[REGIME_WEIGHT] regime 权重观察失败 (不影响训练): {e}")


def main():
    logger.info("=" * 60)
    logger.info(f"  每日 SOTA 模型训练 — {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # ── 工件目录 (P0-3 run_artifact, guarded: 模块缺失/落盘失败不影响训练) ──
    run_ctx = None
    try:
        from modules.run_artifact import start_run
        run_ctx = start_run('sota_train', {
            'stock_code': 'sz300620', 'days': 500, 'n_features': 12,
            'regime_sample_weighting': os.environ.get('REGIME_SAMPLE_WEIGHTING', ''),
        })
    except ImportError:
        logger.info("[run_artifact] 模块不可用, 跳过工件记录")
    except Exception as e:
        logger.error(f"[run_artifact] start_run 失败 (不影响训练): {e}")

    # 链首: rolling 重训策略观察 (只记录不拦截, P1-6)
    _observe_rolling_policy(logger)

    try:
        # 导入并启动调度器
        from modules.sota_training_scheduler import train_sota_models_internal

        result = train_sota_models_internal(
            stock_code='sz300620',
            days=500,
            n_features=12,
            update_flags=True,
        )

        logger.info(f"  训练完成! 耗时: {result.get('elapsed_seconds', 0):.1f}s")
        logger.info(f"  结果已保存: results/sota_training_results.json")

        # 训练后接线 (全部 guarded, 失败不影响训练主链):
        # 1) 版本注册 + online 原子切换 (P1-6)  2) regime 权重观察 (P2-13)
        _register_model_versions(result, run_ctx, logger)
        _observe_regime_weighting(result, logger)
        if run_ctx is not None:
            try:
                models = result.get('models') or {}
                run_ctx.save_metrics({
                    name: {k: v for k, v in entry.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    for name, entry in models.items() if isinstance(entry, dict)
                })
            except Exception as e:
                logger.error(f"[run_artifact] save_metrics 失败 (不影响训练): {e}")

        # 链尾 benchmark (2026-09-11 #31) — 只观测, 失败不影响训练主链结果
        try:
            run_benchmark_chain(logger)
        except Exception as e:
            logger.warning(f"  [BENCHMARK] benchmark 链失败 (不影响训练结果): {e}")

        if run_ctx is not None:
            run_ctx.finish('ok')
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"  训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if run_ctx is not None:
            run_ctx.finish('fail', str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
