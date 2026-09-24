#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SOTA 模型每日训练调度器

功能:
1. 启动时自动训练（如果模型未训练或已过期）
2. 定时重训练（默认 24 小时）
3. 按天固定时间重训练（默认晚上 11 点）
4. 按需触发训练（通过 force_train）
5. 后台线程执行，不阻塞主线程
6. 训练完成后更新模型实例状态

用法:
    from modules.sota_training_scheduler import sota_scheduler
    sota_scheduler.start()
    sota_scheduler.force_train()
    sota_scheduler.stop()
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from modules.logger import logger
from train_sota_models import train_all_models


# ─────────────────────────────────────────────
# 全局模型实例引用（用于训练后更新状态）
# ─────────────────────────────────────────────

# 延迟导入，避免循环依赖
_patchtst_integrator = None
_diffusion_predictor = None
_mamba_hft = None
_self_supervised_pretrainer = None


def _get_model_instances():
    """获取全局模型实例"""
    global _patchtst_integrator, _diffusion_predictor, _mamba_hft, _self_supervised_pretrainer

    try:
        from modules.dependencies import (
            get_patchtst_integrator, get_diffusion_predictor,
            get_mamba_hft, get_self_supervised_pretrainer,
        )
        _patchtst_integrator = get_patchtst_integrator()
        _diffusion_predictor = get_diffusion_predictor()
        _mamba_hft = get_mamba_hft()
        _self_supervised_pretrainer = get_self_supervised_pretrainer()
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] 获取模型实例失败: {e}")


def _update_model_trained_flags():
    """训练完成后更新模型实例的 trained 标志"""
    _get_model_instances()

    updated = []
    try:
        if _patchtst_integrator is not None:
            # 2026-09-10 断链修复: 重训后重载权重 (旧版只置 flag →
            # 随机初始权重被当"已训练"forward, 投票是纯噪声)
            if _patchtst_integrator.load():
                updated.append('patchtst')
            else:
                logger.warning("[SOTATrainingScheduler] PatchTST 权重重载失败, "
                               "保持未训练 (避免随机权重冒充模型)")
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] 更新 PatchTST 状态失败: {e}")

    try:
        if _diffusion_predictor is not None:
            _diffusion_predictor.trained = True
            updated.append('diffusion')
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] 更新 Diffusion 状态失败: {e}")

    try:
        if _mamba_hft is not None:
            _mamba_hft.trained = True
            updated.append('mamba')
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] 更新 Mamba 状态失败: {e}")

    try:
        if _self_supervised_pretrainer is not None:
            _self_supervised_pretrainer.trained = True
            updated.append('self_supervised')
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] 更新 Self-Supervised 状态失败: {e}")

    if updated:
        logger.info(f"[SOTATrainingScheduler] 模型状态已更新: {', '.join(updated)}")


# ─────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# GNN 后台真训练 (2026-09-03, Task #13)
# ─────────────────────────────────────────────

_GNN_LOCK = threading.RLock()


def _train_gnn_background(stock_code: str = 'sz300620'):
    """GNN 真训练链 (纯 numpy, 无需 MPS_LOCK)。

    watchlist 6 票 60d K线 (每票 12s 超时保护) → 每票滚动窗口造样本
    (窗口内收益率→特征, 未来 5d 收益→标签, 无未来函数) →
    GNNPredictor.train (解析梯度反传) → holdout_acc>0.4 门控才置 trained
    → npz 持久化 (重启不丢)。

    fire-and-forget: train_sota_models_internal 起后台线程跑,
    不阻塞 force_train 同步 HTTP 链; 并发由 _GNN_LOCK 串行 (重复触发跳过)。
    """
    if not _GNN_LOCK.acquire(timeout=5):
        logger.info("[GNN] 已有训练在跑, 跳过本轮")
        return
    try:
        from modules.models.gnn_predictor import get_gnn_predictor, GNNPredictor
        from modules.data_fetcher import StockDataFetcher

        gnn = get_gnn_predictor()
        codes = list(dict.fromkeys([stock_code] + [
            'sz300620', 'sh688981', 'sh600519', 'sh601318', 'sz000001', 'sh600036']))[:6]
        fetcher = StockDataFetcher()

        t0 = time.time()
        ks_map = {}
        for code in codes:
            holder = [None]

            def _fetch(c=code):
                try:
                    holder[0] = fetcher.get_kline_data(c, 'daily', 60)
                except Exception:
                    holder[0] = None

            th = threading.Thread(target=_fetch, daemon=True)
            th.start()
            th.join(timeout=12)
            kl = holder[0]
            if kl and len(kl) >= 40:
                ks_map[code] = kl

        Xs, ys, n_pool = [], [], 0
        for kl in ks_map.values():
            X, y = GNNPredictor.build_training_samples(kl, window=20, horizon=5)
            if len(X):
                Xs.append(X)
                ys.append(y)
                n_pool += len(X)

        # 阈值 120 ≈ 4 票 × 30 窗 (watchlist 常态 5 票约 170); 2026-09-03 首跑实测 170
        # 被 200 挡 → 降阈值。真门控在 train: holdout_acc>0.4 才算 trained
        if n_pool < 120:
            logger.info(f"[GNN] 样本池不足: {n_pool} < 120 ({len(ks_map)} 票), 跳过本轮")
            return

        res = gnn.train(np.vstack(Xs), np.concatenate(ys), epochs=200, lr=0.1)
        dt = time.time() - t0
        if res.get('is_trained'):
            gnn.save()
            logger.info(f"[GNN] 后台训练完成: {n_pool} 样本, "
                        f"holdout_acc={res.get('holdout_acc')}, {dt:.1f}s, 已持久化")
    except Exception as e:
        logger.warning(f"[GNN] 后台训练失败: {e}")
    finally:
        _GNN_LOCK.release()


def train_sota_models_internal(
    stock_code: str = 'sz300620',
    days: int = 500,
    n_features: int = 12,
    update_flags: bool = True,
) -> Dict:
    """
    内部训练函数，供调度器调用

    Args:
        stock_code: 股票代码
        days: 训练数据天数
        n_features: 特征数
        update_flags: 训练后是否更新模型实例状态

    Returns:
        训练结果字典
    """
    # 2026-09-02 修复: MPS 重计算串行化 (统一锁点, 覆盖 on_startup/定时/force_train
    # /POST /api/sota/train 全入口) — 与共识决策链并发使用 MPS 会触发
    # Metal 断言崩溃 (A command encoder is already encoding..., 进程 abort)
    from modules.mps_lock import MPS_LOCK
    logger.info(f"[SOTATrainingScheduler] 开始训练 SOTA 模型: {stock_code}")
    if not MPS_LOCK.acquire(timeout=300):
        logger.warning("[SOTATrainingScheduler] MPS 锁等待超时 (300s), 本轮训练跳过")
        return {'success': False, 'skipped': True, 'error': 'MPS 锁等待超时'}
    try:
        result = train_all_models(stock_code=stock_code, days=days, n_features=n_features)

        if update_flags:
            _update_model_trained_flags()
    finally:
        MPS_LOCK.release()

    # 2026-09-03: GNN 纯 numpy 真训练 (无 MPS 需求) — 后台线程 fire-and-forget,
    # 不阻塞本函数同步调用方 (force_train/23:00/on_startup); 并发 _GNN_LOCK 串行
    try:
        threading.Thread(target=_train_gnn_background,
                         args=(stock_code,), daemon=True).start()
    except Exception as e:
        logger.warning(f"[SOTATrainingScheduler] GNN 后台训练启动失败: {e}")

    logger.info(f"[SOTATrainingScheduler] 训练完成, 耗时: {result.get('elapsed_seconds', 0):.1f}s")
    return result


# ─────────────────────────────────────────────
# 调度器类
# ─────────────────────────────────────────────

class SOTATrainingScheduler:
    """
    SOTA 模型训练调度器

    功能:
      - 启动时自动训练（如果模型不存在或已过期）
      - 定时重训练（默认 24 小时）
      - 按天固定时间重训练（默认晚上 11 点）
      - 按需触发训练
      - 后台线程执行，不阻塞主线程

    用法:
      scheduler = SOTATrainingScheduler()
      scheduler.start()
      scheduler.force_train()
      scheduler.stop()
    """

    def __init__(
        self,
        interval_hours: int = 24,
        stock_code: str = 'sz300620',
        target_hour: int = 23,
    ):
        """
        Args:
            interval_hours: 重训练间隔（小时）
            stock_code: 默认训练的股票代码
            target_hour: 按天固定时间的小时数（默认 23 = 晚上 11 点）
        """
        self.interval_hours = interval_hours
        self.stock_code = stock_code
        self.target_hour = target_hour
        self._thread: Optional[threading.Thread] = None
        self._train_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_train_result: Optional[Dict] = None
        self._is_training = False
        self._last_train_time: Optional[datetime] = None

    def start(self, on_startup: bool = True):
        """
        启动后台调度线程

        Args:
            on_startup: 是否在启动时立即检查并训练
        """
        if self._thread and self._thread.is_alive():
            logger.warning("[SOTATrainingScheduler] 调度器已在运行")
            return

        self._stop_event.clear()

        # 启动时检查: 如果模型未训练，立即训练
        if on_startup:
            logger.info("[SOTATrainingScheduler] 启动时触发模型训练...")
            # 在新线程中训练，不阻塞启动
            self._train_thread = threading.Thread(
                target=self._train_and_schedule,
                name="sota-train-startup",
                daemon=True,
            )
            self._train_thread.start()
        else:
            self._schedule_next()

        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="sota-train-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[SOTATrainingScheduler] 调度器已启动，重训练间隔: {self.interval_hours}h, 固定时间: {self.target_hour}:00")

    def stop(self):
        """停止后台调度"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[SOTATrainingScheduler] 调度器已停止")

    def force_train(self, stock_code: Optional[str] = None) -> Dict:
        """
        手动触发模型训练

        Args:
            stock_code: 训练用的股票代码（默认用初始化时的）

        Returns:
            训练结果字典
        """
        with self._lock:
            if self._is_training:
                return {'success': False, 'message': '模型正在训练中，请稍后重试'}
            self._is_training = True

        try:
            code = stock_code or self.stock_code
            logger.info(f"[SOTATrainingScheduler] 手动触发模型训练: {code}")
            result = train_sota_models_internal(stock_code=code)
            self._last_train_result = result
            self._last_train_time = datetime.now()
            self._schedule_next()
            return result
        except Exception as e:
            logger.error(f"[SOTATrainingScheduler] 训练失败: {e}")
            self._last_train_result = {'success': False, 'error': str(e)}
            return self._last_train_result
        finally:
            with self._lock:
                self._is_training = False

    def get_status(self) -> Dict:
        """获取调度器状态"""
        with self._lock:
            is_training = self._is_training

        return {
            'running': self._thread is not None and self._thread.is_alive(),
            'is_training': is_training,
            'interval_hours': self.interval_hours,
            'target_hour': self.target_hour,
            'stock_code': self.stock_code,
            'last_train_time': self._last_train_time.isoformat() if self._last_train_time else None,
            'last_train_result': self._last_train_result,
        }

    def _scheduler_loop(self):
        """后台调度循环: 按固定时间触发训练"""
        while not self._stop_event.is_set():
            wait_seconds = self._next_trigger_delay()
            if self._stop_event.wait(wait_seconds):
                break  # 收到停止信号

            if self._stop_event.is_set():
                break

            logger.info("[SOTATrainingScheduler] 定时重训练触发")
            self._train_and_schedule()

    def _next_trigger_delay(self) -> float:
        """
        计算距离下一次固定时间触发的秒数

        逻辑:
        - 如果今天已经训练过，下一次就是明天 target_hour
        - 如果今天还没到 target_hour，下一次就是今天 target_hour
        - 如果今天已过 target_hour，下一次也是明天 target_hour
        """
        now = datetime.now()
        # 今天 target_hour 的时间
        today_target = now.replace(hour=self.target_hour, minute=0, second=0, microsecond=0)
        # 如果已经过了今天的 target_hour，则下一次是明天
        if now >= today_target:
            next_trigger = today_target + timedelta(days=1)
        else:
            next_trigger = today_target

        delay = (next_trigger - now).total_seconds()
        logger.info(f"[SOTATrainingScheduler] 下次训练将在 {delay:.0f} 秒后 ({next_trigger.strftime('%Y-%m-%d %H:%M')})")
        return delay

    def _schedule_next(self):
        """设置下一次训练的定时器"""
        wait_seconds = self._next_trigger_delay()
        timer = threading.Timer(
            wait_seconds,
            self._on_timer_expire,
        )
        timer.daemon = True
        timer.start()

    def _on_timer_expire(self):
        """Timer 到期回调"""
        if not self._stop_event.is_set():
            logger.info("[SOTATrainingScheduler] 定时重训练触发")
            self._train_and_schedule()

    def _train_and_schedule(self):
        """训练模型并设置下一次调度 (MPS 锁已下沉到 train_sota_models_internal 统一入口)"""
        try:
            result = train_sota_models_internal(stock_code=self.stock_code)
            self._last_train_result = result
            self._last_train_time = datetime.now()
        except Exception as e:
            logger.error(f"[SOTATrainingScheduler] 训练失败: {e}")
            self._last_train_result = {'success': False, 'error': str(e)}
        finally:
            self._schedule_next()


# ─────────────────────────────────────────────
# 全局实例
# ─────────────────────────────────────────────

sota_scheduler = SOTATrainingScheduler()
