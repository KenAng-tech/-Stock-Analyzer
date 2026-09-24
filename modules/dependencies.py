#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
核心依赖单例模块

集中管理所有跨模块共享的核心实例，通过延迟导入消除循环依赖。
其他模块从此处导入，而非直接从 app.py 导入。

使用方式:
    from modules.dependencies import get_data_fetcher, get_analysis_engine, ...
"""

from typing import TYPE_CHECKING

from modules.logger import logger

if TYPE_CHECKING:
    from modules.data_fetcher import StockDataFetcher
    from modules.analysis_engine import AnalysisEngine
    from modules.ml_predictor import MLPredictor
    from flask_socketio import SocketIO

# ── 缓存单例 ────────────────────────────────────────────────
_data_fetcher = None
_analysis_engine = None
_ml_predictor = None
_socketio = None
_websocket_handler = None
_patchtst_integrator = None
_diffusion_predictor = None
_mamba_hft = None
_self_supervised_pretrainer = None


def get_data_fetcher() -> "StockDataFetcher":
    """获取 StockDataFetcher 单例"""
    global _data_fetcher
    if _data_fetcher is None:
        from modules.data_fetcher import StockDataFetcher
        _data_fetcher = StockDataFetcher()
    return _data_fetcher


def get_analysis_engine() -> "AnalysisEngine":
    """获取 AnalysisEngine 单例"""
    global _analysis_engine
    if _analysis_engine is None:
        from modules.analysis_engine import AnalysisEngine
        _analysis_engine = AnalysisEngine()
    return _analysis_engine


def get_ml_predictor() -> "MLPredictor":
    """获取 MLPredictor 单例"""
    global _ml_predictor
    if _ml_predictor is None:
        from modules.ml_predictor import MLPredictor
        _ml_predictor = MLPredictor()
    return _ml_predictor


def get_socketio():
    """获取 SocketIO 实例"""
    global _socketio
    if _socketio is None:
        # SocketIO 在 app.py 中创建，需要外部设置
        raise RuntimeError(
            "SocketIO 未初始化。请通过 set_socketio() 设置，"
            "或直接从 app import socketio"
        )
    return _socketio


def set_socketio(s):
    """由 app.py 调用，设置 SocketIO 实例"""
    global _socketio
    _socketio = s


def get_websocket_handler():
    """获取 WebSocketHandler 实例"""
    global _websocket_handler
    if _websocket_handler is None:
        from modules.websocket_handler import get_websocket_handler as _gh
        _websocket_handler = _gh()
    return _websocket_handler


def set_websocket_handler(h):
    """由 app.py 调用，设置 WebSocketHandler 实例"""
    global _websocket_handler
    _websocket_handler = h


# 2026-09-08 断链横扫: 以下 4 个 getter 原从 modules.sota_integration 导入
# get_patchtst/get_diffusion_predictor/get_mamba_hft_predictor/get_self_supervised_pretrainer —
# 该模块不存在这些函数 (真实定义在 modules/models/ 下) → 每次调用恒 ImportError。
# patchtst 版无 try/except 且排第一 → sota_training_scheduler._get_model_instances
# 整块 try 在首个调用即抛 → 4 个模型实例全部 None (trained 标志更新/推理路由空转,
# 23:00 链日志实锤 "获取模型实例失败")。与 09-03 model_registry 8 处死路径横扫同源漏网。

def get_patchtst_integrator():
    """获取 PatchTST 集成器单例"""
    global _patchtst_integrator
    if _patchtst_integrator is None:
        try:
            from modules.models.patchtst_integrator import get_patchtst
            _patchtst_integrator = get_patchtst()
        except Exception as e:
            logger.warning(f"[dependencies] PatchTST 集成器获取失败: {e}")
    return _patchtst_integrator


def get_diffusion_predictor():
    """获取 Diffusion 预测器单例"""
    global _diffusion_predictor
    if _diffusion_predictor is None:
        try:
            from modules.models.diffusion_model import get_diffusion_predictor as _gdp
            _diffusion_predictor = _gdp()
        except Exception as e:
            logger.warning(f"[dependencies] Diffusion 预测器获取失败: {e}")
    return _diffusion_predictor


def get_mamba_hft():
    """获取 Mamba HFT 预测器单例"""
    global _mamba_hft
    if _mamba_hft is None:
        try:
            from modules.models.hft_mamba import get_mamba_hft_predictor as _gmp
            _mamba_hft = _gmp()
        except Exception as e:
            logger.warning(f"[dependencies] Mamba HFT 预测器获取失败: {e}")
    return _mamba_hft


def get_self_supervised_pretrainer():
    """获取自监督预训练器单例"""
    global _self_supervised_pretrainer
    if _self_supervised_pretrainer is None:
        try:
            from modules.models.self_supervised import get_self_supervised_pretrainer as _gsp
            _self_supervised_pretrainer = _gsp()
        except Exception as e:
            logger.warning(f"[dependencies] 自监督预训练器获取失败: {e}")
    return _self_supervised_pretrainer


def reset_all():
    """重置所有单例（用于测试）"""
    global _data_fetcher, _analysis_engine, _ml_predictor, _socketio
    global _websocket_handler, _patchtst_integrator, _diffusion_predictor
    global _mamba_hft, _self_supervised_pretrainer
    _data_fetcher = None
    _analysis_engine = None
    _ml_predictor = None
    _socketio = None
    _websocket_handler = None
    _patchtst_integrator = None
    _diffusion_predictor = None
    _mamba_hft = None
    _self_supervised_pretrainer = None
