"""
routes/__init__.py - Blueprint 路由注册中心

将 app.py 中的 153+ 条路由按功能拆分为独立 Blueprint，
在 app.py 中统一注册。
"""

from flask import Blueprint

def register_blueprints(app):
    """注册所有 Blueprint 到 Flask app"""
    from . import data_routes
    from . import cache_routes
    from . import backtest_routes
    from . import quant_routes
    from . import config_routes
    from . import sentiment_routes
    from . import factor_routes
    from . import training_routes
    from . import sota_routes
    from . import sota_predict_routes
    from . import sota_status_routes
    from . import sota_advanced_routes
    from . import sota_llm_routes
    from . import time_llm_routes
    from . import regime_routes
    from . import online_learning_routes
    from . import ab_test_routes
    from . import portfolio_routes
    from . import memory_routes
    from . import health_routes
    from . import static_routes
    from . import websocket_routes
    from . import dl_routes
    from . import models_routes
    from . import async_backtest_routes
    from . import monitor_routes
    # 新增 Blueprint
    from . import new_routes
    from . import mlops_routes
    from . import cvar_position_routes
    # 2026-08-13 新增: RD-Agent 因子挖掘
    from . import rd_agent_routes
    # 2026-08-13 新增: TradingAgents 多智能体
    from . import trading_agents_routes
    # 2026-08-13 新增: Kronos 金融基础模型
    from . import kronos_routes
    # 2026-08-14 新增: Consensus / SafeRL / Uncertainty / GNN
    from . import consensus_routes
    from . import xai_routes
    from . import gnn_routes
    # 2026-08-19: 多模态融合预测
    from . import fusion_routes
    # 2026-08-22: 从 app.py 迁移的残留路由
    from . import extra_routes
    # 2026-09-02 补链: 7/27 清理时悬空的 Safe RL / 贝叶斯路由
    from . import safe_rl_routes
    # 2026-09-09 新增: 每日晨报研报
    from . import daily_report_routes
    # 2026-09-17: 同花顺数据 API
    from . import ths_data_routes
    app.register_blueprint(ths_data_routes.bp)

    # Register all blueprints (route 定义已包含 /api 前缀，无需 url_prefix)
    app.register_blueprint(data_routes.bp)
    app.register_blueprint(cache_routes.bp)
    app.register_blueprint(backtest_routes.bp)
    app.register_blueprint(async_backtest_routes.bp)
    app.register_blueprint(quant_routes.bp)
    app.register_blueprint(config_routes.bp)
    app.register_blueprint(sentiment_routes.bp)
    app.register_blueprint(factor_routes.bp)
    app.register_blueprint(training_routes.bp)
    # SOTA: register all 4 sub-blueprints (predict + status + advanced + llm)
    app.register_blueprint(sota_predict_routes.bp)
    app.register_blueprint(sota_status_routes.bp)
    app.register_blueprint(sota_advanced_routes.bp)
    app.register_blueprint(sota_llm_routes.bp)
    app.register_blueprint(time_llm_routes.bp)
    app.register_blueprint(regime_routes.bp)
    app.register_blueprint(online_learning_routes.bp)
    app.register_blueprint(ab_test_routes.bp)
    app.register_blueprint(portfolio_routes.bp)
    app.register_blueprint(memory_routes.bp)
    app.register_blueprint(health_routes.bp)
    app.register_blueprint(dl_routes.bp)
    app.register_blueprint(models_routes.bp)
    app.register_blueprint(monitor_routes.bp)
    # 新增 Blueprint
    app.register_blueprint(new_routes.bp)
    app.register_blueprint(mlops_routes.bp)
    app.register_blueprint(cvar_position_routes.bp)
    # 2026-08-13 新增
    app.register_blueprint(rd_agent_routes.bp)
    app.register_blueprint(trading_agents_routes.bp)
    app.register_blueprint(kronos_routes.bp)
    # 2026-08-19: 注册 Consensus / SafeRL / Uncertainty / GNN
    app.register_blueprint(consensus_routes.bp)
    app.register_blueprint(xai_routes.bp)
    app.register_blueprint(gnn_routes.bp)
    # 2026-08-19: 多模态融合预测
    app.register_blueprint(fusion_routes.bp)
    # 2026-08-22: 从 app.py 迁移的残留路由
    app.register_blueprint(extra_routes.bp)
    # 2026-09-02 补链: /api/safe-rl/status + /api/uncertainty/status
    app.register_blueprint(safe_rl_routes.bp)
    # 2026-09-09: 每日晨报研报 (08:00 链 API)
    app.register_blueprint(daily_report_routes.bp)
    # 2026-09-12: 决策链观察 (校准门控/执行审计/情报雷达/AI洞察, P0-P1)
    from . import decision_routes
    app.register_blueprint(decision_routes.bp)
    # 2026-09-14: 2026 整合层 (qlib 借鉴 + 2026 调研 P0-P2 面板 API)
    from . import integration2026_routes
    app.register_blueprint(integration2026_routes.bp)
    # Static routes
    app.register_blueprint(static_routes.bp)
    # WebSocket routes: init_socketio(socketio) called in app.py after socketio creation
