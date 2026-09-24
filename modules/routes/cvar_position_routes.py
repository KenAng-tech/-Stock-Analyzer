"""
cvar_position_routes.py — CVaR 约束仓位管理 API 路由

提供 CVaR/EVT + Kelly + Black-Litterman 仓位优化接口:
- GET /api/position/cvar/<stock_code> — CVaR 约束仓位建议
- GET /api/position/kelly — Kelly 公式计算
- GET /api/position/black-litterman — Black-Litterman 贝叶斯估计
"""

from flask import Blueprint, jsonify, request
import numpy as np
from datetime import datetime
from modules.logger import logger

bp = Blueprint('cvar_position', __name__)


def _get_closes_from_klines(klines):
    """从 K 线数据提取收盘价"""
    if not klines:
        return None
    closes = [float(k.get('close', 0)) for k in klines if float(k.get('close', 0)) > 0]
    return np.array(closes) if len(closes) >= 60 else None


@bp.route('/api/position/cvar/<stock_code>')
def cvar_position(stock_code):
    """
    获取 CVaR 约束下的最优仓位建议

    查询参数:
        portfolio_value: 组合总价值 (默认 1000000)
        confidence: 置信水平 (默认 0.95)
    """
    try:
        from modules.cvar_position_manager import cvar_position_manager
        from modules.dependencies import get_data_fetcher
        from modules.portfolio_store import portfolio_store

        # 2026-09-09: 真实组合金额优先 (portfolio_value 参数 > 真实总值 > 100 万模拟)
        portfolio_value = request.args.get('portfolio_value', type=float) or \
            portfolio_store.get_total_value() or 1000000
        confidence = request.args.get('confidence', 0.95, type=float)

        # 获取 K 线数据
        klines = get_data_fetcher().get_kline_data(stock_code, 'daily', 120)
        if not klines or len(klines) < 60:
            return jsonify({
                'success': False,
                'error': 'K 线数据不足 (需要至少 60 个交易日)',
                'stock_code': stock_code,
            }), 400

        # 模拟预测结果（从 app 模块获取真实预测）
        predictions = {
            'direction': 'up',
            'confidence': 0.65,
            'target_price': None,
        }
        try:
            from modules.routes.predictor_factory import predictor_factory
            regime = predictor_factory.get('regime')
            if regime:
                regime_status = regime.get_status()
                predictions['confidence'] = regime_status.get('confidence', 0.65)
        except Exception:
            pass

        result = cvar_position_manager.calculate_position(
            stock_code=stock_code,
            klines=klines,
            predictions=predictions,
            portfolio_value=portfolio_value,
            confidence=confidence,
        )

        # 2026-09-09: vs 真实持仓对照 (用户提供真实成本/数量, 非模拟仓位)
        try:
            hold = portfolio_store.get_holding(stock_code)
            if hold and isinstance(result, dict) and result.get('success'):
                price = float(klines[-1].get('close', 0) or 0)
                cost = float(hold.get('cost', 0) or 0)
                qty = int(hold.get('qty', 0) or 0)
                budget = result.get('position', {}).get('vol_adjusted', 0)
                weight = qty * price / portfolio_value if portfolio_value > 0 else 0
                pl_pct = (price - cost) / cost * 100 if cost > 0 else 0
                result['vs_position'] = {
                    'has_position': True,
                    'name': hold.get('name', ''),
                    'cost': cost,
                    'qty': qty,
                    'market_value': round(qty * price, 2),
                    'weight_actual_pct': round(weight * 100, 2),
                    'budget_pct': round(budget * 100, 2),
                    'over_budget': weight > budget,
                    'pl_pct': round(pl_pct, 2),
                    'stop_line': round(price * 0.98, 2),  # 距现价 -2% = 最大亏损容忍线
                    'hint': ('减仓 (超风险预算)' if weight > budget else
                             ('警戒 (浮亏已破容忍线)' if pl_pct <= -2 else
                              ('可加仓 (低于风险预算)' if predictions.get('direction') == 'up' and weight < budget else '持有'))),
                }
        except Exception as ve:
            logger.error(f"[CVaRPosition] vs持仓计算失败 {stock_code}: {ve}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[CVaRPosition] Error for {stock_code}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/position/kelly')
def kelly_calculation():
    """
    Kelly 公式计算

    查询参数:
        win_rate: 胜率 (0-1)
        avg_win: 平均盈利
        avg_loss: 平均亏损 (绝对值)
        kelly_fraction: Kelly 比例 (默认 0.25，半 Kelly)
    """
    try:
        from modules.cvar_position_manager import cvar_position_manager

        win_rate = request.args.get('win_rate', 0.55, type=float)
        avg_win = request.args.get('avg_win', 0.03, type=float)
        avg_loss = request.args.get('avg_loss', 0.02, type=float)

        if avg_loss == 0:
            return jsonify({'error': 'avg_loss 不能为 0'}), 400

        kelly = cvar_position_manager.calculate_kelly(win_rate, avg_win, avg_loss)

        return jsonify({
            'success': True,
            'kelly_fraction': cvar_position_manager.kelly_fraction,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            '盈亏比': round(avg_win / avg_loss, 4),
            'kelly_position': round(kelly, 4),
            'half_kelly': round(kelly / 2, 4),
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"[Kelly] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/position/black-litterman')
def black_litterman():
    """
    Black-Litterman 贝叶斯估计

    查询参数:
        market_caps: 市值权重，逗号分隔 (默认等权)
        views: 观点，格式 "0:0.05,1:-0.03"
        risk_aversion: 风险厌恶系数 (默认 2.5)
    """
    try:
        from modules.cvar_position_manager import cvar_position_manager

        market_caps_str = request.args.get('market_caps', '')
        views_str = request.args.get('views', '')
        risk_aversion = request.args.get('risk_aversion', 2.5, type=float)

        if market_caps_str:
            market_caps = [float(x) for x in market_caps_str.split(',')]
        else:
            market_caps = [1.0] * 5  # 默认 5 只股票等权

        views = {}
        if views_str:
            for pair in views_str.split(','):
                idx, ret = pair.split(':')
                views[int(idx)] = float(ret)

        # 均衡收益
        eq_returns = cvar_position_manager.black_litterman_equilibrium(
            market_caps, risk_aversion
        )

        # 后验收益
        posterior_returns = cvar_position_manager.black_litterman_returns(
            eq_returns, views
        )

        return jsonify({
            'success': True,
            'n_assets': len(market_caps),
            'equilibrium_returns': [round(float(x), 6) for x in eq_returns],
            'posterior_returns': [round(float(x), 6) for x in posterior_returns],
            'views': views,
            'risk_aversion': risk_aversion,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"[BlackLitterman] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/position/status')
def position_manager_status():
    """获取仓位管理器状态"""
    try:
        from modules.cvar_position_manager import cvar_position_manager
        return jsonify(cvar_position_manager.get_status())
    except Exception as e:
        logger.error(f"[PositionStatus] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
