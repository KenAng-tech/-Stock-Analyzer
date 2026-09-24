"""
health_routes.py — /api/health 增强版

返回:
- 整体健康状态
- 模块加载状态 (所有 SOTA 模型)
- GPU/MPS 可用性
- Python/关键依赖版本
- 内存状态
- 数据新鲜度 (K 线最新日期)
- API 响应时间 (P50/P95)
- 凭据状态 (HITHINK_FINANCE_API_KEY)
- 日志健康状态
"""

import sys
import os
import time
import platform
import threading
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify

from modules.logger import logger

bp = Blueprint('health', __name__)

# ── API 响应时间追踪 ──────────────────────────────────────────

_response_times: list[float] = []
_response_times_lock = threading.Lock()


def record_response_time(duration_ms: float) -> None:
    """记录 API 响应时间 (环形缓冲区, 最多 1000 条)"""
    global _response_times
    with _response_times_lock:
        _response_times.append(duration_ms)
        if len(_response_times) > 1000:
            _response_times = _response_times[-1000:]


def get_response_time_stats() -> dict:
    """计算响应时间统计 (P50/P95/P99)"""
    with _response_times_lock:
        if not _response_times:
            return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "avg": 0}
        sorted_times = sorted(_response_times)
        n = len(sorted_times)
        return {
            "count": n,
            "p50": sorted_times[int(n * 0.50)] if n >= 2 else sorted_times[0],
            "p95": sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
            "p99": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
            "avg": sum(sorted_times) / n,
            "min": sorted_times[0],
            "max": sorted_times[-1],
        }


# ── 数据新鲜度检查 ──────────────────────────────────────────

_latest_kline_dates: dict[str, str] = {}
_kline_lock = threading.Lock()


def record_kline_date(stock_code: str, date: str) -> None:
    """记录某只股票的最新 K 线日期"""
    with _kline_lock:
        _latest_kline_dates[stock_code] = date


def get_data_freshness() -> dict:
    """获取数据新鲜度"""
    with _kline_lock:
        if not _latest_kline_dates:
            return {"status": "unknown", "note": "尚无数据更新记录"}

        now = datetime.now()
        issues = []
        latest_date = None

        for code, date_str in _latest_kline_dates.items():
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                days_ago = (now - date_obj).days
                if days_ago > 2:
                    issues.append(f"{code}: {days_ago} 天未更新")
                if latest_date is None or date_str > latest_date:
                    latest_date = date_str
            except (ValueError, TypeError):
                pass

        if issues:
            return {
                "status": "stale",
                "latest_date": latest_date,
                "issues": issues[:5],
                "note": "部分股票数据超过 2 天未更新",
            }
        return {
            "status": "fresh",
            "latest_date": latest_date,
            "stocks_tracked": len(_latest_kline_dates),
        }


# ── 凭据状态检查 ──────────────────────────────────────────

def get_credentials_status() -> dict:
    """检查 API 凭据状态"""
    env_key = os.environ.get("HITHINK_FINANCE_API_KEY", "")
    omlx_key = os.environ.get("OMLX_API_KEY", "")

    result = {
        "hithink_finance": {
            "configured": bool(env_key),
            "length": len(env_key),
        },
        "omlx": {
            "configured": bool(omlx_key),
            "length": len(omlx_key),
        },
    }

    # 检查凭证文件
    cred_file = Path.home() / ".appdata" / "stock_analyzer" / "credentials.env"
    if cred_file.exists():
        result["credential_file"] = {
            "exists": True,
            "path": str(cred_file),
        }
    else:
        result["credential_file"] = {"exists": False}

    return result


# ── 日志健康检查 ──────────────────────────────────────────

def get_log_health() -> dict:
    """检查日志健康状态"""
    logs_dir = Path("/Users/claw/stock_analyzer/logs")
    if not logs_dir.exists():
        return {"exists": False}

    files = list(logs_dir.glob("*.log*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    main_log = logs_dir / "stock_analyzer.log"
    main_size = main_log.stat().st_size if main_log.exists() else 0

    return {
        "exists": True,
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "main_log_size_mb": round(main_size / (1024 * 1024), 2),
        "main_log": str(main_log),
    }


# ── 主健康检查端点 ──────────────────────────────────────────

@bp.route('/api/health')
def health_check():
    """增强健康检查"""
    start_time = time.time()

    result = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'server': {
            'python': platform.python_version(),
            'platform': f"{platform.system()} {platform.release()}",
        },
        'modules': {},
        'gpu': {},
        'dependencies': {},
        'api_performance': get_response_time_stats(),
        'data_freshness': get_data_freshness(),
        'credentials': get_credentials_status(),
        'log_health': get_log_health(),
        'llm': {},
    }

    # ── LLM 链健康 (09-20: 双轨 health 之自检轨 — 熔断态 + token 台账) ──
    try:
        from modules.llm_router import llm_router
        _st = llm_router.get_status()
        result['llm'] = {
            'active_provider': _st.get('active_provider'),
            'budget': _st.get('budget'),
            'circuit_breakers': {k: v.get('state')
                                 for k, v in (_st.get('circuit_breakers') or {}).items()},
            'recent_failures': len(_st.get('recent_failures') or []),
        }
    except Exception as e:
        logger.warning(f"[Health] LLM 状态采集失败 (非断链): {e}")
        result['llm'] = {'error': str(e)[:120]}

    # ── GPU/MPS 检测 ──────────────────────────────────────
    try:
        import torch
        result['gpu']['torch_version'] = torch.__version__
        result['gpu']['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result['gpu']['cuda_device'] = torch.cuda.get_device_name(0)
            result['gpu']['cuda_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        result['gpu']['mps_available'] = hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        result['gpu']['torch'] = 'not installed'
    except Exception as e:
        result['gpu']['error'] = str(e)

    # ── 关键依赖版本 ──────────────────────────────────────
    deps = ['flask', 'flask_socketio', 'flask_cors', 'numpy', 'pandas', 'scipy',
            'lightgbm', 'xgboost', 'sklearn', 'akshare', 'requests', 'socketio']
    for dep in deps:
        try:
            mod = __import__(dep.replace('-', '_'))
            version = getattr(mod, '__version__', 'unknown')
            result['dependencies'][dep] = version
        except ImportError:
            result['dependencies'][dep] = 'NOT INSTALLED'
        except Exception:
            result['dependencies'][dep] = 'version error'

    # ── 模块加载状态 ──────────────────────────────────────
    try:
        app_module = sys.modules.get('app')
        if app_module is None:
            main_mod = sys.modules.get('__main__')
            if main_mod is not None and hasattr(main_mod, 'patchtst_integrator'):
                app_module = main_mod
            else:
                import app as app_mod
                app_module = app_mod

        module_attrs = [
            'patchtst_integrator', 'diffusion_predictor', 'mamba_hft',
            'multi_agent_coordinator', 'gnn_predictor', 'conformal_predictor',
            'timesfm_predictor', 'timesnet_trainer', 'alpha158_calculator',
            'self_supervised_pretrainer', 'rl_trader_v2',
            'drift_monitor', 'sentiment_engine', 'factor_weight_scheduler',
            'cvar_analyzer', 'advanced_drift_detector', 'factor_ic_monitor',
            'memory_manager', 'patchmamba_model', 'causal_discovery_engine',
        ]

        for attr in module_attrs:
            if hasattr(app_module, attr):
                obj = getattr(app_module, attr)
                if obj is None:
                    result['modules'][attr] = 'failed'
                elif hasattr(obj, 'trained') or hasattr(obj, '_trained'):
                    result['modules'][attr] = 'loaded'
                elif hasattr(obj, 'get_status'):
                    result['modules'][attr] = 'loaded'
                else:
                    result['modules'][attr] = 'loaded'
            else:
                result['modules'][attr] = 'not imported'

        _vals = list(result['modules'].values())
        result['modules_summary'] = {
            'total': len(_vals),
            'loaded': _vals.count('loaded'),
            'failed': _vals.count('failed'),
            'not_imported': _vals.count('not imported'),
        }

        # 内存状态
        if hasattr(app_module, 'memory_manager') and app_module.memory_manager:
            try:
                result['memory'] = app_module.memory_manager.get_status()
            except Exception:
                result['memory'] = {'error': 'get_status failed'}
        else:
            result['memory'] = {'status': 'not available'}

    except Exception as e:
        result['modules']['error'] = str(e)

    # ── 标记不健康状态 ────────────────────────────────────
    failed_modules = [k for k, v in result['modules'].items() if v == 'failed']
    if failed_modules:
        result['status'] = 'degraded'
        result['status_message'] = f'{len(failed_modules)} modules failed: {", ".join(failed_modules)}'

    # 数据新鲜度检查
    freshness = result.get('data_freshness', {})
    if freshness.get('status') == 'stale':
        result['status'] = 'degraded'
        result['data_freshness_note'] = freshness.get('note', '')

    # 日志健康检查
    log_health = result.get('log_health', {})
    if log_health.get('total_size_mb', 0) > 200:
        result['log_health']['warning'] = f"总日志大小 {log_health['total_size_mb']}MB > 阈值 200MB"

    elapsed = time.time() - start_time
    result['health_check_duration_ms'] = round(elapsed * 1000, 2)

    return jsonify(result)


@bp.route('/api/health/response-times')
def health_response_times():
    """API 响应时间统计"""
    return jsonify({
        'ok': True,
        'data': get_response_time_stats(),
    })


@bp.route('/api/health/data-freshness')
def health_data_freshness():
    """数据新鲜度"""
    return jsonify({
        'ok': True,
        'data': get_data_freshness(),
    })
