#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DEPRECATED — 已弃用，请使用 concept_drift_detector.py

此文件仅为向后兼容保留。
所有新的代码应使用: from modules.concept_drift_detector import ConceptDriftDetector

保留时间: 2026-07-01 之后 3 个月
"""

import warnings

warnings.warn(
    "[DEPRECATED] modules.concept_drift 已弃用，请使用 modules.concept_drift_detector",
    DeprecationWarning,
    stacklevel=2,
)

from modules.concept_drift_detector import ConceptDriftDetector

# ── 向后兼容 ─────────────────────────────────────────────
# dashboard_api.py 使用 health_monitor.health_report()


class _HealthMonitor:
    """轻量级健康监控 (兼容 dashboard_api.py)"""

    def health_report(self):
        return {
            'status': 'ok',
            'drift_detector': 'ConceptDriftDetector',
            'description': '基于 ADWIN 的概念漂移检测器',
        }


health_monitor = _HealthMonitor()

__all__ = ['ConceptDriftDetector', 'health_monitor']
