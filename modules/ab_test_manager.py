#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
A/B 测试框架 — 比较不同 SOTA 模型表现，自动切换最佳模型到生产流量

功能:
- 创建 A/B 测试实验 (多个模型并行比较)
- 流量分配 (随机将请求分发到不同模型)
- 指标收集 (准确率、收益、延迟等)
- 统计显著性检验
- 自动最优模型选择
- 实验报告生成

实验设计:
- 实验组: 新模型 (待测试)
- 对照组: 当前生产模型
- 流量分配: 50/50 或可配置比例

用法:
    ab_tester = ABTestManager()
    ab_tester.create_experiment('test_diffusion_vs_drl', [
        {'name': 'control', 'model': 'drl', 'weight': 0.5},
        {'name': 'treatment', 'model': 'diffusion', 'weight': 0.5},
    ])
    winner = ab_tester.get_winner('test_diffusion_vs_drl')
"""

import os
import json
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

from modules.logger import logger


class ExperimentStatus(Enum):
    """实验状态"""
    ACTIVE = 'active'
    COMPLETED = 'completed'
    TERMINATED = 'terminated'
    FAILED = 'failed'


@dataclass
class Variant:
    """实验变体 (一个模型)"""
    name: str
    model_name: str
    weight: float = 0.5  # 流量权重
    is_control: bool = False
    description: str = ''


@dataclass
class ExperimentResult:
    """实验结果"""
    variant_name: str
    n_samples: int = 0
    correct_predictions: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    win_rate: float = 0.0  # 方向准确率

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.n_samples if self.n_samples > 0 else 0.0


@dataclass
class ABEExperiment:
    """A/B 测试实验"""
    experiment_id: str
    name: str
    variants: List[Variant]
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    created_at: str = ''
    started_at: str = ''
    completed_at: str = ''

    # 结果存储
    results: Dict[str, ExperimentResult] = field(default_factory=dict)

    # 配置
    min_sample_size: int = 100  # 最小样本数
    confidence_level: float = 0.95  # 显著性水平
    traffic_split: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['variants'] = [asdict(v) for v in self.variants]
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'ABEExperiment':
        variants = [
            Variant(**v) if isinstance(v, dict) else v
            for v in data.get('variants', [])
        ]
        return cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            variants=variants,
            status=ExperimentStatus(data.get('status', 'active')),
            created_at=data.get('created_at', ''),
            started_at=data.get('started_at', ''),
            completed_at=data.get('completed_at', ''),
            traffic_split=data.get('traffic_split', {}),
        )


class ABTestManager:
    """A/B 测试管理器"""

    def __init__(self, experiments_path: Optional[str] = None):
        if experiments_path is None:
            experiments_path = os.path.join(
                os.path.dirname(__file__), 'dl_models', 'ab_experiments.json'
            )
        self.experiments_path = experiments_path
        self._experiments: Dict[str, ABEExperiment] = {}
        self._load_experiments()

        # 流量路由缓存 (stock_code → variant_name)
        self._traffic_cache: Dict[str, str] = {}

        logger.info(f"[ABTest] 管理器初始化: {len(self._experiments)} 个实验")

    def _load_experiments(self):
        """加载实验数据"""
        if os.path.exists(self.experiments_path):
            try:
                with open(self.experiments_path) as f:
                    data = json.load(f)
                for exp_id, exp_data in data.items():
                    self._experiments[exp_id] = ABEExperiment.from_dict(exp_data)
            except Exception as e:
                logger.warning(f"[ABTest] 加载失败: {e}")

    def _save_experiments(self):
        """保存实验数据"""
        try:
            os.makedirs(os.path.dirname(self.experiments_path), exist_ok=True)
            with open(self.experiments_path, 'w') as f:
                json.dump(
                    {eid: exp.to_dict() for eid, exp in self._experiments.items()},
                    f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"[ABTest] 保存失败: {e}")

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        min_samples: int = 100,
        confidence_level: float = 0.95,
    ) -> str:
        """
        创建 A/B 测试实验

        Args:
            name: 实验名称
            variants: [{'name': 'control', 'model_name': 'drl', 'weight': 0.5, 'is_control': True}, ...]
            min_samples: 最小样本数
            confidence_level: 显著性水平

        Returns:
            experiment_id
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        variant_objects = []
        total_weight = 0
        for v in variants:
            variant = Variant(
                name=v.get('name', ''),
                model_name=v.get('model_name', ''),
                weight=v.get('weight', 1.0 / len(variants)),
                is_control=v.get('is_control', False),
                description=v.get('description', ''),
            )
            variant_objects.append(variant)
            total_weight += variant.weight

        # 归一化权重
        if total_weight > 0:
            for v in variant_objects:
                v.weight /= total_weight

        experiment = ABEExperiment(
            experiment_id=experiment_id,
            name=name,
            variants=variant_objects,
            status=ExperimentStatus.ACTIVE,
            created_at=datetime.now().isoformat(),
            min_sample_size=min_samples,
            confidence_level=confidence_level,
        )

        self._experiments[experiment_id] = experiment
        self._save_experiments()

        logger.info(
            f"[ABTest] 创建实验: {experiment_id} ({name}) "
            f"variants={[v.name for v in variant_objects]}"
        )
        return experiment_id

    def get_variant(self, experiment_id: str, stock_code: str = '') -> Optional[Variant]:
        """
        为请求选择变体 (权重随机)

        Args:
            experiment_id: 实验 ID
            stock_code: 股票代码 (用于缓存一致性)

        Returns:
            选中的变体
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.ACTIVE:
            return None

        # 缓存: 同一股票在实验期间使用同一变体
        cache_key = f"{experiment_id}:{stock_code}"
        if cache_key in self._traffic_cache:
            name = self._traffic_cache[cache_key]
            return next((v for v in experiment.variants if v.name == name), None)

        # 权重随机选择
        r = random.random()
        cumulative = 0
        for v in experiment.variants:
            cumulative += v.weight
            if r <= cumulative:
                self._traffic_cache[cache_key] = v.name
                return v

        return experiment.variants[-1]

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        is_correct: bool,
        return_pct: float = 0.0,
        confidence: float = 0.5,
        latency_ms: float = 0.0,
    ):
        """
        记录实验结果

        Args:
            experiment_id: 实验 ID
            variant_name: 变体名称
            is_correct: 预测是否正确
            return_pct: 收益率
            confidence: 预测置信度
            latency_ms: 推理延迟
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return

        if variant_name not in experiment.results:
            experiment.results[variant_name] = ExperimentResult(
                variant_name=variant_name,
            )

        result = experiment.results[variant_name]
        result.n_samples += 1
        if is_correct:
            result.correct_predictions += 1
        result.total_return += return_pct
        result.avg_confidence = (
            (result.avg_confidence * (result.n_samples - 1) + confidence) / result.n_samples
        )
        result.avg_latency_ms = (
            (result.avg_latency_ms * (result.n_samples - 1) + latency_ms) / result.n_samples
        )

    def get_winner(self, experiment_id: str) -> Optional[str]:
        """
        获取实验优胜者 (基于统计显著性检验)

        使用两比例 Z 检验判断是否有显著差异

        Returns:
            优胜者变体名称，或 None (无显著差异)
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        results = experiment.results
        if len(results) < 2:
            return None

        # 检查是否达到最小样本数
        for r in results.values():
            if r.n_samples < experiment.min_sample_size:
                return None  # 样本不足

        # 按准确率排序
        sorted_variants = sorted(
            results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True,
        )

        best = sorted_variants[0]
        second = sorted_variants[1]

        # Z 检验
        z_stat = self._z_test(
            best[1].correct_predictions, best[1].n_samples,
            second[1].correct_predictions, second[1].n_samples,
        )

        # 临界值 (95% 置信度)
        critical_value = 1.96 if experiment.confidence_level == 0.95 else 2.58

        if abs(z_stat) > critical_value:
            winner = best[0]
            logger.info(
                f"[ABTest] 实验 {experiment_id} 优胜者: {winner} "
                f"(z={z_stat:.3f}, acc={best[1].accuracy:.3f} vs {second[1].accuracy:.3f})"
            )
            return winner

        logger.info(
            f"[ABTest] 实验 {experiment_id} 无显著差异 "
            f"(z={abs(z_stat):.3f} < {critical_value})"
        )
        return None

    def _z_test(self, x1: int, n1: int, x2: int, n2: int) -> float:
        """两比例 Z 检验"""
        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0

        p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if (n1 + n2) > 0 else 0

        if se == 0:
            return 0

        return (p1 - p2) / se

    def complete_experiment(self, experiment_id: str):
        """完成实验"""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now().isoformat()
            self._save_experiments()

    def terminate_experiment(self, experiment_id: str, reason: str = ''):
        """终止实验"""
        experiment = self._experiments.get(experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.TERMINATED
            experiment.completed_at = datetime.now().isoformat()
            self._save_experiments()
            logger.info(f"[ABTest] 终止实验: {experiment_id} ({reason})")

    def get_report(self, experiment_id: str) -> Dict:
        """获取实验报告"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {'success': False, 'error': 'Experiment not found'}

        winner = self.get_winner(experiment_id)

        variants_report = []
        for name, result in experiment.results.items():
            variants_report.append({
                'name': name,
                'n_samples': result.n_samples,
                'accuracy': round(result.accuracy, 4),
                'win_rate': round(result.win_rate, 4),
                'total_return': round(result.total_return, 4),
                'sharpe_ratio': round(result.sharpe_ratio, 4),
                'max_drawdown': round(result.max_drawdown, 4),
                'avg_confidence': round(result.avg_confidence, 4),
                'avg_latency_ms': round(result.avg_latency_ms, 2),
            })

        return {
            'success': True,
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'winner': winner,
            'variants': variants_report,
            'created_at': experiment.created_at,
            'completed_at': experiment.completed_at,
        }

    def list_experiments(self) -> List[Dict]:
        """列出所有实验"""
        return [
            {
                'experiment_id': eid,
                'name': exp.name,
                'status': exp.status.value,
                'n_variants': len(exp.variants),
                'n_results': sum(len(exp.results) for exp in [exp_data]),
                'created_at': exp_data.created_at,
            }
            for eid, exp_data in self._experiments.items()
        ]

    def get_status(self) -> Dict:
        """获取管理器状态"""
        active = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.ACTIVE)
        return {
            'total_experiments': len(self._experiments),
            'active_experiments': active,
            'experiments': self.list_experiments(),
        }


# 全局单例
_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    """获取 A/B 测试管理器全局实例"""
    global _manager
    if _manager is None:
        _manager = ABTestManager()
    return _manager


def reset_ab_test_manager():
    """重置全局实例"""
    global _manager
    _manager = None
