#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
mlops_orchestrator.py — MLOps 管道编排器 (2026-08-12)

串联现有 MLOps 模块，形成完整的自动训练管道:

    数据 → 概念漂移检测 → 自动触发训练 → A/B 测试 → 模型注册 → 部署

模块依赖:
    - drift_aware_pipeline.py: 概念漂移检测
    - auto_retrain_trigger.py: 自动重训练触发
    - ab_test_manager.py: A/B 测试
    - model_registry.py: 模型注册表
    - async_training_pipeline.py: 异步训练

使用方式:
    from modules.models.mlops_orchestrator import mlops_orchestrator

    # 启动编排器
    mlops_orchestrator.start()

    # 检查是否需要重训练
    need_retrain = mlops_orchestrator.check_retrain_trigger('sz300620')

    # 获取管道状态
    status = mlops_orchestrator.get_status()
"""

import time
import threading
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum

from modules.logger import logger


class OrchestratorStatus(Enum):
    """编排器状态"""
    IDLE = "idle"
    MONITORING = "monitoring"
    TRAINING = "training"
    A_B_TESTING = "a_b_testing"
    DEPLOYING = "deploying"
    ERROR = "error"


class MLOpsOrchestrator:
    """
    MLOps 管道编排器

    核心功能:
    1. 概念漂移检测 → 触发重训练
    2. 新模型 A/B 测试 → 选择优胜者
    3. 模型注册 → 一键回滚
    4. 性能监控 → 自动告警
    """

    def __init__(self):
        self.status = OrchestratorStatus.IDLE
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None

        # 各模块实例 (延迟初始化)
        self._drift_detector = None
        self._retrain_trigger = None
        self._ab_manager = None
        self._model_registry = None
        self._training_pipeline = None

        # 配置
        self.drift_check_interval = 300  # 5 分钟
        self.retrain_threshold = 0.65    # 准确率低于此值触发重训练
        self.ab_test_duration = 86400    # 24 小时

        # 统计
        self._retrain_count = 0
        self._deployment_count = 0
        self._last_drift_time = None
        self._last_deployment_time = None

    def _ensure_modules(self):
        """延迟初始化各模块"""
        if self._drift_detector is None:
            try:
                from modules.drift_aware_pipeline import get_drift_aware_pipeline
                self._drift_detector = get_drift_aware_pipeline()
                logger.info("[MLOpsOrchestrator] 概念漂移检测器已加载")
            except Exception as e:
                logger.warning(f"[MLOpsOrchestrator] 概念漂移检测器加载失败: {e}")

        if self._retrain_trigger is None:
            try:
                from modules.auto_retrain_trigger import get_auto_retrain_trigger
                self._retrain_trigger = get_auto_retrain_trigger()
                logger.info("[MLOpsOrchestrator] 自动重训练触发器已加载")
            except Exception as e:
                logger.warning(f"[MLOpsOrchestrator] 自动重训练触发器加载失败: {e}")

        if self._ab_manager is None:
            try:
                from modules.ab_test_manager import get_ab_test_manager
                self._ab_manager = get_ab_test_manager()
                logger.info("[MLOpsOrchestrator] A/B 测试管理器已加载")
            except Exception as e:
                logger.warning(f"[MLOpsOrchestrator] A/B 测试管理器加载失败: {e}")

        if self._model_registry is None:
            try:
                from modules.model_registry import get_model_registry
                self._model_registry = get_model_registry()
                logger.info("[MLOpsOrchestrator] 模型注册表已加载")
            except Exception as e:
                logger.warning(f"[MLOpsOrchestrator] 模型注册表加载失败: {e}")

        if self._training_pipeline is None:
            try:
                from modules.async_training_pipeline import get_training_pipeline
                self._training_pipeline = get_training_pipeline()
                logger.info("[MLOpsOrchestrator] 异步训练管道已加载")
            except Exception as e:
                logger.warning(f"[MLOpsOrchestrator] 异步训练管道加载失败: {e}")

    def check_retrain_trigger(self, stock_code: str) -> Dict:
        """
        检查是否需要重训练

        触发条件:
        1. 概念漂移检测触发
        2. 模型准确率低于阈值
        3. 距离上次重训练超过 N 天
        """
        self._ensure_modules()

        reasons = []

        # 1. 概念漂移检测 (2026-09-08 断链修复: get_status() 返回 PipelineStatus
        # dataclass 而非 dict, 原 .get('drift_detected') 抛 AttributeError 被
        # except+debug 静默 — 此路径自写入起从未工作过)
        if self._drift_detector:
            try:
                status = self._drift_detector.get_status()
                if status.drift_count > 0:
                    reasons.append(f"概念漂移 (drift 事件 {status.drift_count} 次)")
                    self._last_drift_time = time.time()
            except Exception as e:
                logger.debug(f"[MLOpsOrchestrator] 漂移检测失败: {e}")

        # 2. 模型性能检查
        if self._model_registry:
            try:
                models = self._model_registry.list_models()
                for model_name, model_info in models.items():
                    accuracy = model_info.get('accuracy', 1.0)
                    if accuracy < self.retrain_threshold:
                        reasons.append(f"{model_name} 准确率过低 ({accuracy:.3f})")
            except Exception as e:
                logger.debug(f"[MLOpsOrchestrator] 模型检查失败: {e}")

        # 3. 时间检查
        if self._last_deployment_time:
            days_since = (time.time() - self._last_deployment_time) / 86400
            if days_since > 7:  # 超过 7 天
                reasons.append(f"超过 {days_since:.0f} 天未更新")

        return {
            'need_retrain': len(reasons) > 0,
            'reasons': reasons,
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
        }

    def trigger_retrain(self, stock_code: str, model_name: str = None) -> Dict:
        """
        触发重训练

        流程:
        1. 创建 A/B 测试实验
        2. 启动异步训练
        3. 训练完成后自动进入 A/B 测试
        """
        self._ensure_modules()
        self.status = OrchestratorStatus.TRAINING

        result = {
            'success': False,
            'stock_code': stock_code,
            'model_name': model_name or 'default',
            'steps': [],
        }

        try:
            # 模型类型归一 (2026-09-08: 原 'default' 非法 — 注册表/训练管线两处
            # 都按合法类型消费; 与 AsyncTrainingPipeline._execute_training 的白名单一致)
            model_type = result.get('model_name') or 'drl'
            if model_type not in ('drl', 'patchtst', 'moirai'):
                model_type = 'drl'

            # Step 1: 创建 A/B 实验
            if self._ab_manager:
                experiment = self._ab_manager.create_experiment(
                    name=f"retrain_{stock_code}_{int(time.time())}",
                    variants=[
                        {'name': 'current', 'model': 'current'},
                        {'name': 'new', 'model': 'pending'},
                    ],
                )
                # create_experiment 返回 str (实验 id), 非对象 — 原 .experiment_id
                # 访问抛 AttributeError 被外层 except 吞 (2026-09-08 断链修复)
                result['steps'].append(f'A/B 实验创建: {experiment}')
                result['experiment_id'] = experiment

            # Step 2: 注册待训新模型占位版本 (2026-09-08 断链修复: ModelRegistry
            # 没有 register_model/get_model/update_model_status 方法 — 原调用抛
            # AttributeError 被外层 except 吞。真实 API = register(name, ModelMetadata))
            if self._model_registry:
                from modules.model_registry import ModelMetadata, ModelType, ModelStatus
                version = '1.0.0'
                self._model_registry.register(
                    f"{stock_code}_retrain",
                    ModelMetadata(
                        name=f"{stock_code}_retrain",
                        model_type={'drl': ModelType.DRL,
                                    'patchtst': ModelType.TRANSFORMER,
                                    'moirai': ModelType.MOIRAI}[model_type],
                        version=version,
                        status=ModelStatus.TRAINING,
                        description=f"trigger_retrain 重训占位 ({stock_code})",
                    ),
                )
                result['steps'].append(f'模型版本注册: {stock_code}_retrain v{version}')
                result['version'] = version

            # Step 3: 启动异步训练 (2026-09-08 断链修复: AsyncTrainingPipeline
            # 真实 API = submit(model_type, stock_code, params) → 返回 task_id str;
            # 原 submit_training 不存在 + 返回 str 却按 task.task_id 访问)
            if self._training_pipeline:
                task_id = self._training_pipeline.submit(
                    model_type, stock_code,
                    params={'epochs': 50, 'days': 365, 'batch_size': 32},
                )
                result['steps'].append(f'训练任务提交: {task_id}')
                result['task_id'] = task_id
                result['success'] = True

            self._retrain_count += 1
            logger.info(f"[MLOpsOrchestrator] 重训练已触发: {stock_code}, steps={result['steps']}")

        except Exception as e:
            logger.error(f"[MLOpsOrchestrator] 重训练触发失败: {e}")
            result['error'] = str(e)

        return result

    def deploy_model(self, stock_code: str, version: str = None) -> Dict:
        """
        部署模型

        流程:
        1. 从注册表获取指定版本
        2. 替换当前模型
        3. 更新 A/B 测试结果
        """
        self._ensure_modules()
        self.status = OrchestratorStatus.DEPLOYING

        result = {'success': False, 'stock_code': stock_code}

        try:
            # 2026-09-08 断链修复: get_model/update_model_status 不存在 (抛
            # AttributeError 被 except 吞)。真实 API = get(name) + set_active(name)
            # (set_active 将目标置 ACTIVE 并把同类型其他模型降为 READY)
            if self._model_registry:
                name = f"{stock_code}_retrain"
                model = self._model_registry.get(name)
                if model and self._model_registry.set_active(name):
                    self._last_deployment_time = time.time()
                    self._deployment_count += 1
                    result['success'] = True
                    result['version'] = model.version
                    logger.info(f"[MLOpsOrchestrator] 模型已部署: {stock_code} v{model.version}")

        except Exception as e:
            logger.error(f"[MLOpsOrchestrator] 部署失败: {e}")
            result['error'] = str(e)

        self.status = OrchestratorStatus.MONITORING
        return result

    def get_status(self) -> Dict:
        """获取编排器状态"""
        self._ensure_modules()

        return {
            'status': self.status.value,
            'retrain_count': self._retrain_count,
            'deployment_count': self._deployment_count,
            'last_drift_time': datetime.fromtimestamp(self._last_drift_time).isoformat() if self._last_drift_time else None,
            'last_deployment_time': datetime.fromtimestamp(self._last_deployment_time).isoformat() if self._last_deployment_time else None,
            'modules': {
                'drift_detector': self._drift_detector is not None,
                'retrain_trigger': self._retrain_trigger is not None,
                'ab_test': self._ab_manager is not None,
                'model_registry': self._model_registry is not None,
                'training_pipeline': self._training_pipeline is not None,
            },
            'timestamp': datetime.now().isoformat(),
        }

    def start(self):
        """启动编排器 (后台监控线程)"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MLOpsOrchestrator",
        )
        self._monitor_thread.start()
        logger.info("[MLOpsOrchestrator] 编排器已启动")

    def stop(self):
        """停止编排器"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("[MLOpsOrchestrator] 编排器已停止")

    def _monitor_loop(self):
        """监控循环 — 自动 drift 检测 → 重训练触发

        2026-09-08 四层断链修复 (自动重训练链自写入起从未工作过,
        四层全被 except+logger.debug 静默):
        ① self.drift_monitor 从未被赋值 (真实属性是 _drift_detector) → AttributeError
        ② start() 不调 _ensure_modules() → 检测器在监控线程里永远不会加载
        ③ get_drift_report() 不是 DriftAwarePipeline 的方法 (那是 DriftAwareEnsemble
           的 API); DriftAwarePipeline.get_status() 返回 PipelineStatus dataclass
           (drift_count 是属性非键) → 原 .get('drift_count') 必抛 AttributeError
        ④ trigger_retrain() 缺必填位置参数 stock_code → TypeError
        """
        self._ensure_modules()
        while self._running:
            try:
                self.status = OrchestratorStatus.MONITORING

                # 检查漂移 (drift_count = 漂移事件计数, ≥1 即在发生漂移)
                if self._drift_detector is not None:
                    try:
                        status = self._drift_detector.get_status()
                        drift_count = status.drift_count

                        # 检测到漂移 → 对当前金丝雀部署的股票触发重训练
                        if drift_count >= 1:
                            codes = list(self.canary_deployments.keys())[:1]
                            if codes:
                                logger.warning(
                                    f"[MLOps] 检测到漂移 (count={drift_count}), "
                                    f"触发重训练: {codes[0]}"
                                )
                                self.trigger_retrain(codes[0])
                            else:
                                logger.debug(
                                    f"[MLOps] 检测到漂移 (count={drift_count}) 但无"
                                    f"金丝雀部署股票, 跳过重训练"
                                )
                    except Exception as e:
                        logger.debug(f"[MLOps] 漂移检查失败: {e}")

                # 检查金丝雀部署是否需要评估
                for stock_code in list(self.canary_deployments.keys()):
                    try:
                        self.canary_deployments[stock_code]['canary_n'] += 1
                        if (self.canary_deployments[stock_code]['canary_n']
                                % self.evaluation_threshold == 0):
                            self.canary_evaluate(stock_code)
                    except Exception as e:
                        logger.debug(f"[MLOps] 金丝雀评估失败 {stock_code}: {e}")

                time.sleep(self.drift_check_interval)

            except Exception as e:
                logger.error(f"[MLOpsOrchestrator] 监控循环错误: {e}")
                self.status = OrchestratorStatus.ERROR
                time.sleep(60)  # 出错后等待 1 分钟


# ── 金丝雀部署 ──────────────────────────────────────────────

class CanaryDeployment:
    """
    金丝雀部署 (Canary Deployment) + 自动回滚

    流程:
    1. 新模型以低流量比例 (5%) 部署 (canary)
    2. 监控 canary 模型 vs 稳定模型的性能指标
    3. 如果 canary 性能达标 → 逐步增加流量 (5% → 25% → 50% → 100%)
    4. 如果 canary 性能不达标 → 自动回滚到稳定模型

    参考:
    - Google SRE: Canary Deployments
    - Netflix: Spinnaker 金丝雀发布
    - 2026 量化最佳实践: 模型部署灰度 + 自动回滚
    """

    # 流量阶梯
    TRAFFIC_STEPS = [0.05, 0.25, 0.50, 1.0]

    # 性能阈值
    MIN_ACCURACY = 0.55       # 最低准确率
    MIN_IC = 0.03             # 最低 IC 值
    MAX_DD = 0.05             # 最大回撤

    def __init__(self):
        self._deployments: Dict[str, Dict] = {}  # stock_code → deployment info
        self._lock = threading.Lock()

    def deploy_canary(
        self,
        stock_code: str,
        new_version: str,
        old_version: str,
        initial_traffic: float = 0.05,
    ) -> Dict:
        """
        启动金丝雀部署

        Args:
            stock_code: 股票代码
            new_version: 新模型版本
            old_version: 当前稳定版本
            initial_traffic: 初始流量比例 (默认 5%)

        Returns:
            部署信息
        """
        with self._lock:
            if stock_code in self._deployments:
                existing = self._deployments[stock_code]
                if existing.get('status') in ('active', 'scaling'):
                    return {
                        'success': False,
                        'error': f'{stock_code} 已有活跃金丝雀部署 (version={existing.get("new_version")})',
                    }

            deployment = {
                'stock_code': stock_code,
                'new_version': new_version,
                'old_version': old_version,
                'traffic_step': 0,  # 索引，指向 TRAFFIC_STEPS[0] = 0.05
                'traffic_ratio': initial_traffic,
                'status': 'active',  # active → scaling → promoted / rolled_back
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'evaluations': [],  # 每次评估记录
                'canary_correct': 0,
                'canary_total': 0,
                'stable_correct': 0,
                'stable_total': 0,
            }

            self._deployments[stock_code] = deployment
            logger.info(
                f"[CanaryDeploy] 金丝雀部署启动: {stock_code} "
                f"{old_version} → {new_version} (流量={initial_traffic:.0%})"
            )

            return {
                'success': True,
                'deployment': deployment,
                'message': f'金丝雀部署已启动: {old_version} → {new_version}',
            }

    def record_prediction(
        self,
        stock_code: str,
        model_version: str,
        predicted: int,      # 0=down, 1=neutral, 2=up
        actual: int,         # 0=down, 1=neutral, 2=up
    ) -> Dict:
        """
        记录一次预测结果 (用于性能评估)

        Args:
            stock_code: 股票代码
            model_version: 模型版本 (canary 或 stable)
            predicted: 预测结果
            actual: 实际结果

        Returns:
            评估结果 (如果达到评估周期)
        """
        with self._lock:
            if stock_code not in self._deployments:
                return {'success': False, 'error': '无活跃金丝雀部署'}

            deployment = self._deployments[stock_code]
            if deployment['status'] not in ('active', 'scaling'):
                return {'success': False, 'error': '部署已结束'}

            # 确定记录到哪个模型
            if model_version == deployment['new_version']:
                deployment['canary_total'] += 1
                if predicted == actual:
                    deployment['canary_correct'] += 1
            elif model_version == deployment['old_version']:
                deployment['stable_total'] += 1
                if predicted == actual:
                    deployment['stable_correct'] += 1
            else:
                return {'success': False, 'error': f'未知模型版本: {model_version}'}

            # 检查是否达到评估周期 (每次评估需要至少 50 次预测)
            evaluation_threshold = 50
            canary_n = deployment['canary_total']

            if canary_n % evaluation_threshold == 0:
                return self._evaluate_deployment(stock_code)

            return {'success': True, 'message': f'已记录，还需 {evaluation_threshold - (canary_n % evaluation_threshold)} 次评估'}

    def _evaluate_deployment(self, stock_code: str) -> Dict:
        """
        评估金丝雀部署性能 — 多维度 (准确率 + IC + Sharpe + MaxDD)

        Returns:
            评估结果
        """
        deployment = self._deployments[stock_code]

        canary_acc = (
            deployment['canary_correct'] / deployment['canary_total']
            if deployment['canary_total'] > 0 else 0.0
        )
        stable_acc = (
            deployment['stable_correct'] / deployment['stable_total']
            if deployment['stable_total'] > 0 else 0.0
        )

        # 计算 IC (预测方向 vs 实际方向)
        canary_ic = self._compute_ic(deployment.get('canary_predictions', []))
        stable_ic = self._compute_ic(deployment.get('stable_predictions', []))

        # 计算 Sharpe 比率 (基于预测收益)
        canary_sharpe = self._compute_sharpe(deployment.get('canary_returns', []))
        stable_sharpe = self._compute_sharpe(deployment.get('stable_returns', []))

        # 计算 MaxDD (最大回撤)
        canary_maxdd = self._compute_maxdd(deployment.get('canary_returns', []))
        stable_maxdd = self._compute_maxdd(deployment.get('stable_returns', []))

        # 综合评分: 加权多维度
        canary_score = (
            0.3 * canary_acc +
            0.3 * abs(canary_ic) +
            0.2 * max(canary_sharpe, 0) +
            0.2 * max(1.0 - canary_maxdd, 0)
        )
        stable_score = (
            0.3 * stable_acc +
            0.3 * abs(stable_ic) +
            0.2 * max(stable_sharpe, 0) +
            0.2 * max(1.0 - stable_maxdd, 0)
        )

        evaluation = {
            'timestamp': datetime.now().isoformat(),
            # 准确率
            'canary_accuracy': round(canary_acc, 4),
            'stable_accuracy': round(stable_acc, 4),
            'accuracy_diff': round(canary_acc - stable_acc, 4),
            # IC
            'canary_ic': round(canary_ic, 4),
            'stable_ic': round(stable_ic, 4),
            'ic_diff': round(canary_ic - stable_ic, 4),
            # Sharpe
            'canary_sharpe': round(canary_sharpe, 4),
            'stable_sharpe': round(stable_sharpe, 4),
            'sharpe_diff': round(canary_sharpe - stable_sharpe, 4),
            # MaxDD
            'canary_maxdd': round(canary_maxdd, 4),
            'stable_maxdd': round(stable_maxdd, 4),
            'maxdd_diff': round(canary_maxdd - stable_maxdd, 4),
            # 综合
            'canary_score': round(canary_score, 4),
            'stable_score': round(stable_score, 4),
            'score_diff': round(canary_score - stable_score, 4),
            'canary_n': deployment['canary_total'],
            'stable_n': deployment['stable_total'],
        }
        deployment['evaluations'].append(evaluation)

        # 多维度决策
        if canary_acc < self.MIN_ACCURACY:
            self._rollback(stock_code, evaluation)
            return {**evaluation, 'decision': 'rolled_back',
                    'reason': f'准确率 {canary_acc:.3f} < 阈值 {self.MIN_ACCURACY}'}

        if canary_score < stable_score - 0.05:
            self._rollback(stock_code, evaluation)
            return {**evaluation, 'decision': 'rolled_back',
                    'reason': f'综合评分比稳定模型低 > 0.05 ({canary_score:.3f} vs {stable_score:.3f})'}

        if canary_score >= stable_score:
            self._scale_up(stock_code, evaluation)
            return {**evaluation, 'decision': 'scaled_up',
                    'reason': f'综合评分 >= 稳定模型 ({canary_score:.3f} vs {stable_score:.3f})'}

        return {**evaluation, 'decision': 'continue', 'reason': '继续观察'}

    def _compute_ic(self, predictions: List[Dict]) -> float:
        """计算 IC (预测方向与实际方向的相关系数)"""
        if len(predictions) < 5:
            return 0.0
        pred_dirs = [p.get('pred_dir', 0) for p in predictions]
        actual_dirs = [p.get('actual_dir', 0) for p in predictions]
        if len(set(pred_dirs)) < 2 or len(set(actual_dirs)) < 2:
            return 0.0
        try:
            return float(np.corrcoef(pred_dirs, actual_dirs)[0, 1])
        except Exception:
            return 0.0

    def _compute_sharpe(self, returns: List[float]) -> float:
        """计算 Sharpe 比率 (假设无风险利率=0)"""
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std)

    def _compute_maxdd(self, returns: List[float]) -> float:
        """计算最大回撤"""
        if len(returns) < 2:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / (running_max + 1e-8)
        return float(np.max(drawdowns))

    def _scale_up(self, stock_code: str, evaluation: Dict) -> None:
        """提升流量"""
        deployment = self._deployments[stock_code]
        current_step = deployment['traffic_step']

        if current_step + 1 >= len(self.TRAFFIC_STEPS):
            # 已达到 100% → 提升为正式版本
            self._promote(stock_code, evaluation)
        else:
            deployment['traffic_step'] += 1
            deployment['traffic_ratio'] = self.TRAFFIC_STEPS[current_step + 1]
            deployment['status'] = 'scaling'
            deployment['updated_at'] = datetime.now().isoformat()

            logger.info(
                f"[CanaryDeploy] 流量提升: {stock_code} → "
                f"{deployment['traffic_ratio']:.0%} (step {deployment['traffic_step']}/{len(self.TRAFFIC_STEPS)-1})"
            )

    def _promote(self, stock_code: str, evaluation: Dict) -> None:
        """提升为正式版本"""
        deployment = self._deployments[stock_code]
        deployment['status'] = 'promoted'
        deployment['traffic_ratio'] = 1.0
        deployment['updated_at'] = datetime.now().isoformat()

        logger.info(
            f"[CanaryDeploy] 金丝雀已提升为正式版本: {stock_code} "
            f"{deployment['old_version']} → {deployment['new_version']}"
        )

    def _rollback(self, stock_code: str, evaluation: Dict) -> None:
        """回滚到稳定版本"""
        deployment = self._deployments[stock_code]
        deployment['status'] = 'rolled_back'
        deployment['traffic_ratio'] = 0.0
        deployment['updated_at'] = datetime.now().isoformat()

        logger.warning(
            f"[CanaryDeploy] 自动回滚: {stock_code} "
            f"{deployment['new_version']} → {deployment['old_version']}"
        )

    def get_status(self, stock_code: str = None) -> Dict:
        """获取金丝雀部署状态"""
        with self._lock:
            if stock_code:
                deployment = self._deployments.get(stock_code, {})
                return {
                    'success': True,
                    'deployment': dict(deployment),
                }
            else:
                return {
                    'success': True,
                    'deployments': {k: dict(v) for k, v in self._deployments.items()},
                    'active_count': sum(
                        1 for d in self._deployments.values()
                        if d['status'] in ('active', 'scaling')
                    ),
                }

    def end_deployment(self, stock_code: str, reason: str = 'manual') -> Dict:
        """手动结束金丝雀部署"""
        with self._lock:
            if stock_code not in self._deployments:
                return {'success': False, 'error': '无活跃部署'}

            deployment = self._deployments[stock_code]
            if deployment['status'] not in ('active', 'scaling'):
                return {'success': False, 'error': '部署已结束'}

            deployment['status'] = 'ended'
            deployment['end_reason'] = reason
            deployment['updated_at'] = datetime.now().isoformat()

            logger.info(f"[CanaryDeploy] 手动结束部署: {stock_code} (reason={reason})")
            return {'success': True, 'deployment': dict(deployment)}


# ── 全局单例 ─────────────────────────────────────────────────

mlops_orchestrator = MLOpsOrchestrator()
canary_deploy = CanaryDeployment()


def get_canary_deploy() -> CanaryDeployment:
    """获取全局 CanaryDeployment 实例"""
    return canary_deploy
