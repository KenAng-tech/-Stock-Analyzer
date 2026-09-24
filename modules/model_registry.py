#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
模型元数据管理 — 注册、追踪、查询所有 ML 模型

管理内容:
- 模型基本信息 (名称、类型、版本)
- 训练参数 (超参数、数据集)
- 性能指标 (准确率、夏普比率、最大回撤等)
- 文件路径和大小
- 训练时间和最后更新时间
- 模型状态 (active/deprecated/archived)

用法:
    registry = ModelRegistry()
    registry.register('drl_agent', {
        'type': 'drl',
        'version': '1.0.0',
        'path': 'modules/dl_models/drl_agent.npy',
        'metrics': {'accuracy': 0.55, 'sharpe': 0.3},
    })
    models = registry.list_models()
"""

import os
import json
import time
import hashlib
import shutil
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from modules.logger import logger


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = 'training'
    READY = 'ready'
    ACTIVE = 'active'
    DEPRECATED = 'deprecated'
    ARCHIVED = 'archived'


class ModelType(Enum):
    """模型类型"""
    DRL = 'drl'
    TRANSFORMER = 'transformer'
    MAMBA = 'mamba'
    DIFFUSION = 'diffusion'
    SELF_SUPERVISED = 'self_supervised'
    GNN = 'gnn'
    TIMESFM = 'timesfm'
    MOIRAI = 'moirai'
    CONFORMAL = 'conformal'
    TIME_LLM = 'time_llm'
    REGIME_SWITCHING = 'regime_switching'
    MULTI_FACTOR = 'multi_factor'
    ALPHA158 = 'alpha158'
    CROSS_MARKET = 'cross_market'
    MULTI_AGENT = 'multi_agent'
    UNIFIED_DECISION = 'unified_decision'


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    model_type: ModelType
    version: str = '1.0.0'
    status: ModelStatus = ModelStatus.READY
    path: str = ''
    file_size_bytes: int = 0
    file_hash: str = ''

    # 训练信息
    trained_at: str = ''
    trained_on: str = ''           # 训练数据集
    epochs: int = 0
    learning_rate: float = 0.0
    batch_size: int = 0

    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)

    # 超参数
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    # 描述
    description: str = ''
    tags: List[str] = field(default_factory=list)

    # 跟踪
    created_at: str = ''
    updated_at: str = ''

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['model_type'] = self.model_type.value
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        mt = data.get('model_type', 'drl')
        if isinstance(mt, str):
            try:
                mt = ModelType(mt)
            except ValueError:
                mt = ModelType.DRL
        st = data.get('status', 'ready')
        if isinstance(st, str):
            try:
                st = ModelStatus(st)
            except ValueError:
                st = ModelStatus.READY
        return cls(
            name=data['name'],
            model_type=mt,
            version=data.get('version', '1.0.0'),
            status=st,
            path=data.get('path', ''),
            file_size_bytes=data.get('file_size_bytes', 0),
            file_hash=data.get('file_hash', ''),
            trained_at=data.get('trained_at', ''),
            trained_on=data.get('trained_on', ''),
            epochs=data.get('epochs', 0),
            learning_rate=data.get('learning_rate', 0.0),
            batch_size=data.get('batch_size', 0),
            metrics=data.get('metrics', {}),
            hyperparams=data.get('hyperparams', {}),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
        )


class ModelRegistry:
    """模型元数据注册表"""

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = os.path.join(
                os.path.dirname(__file__), 'dl_models', 'model_registry.json'
            )
        self.registry_path = registry_path
        self._models: Dict[str, ModelMetadata] = {}
        self._load_registry()
        logger.info(f"[ModelRegistry] 已加载 {len(self._models)} 个模型")

    def _load_registry(self):
        """加载注册表"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                for name, meta in data.items():
                    self._models[name] = ModelMetadata.from_dict(meta)
            except Exception as e:
                logger.warning(f"[ModelRegistry] 加载失败: {e}")

    def _save_registry(self):
        """保存注册表"""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(
                    {name: meta.to_dict() for name, meta in self._models.items()},
                    f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"[ModelRegistry] 保存失败: {e}")

    def _compute_file_hash(self, path: str) -> str:
        """计算文件 MD5 哈希"""
        if not os.path.exists(path):
            return ''
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def register(self, name: str, metadata: ModelMetadata) -> bool:
        """注册模型"""
        # 自动更新文件信息
        if metadata.path and os.path.exists(metadata.path):
            metadata.file_size_bytes = os.path.getsize(metadata.path)
            metadata.file_hash = self._compute_file_hash(metadata.path)

        now = datetime.now().isoformat()
        if not metadata.created_at:
            metadata.created_at = now
        metadata.updated_at = now

        self._models[name] = metadata
        self._save_registry()
        logger.info(f"[ModelRegistry] 注册模型: {name} ({metadata.model_type.value})")
        return True

    def unregister(self, name: str) -> bool:
        """注销模型"""
        if name in self._models:
            del self._models[name]
            self._save_registry()
            logger.info(f"[ModelRegistry] 注销模型: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self._models.get(name)

    def list_models(self,
                    model_type: Optional[ModelType] = None,
                    status: Optional[ModelStatus] = None,
                    active_only: bool = True) -> List[ModelMetadata]:
        """列出模型"""
        models = list(self._models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]
        if status:
            models = [m for m in models if m.status == status]
        if active_only:
            models = [m for m in models if m.status in (ModelStatus.READY, ModelStatus.ACTIVE)]

        return sorted(models, key=lambda m: m.updated_at, reverse=True)

    def update_metrics(self, name: str, metrics: Dict[str, float]) -> bool:
        """更新模型指标"""
        if name in self._models:
            self._models[name].metrics.update(metrics)
            self._models[name].updated_at = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"[ModelRegistry] 更新指标: {name} → {metrics}")
            return True
        return False

    def set_active(self, name: str) -> bool:
        """设置为活跃模型"""
        if name in self._models:
            self._models[name].status = ModelStatus.ACTIVE
            # 其他同类型模型设为非活跃
            for m in self._models.values():
                if m.model_type == self._models[name].model_type and m.name != name:
                    m.status = ModelStatus.READY
            self._save_registry()
            logger.info(f"[ModelRegistry] 设置活跃模型: {name}")
            return True
        return False

    def scan_directory(self, dl_models_dir: Optional[str] = None) -> List[str]:
        """扫描 dl_models 目录，自动注册发现的模型文件"""
        if dl_models_dir is None:
            dl_models_dir = os.path.join(os.path.dirname(__file__), 'dl_models')

        if not os.path.exists(dl_models_dir):
            return []

        type_map = {
            'drl_agent': ModelType.DRL,
            'patchtst': ModelType.TRANSFORMER,
            'patchtst_best': ModelType.TRANSFORMER,
            'mamba_hft': ModelType.MAMBA,
            'diffusion_model': ModelType.DIFFUSION,
            'self_supervised': ModelType.SELF_SUPERVISED,
            'timesfm': ModelType.TIMESFM,
            'moirai': ModelType.MOIRAI,
        }

        registered = []
        for filename in os.listdir(dl_models_dir):
            if not filename.endswith(('.pth', '.npy', '.pt', '.bin', '.h5')):
                continue

            # 尝试匹配类型
            model_type = None
            for prefix, mt in type_map.items():
                if prefix in filename:
                    model_type = mt
                    break

            if model_type is None:
                model_type = ModelType.TRANSFORMER  # 默认

            name = filename.rsplit('.', 1)[0]  # 去掉扩展名
            path = os.path.join(dl_models_dir, filename)

            if name not in self._models:
                self.register(name, ModelMetadata(
                    name=name,
                    model_type=model_type,
                    path=path,
                    description=f'自动扫描注册: {filename}',
                    tags=['auto-registered'],
                ))
                registered.append(name)

        if registered:
            logger.info(f"[ModelRegistry] 自动注册 {len(registered)} 个模型: {registered}")

        return registered

    def get_summary(self) -> Dict:
        """获取注册表摘要"""
        models = self.list_models(active_only=False)
        by_type = {}
        by_status = {}

        for m in models:
            by_type[m.model_type.value] = by_type.get(m.model_type.value, 0) + 1
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1

        total_size = sum(m.file_size_bytes for m in models)

        return {
            'total_models': len(models),
            'by_type': by_type,
            'by_status': by_status,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'models': [m.to_dict() for m in models],
        }


# 全局单例
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """获取模型注册表全局实例"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        # 自动扫描
        _registry.scan_directory()
    return _registry


def reset_model_registry():
    """重置全局实例"""
    global _registry
    _registry = None


class ModelFactory:
    """
    模型工厂 — 自动发现 + 条件加载 + 依赖检查

    功能:
    1. 自动扫描 modules/ 目录，发现所有 *predictor*.py 和 *model*.py
    2. 检查依赖是否可用 (torch, transformers, etc.)
    3. 条件加载模型
    4. 提供统一接口获取模型实例

    用法:
        factory = ModelFactory()
        factory.scan()
        models = factory.get_available_models()
        instance = factory.create('patchtst')
    """

    # 模型文件到依赖的映射
    MODEL_DEPENDENCIES = {
        'patchtst': ['torch'],
        'mamba': ['torch', 'mamba_ssm'],
        'diffusion': ['torch'],
        'drl': ['torch', 'gym'],
        'moirai': ['torch', 'huggingface_hub'],
        'gnn': ['torch', 'torch_geometric'],
        'conformal': ['torch'],
        'timesfm': ['torch', 'transformers'],
        'chronos': ['torch', 'transformers'],
        'patchmamba': ['torch', 'mamba_ssm'],
    }

    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._dependencies: Dict[str, List[str]] = dict(self.MODEL_DEPENDENCIES)
        self._scan_dir = os.path.join(os.path.dirname(__file__))

    def scan(self) -> List[str]:
        """
        自动扫描 modules/ 目录，发现所有模型文件

        匹配规则:
        - patchtst_integrator.py → patchtst
        - hft_mamba.py → mamba
        - diffusion_model.py → diffusion
        - drl_agent.py → drl
        - moirai_predictor.py → moirai
        - gnn_predictor.py → gnn
        - conformal_predictor.py → conformal
        - timesfm_predictor.py → timesfm
        - chronos_predictor.py → chronos
        - patchmamba.py → patchmamba
        - self_supervised.py → self_supervised
        - foundation_model.py → foundation

        Returns:
            发现的模型名列表
        """
        # 文件名前缀 → 模型名 映射
        FILE_TO_MODEL = {
            'patchtst_integrator': 'patchtst',
            'hft_mamba': 'mamba',
            'diffusion_model': 'diffusion',
            'drl_agent': 'drl',
            'moirai_predictor': 'moirai',
            'gnn_predictor': 'gnn',
            'conformal_predictor': 'conformal',
            'timesfm_predictor': 'timesfm',
            'chronos_predictor': 'chronos',
            'patchmamba': 'patchmamba',
            'self_supervised': 'self_supervised',
            'foundation_model': 'foundation',
        }

        discovered = []
        try:
            for fname in os.listdir(self._scan_dir):
                if not fname.endswith('.py') or fname.startswith('_'):
                    continue
                name = fname[:-3]  # 去掉 .py
                if name in FILE_TO_MODEL:
                    model_name = FILE_TO_MODEL[name]
                    discovered.append(model_name)
                    self._registry[model_name] = {'file': fname, 'status': 'pending'}
        except Exception as e:
            logger.error(f"[ModelFactory] 扫描失败: {e}")

        logger.info(f"[ModelFactory] 发现 {len(discovered)} 个模型: {discovered}")
        return discovered

    def check_dependencies(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        检查模型依赖是否可用

        Returns:
            (all_available, missing_deps)
        """
        deps = self._dependencies.get(model_name, [])
        missing = []
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        available = len(missing) == 0
        if not available:
            logger.warning(f"[ModelFactory] {model_name} 缺少依赖: {missing}")

        return available, missing

    def get_available_models(self) -> Dict[str, Dict]:
        """
        获取所有可用模型及其状态

        Returns:
            {model_name: {'status': 'available'|'missing_deps'|'not_found',
                          'dependencies': [...],
                          'missing': [...]}}
        """
        result = {}
        for name in self._registry:
            available, missing = self.check_dependencies(name)
            status = 'available' if available else 'missing_deps'
            result[name] = {
                'status': status,
                'dependencies': self._dependencies.get(name, []),
                'missing': missing,
            }

        available_count = sum(1 for v in result.values() if v['status'] == 'available')
        logger.info(f"[ModelFactory] 可用模型: {available_count}/{len(result)}")
        return result

    def create(self, model_name: str, **kwargs) -> Any:
        """
        创建模型实例

        Args:
            model_name: 模型名
            **kwargs: 传递给模型构造函数的参数

        Returns:
            模型实例
        """
        available, missing = self.check_dependencies(model_name)
        if not available:
            logger.error(f"[ModelFactory] {model_name} 依赖不可用: {missing}")
            return None

        # 根据模型名动态导入
        # 2026-09-03: 8 处死路径修复 — 模型文件 08-23 已迁入 modules/models/,
        # 本表路径未同步 → get_model() 动态 import 恒 ModuleNotFoundError →
        # 静默返回 None (MLOps/在线学习/调度器三链丢模型)。逐条核实:
        # timesfm/patchmamba 确在 modules/ 根, 保持原样。
        import_map = {
            'patchtst': ('modules.models.patchtst_integrator', 'get_patchtst'),
            'mamba': ('modules.models.hft_mamba', 'get_mamba_hft_predictor'),
            'diffusion': ('modules.models.diffusion_model', 'get_diffusion_predictor'),
            'drl': ('modules.models.drl_agent', 'get_drl_agent'),
            'moirai': ('modules.models.moirai_predictor', 'get_moirai_predictor'),
            'gnn': ('modules.models.gnn_predictor', 'get_gnn_predictor'),
            'conformal': ('modules.models.conformal_predictor', 'get_conformal_predictor'),
            'timesfm': ('modules.timesfm_predictor', 'get_timesfm'),
            'chronos': ('modules.models.chronos_predictor', 'ChronosPredictor'),
            'patchmamba': ('modules.patchmamba', 'get_patchmamba'),
        }

        if model_name not in import_map:
            logger.error(f"[ModelFactory] 未知模型: {model_name}")
            return None

        module_name, class_name = import_map[model_name]
        try:
            module = __import__(module_name, fromlist=[class_name])
            factory_func = getattr(module, class_name)
            if callable(factory_func):
                instance = factory_func(**kwargs)
            else:
                instance = factory_func
            logger.info(f"[ModelFactory] 模型 {model_name} 创建成功")
            return instance
        except Exception as e:
            logger.error(f"[ModelFactory] 模型 {model_name} 创建失败: {e}")
            return None


# 全局模型工厂
model_factory = ModelFactory()
model_factory.scan()


# ══════════════════════════════════════════════════════════════════════
# 模型版本流水线 (qlib online tag 思想, P1-6, 2026-09-14)
#
# qlib 的 RecordTempProvider 用 online 标签标记"当前生效版本", 新版本注册后
# 原子切换 online, 旧版本保留可回滚。本段把该纪律落到本项目:
# - 注册表独立于上方 ModelMetadata 注册表, 落盘 data/model_versions.json;
# - 所有写操作在 文件锁 + 线程锁 下做 读-改-写, tmp+os.replace 原子落盘;
# - set_online 原子保证同模型至多一个 online 版本 (回滚 = 对旧 version_id 再
#   调一次 set_online)。
# ══════════════════════════════════════════════════════════════════════

_STATUS_ONLINE = 'online'
_STATUS_OFFLINE = 'offline'


class _FileLock:
    """跨进程文件锁 (O_CREAT|O_EXCL), 同进程线程也互斥; 超时 fail-open 只告警。

    设计取舍: 版本注册表是观察/回滚辅助设施, 锁不可用时宁可裸写
    (tmp+replace 保证单写不撕裂) 也不阻塞训练主链 — fail-open + logger.error。
    """

    def __init__(self, lock_path: str, timeout: float = 10.0, stale_seconds: float = 30.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self.stale_seconds = stale_seconds
        self._acquired = False

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._acquired = True
                return self
            except FileExistsError:
                try:  # 陈旧锁清理 (持有者崩溃残留)
                    if time.time() - os.path.getmtime(self.lock_path) > self.stale_seconds:
                        logger.warning(f"[VersionRegistry] 清理陈旧锁: {self.lock_path}")
                        os.remove(self.lock_path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    logger.error(f"[VersionRegistry] 文件锁超时 {self.timeout}s, fail-open 裸写: {self.lock_path}")
                    self._acquired = False
                    return self
                time.sleep(0.05)
            except OSError as e:
                logger.error(f"[VersionRegistry] 文件锁异常, fail-open 裸写: {e}")
                self._acquired = False
                return self

    def __exit__(self, *exc):
        if self._acquired:
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
        return False


class ModelVersionRegistry:
    """模型版本注册表 — 每次训练产出版本, online 标签原子切换 (qlib 思想)"""

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'model_versions.json'
            )
        self.registry_path = registry_path
        self._lock_path = registry_path + '.lock'
        self._tlock = threading.RLock()  # 同进程线程锁 (文件锁之外的快速路径)

    # ── 内部 ──

    def _read(self) -> Dict[str, Any]:
        """读注册表 (原子写保证不读到半截文件); 缺失/损坏返回空表"""
        if not os.path.exists(self.registry_path):
            return {'versions': [], 'updated_at': ''}
        try:
            with open(self.registry_path, encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get('versions'), list):
                logger.warning("[VersionRegistry] 注册表结构异常, 重建空表")
                return {'versions': [], 'updated_at': ''}
            return data
        except Exception as e:
            logger.error(f"[VersionRegistry] 注册表读取失败, 重建空表: {e}")
            return {'versions': [], 'updated_at': ''}

    def _write(self, data: Dict[str, Any]) -> None:
        """原子写 (tmp + os.replace, 同目录保证同文件系统)"""
        data['updated_at'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        tmp = f"{self.registry_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.registry_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ── 公开 API ──

    def register_version(self, model_name: str, path: str, metrics: Dict[str, float],
                         train_range: Optional[List[str]] = None,
                         extra: Optional[Dict] = None) -> str:
        """注册一个训练产出版本 (初始 offline), 返回 version_id。

        Args:
            model_name: 模型名 (如 'patchtst')
            path: 模型文件路径
            metrics: 训练指标 (val_acc 等)
            train_range: 训练数据日期范围 [start, end]
            extra: 附加信息 (stock_code/耗时等)

        Returns:
            version_id: '{model_name}@{YYYYmmdd_HHMMSS}' (同秒冲突加 _k 后缀)
        """
        with self._tlock, _FileLock(self._lock_path):
            data = self._read()
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_id = f"{model_name}@{ts}"
            suffix = 1
            existing_ids = {v.get('version_id') for v in data['versions']}
            while version_id in existing_ids:
                version_id = f"{model_name}@{ts}_{suffix}"
                suffix += 1
            record = {
                'version_id': version_id,
                'model_name': model_name,
                'path': str(path),
                'metrics': metrics or {},
                'train_range': train_range or [],
                'extra': extra or {},
                'status': _STATUS_OFFLINE,
                'trained_at': datetime.now().isoformat(),
            }
            data['versions'].append(record)
            self._write(data)
        logger.info(f"[VersionRegistry] 注册版本: {version_id} (metrics={metrics})")
        return version_id

    def set_online(self, model_name: str, version_id: str) -> bool:
        """原子把 version_id 设为 online, 同模型其他版本转 offline (保留可回滚)"""
        with self._tlock, _FileLock(self._lock_path):
            data = self._read()
            target = None
            for v in data['versions']:
                if v.get('model_name') != model_name:
                    continue
                if v.get('version_id') == version_id:
                    target = v
            if target is None:
                logger.error(f"[VersionRegistry] set_online 失败, 版本不存在: {version_id}")
                return False
            for v in data['versions']:
                if v.get('model_name') == model_name:
                    v['status'] = _STATUS_ONLINE if v is target else _STATUS_OFFLINE
            self._write(data)
        logger.info(f"[VersionRegistry] online 切换: {model_name} → {version_id}")
        return True

    def get_online(self, model_name: str) -> Optional[Dict]:
        """获取当前 online 版本记录 (无则 None)"""
        with self._tlock:
            data = self._read()
        for v in reversed(data['versions']):
            if v.get('model_name') == model_name and v.get('status') == _STATUS_ONLINE:
                return v
        return None

    def list_versions(self, model_name: str) -> List[Dict]:
        """列出模型全部版本 (新→旧)"""
        with self._tlock:
            data = self._read()
        versions = [v for v in data['versions'] if v.get('model_name') == model_name]
        return sorted(versions, key=lambda v: v.get('trained_at', ''), reverse=True)

    def days_since_last_train(self, model_name: str) -> Optional[float]:
        """距最近一次注册版本的训练天数; 无记录返回 None"""
        versions = self.list_versions(model_name)
        if not versions:
            return None
        latest = max(v.get('trained_at', '') for v in versions)
        try:
            last_dt = datetime.fromisoformat(latest)
        except ValueError:
            logger.error(f"[VersionRegistry] trained_at 解析失败: {latest}")
            return None
        return (datetime.now() - last_dt).total_seconds() / 86400.0


# 全局版本注册表单例
_version_registry: Optional[ModelVersionRegistry] = None


def get_version_registry() -> ModelVersionRegistry:
    """获取模型版本注册表全局实例"""
    global _version_registry
    if _version_registry is None:
        _version_registry = ModelVersionRegistry()
    return _version_registry


def reset_version_registry():
    """重置版本注册表全局实例 (测试用)"""
    global _version_registry
    _version_registry = None
