#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""训练工件目录 — qlib trainer 工件纪律 (P0-3, 2026-09-14)

背景 (qlib 参考): qlib workflow 的 trainer 每次训练固定落 conf + params +
metrics 工件, 下游组件生成前用 check() 校验上游工件存在 — 断点续跑与复盘
链都基于磁盘工件而非内存态。本模块把该纪律落到本项目: 训练/调参链每次运行
建 runs/{name}/{YYYYmmdd_HHMMSS}/ 目录, 生命周期写 conf.json / metrics.json /
pred.pkl / model/ / status.json, check() 列出最近一次工件齐全的完整 run。

设计纪律:
- 轻量: 仅 stdlib (os/json/shutil/hashlib/pickle), 不 import torch/akshare
  等重模块, 可被任何进程 (02:00 链 / worker / Flask) 安全调用;
- save_pred/save_model 不反序列化大对象进内存: 给定路径时流式哈希+拷贝,
  给定 DataFrame/ndarray 时 pickle 直接落盘, sha256 分块读文件计算;
- sha256 + shape 记入 run 目录 meta.json (metrics.json 旁边), 供下游校验
  工件未被篡改/截断, 无需加载大对象;
- save_*/finish 返回 bool 不抛异常 (记账类操作不破坏调用链);
  start_run 失败抛 OSError (调用方在链尾 guarded 使用)。

目录结构:
    runs/{name}/{YYYYmmdd_HHMMSS}[_k]/
        conf.json      — start_run 时冻结的运行配置
        metrics.json   — save_metrics 落盘
        pred.pkl       — save_pred (pickle), 元信息在 meta.json
        model/         — save_model 拷贝的模型文件
        meta.json      — pred/model 的 sha256+shape+size 元信息
        status.json    — finish 落盘 (ok|fail, error, 起止时间)

API:
    start_run(name, conf) -> RunContext
    RunContext.save_metrics(dict) / save_pred(df_or_path) /
               save_model(path_or_str) / finish(status='ok', error=None)
    check(name, need=('conf','metrics')) -> dict | None   # 最近完整 run (qlib 思想)
    list_runs(name, limit=10) -> list[dict]
"""

import hashlib
import json
import os
import pickle
import re
import shutil
from datetime import datetime

try:
    from modules.logger import logger
except Exception:  # pragma: no cover - 项目外独立运行回退
    import logging
    logger = logging.getLogger('run_artifact')

__all__ = ['start_run', 'RunContext', 'check', 'list_runs', 'RUNS_ROOT']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(PROJECT_ROOT, 'runs')

# check() 的 need 名 → 工件存在性判定 (qlib check() 的上游工件校验思想)
_ARTIFACT_FILES = {'conf': 'conf.json', 'metrics': 'metrics.json',
                   'pred': 'pred.pkl'}


def _runs_root() -> str:
    """运行期读取模块全局 RUNS_ROOT (测试可 monkeypatch 到临时目录)"""
    return RUNS_ROOT


def _now_iso() -> str:
    """统一秒级 ISO 时间戳"""
    return datetime.now().isoformat(timespec='seconds')


def _atomic_write_json(path: str, payload) -> None:
    """原子写 JSON (tmp+replace, 防读写撕裂 — 同 hyperparam_optimizer 快照模式)"""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def _sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    """分块流式 sha256 — 大模型/大预测文件不进内存"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


class RunContext:
    """一次运行的工件上下文 (start_run 返回)

    工件写方法 (save_metrics/save_pred/save_model/finish) 均返回 bool 且
    内部吞异常 + logger.error — 工件记账失败绝不炸掉训练/调参主链。
    """

    def __init__(self, name: str, run_id: str, path: str, started_at: str):
        self.name = name
        self.run_id = run_id
        self.path = path
        self.started_at = started_at

    # ── 内部 ──

    def _update_meta(self, key: str, value) -> None:
        """读-改-写 meta.json (pred/model 元信息, 与 metrics.json 并列)"""
        meta_path = os.path.join(self.path, 'meta.json')
        meta = {}
        try:
            if os.path.exists(meta_path):
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.load(f)
        except Exception as e:
            logger.error(f"[run_artifact] meta.json 读取失败 (重建): {e}")
        meta[key] = value
        _atomic_write_json(meta_path, meta)

    # ── 工件写入 ──

    def save_metrics(self, metrics: dict) -> bool:
        """落盘 metrics.json (best params / trial 数 / 诊断键等, JSON 可序列化)"""
        try:
            _atomic_write_json(os.path.join(self.path, 'metrics.json'), metrics)
            return True
        except Exception as e:
            logger.error(f"[run_artifact] save_metrics 失败 ({self.run_id}): {e}")
            return False

    def save_pred(self, pred) -> bool:
        """落盘预测工件 pred.pkl + sha256/shape 记入 meta.json。

        Args:
            pred: DataFrame/ndarray (pickle 直接落盘, shape 取属性不复制数据)
                  或 文件路径 str/PathLike (流式拷贝, 不反序列化进内存)

        Returns:
            bool: 成功与否 (失败已 logger.error, 不抛)
        """
        try:
            dst = os.path.join(self.path, 'pred.pkl')
            shape = None
            if isinstance(pred, (str, os.PathLike)):
                if not os.path.isfile(pred):
                    raise FileNotFoundError(f'pred 路径不存在: {pred}')
                shutil.copyfile(pred, dst)   # 流式拷贝, 不进内存
            else:
                shape = list(getattr(pred, 'shape', [])) or None
                with open(dst, 'wb') as f:    # pickle 直接写文件, 不在内存拼 bytes
                    pickle.dump(pred, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._update_meta('pred', {
                'file': 'pred.pkl',
                'sha256': _sha256_file(dst),
                'shape': shape,
                'size_bytes': os.path.getsize(dst),
                'saved_at': _now_iso(),
            })
            return True
        except Exception as e:
            logger.error(f"[run_artifact] save_pred 失败 ({self.run_id}): {e}")
            return False

    def save_model(self, path_or_str) -> bool:
        """拷贝模型文件/目录到 run 目录 model/ 子目录, 元信息记入 meta.json。

        Args:
            path_or_str: 模型文件或目录路径 (str/PathLike)

        Returns:
            bool: 成功与否 (失败已 logger.error, 不抛)
        """
        try:
            src = str(path_or_str)
            if not os.path.exists(src):
                raise FileNotFoundError(f'model 路径不存在: {src}')
            model_dir = os.path.join(self.path, 'model')
            os.makedirs(model_dir, exist_ok=True)
            base = os.path.basename(os.path.normpath(src))
            dst = os.path.join(model_dir, base)
            is_dir = os.path.isdir(src)
            if is_dir:
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copyfile(src, dst)
            self._update_meta('model', {
                'file': os.path.join('model', base),
                'is_dir': is_dir,
                'sha256': None if is_dir else _sha256_file(dst),
                'size_bytes': sum(os.path.getsize(os.path.join(dp, fn))
                                  for dp, _, fns in os.walk(dst) for fn in fns),
                'saved_at': _now_iso(),
            })
            return True
        except Exception as e:
            logger.error(f"[run_artifact] save_model 失败 ({self.run_id}): {e}")
            return False

    def finish(self, status: str = 'ok', error: str = None) -> bool:
        """落盘 status.json (ok|fail) 并把状态并入 meta.json — 运行闭环标记。

        Args:
            status: 'ok' 或 'fail' (其他值原样记录但不推荐)
            error: fail 时的错误描述
        """
        try:
            payload = {'run_id': self.run_id, 'name': self.name,
                       'status': status, 'error': error,
                       'started_at': self.started_at, 'finished_at': _now_iso()}
            _atomic_write_json(os.path.join(self.path, 'status.json'), payload)
            self._update_meta('status', {'status': status, 'error': error,
                                         'finished_at': payload['finished_at']})
            return True
        except Exception as e:
            logger.error(f"[run_artifact] finish 失败 ({self.run_id}): {e}")
            return False


def start_run(name: str, conf: dict) -> RunContext:
    """开一次运行: 建 runs/{name}/{YYYYmmdd_HHMMSS}/ + 冻结 conf.json。

    与 save_*/finish 不同, 本函数失败会抛 OSError (目录/conf 写不进磁盘时
    调用方应知道 — 链尾使用请包 try/except)。

    Args:
        name: 运行类别名 (如 'hyperparam'); 非 [A-Za-z0-9_-] 字符替换为 _
        conf: 运行配置 dict (数据窗口、trial 预算、ledger_run_id 等)

    Returns:
        RunContext: 工件上下文
    """
    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', str(name)) or 'run'
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = os.path.join(_runs_root(), safe_name)
    os.makedirs(run_root, exist_ok=True)
    run_id, run_dir = run_ts, os.path.join(run_root, run_ts)
    suffix = 1
    while os.path.exists(run_dir):        # 同秒多次运行 → _1/_2/... 后缀
        run_id = f'{run_ts}_{suffix}'
        run_dir = os.path.join(run_root, run_id)
        suffix += 1
    os.makedirs(run_dir)
    started_at = _now_iso()
    _atomic_write_json(os.path.join(run_dir, 'conf.json'),
                       {'name': safe_name, 'run_id': run_id,
                        'started_at': started_at, 'conf': conf or {}})
    logger.info(f"[run_artifact] run 开始: {safe_name}/{run_id}")
    return RunContext(safe_name, run_id, run_dir, started_at)


def _artifact_ok(run_dir: str, need: str):
    """单工件存在性判定: 返回 (ok, path_or_None)"""
    if need in _ARTIFACT_FILES:
        p = os.path.join(run_dir, _ARTIFACT_FILES[need])
        return (os.path.isfile(p), p if os.path.isfile(p) else None)
    if need == 'model':
        d = os.path.join(run_dir, 'model')
        ok = os.path.isdir(d) and bool(os.listdir(d))
        return (ok, d if ok else None)
    logger.error(f"[run_artifact] check 未知工件名: {need} (视为缺失)")
    return (False, None)


def check(name: str, need=('conf', 'metrics')):
    """列出最近一次工件齐全的完整 run (qlib check() 思想, 断点续跑/复盘校验)。

    Args:
        name: 运行类别名
        need: 必备工件名序列, 支持 'conf'/'metrics'/'pred'/'model'

    Returns:
        dict | None: {name, run_id, path, artifacts: {need: path}};
                     无完整 run / 任何异常 → None (防御式)
    """
    try:
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', str(name)) or 'run'
        run_root = os.path.join(_runs_root(), safe_name)
        if not os.path.isdir(run_root):
            return None
        for run_id in sorted(os.listdir(run_root), reverse=True):  # 新→旧 (时间序)
            run_dir = os.path.join(run_root, run_id)
            if not os.path.isdir(run_dir):
                continue
            artifacts, ok = {}, True
            for need_name in need:
                found, path = _artifact_ok(run_dir, need_name)
                if not found:
                    ok = False
                    break
                artifacts[need_name] = path
            if ok:
                return {'name': safe_name, 'run_id': run_id,
                        'path': run_dir, 'artifacts': artifacts}
        return None
    except Exception as e:
        logger.error(f"[run_artifact] check 失败 (返回 None): {e}")
        return None


def list_runs(name: str, limit: int = 10) -> list:
    """列出最近 runs (新→旧) 及状态摘要, 供面板/复盘链浏览。

    Args:
        name: 运行类别名
        limit: 最大条数

    Returns:
        list[dict]: [{run_id, path, status, finished_at, error}];
                    status 缺失 (未 finish/崩溃) → 'running'; 失败 → []
    """
    try:
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', str(name)) or 'run'
        run_root = os.path.join(_runs_root(), safe_name)
        if not os.path.isdir(run_root):
            return []
        out = []
        for run_id in sorted(os.listdir(run_root), reverse=True)[:max(0, int(limit))]:
            run_dir = os.path.join(run_root, run_id)
            if not os.path.isdir(run_dir):
                continue
            entry = {'run_id': run_id, 'path': run_dir, 'status': 'running',
                     'finished_at': None, 'error': None}
            status_path = os.path.join(run_dir, 'status.json')
            if os.path.isfile(status_path):
                try:
                    with open(status_path, encoding='utf-8') as f:
                        st = json.load(f)
                    entry.update({'status': st.get('status', 'running'),
                                  'finished_at': st.get('finished_at'),
                                  'error': st.get('error')})
                except Exception as e:
                    logger.error(f"[run_artifact] status.json 损坏 ({run_id}): {e}")
            out.append(entry)
        return out
    except Exception as e:
        logger.error(f"[run_artifact] list_runs 失败 (返回 []): {e}")
        return []
