#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
因子挖掘三级反思记忆 (2026-09-14 P1-8)

2026 研究 (AgonAlpha/XAlpha): LLM 因子挖掘两大风险是「背诵假新颖」与
「同质化拥挤」, 2026 标配 = fresh-context 对抗审查员 + 三级反思记忆。
本模块实现三级记忆, 供 rd_agent_miner 在挖掘链中读写, 并把简短线索注入
下一轮生成 prompt, 让挖掘器"记得"试过什么/哪些被否/教训是什么:

- generation (每个候选): {name, expr, family, ic, verdict, reviewer_notes, ts}
    verdict: pass / reject / rejected_reviewer
- cycle (每轮摘要):      {themes, n_candidates, pass_rate, top_families, ts}
- archetype (家族线索):  {family: {family, tried_n, pass_n, pass_rate, lessons[]}}

存储: data/miner_memory.json (单文件 JSON)
- 原子写: tmp 文件 + os.replace, 崩溃不留半截文件
- 损坏自愈: 解析失败先备份为 <path>.corrupt.<ts> 再重建空记忆
- 线程安全: 实例级 threading.Lock (挖掘链单线程, 防御性加锁)

API:
    get_miner_memory() → MinerMemory 单例
    record_generation(name, expr, family, ic, verdict, reviewer_notes)
    summarize_cycle(themes, n_candidates, pass_rate, top_families, lessons)
    cues_for(family=None) → 供 prompt 注入的简短线索文本 (无记忆时返回 '')
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.logger import logger

# 各级记忆上限 (防 JSON 无限膨胀; 环形截断保留最新)
MAX_GENERATIONS = 500
MAX_CYCLES = 200
MAX_LESSONS_PER_FAMILY = 5

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'miner_memory.json',
)


class MinerMemory:
    """三级反思记忆: generation 条目 / cycle 摘要 / archetype 家族线索"""

    def __init__(self, path: str = None):
        """
        Args:
            path: JSON 文件路径, 默认 data/miner_memory.json (测试可注入 tmp 路径)
        """
        self.path = path or _DEFAULT_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = self._load()

    # ── 内部: 加载 / 原子写 ─────────────────────────────────────

    @staticmethod
    def _empty() -> Dict[str, Any]:
        """空记忆骨架"""
        return {'generations': [], 'cycles': [], 'archetypes': {}}

    def _load(self) -> Dict[str, Any]:
        """
        加载记忆 JSON。

        损坏自愈: 文件存在但解析失败 → 先 os.replace 备份为
        <path>.corrupt.<时间戳>, 再返回空骨架 (绝不因坏文件阻断挖掘链)。
        """
        if not os.path.exists(self.path):
            return self._empty()
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f'顶层不是 dict: {type(data)}')
            # 结构校验: 缺键/错型 → 就地补默认, 不整体丢弃 (宽容加载)
            if not isinstance(data.get('generations'), list):
                data['generations'] = []
            if not isinstance(data.get('cycles'), list):
                data['cycles'] = []
            if not isinstance(data.get('archetypes'), dict):
                data['archetypes'] = {}
            return data
        except Exception as e:
            logger.error(f'[MinerMemory] 记忆文件损坏, 尝试自愈重建: {e}')
            try:
                bak = f'{self.path}.corrupt.{datetime.now().strftime("%Y%m%d%H%M%S")}'
                os.replace(self.path, bak)
                logger.warning(f'[MinerMemory] 损坏文件已备份: {bak}, 已重建空记忆')
            except Exception as e2:
                logger.error(f'[MinerMemory] 损坏文件备份失败 (直接重建): {e2}')
            return self._empty()

    def _save(self) -> None:
        """原子写 (调用方持锁): tmp + os.replace, 失败清理 tmp 并上抛由调用方兜底"""
        d = os.path.dirname(self.path) or '.'
        os.makedirs(d, exist_ok=True)
        tmp = f'{self.path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise

    # ── 三级写入 API ─────────────────────────────────────────────

    def record_generation(self, name: str, expr: str, family: str,
                          ic: float, verdict: str, reviewer_notes: str = '') -> None:
        """
        记录一个挖掘候选 (generation 级), 并同步累计 archetype 家族计数。

        Args:
            name: 因子名
            expr: 公式/表达式 (自然语言 formula 或 code 摘要)
            family: 因子家族 (category)
            ic: 样本内 IC
            verdict: pass / reject / rejected_reviewer
            reviewer_notes: 对抗审查员否决理由 (可空)
        """
        try:
            with self._lock:
                self._data['generations'].append({
                    'name': name,
                    'expr': (expr or '')[:300],
                    'family': family,
                    'ic': round(float(ic), 4),
                    'verdict': verdict,
                    'reviewer_notes': (reviewer_notes or '')[:300],
                    'ts': datetime.now().isoformat(timespec='seconds'),
                })
                if len(self._data['generations']) > MAX_GENERATIONS:
                    self._data['generations'] = self._data['generations'][-MAX_GENERATIONS:]

                arch = self._data['archetypes'].setdefault(family, {
                    'family': family, 'tried_n': 0, 'pass_n': 0,
                    'pass_rate': 0.0, 'lessons': [],
                })
                arch['tried_n'] = int(arch.get('tried_n', 0)) + 1
                if verdict == 'pass':
                    arch['pass_n'] = int(arch.get('pass_n', 0)) + 1
                arch['pass_rate'] = round(arch['pass_n'] / max(arch['tried_n'], 1), 3)
                self._save()
        except Exception as e:
            logger.error(f'[MinerMemory] record_generation 失败: {e}')

    def summarize_cycle(self, themes: List[str], n_candidates: int,
                        pass_rate: float, top_families: List[str],
                        lessons: Dict[str, str] = None) -> None:
        """
        记录一轮挖掘摘要 (cycle 级), 可选附带家族教训 (写入 archetype)。

        Args:
            themes: 本轮挖掘主题 (类别列表)
            n_candidates: 本轮候选数
            pass_rate: 通过率 (validated / 有效候选)
            top_families: 通过最多的家族 Top-N
            lessons: {family: 教训文本} — 追加到对应 archetype.lessons (环形截断)
        """
        try:
            with self._lock:
                self._data['cycles'].append({
                    'themes': list(themes or []),
                    'n_candidates': int(n_candidates),
                    'pass_rate': round(float(pass_rate), 3),
                    'top_families': list(top_families or []),
                    'ts': datetime.now().isoformat(timespec='seconds'),
                })
                if len(self._data['cycles']) > MAX_CYCLES:
                    self._data['cycles'] = self._data['cycles'][-MAX_CYCLES:]
                for fam, lesson in (lessons or {}).items():
                    if not lesson:
                        continue
                    arch = self._data['archetypes'].setdefault(fam, {
                        'family': fam, 'tried_n': 0, 'pass_n': 0,
                        'pass_rate': 0.0, 'lessons': [],
                    })
                    lessons_list = arch.setdefault('lessons', [])
                    lessons_list.append(str(lesson)[:200])
                    if len(lessons_list) > MAX_LESSONS_PER_FAMILY:
                        arch['lessons'] = lessons_list[-MAX_LESSONS_PER_FAMILY:]
                self._save()
        except Exception as e:
            logger.error(f'[MinerMemory] summarize_cycle 失败: {e}')

    # ── 读取 API (prompt 注入) ───────────────────────────────────

    def cues_for(self, family: str = None) -> str:
        """
        生成供下一轮挖掘 prompt 注入的简短反思线索文本。

        Args:
            family: 指定家族 → 只给该家族线索; None → 全局线索 (活跃家族 + 上轮摘要)

        Returns:
            多行简短文本; 无任何记忆时返回 '' (调用方据此跳过注入)
        """
        try:
            with self._lock:
                lines: List[str] = []
                archetypes: Dict[str, Any] = self._data.get('archetypes', {})

                if family:
                    arch = archetypes.get(family)
                    if arch:
                        lines.extend(self._archetype_line(arch))
                    recent = [g for g in self._data.get('generations', [])
                              if g.get('family') == family][-5:]
                else:
                    top = sorted(archetypes.values(),
                                 key=lambda a: a.get('tried_n', 0), reverse=True)[:3]
                    for arch in top:
                        lines.extend(self._archetype_line(arch))
                    recent = self._data.get('generations', [])[-4:]

                for g in recent:
                    lines.append(
                        f"- 候选 {g.get('name')} [{g.get('family')}] "
                        f"IC={g.get('ic')} 判定={g.get('verdict')}"
                        + (f" 否决理由: {g.get('reviewer_notes')[:80]}"
                           if g.get('reviewer_notes') else ''))

                cycles = self._data.get('cycles', [])
                if cycles:
                    c = cycles[-1]
                    lines.append(
                        f"- 上轮({c.get('ts')}): 候选{c.get('n_candidates')} "
                        f"通过率{c.get('pass_rate')} "
                        f"家族{'/'.join(c.get('top_families') or []) or '-'}")

                return '\n'.join(lines)
        except Exception as e:
            logger.error(f'[MinerMemory] cues_for 失败: {e}')
            return ''

    @staticmethod
    def _archetype_line(arch: Dict[str, Any]) -> List[str]:
        """archetype → 单行线索文本"""
        line = (f"- [{arch.get('family')}] 已试{arch.get('tried_n', 0)}次 "
                f"通过率{arch.get('pass_rate', 0)}")
        lessons = arch.get('lessons') or []
        if lessons:
            line += f" 教训: {'; '.join(lessons[-2:])}"
        return [line]

    # ── 只读快照 (状态 API/调试用) ────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """返回记忆统计摘要 (不含明细, 供状态端点)"""
        with self._lock:
            return {
                'n_generations': len(self._data.get('generations', [])),
                'n_cycles': len(self._data.get('cycles', [])),
                'n_families': len(self._data.get('archetypes', {})),
                'archetypes': {
                    fam: {'tried_n': a.get('tried_n'), 'pass_rate': a.get('pass_rate'),
                          'n_lessons': len(a.get('lessons', []))}
                    for fam, a in self._data.get('archetypes', {}).items()
                },
            }


# ── 单例 ────────────────────────────────────────────────────────

_miner_memory: Optional[MinerMemory] = None
_miner_memory_lock = threading.Lock()


def get_miner_memory() -> MinerMemory:
    """获取 MinerMemory 单例"""
    global _miner_memory
    with _miner_memory_lock:
        if _miner_memory is None:
            _miner_memory = MinerMemory()
            logger.info(f'[MinerMemory] 单例已创建: {_miner_memory.path}')
        return _miner_memory
