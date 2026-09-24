#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
RD-Agent 风格的 LLM 因子挖掘 (2026-08-13 新增)

参考: Microsoft Qlib RD-Agent (2025) - "LLM-Based Autonomous Evolving Agents"

闭环流程:
    1. Hypothesis (假设): LLM 生成因子假设
    2. Code (代码): LLM 生成因子计算代码
    3. Backtest (回测): 在历史数据上验证 IC
    4. Decision (决策): |IC| >= 阈值 → validated 入库; < 阈值 → rejected;
       exec 失败/面板 <2 票 → error (验证无效, 不判 alpha 优劣, 2026-09-02)

使用 OMLX Qwen3.6-35B 作为 LLM 后端 (本地优先)

API 端点:
    GET  /api/rdagent/status          - 挖掘器状态
    POST /api/rdagent/mine            - 触发一轮挖掘
    GET  /api/rdagent/factors         - 已挖掘因子列表
    POST /api/rdagent/validate/<name> - 手动验证因子
"""

import json
import time
import threading
import sqlite3
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from modules.logger import logger

# 尝试导入 OMLX 客户端
try:
    from modules.llm_router import llm_router
    HAS_LLM_ROUTER = True
except ImportError:
    HAS_LLM_ROUTER = False

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# 2026-09-14 P1-8: 三级反思记忆 (AgonAlpha/XAlpha generation/cycle/archetype)
try:
    from modules.miner_memory import get_miner_memory
    HAS_MINER_MEMORY = True
except ImportError:
    HAS_MINER_MEMORY = False


# ── 2026-09-14 P1-8: 对抗审查员模块级纯函数 ─────────────────────────

# 面板基线列 (定位 LLM 新生成因子列时先排除)
BASE_COLS = ('date', 'open', 'high', 'low', 'close', 'volume', 'returns', 'returns_fwd')

# exec 沙箱内置白名单 (validate_factor 与保留窗口重执行共用, 单一来源 2026-09-14)
SAFE_BUILTINS = {
    '__builtins__': {
        'len': len, 'str': str, 'int': int, 'float': float,
        'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'range': range, 'enumerate': enumerate,
        'zip': zip, 'sorted': sorted, 'min': min, 'max': max,
        'sum': sum, 'abs': abs, 'round': round,
        'isinstance': isinstance, 'type': type,
        'map': map, 'filter': filter,
        'any': any, 'all': all,
        'Exception': Exception, 'ValueError': ValueError,
        'TypeError': TypeError, 'KeyError': KeyError,
        'IndexError': IndexError, 'RuntimeError': RuntimeError,
    },
}


def ic_decay_ratio(declared_ic: float, holdout_ic: Optional[float]) -> Optional[float]:
    """
    IC 衰减比例 = 1 - |holdout| / |declared| (纯函数, 可单测)。

    Args:
        declared_ic: validate_factor 申报的样本内 IC
        holdout_ic: 保留窗口重执行 IC (None = 无法重执行)

    Returns:
        衰减比例 (可为负 = holdout 更强); holdout_ic 为 None → None;
        declared≈0 时无有效信号可比 → 视为全衰减 1.0
    """
    try:
        if holdout_ic is None:
            return None
        if abs(float(declared_ic)) < 1e-6:
            return 1.0
        return 1.0 - abs(float(holdout_ic)) / abs(float(declared_ic))
    except Exception as e:
        logger.error(f'[RDAgent] ic_decay_ratio 计算失败: {e}')
        return None


def compute_holdout_ic(dfs: List[Any], code: str, factor_col: str,
                       holdout_days: int = 60,
                       fwd_days: int = 5) -> Tuple[Optional[float], int, List[str]]:
    """
    保留窗口重执行 (纯函数, 不调 LLM): 在每票最近 holdout_days 个交易日
    (训练窗之外) 重跑因子代码并重算 IC。

    因子代码在完整序列上 exec (滚动窗口需要历史), IC 只取尾部 holdout 窗口。

    Args:
        dfs: 每票日线 DataFrame 列表 (列: date/open/high/low/close/volume)
        code: 因子计算代码 (沙箱 exec)
        factor_col: 期望的因子列名 (缺失时取首个新增列, 与 validate_factor 同逻辑)
        holdout_days: 保留窗口交易日数
        fwd_days: 前瞻收益窗口 (close.pct_change(fwd).shift(-fwd))

    Returns:
        (holdout_ic_mean 或 None, n_valid, errors[:2]) — 全票失败 → (None, 0, errs)
    """
    if not HAS_PANDAS:
        return None, 0, ['pandas 不可用']
    ics = []
    failed = []
    for df0 in dfs:
        try:
            df0 = df0.copy()
            if len(df0) < max(30, holdout_days // 4) or 'close' not in df0.columns:
                failed.append('单票数据不足或缺 close 列')
                continue
            if 'returns' not in df0.columns:
                df0['returns'] = df0['close'].pct_change()
            local_ns = {'df': df0, 'pd': pd, 'np': np}
            exec(code, SAFE_BUILTINS, local_ns)
            dfx = local_ns['df']
            col = factor_col if factor_col in dfx.columns else next(
                (c for c in dfx.columns if c not in BASE_COLS), None)
            if not col:
                failed.append('未生成新因子列')
                continue
            fwd = dfx['close'].pct_change(fwd_days).shift(-fwd_days)
            tail = min(holdout_days, len(dfx))
            valid = pd.DataFrame(
                {'f': dfx[col].iloc[-tail:], 'r': fwd.iloc[-tail:]}).dropna()
            if len(valid) < 10:
                failed.append(f'保留窗口有效样本不足 ({len(valid)}<10)')
                continue
            ic_v = valid['f'].corr(valid['r'])
            if pd.isna(ic_v):
                failed.append('保留窗口 IC 为 NaN (因子或收益零方差)')
                continue
            ics.append(float(ic_v))
        except Exception as e:
            failed.append(str(e)[:80])
            logger.debug(f'[RDAgent] 保留窗口单票重执行失败: {e}')
            continue
    if not ics:
        return None, 0, failed[:2]
    return float(np.mean(ics)), len(ics), failed[:2]


def _extract_json_obj(text: str) -> Optional[dict]:
    """从 LLM 输出提取首个平衡 {} JSON 对象 (剥 markdown fence + 大括号配对)"""
    try:
        t = (text or '').strip()
        if t.startswith('```'):
            t = t.split('\n', 1)[1].rsplit('```', 1)[0] if '\n' in t else t
        try:
            d = json.loads(t)
            return d if isinstance(d, dict) else None
        except Exception:
            pass
        start = t.find('{')
        while start != -1:
            depth = 0
            for i in range(start, len(t)):
                if t[i] == '{':
                    depth += 1
                elif t[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except Exception:
                            break
            start = t.find('{', start + 1)
        return None
    except Exception as e:
        logger.error(f'[RDAgent] JSON 提取失败: {e}')
        return None


@dataclass
class FactorHypothesis:
    """因子假设"""
    name: str
    formula: str  # 因子计算公式 (自然语言)
    category: str  # momentum/value/volatility/volume/quality
    rationale: str  # LLM 给出的经济学原理
    code: str = ''  # LLM 生成的代码
    ic: float = 0.0  # Information Coefficient
    icir: float = 0.0  # IC * sqrt(N) / std(IC)
    status: str = 'pending'  # pending/validated/rejected/error/
                              # rejected_reviewer(_observe) (2026-09-14 P1-8)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validated_at: str = ''
    # ── 2026-09-14 P1-8 对抗审查员 (DB schema 不变, 仅内存/API 可见) ──
    review_verdict: str = ''   # accept/veto/review_skipped
    reviewer_notes: str = ''   # 否决理由 (reexec 衰减 或 LLM reasons)
    holdout_ic: float = 0.0    # 保留窗口重执行 IC


@dataclass
class RDAgentStatus:
    """挖掘器状态"""
    total_hypotheses: int = 0
    validated_factors: int = 0
    rejected_factors: int = 0
    error_factors: int = 0  # 2026-09-02: exec 失败/面板不足 ≠ 真无预测力, 单独计数
    last_mine_time: str = ''
    llm_provider: str = 'omlx'
    llm_available: bool = False
    ic_threshold: float = 0.03  # IC > 0.03 才入库
    mining_round: int = 0
    adversarial_mode: str = 'observe'  # 2026-09-14 P1-8: observe(默认,只标记)/enforce(拦截)


class RDAgentMiner:
    """
    RD-Agent 风格的因子挖掘器

    核心特性:
    1. LLM 生成因子假设 (Hypothesis)
    2. LLM 生成因子代码 (Code)
    3. 在历史数据上验证 IC (Backtest)
    4. IC > 阈值则入库 (Decision)
    """

    # IC 阈值
    DEFAULT_IC_THRESHOLD = 0.03
    # 最大并发挖掘数
    MAX_CONCURRENT = 3
    # 单次 LLM 调用超时
    LLM_TIMEOUT = 90.0
    # ── 2026-09-14 P1-8 对抗审查员 ──
    HOLDOUT_DAYS = 60        # 保留窗口: 最近 60 交易日 (训练窗之外)
    REVIEW_DECAY_VETO = 0.5  # 保留窗口 IC 衰减 >50% → 重执行直接 veto (不依赖 LLM)
    REVIEW_TIMEOUT = 30.0    # 审查员 LLM 调用显式超时

    # 因子类别
    CATEGORIES = ('momentum', 'value', 'volatility', 'volume', 'quality', 'sentiment')

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'rdagent_factors.db'
        )
        self._lock = threading.Lock()
        self._hypotheses: Dict[str, FactorHypothesis] = {}
        self._status = RDAgentStatus()
        self._last_panel: List[Any] = []  # 最近 validate_factor 面板 (P1-8 重执行复用)
        self._init_db()
        self._load_from_db()
        self._check_llm()
        self._status.adversarial_mode = self._adversarial_mode()
        logger.info(f"[RDAgent] 对抗审查模式: miner_adversarial_mode="
                    f"'{self._status.adversarial_mode}'")
        self._start_auto_mine()

    @staticmethod
    def _adversarial_mode() -> str:
        """
        读取对抗审查模式 flag miner_adversarial_mode (2026-09-14 P1-8)。

        解析顺序: config_keys.get → env MINER_ADVERSARIAL_MODE → 'observe'。
        - observe (默认): veto 只标记 status='rejected_reviewer_observe' + warning,
          不拦截入库 (今晚挖掘链安全);
        - enforce: veto 改 status='rejected_reviewer', 拦截 validated 入库。
        """
        mode = None
        try:
            from modules.config_keys import get as cfg_get  # type: ignore
            mode = cfg_get('miner_adversarial_mode', None)
        except Exception:
            mode = None
        if not mode:
            mode = os.environ.get('MINER_ADVERSARIAL_MODE', 'observe')
        mode = str(mode).strip().lower()
        return mode if mode in ('observe', 'enforce') else 'observe'

    def _init_db(self):
        """初始化 SQLite 数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS factors (
                    name TEXT PRIMARY KEY,
                    formula TEXT,
                    category TEXT,
                    rationale TEXT,
                    code TEXT,
                    ic REAL,
                    icir REAL,
                    status TEXT,
                    created_at TEXT,
                    validated_at TEXT
                )
            ''')
            conn.commit()

    def _load_from_db(self):
        """从数据库加载已挖掘因子"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    'SELECT name, formula, category, rationale, code, ic, icir, status, created_at, validated_at FROM factors'
                ).fetchall()
                for row in rows:
                    h = FactorHypothesis(
                        name=row[0],
                        formula=row[1],
                        category=row[2],
                        rationale=row[3],
                        code=row[4] or '',
                        ic=row[5] or 0.0,
                        icir=row[6] or 0.0,
                        status=row[7] or 'pending',
                        created_at=row[8] or '',
                        validated_at=row[9] or '',
                    )
                    self._hypotheses[h.name] = h
            self._status.total_hypotheses = len(self._hypotheses)
            self._status.validated_factors = sum(1 for h in self._hypotheses.values() if h.status == 'validated')
            self._status.rejected_factors = sum(1 for h in self._hypotheses.values() if h.status == 'rejected')
            self._status.error_factors = sum(1 for h in self._hypotheses.values() if h.status == 'error')
        except Exception as e:
            logger.warning(f"[RDAgent] 加载数据库失败: {e}")

    def _check_llm(self):
        """检查 LLM 可用性"""
        if HAS_LLM_ROUTER:
            try:
                self._status.llm_available = llm_router._configs.get('omlx', None) is not None
                self._status.llm_provider = 'omlx'
            except Exception:
                self._status.llm_available = False
        logger.info(f"[RDAgent] LLM 可用: {self._status.llm_available}, provider={self._status.llm_provider}")

    def _start_auto_mine(self):
        """夜间自动挖掘调度 (2026-09-02)

        23:17 每日触发 + 启动 5 分钟后首跑; 开关 config['rdagent_auto_mine'] 默认 True.
        串行单轮 (~2-5min), 与 llm_router Semaphore(2) 反压兼容, 不争用交互会话.
        """
        if getattr(self, '_auto_mine_started', False):
            return
        self._auto_mine_started = True
        try:
            from modules.config_keys import get as cfg_get  # type: ignore
            enabled = cfg_get('rdagent_auto_mine', True)
        except Exception:
            enabled = True
        if not enabled:
            logger.info('[RDAgent] 夜间自动挖掘已禁用 (rdagent_auto_mine=False)')
            return

        def _loop():
            time.sleep(300)  # 启动 5 分钟后首跑
            while True:
                try:
                    r = self.mine_round()
                    logger.info(
                        f"[RDAgent] 自动挖掘 round={r.get('round')}: "
                        f"validated={r.get('validated')} rejected={r.get('rejected')} "
                        f"error={r.get('error')}")
                except Exception as e:
                    logger.error(f'[RDAgent] 自动挖掘失败: {e}')
                # 下一个 23:17
                now = datetime.now()
                nxt = now.replace(hour=23, minute=17, second=0, microsecond=0)
                if nxt <= now:
                    nxt += timedelta(days=1)
                time.sleep(max(60.0, (nxt - now).total_seconds()))

        t = threading.Thread(target=_loop, daemon=True, name='rdagent-auto-mine')
        t.start()
        logger.info('[RDAgent] 夜间自动挖掘调度线程已启动 (首跑 5min 后, 之后每日 23:17)')

    def _call_llm(self, prompt: str, timeout: float = None) -> str:
        """调用 LLM"""
        if not HAS_LLM_ROUTER:
            return self._rule_engine_generate(prompt)

        timeout = timeout or self.LLM_TIMEOUT
        try:
            result = llm_router.route(prompt, timeout=timeout)
            if result.get('success'):
                return result.get('content', '')
            logger.warning(f"[RDAgent] LLM 调用失败: {result.get('error', 'unknown')}")
            return self._rule_engine_generate(prompt)
        except Exception as e:
            logger.warning(f"[RDAgent] LLM 调用异常: {e}")
            return self._rule_engine_generate(prompt)

    def _rule_engine_generate(self, prompt: str) -> str:
        """规则引擎降级 (LLM 不可用时)"""
        # 基于 prompt 中的关键词返回预设因子
        if 'momentum' in prompt.lower() or '动量' in prompt:
            return json.dumps({
                'name': 'momentum_20d',
                'formula': '(close - close_20d_ago) / close_20d_ago',
                'category': 'momentum',
                'rationale': '20日动量反映短期趋势',
                'code': 'df["momentum_20d"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)',
            }, ensure_ascii=False)
        elif 'value' in prompt.lower() or '价值' in prompt:
            return json.dumps({
                'name': 'ep_ratio',
                'formula': 'eps / price',
                'category': 'value',
                'rationale': '盈利收益率 (E/P) 经典价值因子',
                'code': 'df["ep_ratio"] = df["eps"] / df["close"]',
            }, ensure_ascii=False)
        elif 'volatility' in prompt.lower() or '波动' in prompt:
            return json.dumps({
                'name': 'volatility_20d',
                'formula': 'std(returns_20d)',
                'category': 'volatility',
                'rationale': '20日波动率反映风险',
                'code': 'df["volatility_20d"] = df["returns"].rolling(20).std()',
            }, ensure_ascii=False)
        else:
            return json.dumps({
                'name': 'volume_ratio',
                'formula': 'volume / volume_ma_20',
                'category': 'volume',
                'rationale': '成交量相对均量反映资金关注度',
                'code': 'df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()',
            }, ensure_ascii=False)

    def generate_hypothesis(self, category: str = 'momentum') -> FactorHypothesis:
        """
        步骤 1: LLM 生成因子假设

        Args:
            category: 因子类别

        Returns:
            FactorHypothesis 对象
        """
        # 2026-09-14 P1-8: 三级反思记忆线索注入 (guarded, 失败跳过不阻断)
        cues_block = ''
        if HAS_MINER_MEMORY:
            try:
                cues = get_miner_memory().cues_for(category)
                if cues:
                    cues_block = (
                        f"历史挖掘反思线索 (同家族已试过, 勿重复背诵被否决的变体):\n"
                        f"{cues}\n\n")
            except Exception as e:
                logger.debug(f'[RDAgent] 记忆线索读取失败 (跳过注入): {e}')

        prompt = f"""你是一个量化研究专家。基于以下类别，生成一个全新的 alpha 因子假设。

类别: {category}

{cues_block}要求:
1. 因子名称 (英文, snake_case)
2. 计算公式 (自然语言描述)
3. 经济学原理 (为什么这个因子能预测收益)
4. Python 代码: 直接可执行的 pandas 语句片段, 把因子值直接写入 df['因子名称'];
   禁止 def/import/类/函数定义/注释 (沙箱无 __import__ 且不执行函数定义);
   df 是单只股票的日线时序 (列: date/open/high/low/close/volume), 只能用该票自身列,
   禁止 groupby/merge 多票横截面操作

只输出一个 JSON 对象, 禁止任何解释/分析/编号列表:
{{"name": "...", "formula": "...", "category": "{category}", "rationale": "...", "code": "..."}}
"""
        response = self._call_llm(prompt)

        try:
            # JSON 提取: 直接 loads → 剥 markdown fence → 大括号配对切片 (2026-09-02)
            text = (response or '').strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0] if '\n' in text else text
            data = None
            try:
                data = json.loads(text)
            except Exception:
                start = text.find('{')
                while start != -1 and data is None:
                    depth = 0
                    for i in range(start, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    data = json.loads(text[start:i + 1])
                                except Exception:
                                    data = None
                                break
                    if data is None:
                        start = text.find('{', start + 1)
            if data is None:
                raise ValueError('JSON 切片失败')
            # 空壳校验 (2026-09-02): 降级链 (route→rule_engine) 返回的 direction/决策类
            # JSON 或 echo 式垃圾在此被拒 → 落入 except 走 _rule_engine_generate replay,
            # 避免生成无 code 的空壳假设浪费整轮挖掘
            if not (isinstance(data, dict) and data.get('name') and data.get('code')):
                raise ValueError('JSON 缺少因子字段 (空壳/决策类响应)')

            h = FactorHypothesis(
                name=data.get('name', f'{category}_auto_{int(time.time())}'),
                formula=data.get('formula', ''),
                category=data.get('category', category),
                rationale=data.get('rationale', ''),
                code=data.get('code', ''),
            )
            logger.info(f"[RDAgent] 生成假设: {h.name} ({h.category})")
            return h
        except Exception as e:
            logger.warning(f"[RDAgent] 解析 LLM 响应失败: {e}, response={response[:200]}")
            # 降级到规则引擎
            data = json.loads(self._rule_engine_generate(prompt))
            return FactorHypothesis(
                name=data['name'],
                formula=data['formula'],
                category=data['category'],
                rationale=data['rationale'],
                code=data['code'],
            )

    def _fetch_panel(self, stocks: List[str] = None) -> List[Any]:
        """
        获取真实数据多票 K 线面板 (2026-09-14 从 validate_factor 抽出,
        与对抗审查员保留窗口重执行共用同一取数路径)。

        Args:
            stocks: 股票池, 默认 ['sz300620', 'sh688981', 'sh600519', 'sz000001']

        Returns:
            DataFrame 列表 (可能为空 — 调用方自行判足够, 本函数绝不抛)
        """
        dfs = []
        stocks = stocks or ['sz300620', 'sh688981', 'sh600519', 'sz000001']
        try:
            from modules.dependencies import get_data_fetcher
            fetcher = get_data_fetcher()
            for code in stocks:
                try:
                    kl = fetcher.get_kline_data(code, period='daily', count=120)
                    if kl and len(kl) >= 60:
                        dfs.append(pd.DataFrame(kl))
                except Exception as e:
                    logger.debug(f"[RDAgent] {code} K线获取失败: {e}")
        except Exception as e:
            logger.warning(f"[RDAgent] get_data_fetcher 不可用: {e}")
        return dfs

    def adversarial_review(self, hypothesis: FactorHypothesis) -> Dict[str, Any]:
        """
        对抗审查员 (2026-09-14 P1-8, AgonAlpha/XAlpha 2026 标配) — 两票独立否决:

        1. 重执行 (不依赖 LLM): 在最近 HOLDOUT_DAYS 交易日保留窗口 (训练窗之外)
           重算因子 IC, 与申报 IC 衰减 >REVIEW_DECAY_VETO (50%) → 直接 veto;
        2. fresh-context LLM 怀疑者: 单次调用无历史对话, 从 (a) 教科书已知因子/
           背诵嫌疑 (b) 数据窥探嫌疑 (c) 经济逻辑缺失 三方面质疑,
           输出 {"verdict": "accept|veto", "reasons": [...]}。
           走 llm_router 现成 route() API, timeout 显式 (REVIEW_TIMEOUT);
           enable_thinking=false 沿用项目惯例 — OmlxClient 不注入 chat_template_kwargs
           走模型原生模板 (2026-09-02 坍缩教训), 由路由层保证。
        3. 任何不可用 (LLM 失败 / rule_engine 降级响应 / JSON 不可解析) →
           verdict='review_skipped', 绝不阻断挖掘链。

        Args:
            hypothesis: 已通过 IC 阈值、即将入库的候选

        Returns:
            {verdict: accept|veto|review_skipped, reasons: [...],
             source: reexec|llm|skipped, holdout_ic: float|None, decay: float|None}
        """
        result: Dict[str, Any] = {'verdict': 'review_skipped', 'reasons': [],
                                  'source': 'skipped', 'holdout_ic': None, 'decay': None}
        try:
            # ── 1. 重执行否决 (独立于 LLM 的一票) ──
            panel = self._last_panel if self._last_panel else self._fetch_panel()
            holdout_ic, n_valid, errs = compute_holdout_ic(
                panel, hypothesis.code, hypothesis.name,
                holdout_days=self.HOLDOUT_DAYS)
            result['holdout_ic'] = holdout_ic
            decay = ic_decay_ratio(hypothesis.ic, holdout_ic)
            result['decay'] = decay
            if decay is not None and decay > self.REVIEW_DECAY_VETO:
                result['verdict'] = 'veto'
                result['source'] = 'reexec'
                result['reasons'] = [
                    f'保留窗口重执行衰减 {decay:.0%} (>50%): '
                    f'申报 IC={hypothesis.ic:.4f} → '
                    f'holdout({self.HOLDOUT_DAYS}d, {n_valid}票) IC={holdout_ic:.4f}']
                return result

            # ── 2. fresh-context LLM 怀疑者 ──
            if not HAS_LLM_ROUTER:
                if errs:
                    result['reasons'] = [f'重执行不可用: {"; ".join(errs)}']
                return result
            holdout_str = (f'{holdout_ic:.4f}' if holdout_ic is not None
                           else '不可用 (面板不足/零方差)')
            prompt = f"""你是冷酷无情的量化审查员 (fresh context, 无任何先前轮次记忆)。你的任务不是夸奖, 而是找出理由否决下面这个 LLM 挖掘的因子候选。

因子名称: {hypothesis.name}
家族: {hypothesis.category}
公式: {hypothesis.formula}
计算代码: {hypothesis.code[:400]}
经济学理由: {hypothesis.rationale[:300]}
样本内 IC: {hypothesis.ic:.4f}
保留窗口 (最近 {self.HOLDOUT_DAYS} 交易日, 训练窗之外) 重执行 IC: {holdout_str}

请从三方面质疑:
(a) 背诵嫌疑: 这是否其实是教科书已知因子 (动量/反转/波动率/换手率等) 的换皮, 却包装成"全新"?
(b) 数据窥探: IC 是否可能来自参数挖掘/前视偏差? 保留窗口 IC 是否支撑样本内?
(c) 经济逻辑缺失: 理由是否只是词藻堆砌, 无法解释超额收益为何存在?

只输出一个 JSON 对象, 禁止任何解释:
{{"verdict": "accept 或 veto", "reasons": ["理由1", "理由2"]}}
"""
            resp = llm_router.route(prompt, timeout=self.REVIEW_TIMEOUT)
            if not resp.get('success') or resp.get('fallback') \
                    or resp.get('provider') == 'rule_engine':
                # 规则引擎降级输出是交易决策 JSON, 不是审查结论 → skipped
                return result
            data = _extract_json_obj(resp.get('content', ''))
            if not isinstance(data, dict):
                return result
            verdict = str(data.get('verdict', '')).strip().lower()
            if verdict not in ('accept', 'veto'):
                return result
            reasons = data.get('reasons') or []
            if isinstance(reasons, str):
                reasons = [reasons]
            result['verdict'] = verdict
            result['source'] = 'llm'
            result['reasons'] = [str(r)[:200] for r in list(reasons)[:4]]
            if decay is not None:
                result['reasons'].append(
                    f'(重执行参考: holdout IC={holdout_ic:.4f}, 衰减 {decay:.0%})')
            return result
        except Exception as e:
            logger.error(f'[RDAgent][Reviewer] 审查异常 (降级 review_skipped): {e}')
            return {'verdict': 'review_skipped', 'reasons': [str(e)[:120]],
                    'source': 'skipped', 'holdout_ic': None, 'decay': None}

    def _record_generation_memory(self, h: FactorHypothesis) -> None:
        """
        generation 级记忆写入 (2026-09-14 P1-8): 候选出 IC 后记录。
        error 候选无有效 IC 不记录; 任何失败只记日志, 绝不阻断挖掘链。
        """
        if not HAS_MINER_MEMORY or h.status == 'error':
            return
        try:
            verdict_map = {'validated': 'pass', 'rejected': 'reject'}
            verdict = verdict_map.get(h.status, 'rejected_reviewer')
            get_miner_memory().record_generation(
                name=h.name, expr=h.formula or (h.code or '')[:200],
                family=h.category, ic=h.ic, verdict=verdict,
                reviewer_notes=h.reviewer_notes)
        except Exception as e:
            logger.error(f'[RDAgent] generation 记忆写入失败: {e}')

    def _summarize_cycle_memory(self, categories: List[str],
                                results: List[Dict]) -> None:
        """cycle 级摘要 + 家族 Top (2026-09-14 P1-8, 每轮挖掘结束调用, 失败不阻断)"""
        if not HAS_MINER_MEMORY:
            return
        try:
            from collections import Counter
            cands = [r for r in results if r.get('status')]
            n = len(cands)
            n_pass = sum(1 for r in cands if r.get('status') == 'validated')
            fam = Counter(r.get('category') for r in cands
                          if r.get('status') in ('validated', 'rejected_reviewer_observe'))
            if not fam:
                fam = Counter(r.get('category') for r in cands)
            top = [f for f, _ in fam.most_common(3)]
            get_miner_memory().summarize_cycle(
                themes=list(categories), n_candidates=n,
                pass_rate=(n_pass / n if n else 0.0), top_families=top)
        except Exception as e:
            logger.error(f'[RDAgent] cycle 记忆摘要失败: {e}')

    def validate_factor(self, hypothesis: FactorHypothesis) -> Tuple[float, float, Optional[str]]:
        """
        步骤 2-3: 在历史数据上验证 IC

        Args:
            hypothesis: 因子假设

        Returns:
            (ic, icir, error) — error 非 None 时验证无效 (exec 未成功/面板不足),
            不得据此判 rejected. 2026-09-02: 不再用模拟随机漫步数据伪造 IC≈0 样本,
            面板 <2 票或 exec 全票失败直接判 error, 本轮作废可重试.
        """
        if not HAS_PANDAS:
            return 0.0, 0.0, 'pandas 不可用'

        try:
            # ── 2026-09-02: 真实数据多票面板 (默认股票池, 避免单票/模拟过拟合) ──
            dfs = self._fetch_panel()
            if len(dfs) < 2:
                # 2026-09-02: 面板不足 2 票不再 fallback 模拟数据 — 随机漫步上 IC 恒≈0,
                # 等于把"没测过"伪装成"真无预测力"的 rejected. 本轮作废 (下轮或手动可重试验证).
                return 0.0, 0.0, f'面板数据不足 ({len(dfs)}/4 票可取, 本轮作废)'
            # 2026-09-14 P1-8: 缓存面板供对抗审查员保留窗口重执行复用
            self._last_panel = dfs

            # 执行 LLM 生成的代码 (严格沙箱) + 多票面板 IC 评估
            safe_builtins = SAFE_BUILTINS
            factor_col = hypothesis.name
            ics = []
            failed = []
            for df0 in dfs:
                try:
                    df0 = df0.copy()
                    if len(df0) < 30 or 'close' not in df0.columns:
                        failed.append('单票数据<30行或缺 close 列')
                        continue
                    if 'returns' not in df0.columns and 'close' in df0.columns:
                        df0['returns'] = df0['close'].pct_change()
                    local_ns = {'df': df0, 'pd': pd, 'np': np}
                    exec(hypothesis.code, safe_builtins, local_ns)
                    dfx = local_ns['df']
                    col = factor_col if factor_col in dfx.columns else next(
                        (c for c in dfx.columns
                         if c not in ('date', 'open', 'high', 'low', 'close', 'volume',
                                      'returns', 'returns_fwd')),
                        None)
                    if not col:
                        failed.append('未生成新因子列')
                        continue
                    fwd = dfx['close'].pct_change(5).shift(-5)
                    valid = pd.DataFrame({'f': dfx[col], 'r': fwd}).dropna()
                    if len(valid) < 20:
                        failed.append(f'有效样本不足 ({len(valid)}<20)')
                        continue
                    ic_v = valid['f'].corr(valid['r'])
                    if pd.isna(ic_v):
                        failed.append('IC 计算为 NaN')
                        continue
                    ics.append(float(ic_v))
                except Exception as e:
                    failed.append(str(e)[:80])
                    logger.debug(f"[RDAgent] 面板单票执行失败: {e}")
                    continue
            if len(ics) < 2:
                # 全部 (或仅 1 票) 未成功 → 验证无效, 不算真无预测力
                return 0.0, 0.0, 'exec 失败或有效样本 <2 票: ' + '; '.join(failed[:2])
            ic_mean = float(np.mean(ics))
            icir = ic_mean * float(np.sqrt(len(ics)))
            return ic_mean, icir, None
        except Exception as e:
            logger.warning(f"[RDAgent] 验证异常: {e}")
            return 0.0, 0.0, f'验证异常: {e}'

    def mine_round(self, categories: List[str] = None) -> Dict[str, Any]:
        """
        执行一轮挖掘 (生成 → 验证 → 入库)

        Args:
            categories: 要挖掘的类别列表，默认全部

        Returns:
            本轮挖掘结果
        """
        categories = categories or list(self.CATEGORIES)
        results = []

        with self._lock:
            self._status.mining_round += 1
            self._status.last_mine_time = datetime.now().isoformat()

        for category in categories[:self.MAX_CONCURRENT]:
            try:
                # 1. 生成假设
                h = self.generate_hypothesis(category)

                # 2-3. 验证 (error = exec 失败/面板不足, 区别于真无预测力)
                ic, icir, err = self.validate_factor(h)
                h.ic = ic
                h.icir = icir

                # 4. 决策
                if err is not None:
                    h.status = 'error'
                    logger.warning(f"[RDAgent] ⚠ 验证无效: {h.name} — {err} (不判 rejected, 可重试验证)")
                elif abs(ic) >= self._status.ic_threshold:
                    h.status = 'validated'
                    h.validated_at = datetime.now().isoformat()
                    logger.info(f"[RDAgent] ✓ 因子入库: {h.name} IC={ic:.4f}")
                else:
                    h.status = 'rejected'
                    logger.info(f"[RDAgent] ✗ 因子拒绝: {h.name} IC={ic:.4f} < {self._status.ic_threshold}")

                # 4.5 对抗审查 (2026-09-14 P1-8): 只审即将 validated 入库的候选
                if h.status == 'validated':
                    review = self.adversarial_review(h)
                    h.review_verdict = review.get('verdict', 'review_skipped')
                    if review.get('holdout_ic') is not None:
                        h.holdout_ic = review['holdout_ic']
                    if review['verdict'] == 'veto':
                        h.reviewer_notes = '; '.join(review.get('reasons', []))[:500]
                        mode = self._adversarial_mode()
                        if mode == 'enforce':
                            h.status = 'rejected_reviewer'
                            logger.warning(
                                f"[RDAgent][Reviewer] ✗ 否决 (enforce, 拦截 validated 入库): "
                                f"{h.name} — {h.reviewer_notes[:150]}")
                        else:
                            # observe (默认): 只标记 + warning, 不拦截入库
                            h.status = 'rejected_reviewer_observe'
                            logger.warning(
                                f"[RDAgent][Reviewer] ⚠ 否决 (observe, 不拦截入库): "
                                f"{h.name} — {h.reviewer_notes[:150]}")
                    elif review['verdict'] == 'review_skipped':
                        logger.info(
                            f"[RDAgent][Reviewer] 审查不可用 (review_skipped): {h.name}")

                # 入库
                self._hypotheses[h.name] = h
                self._save_to_db(h)

                # 5. 三级反思记忆 — generation 级 (2026-09-14 P1-8)
                self._record_generation_memory(h)

                results.append({
                    'name': h.name,
                    'category': h.category,
                    'ic': ic,
                    'icir': icir,
                    'status': h.status,
                    'rationale': h.rationale,
                    'review': h.review_verdict,
                })
            except Exception as e:
                logger.warning(f"[RDAgent] 挖掘 {category} 失败: {e}")
                results.append({'category': category, 'error': str(e)})

        # 更新统计
        with self._lock:
            self._status.total_hypotheses = len(self._hypotheses)
            self._status.validated_factors = sum(1 for h in self._hypotheses.values() if h.status == 'validated')
            self._status.rejected_factors = sum(1 for h in self._hypotheses.values() if h.status == 'rejected')
            self._status.error_factors = sum(1 for h in self._hypotheses.values() if h.status == 'error')

        # 6. 三级反思记忆 — cycle 级摘要 (2026-09-14 P1-8, 每轮挖掘结束)
        self._summarize_cycle_memory(categories, results)

        return {
            'success': True,
            'round': self._status.mining_round,
            'mined': len(results),
            'validated': sum(1 for r in results if r.get('status') == 'validated'),
            'rejected': sum(1 for r in results if r.get('status') == 'rejected'),
            'error': sum(1 for r in results if r.get('status') == 'error'),
            # 2026-09-14 P1-8 对抗审查统计 (observe 模式供今晚链观察, 不影响上面计数)
            'reviewer_veto_observe': sum(
                1 for r in results if r.get('status') == 'rejected_reviewer_observe'),
            'reviewer_veto_blocked': sum(
                1 for r in results if r.get('status') == 'rejected_reviewer'),
            'review_skipped': sum(1 for r in results if r.get('review') == 'review_skipped'),
            'adversarial_mode': self._adversarial_mode(),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }

    def _save_to_db(self, h: FactorHypothesis):
        """保存到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO factors
                    (name, formula, category, rationale, code, ic, icir, status, created_at, validated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    h.name, h.formula, h.category, h.rationale, h.code,
                    h.ic, h.icir, h.status, h.created_at, h.validated_at,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"[RDAgent] 保存失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        with self._lock:
            return {
                'success': True,
                'data': asdict(self._status),
                'timestamp': datetime.now().isoformat(),
            }

    def get_factors(self, status: str = None) -> Dict[str, Any]:
        """获取已挖掘因子"""
        with self._lock:
            factors = list(self._hypotheses.values())
            if status:
                factors = [f for f in factors if f.status == status]

            return {
                'success': True,
                'data': {
                    'factors': [asdict(f) for f in factors],
                    'total': len(factors),
                    'timestamp': datetime.now().isoformat(),
                },
            }


# ── 单例 ────────────────────────────────────────────────────────
_rd_agent_miner: Optional[RDAgentMiner] = None
_rd_agent_lock = threading.Lock()


def get_rd_agent_miner() -> RDAgentMiner:
    """获取单例"""
    global _rd_agent_miner
    with _rd_agent_lock:
        if _rd_agent_miner is None:
            _rd_agent_miner = RDAgentMiner()
            logger.info("[RDAgent] 单例已创建")
        return _rd_agent_miner