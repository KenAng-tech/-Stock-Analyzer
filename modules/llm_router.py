#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
LLM 路由模块 — LLM Router (2026-07-24)

多 LLM 故障转移 + 降级策略:
1. 健康检查路由: 基于健康状态选择可用 LLM
2. 熔断器: 连续失败后自动切换 + 冷却期
3. 分层超时: 不同 LLM 不同超时时间
4. 快速降级路径: 模型不可用时返回 fallback (规则引擎/统计模型)
5. 监控 API: 返回各 LLM 状态、切换历史、熔断状态

使用方式:
    from modules.llm_router import llm_router
    result = llm_router.route("chat", prompt="...")
    status = llm_router.get_status()
"""

import os
import time
import json
import time as _time
import threading
import http.client
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from modules.logger import logger
from modules.llm_budget import LlmBudget, CST

# 全局 LLM 并发反压 (2026-09-02, 防与交互会话争用 oMLX 8080, 其 max_concurrent=4)
_LLM_SEMAPHORE = __import__('threading').Semaphore(2)
_SEM_ACQUIRE_TIMEOUT = 60.0
# 09-20 token 预算 (PanWatch 式): 默认 0 = 纯观测不拦截; env 设数字即开门
_LLM_BUDGET_LIMIT = int(os.environ.get('LLM_DAILY_TOKEN_LIMIT', '0') or 0)


# ── 枚举 ──────────────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    OMLX = "omlx"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    RULE_ENGINE = "rule_engine"


class CircuitState(str, Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open" # 探测中


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """LLM 配置"""
    name: str
    provider: LLMProvider
    url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 60.0  # 超时时间 (秒)
    enabled: bool = True


@dataclass
class CircuitBreaker:
    """熔断器"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    failure_threshold: int = 3       # 连续失败 N 次后熔断
    recovery_timeout: float = 30.0   # 冷却期 (秒)
    half_open_max_calls: int = 1     # HALF_OPEN 状态最大探测次数


@dataclass
class RoutingRecord:
    """路由记录"""
    timestamp: float
    provider: str
    success: bool
    latency: float
    error: str = ""


@dataclass
class LLMRouterStatus:
    """路由状态"""
    active_provider: str = "omlx"
    total_requests: int = 0
    total_success: int = 0
    total_failure: int = 0
    avg_latency: float = 0.0
    providers: Dict[str, Dict] = field(default_factory=dict)
    recent_failures: List[Dict] = field(default_factory=list)
    switch_history: List[Dict] = field(default_factory=dict)
    circuit_breakers: Dict[str, Dict] = field(default_factory=dict)


# ── 规则引擎 (降级) ────────────────────────────────────────────────

class RuleEngineFallback:
    """
    规则引擎降级 — 当所有 LLM 不可用时使用

    基于技术指标生成交易建议:
    - RSI: 超买/超卖信号
    - MACD: 金叉/死叉信号
    - 均线: 趋势判断
    - 成交量: 确认信号
    """

    @staticmethod
    def decide(stock_data: Dict) -> Dict:
        """
        基于技术指标生成交易决策

        Args:
            stock_data: 包含 RSI, MACD, 均线等指标

        Returns:
            {direction, confidence, reasoning}
        """
        score = 0.0
        reasons = []

        # RSI
        rsi = stock_data.get('rsi_14', stock_data.get('rsi', 50))
        if rsi < 30:
            score += 0.3
            reasons.append("RSI 超卖")
        elif rsi > 70:
            score -= 0.3
            reasons.append("RSI 超买")

        # MACD
        macd = stock_data.get('macd', 0)
        macd_signal = stock_data.get('macd_signal', 0)
        if macd > macd_signal:
            score += 0.2
            reasons.append("MACD 金叉")
        elif macd < macd_signal:
            score -= 0.2
            reasons.append("MACD 死叉")

        # 均线
        ma5 = stock_data.get('ma_5', stock_data.get('ma5', 0))
        ma20 = stock_data.get('ma_20', stock_data.get('ma20', 0))
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20 * 1.02:
                score += 0.15
                reasons.append("MA5 > MA20")
            elif ma5 < ma20 * 0.98:
                score -= 0.15
                reasons.append("MA5 < MA20")

        # 成交量
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        change_pct = stock_data.get('change_pct', 0)
        if volume_ratio > 1.5:
            score += 0.05 * (1 if change_pct > 0 else -1)
            reasons.append(f"放量(量比{volume_ratio:.1f})")

        # 综合决策
        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(0.7, abs(score) + 0.3)

        return {
            'direction': direction,
            'confidence': round(confidence, 3),
            'score': round(score, 3),
            'reasoning': '; '.join(reasons) if reasons else '无明显信号',
            'fallback': True,
            'method': 'rule_engine',
        }


# ── LLM 客户端 ──────────────────────────────────────────────────────

def _read_omlx_key() -> str:
    """读取 OMLX API key — 优先 settings.json, 环境变量兜底 (2026-09-02)

    修复: 旧版仅从 OMLX_API_KEY env 读取; launchd/cron 等非 shell 启动上下文
    无该 env → 8080 全请求 401 (server skip_api_key_verification=False) →
    路由静默级联降级 rule_engine (预设因子 replay).
    """
    import os as _os
    try:
        with open(_os.path.expanduser('~/.omlx/settings.json')) as f:
            key = (json.load(f).get('auth') or {}).get('api_key', '')
        if key:
            return key
        logger.warning('[LLMRouter] settings.json auth.api_key 为空, 回退 env')
    except Exception as e:
        logger.debug(f'[LLMRouter] settings.json 读取失败: {e}, 回退 env')
    return _os.environ.get('OMLX_API_KEY', '')


class OmlxClient:
    """OMLX (本地 MLX) 客户端"""

    @staticmethod
    def chat(url: str, prompt: str, model: str = "Qwen3.6-35B-A3B-TurboQuant-MLX-8bit",
             api_key: str = "", timeout: float = 60.0) -> Dict:
        """调用 OMLX 聊天接口"""
        try:
            parsed = urlparse(url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                            timeout=timeout)

            headers = {
                'Content-Type': 'application/json',
            }
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            # 单 user 轮, 无 system 前缀 (2026-09-02 质量矩阵 B/C/D 验证: 裸调无坍缩;
            # 原 _AGENT_SCAFFOLD 系统前缀 ~500 tok/次 无增益, 已移除)
            payload = json.dumps({
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 2048,
                'temperature': 0.3,
                # 2026-09-02 实测: 手动注入 chat_template_kwargs 使 oMLX 以 default 模板重组 prompt,
                # 与 qwen35 自带 chat_template 冲突 → 全模型坍缩. 不注入走模型原生模板 (S2 对照).
            })

            conn.request('POST', '/v1/chat/completions', payload, headers)
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            conn.close()

            if resp.status == 200:
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return {'success': True, 'content': content,
                        'usage': data.get('usage') or {}}
            else:
                return {'success': False, 'error': f'HTTP {resp.status}: {data}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


class OpenAIClient:
    """OpenAI 兼容客户端"""

    @staticmethod
    def chat(url: str, prompt: str, model: str = "gpt-5.5",
             api_key: str = "", timeout: float = 60.0) -> Dict:
        """调用 OpenAI 兼容接口"""
        try:
            parsed = urlparse(url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                            timeout=timeout)

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}' if api_key else '',
            }

            payload = json.dumps({
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 500,
                'temperature': 0.3,
            })

            conn.request('POST', '/v1/chat/completions', payload, headers)
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            conn.close()

            if resp.status == 200:
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                return {'success': True, 'content': content,
                        'usage': data.get('usage') or {}}
            else:
                return {'success': False, 'error': f'HTTP {resp.status}: {data}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


class OllamaClient:
    """Ollama 客户端"""

    @staticmethod
    def chat(url: str, prompt: str, model: str = "qwen3.5",
             timeout: float = 15.0) -> Dict:
        """调用 Ollama 接口"""
        try:
            parsed = urlparse(url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port,
                                            timeout=timeout)

            payload = json.dumps({
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
            })

            conn.request('POST', '/api/chat', payload,
                        {'Content-Type': 'application/json'})
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            conn.close()

            content = data.get('message', {}).get('content', '')
            if content:
                return {'success': True, 'content': content,
                        'usage': data.get('usage') or {}}
            return {'success': False, 'error': 'Empty response'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


# ── LLM 路由服务 ────────────────────────────────────────────────────

class LLMRouter:
    """
    LLM 路由服务 — 多 LLM 故障转移 + 降级

    路由优先级:
    1. OMLX (本地 MLX, 最快)
    2. OpenAI 兼容 (如果有)
    3. Ollama (如果有)
    4. RuleEngine (最终降级)

    熔断器:
    - 每个 Provider 独立熔断器
    - 连续 3 次失败 → 熔断 30 秒
    - 熔断期间自动跳过该 Provider
    """

    def __init__(self):
        # LLM 配置
        self._configs: Dict[str, LLMConfig] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._records: List[RoutingRecord] = []
        self._switch_history: List[Dict] = []
        self._lock = threading.Lock()

        # 默认配置
        self._setup_defaults()

        # 09-20: token 台账 + 预算门 (默认 limit=0 纯观测; test 可替换 _budget)
        self._budget = LlmBudget(daily_limit=_LLM_BUDGET_LIMIT)

        # 降级引擎
        self._rule_engine = RuleEngineFallback()

        # 统计
        self._total_requests = 0
        self._total_success = 0
        self._total_failure = 0
        self._total_latency = 0.0

    def _setup_defaults(self):
        """设置默认 LLM 配置"""
        # OMLX (主) — key 读自 settings.json, env 兜底 (修复 launchd/cron 无 env 时 401)
        omlx_key = _read_omlx_key()
        self.add_provider(LLMConfig(
            name="OMLX",
            provider=LLMProvider.OMLX,
            url="http://127.0.0.1:8080",
            api_key=omlx_key,
            model="Qwen3.6-35B-A3B-TurboQuant-MLX-8bit",
            timeout=60.0,
            enabled=True,
        ))

        # OpenAI 兼容 (次)
        self.add_provider(LLMConfig(
            name="OpenAI",
            provider=LLMProvider.OPENAI,
            url="http://127.0.0.1:8080/v1",
            api_key=omlx_key,
            model="gpt-5.5",
            timeout=15.0,
            enabled=False,  # 默认禁用 (同一服务器)
        ))

        # Ollama (备选)
        self.add_provider(LLMConfig(
            name="Ollama",
            provider=LLMProvider.OLLAMA,
            url="http://127.0.0.1:11434",
            model="qwen3.5",
            timeout=20.0,
            enabled=False,
        ))

    def add_provider(self, config: LLMConfig) -> None:
        """添加 LLM 提供者"""
        key = config.provider.value
        self._configs[key] = config
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker()

    def remove_provider(self, provider: LLMProvider) -> None:
        """移除 LLM 提供者"""
        key = provider.value
        self._configs.pop(key, None)
        self._circuit_breakers.pop(key, None)

    def enable_provider(self, provider: LLMProvider, enabled: bool = True) -> None:
        """启用/禁用 LLM 提供者"""
        if provider in self._configs:
            self._configs[provider].enabled = enabled

    def _get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """获取熔断器"""
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker()
        return self._circuit_breakers[key]

    def _can_use_provider(self, key: str) -> bool:
        """检查是否可以使用该 Provider"""
        if key not in self._configs:
            return False
        config = self._configs[key]
        if not config.enabled:
            return False

        cb = self._get_circuit_breaker(key)

        if cb.state == CircuitState.CLOSED:
            return True

        if cb.state == CircuitState.OPEN:
            # 检查冷却期是否已过
            if time.time() - cb.last_failure_time >= cb.recovery_timeout:
                cb.state = CircuitState.HALF_OPEN
                cb.last_state_change = time.time()
                logger.info(f"[LLMRouter] {key} 熔断器进入 HALF_OPEN 状态")
                return True
            return False

        if cb.state == CircuitState.HALF_OPEN:
            return True

        return False

    def _record_result(self, key: str, success: bool, latency: float,
                       error: str = "") -> None:
        """记录路由结果"""
        record = RoutingRecord(
            timestamp=time.time(),
            provider=key,
            success=success,
            latency=latency,
            error=error,
        )
        self._records.append(record)

        # 保持最近 100 条记录
        if len(self._records) > 100:
            self._records = self._records[-100:]

        # 更新熔断器状态
        cb = self._get_circuit_breaker(key)
        if success:
            cb.failure_count = 0
            cb.success_count += 1
            if cb.state == CircuitState.HALF_OPEN:
                cb.state = CircuitState.CLOSED
                cb.last_state_change = time.time()
                logger.info(f"[LLMRouter] {key} 熔断器恢复 CLOSED")
        else:
            cb.failure_count += 1
            cb.last_failure_time = time.time()
            if cb.failure_count >= cb.failure_threshold and cb.state != CircuitState.OPEN:
                cb.state = CircuitState.OPEN
                cb.last_state_change = time.time()
                logger.warning(f"[LLMRouter] {key} 熔断器触发 OPEN (失败 {cb.failure_count} 次)")

        # 更新统计
        self._total_requests += 1
        self._total_latency += latency
        if success:
            self._total_success += 1
        else:
            self._total_failure += 1

    def route(self, prompt: str, context: Dict = None, timeout: float = None) -> Dict:
        """路由 LLM 请求 — Semaphore(2) 反压包装 (饱和 >25s → 规则引擎降级)."""
        _t0 = _time.time()
        if self._budget.is_exhausted():
            # 09-20 预算门: 耗尽 → 不发起任何 LLM 调用, 直落规则引擎 (非断链)
            logger.warning(
                f"[LLMRouter] token 预算耗尽 (limit={_LLM_BUDGET_LIMIT}), 降级规则引擎")
            self._record_result('rule_engine', True, 0.0)
            return {
                'success': True,
                'content': json.dumps(self._rule_engine.decide(context or {}),
                                      ensure_ascii=False),
                'provider': 'rule_engine', 'latency': 0.0, 'fallback': True,
            }
        if not _LLM_SEMAPHORE.acquire(timeout=_SEM_ACQUIRE_TIMEOUT):
            latency = _time.time() - _t0
            logger.warning(
                f"[LLMRouter] LLM 并发饱和 (semaphore 25s 超时), 降级规则引擎 ({latency:.1f}s)")
            self._record_result('rule_engine', True, latency)
            return {
                'success': True,
                'content': json.dumps(self._rule_engine.decide({}), ensure_ascii=False),
                'provider': 'rule_engine', 'latency': round(latency, 3), 'fallback': True,
            }
        try:
            return self._route_locked(prompt, context, timeout or 60.0)  # 2026-09-15: 25→60, 8080 实测 17s+ 辩论常超时
        finally:
            _LLM_SEMAPHORE.release()

    def _route_locked(self, prompt: str, context: Dict, timeout: float = 60.0) -> Dict:
        """
        路由 LLM 请求 — 按优先级尝试各 Provider

        Args:
            prompt: 用户提示
            context: 上下文数据 (用于规则引擎)

        Returns:
            {success, content, provider, latency, fallback}
        """
        start_time = time.time()

        # 按优先级排序 Provider
        priority_order = [
            LLMProvider.OMLX,
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.OLLAMA,
        ]

        last_error = ""

        for provider in priority_order:
            key = provider.value
            if not self._can_use_provider(key):
                continue

            config = self._configs[key]
            try:
                # 根据 Provider 类型选择客户端
                if provider == LLMProvider.OMLX:
                    result = OmlxClient.chat(
                        config.url, prompt, config.model,
                        config.api_key, timeout or config.timeout,
                    )
                elif provider == LLMProvider.OPENAI:
                    result = OpenAIClient.chat(
                        config.url, prompt, config.model,
                        config.api_key, timeout or config.timeout,
                    )
                elif provider == LLMProvider.ANTHROPIC:
                    # Anthropic 暂不支持 (需要 anthropic SDK)
                    result = {'success': False, 'error': 'Anthropic not implemented'}
                elif provider == LLMProvider.OLLAMA:
                    result = OllamaClient.chat(
                        config.url, prompt, config.model,
                        timeout or config.timeout,
                    )
                else:
                    result = {'success': False, 'error': f'Unknown provider: {provider}'}

                latency = time.time() - start_time

                if result.get('success'):
                    # 09-20: 成功响应带 usage → token 入账 (无 usage 键则 0 计入调用数)
                    _u = result.get('usage') or {}
                    self._budget.record(key, _u.get('prompt_tokens', 0),
                                        _u.get('completion_tokens', 0))
                    self._record_result(key, True, latency)
                    logger.info(f"[LLMRouter] {key} 成功 ({latency:.1f}s)")
                    return {
                        'success': True,
                        'content': result.get('content', ''),
                        'provider': key,
                        'latency': round(latency, 3),
                        'fallback': False,
                    }
                else:
                    last_error = result.get('error', 'Unknown error')
                    self._record_result(key, False, latency, last_error)
                    logger.warning(f"[LLMRouter] {key} 失败: {last_error}")

            except Exception as e:
                latency = time.time() - start_time
                last_error = str(e)
                self._record_result(key, False, latency, last_error)
                logger.warning(f"[LLMRouter] {key} 异常: {e}")

        # 所有 LLM 都失败了, 使用规则引擎降级
        latency = time.time() - start_time
        stock_data = context or {}
        fallback_result = self._rule_engine.decide(stock_data)

        self._record_result("rule_engine", True, latency)
        logger.info(f"[LLMRouter] 降级到规则引擎 ({latency:.1f}s)")

        return {
            'success': True,
            'content': json.dumps(fallback_result, ensure_ascii=False),
            'provider': 'rule_engine',
            'latency': round(latency, 3),
            'fallback': True,
        }

    def health_check(self, provider: LLMProvider = None) -> Dict:
        """
        健康检查

        Args:
            provider: 指定 Provider, None 则检查所有

        Returns:
            {provider: {healthy, latency, error}}
        """
        results = {}

        providers = [provider] if provider else list(LLMProvider)

        for p in providers:
            key = p.value
            if key not in self._configs:
                results[key] = {'healthy': False, 'error': 'Not configured'}
                continue

            config = self._configs[key]
            start = time.time()

            try:
                if p == LLMProvider.OMLX:
                    result = OmlxClient.chat(config.url, "Hello", config.model,
                                           config.api_key, 3.0)
                elif p == LLMProvider.OPENAI:
                    result = OpenAIClient.chat(config.url, "Hello", config.model,
                                              config.api_key, 3.0)
                elif p == LLMProvider.OLLAMA:
                    result = OllamaClient.chat(config.url, "Hello", config.model, 3.0)
                else:
                    result = {'success': False, 'error': 'Not implemented'}

                latency = time.time() - start
                results[key] = {
                    'healthy': result.get('success', False),
                    'latency': round(latency, 3),
                    'error': '' if result.get('success') else result.get('error', ''),
                }

            except Exception as e:
                latency = time.time() - start
                results[key] = {
                    'healthy': False,
                    'latency': round(latency, 3),
                    'error': str(e),
                }

        return results

    def get_status(self) -> Dict:
        """获取路由状态"""
        # 当前活跃 Provider
        active = None
        for p in [LLMProvider.OMLX, LLMProvider.OPENAI, LLMProvider.OLLAMA]:
            key = p.value
            if self._can_use_provider(key):
                active = key
                break

        if not active:
            active = "rule_engine"

        # 平均延迟
        avg_latency = (self._total_latency / self._total_requests
                       if self._total_requests > 0 else 0)

        # 最近失败
        recent_failures = [
            {
                'provider': r.provider,
                'error': r.error,
                'time': r.timestamp,
            }
            for r in self._records[-10:]
            if not r.success
        ]

        # 熔断器状态
        cb_status = {}
        for key, cb in self._circuit_breakers.items():
            cb_status[key] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count,
                'last_failure': cb.last_failure_time,
            }

        return {
            'active_provider': active,
            'total_requests': self._total_requests,
            'total_success': self._total_success,
            'total_failure': self._total_failure,
            'success_rate': round(
                self._total_success / max(self._total_requests, 1), 4),
            'avg_latency': round(avg_latency, 3),
            'providers': {
                key: {
                    'name': c.name,
                    'enabled': c.enabled,
                    'url': c.url,
                    'model': c.model,
                }
                for key, c in self._configs.items()
            },
            'recent_failures': recent_failures,
            'circuit_breakers': cb_status,
            'budget': self._budget.stats(),   # 09-20: token 台账 (观测非拦截, 默认)
        }


# ── 全局单例 ───────────────────────────────────────────────────────

llm_router = LLMRouter()
