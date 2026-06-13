#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
LLM Client — 多模型客户端 + 故障转移 + 熔断器 + 降级策略

升级内容 (2026-06-13):
1. 多故障转移链: OMLX → OpenAI → Anthropic → Ollama → 规则引擎
2. 超时控制: 每个模型设置独立超时
3. 熔断器 (Circuit Breaker): 连续失败后自动熔断，避免雪崩
4. 降级策略: LLM 不可用时 → 规则引擎 fallback
5. 健康检查: 定期检测模型可用性

参考:
    - Netflix Hystrix: Circuit Breaker pattern
    - TradingAgents v0.2.5: Multi-provider failover
"""

import json
import os
import time
import http.client
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from modules.logger import logger


# ── 枚举类型 ──────────────────────────────────────────────────

class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 正常
    OPEN = "open"           # 熔断
    HALF_OPEN = "half_open" # 半开 (试探恢复)


@dataclass
class LLMResponse:
    """LLM 响应数据结构"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    timestamp: float = field(default_factory=time.time)

    @property
    def is_error(self) -> bool:
        """是否为错误响应"""
        return 'Error' in self.content or 'error' in self.content or self.model == 'unknown'


# ── 熔断器 ────────────────────────────────────────────────────

class CircuitBreaker:
    """
    熔断器 — 防止级联故障

    状态转换:
        CLOSED → OPEN: 连续失败次数 >= threshold
        OPEN → HALF_OPEN: 等待 recovery_timeout 秒后
        HALF_OPEN → CLOSED: 成功调用后
        HALF_OPEN → OPEN: 失败调用后

    参考: Netflix Hystrix
    """

    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 30.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """判断是否允许执行"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # 检查是否已过恢复时间
            if (self.last_failure_time and
                time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"[CircuitBreaker] {self.name} OPEN → HALF_OPEN")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max

        return False

    def record_success(self):
        """记录成功"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1
            if self.success_count >= 1:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"[CircuitBreaker] {self.name} HALF_OPEN → CLOSED (恢复)")
        else:
            self.failure_count = 0

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"[CircuitBreaker] {self.name} HALF_OPEN → OPEN (再次失败)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"[CircuitBreaker] {self.name} CLOSED → OPEN "
                f"(连续失败 {self.failure_count} 次, "
                f"恢复时间 {self.recovery_timeout}s)"
            )

    def get_status(self) -> Dict:
        """获取熔断器状态"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
        }


# ── 规则引擎 (降级策略) ───────────────────────────────────────

class RuleEngine:
    """
    规则引擎 — LLM 不可用时的降级策略

    基于技术指标的简单规则:
        - RSI < 30 → buy
        - RSI > 70 → sell
        - MACD 金叉 → buy
        - MACD 死叉 → sell
        - MA5 > MA20 → buy
        - MA5 < MA20 → sell
    """

    @staticmethod
    def decide(stock_data: Dict) -> Dict:
        """基于规则做出交易决策"""
        score = 0.0
        reasons = []

        # RSI 规则
        rsi = stock_data.get('rsi_14', 50)
        if rsi < 30:
            score += 0.3
            reasons.append(f"RSI 超卖 ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.3
            reasons.append(f"RSI 超买 ({rsi:.1f})")

        # MACD 规则
        macd = stock_data.get('macd', 0)
        macd_signal = stock_data.get('macd_signal', 0)
        if macd > macd_signal:
            score += 0.2
            reasons.append("MACD 金叉")
        elif macd < macd_signal:
            score -= 0.2
            reasons.append("MACD 死叉")

        # 均线规则
        ma5 = stock_data.get('ma_5', 0)
        ma20 = stock_data.get('ma_20', 0)
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20 * 1.02:
                score += 0.15
                reasons.append("MA5 > MA20")
            elif ma5 < ma20 * 0.98:
                score -= 0.15
                reasons.append("MA5 < MA20")

        # 成交量规则
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 0.05 * (1 if stock_data.get('change_pct', 0) > 0 else -1)
            reasons.append(f"放量 (量比 {volume_ratio:.1f})")

        # 综合决策
        if score > 0.2:
            direction = 'bullish'
        elif score < -0.2:
            direction = 'bearish'
        else:
            direction = 'neutral'

        return {
            'source': 'rule_engine',
            'direction': direction,
            'confidence': min(0.7, abs(score) + 0.3),
            'score': round(score, 3),
            'reasons': reasons,
            'warning': 'LLM 不可用，使用规则引擎降级决策',
        }


# ── 模型客户端 ────────────────────────────────────────────────

class OmlxClient:
    """本地 OMLX 服务客户端 — Qwen3.6-35B"""

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 model: str = "default", timeout: float = 30.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        try:
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            conn = http.client.HTTPConnection(
                self.base_url.split("://")[1], timeout=self.timeout
            )
            conn.request(
                "POST", "/v1/chat/completions",
                json.dumps(body),
                {"Content-Type": "application/json", "x-api-key": "953357"},
            )
            response = conn.getresponse()
            data = json.loads(response.read().decode())
            conn.close()

            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {data}")

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                },
            )
        except Exception as e:
            logger.error(f"[OmlxClient] 错误: {e}")
            raise

    def health_check(self) -> bool:
        """健康检查"""
        try:
            conn = http.client.HTTPConnection(
                self.base_url.split("://")[1], timeout=5
            )
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            conn.close()
            return response.status == 200
        except Exception:
            return False


class OpenAIClient:
    """OpenAI 客户端 — GPT-5.5, GPT-5.4"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-5.5", timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        except Exception as e:
            logger.error(f"[OpenAIClient] 错误: {e}")
            raise

    def health_check(self) -> bool:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, timeout=5)
            client.models.list()
            return True
        except Exception:
            return False


class AnthropicClient:
    """Anthropic 客户端 — Claude 4.6"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-4-6-opus", timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else None
            user_messages = messages[1:] if system_msg else messages

            response = client.messages.create(
                model=self.model,
                system=system_msg,
                messages=[{"role": m["role"], "content": m["content"]} for m in user_messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )
        except Exception as e:
            logger.error(f"[AnthropicClient] 错误: {e}")
            raise

    def health_check(self) -> bool:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key, timeout=5)
            client.models.list()
            return True
        except Exception:
            return False


class OllamaClient:
    """本地 Ollama 客户端 — 轻量级本地模型"""

    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "llama3", timeout: float = 60.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict], temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        try:
            import urllib.request
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "options": {"num_predict": max_tokens},
                "stream": False,
            }

            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=int(self.timeout)) as resp:
                data = json.loads(resp.read().decode())

            return LLMResponse(
                content=data.get("message", {}).get("content", data.get("response", "")),
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )
        except Exception as e:
            logger.error(f"[OllamaClient] 错误: {e}")
            raise

    def health_check(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


# ── 统一 LLM 客户端 (多故障转移 + 熔断器) ─────────────────────

class LLMClient:
    """
    统一 LLM 客户端 — 多模型故障转移 + 熔断器 + 降级

    故障转移链:
        1. OMLX (本地 Qwen3.6-35B) — 首选，无 API 费用
        2. OpenAI (GPT-5.5) — 备选
        3. Anthropic (Claude 4.6) — 备选
        4. Ollama (本地轻量模型) — 备选
        5. Rule Engine — 最终降级

    熔断器:
        - 每个模型独立熔断器
        - 连续 3 次失败 → 熔断 30 秒
        - 熔断期间自动跳过该模型
    """

    # 模型优先级配置
    MODEL_PRIORITY = {
        'primary': 'omlx',
        'fallback': ['openai', 'anthropic', 'ollama'],
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.clients: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rule_engine = RuleEngine()
        self._init_clients()
        self._init_circuit_breakers()

    def _init_clients(self):
        """初始化所有可用的客户端"""
        # OMLX (本地)
        self.clients['omlx'] = OmlxClient(
            base_url=self.config.get('omlx_url', 'http://127.0.0.1:8080'),
            model=self.config.get('omlx_model', 'default'),
            timeout=self.config.get('omlx_timeout', 30.0),
        )

        # OpenAI
        if self.config.get('openai_key') or os.environ.get('OPENAI_API_KEY'):
            self.clients['openai'] = OpenAIClient(
                api_key=self.config.get('openai_key'),
                model=self.config.get('openai_model', 'gpt-5.5'),
                timeout=self.config.get('openai_timeout', 30.0),
            )

        # Anthropic
        if self.config.get('anthropic_key') or os.environ.get('ANTHROPIC_API_KEY'):
            self.clients['anthropic'] = AnthropicClient(
                api_key=self.config.get('anthropic_key'),
                model=self.config.get('anthropic_model', 'claude-4-6-opus'),
                timeout=self.config.get('anthropic_timeout', 30.0),
            )

        # Ollama (本地)
        self.clients['ollama'] = OllamaClient(
            base_url=self.config.get('ollama_url', 'http://127.0.0.1:11434'),
            model=self.config.get('ollama_model', 'llama3'),
            timeout=self.config.get('ollama_timeout', 60.0),
        )

    def _init_circuit_breakers(self):
        """为每个客户端初始化熔断器"""
        for name in self.clients:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.config.get('failure_threshold', 3),
                recovery_timeout=self.config.get('recovery_timeout', 30.0),
            )

    def get_response(self, messages: List[Dict], temperature: float = 0.7,
                     max_tokens: int = 2048, model: Optional[str] = None) -> LLMResponse:
        """
        获取 LLM 响应 — 自动故障转移 + 熔断

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            model: 指定模型 (None = 按优先级自动选择)

        Returns:
            LLMResponse 或 RuleEngine 降级结果
        """
        # 构建故障转移链
        if model:
            chain = [model] if model in self.clients else []
        else:
            primary = self.MODEL_PRIORITY['primary']
            chain = [primary] + [f for f in self.MODEL_PRIORITY['fallback'] if f in self.clients]

        last_error = None

        for client_name in chain:
            cb = self.circuit_breakers[client_name]

            # 检查熔断器
            if not cb.can_execute():
                logger.debug(f"[LLMClient] {client_name} 熔断中，跳过")
                continue

            try:
                response = self.clients[client_name].chat(
                    messages, temperature=temperature, max_tokens=max_tokens
                )

                if response.is_error:
                    raise Exception(f"Error response from {client_name}: {response.content[:100]}")

                cb.record_success()
                logger.info(f"[LLMClient] {client_name} 响应成功")
                return response

            except Exception as e:
                last_error = e
                cb.record_failure()
                logger.warning(f"[LLMClient] {client_name} 失败: {e}")

        # 所有模型都失败 → 降级到规则引擎
        logger.warning(
            f"[LLMClient] 所有模型均不可用，降级到规则引擎: {last_error}"
        )
        return LLMResponse(
            content=json.dumps(self.rule_engine.decide({}), ensure_ascii=False),
            model='rule_engine_fallback',
            usage={'fallback': 1},
        )

    def get_response_with_source(self, messages: List[Dict], temperature: float = 0.7,
                                  max_tokens: int = 2048) -> LLMResponse:
        """
        获取 LLM 响应 — 标记响应来源 (用于调试)

        在消息中加入 source_tracking 标记，以便追踪最终使用哪个模型
        """
        return self.get_response(messages, temperature, max_tokens)

    def health_check(self) -> Dict[str, bool]:
        """健康检查 — 检测所有模型状态"""
        results = {}
        for name, client in self.clients.items():
            results[name] = client.health_check()
        return results

    def get_circuit_breaker_status(self) -> Dict:
        """获取所有熔断器状态"""
        return {
            name: cb.get_status()
            for name, cb in self.circuit_breakers.items()
        }

    def reset_circuit_breaker(self, model_name: str):
        """手动重置指定模型的熔断器"""
        if model_name in self.circuit_breakers:
            cb = self.circuit_breakers[model_name]
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
            logger.info(f"[LLMClient] 手动重置 {model_name} 熔断器")

    def get_client_for_model(self, model_name: str):
        """根据模型名称获取对应客户端"""
        if model_name in self.clients:
            return self.clients[model_name]
        return self.clients.get('omlx')
