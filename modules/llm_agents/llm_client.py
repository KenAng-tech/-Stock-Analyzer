#!/usr/bin/env python3
"""
LLM Client - 多模型客户端

支持多种 LLM 后端:
- omlx (本地 Qwen3.6-35B 服务)
- OpenAI (GPT-5.5, GPT-5.4)
- Anthropic (Claude 4.6)
- Google (Gemini 3.1)
- 本地 Ollama

SOTA Reference:
- TradingAgents v0.2.5: GPT-5.5, Gemini 3.1, Claude 4.6, Qwen, GLM, MiniMax
- RETuning: Inference-Time Scaling for Stock Movement Prediction
"""

import json
import os
import time
import http.client
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from modules.logger import logger


@dataclass
class LLMResponse:
    """LLM 响应数据结构"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    timestamp: float = field(default_factory=time.time)


class OmlxClient:
    """本地 omlx 服务客户端 - Qwen3.6-35B"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "default"):
        self.base_url = base_url
        self.model = model
        
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        """通过 HTTP 连接调用 omlx 服务"""
        try:
            # omlx 使用 OpenAI 兼容 API
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            conn = http.client.HTTPConnection(self.base_url.split("://")[1], timeout=30)
            conn.request("POST", "/v1/chat/completions", json.dumps(body), 
                        {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = json.loads(response.read().decode())
            conn.close()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                usage={"prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                       "completion_tokens": data.get("usage", {}).get("completion_tokens", 0)}
            )
        except Exception as e:
            logger.error(f"omlx client error: {e}")
            return LLMResponse(content="Error connecting to omlx service", model=self.model)


class OpenAIClient:
    """OpenAI 客户端 - GPT-5.5, GPT-5.4"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5.5"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage={"prompt_tokens": response.usage.prompt_tokens,
                       "completion_tokens": response.usage.completion_tokens}
            )
        except Exception as e:
            logger.error(f"OpenAI client error: {e}")
            return LLMResponse(content="OpenAI error", model=self.model)


class AnthropicClient:
    """Anthropic 客户端 - Claude 4.6"""
    
    def __init__(self, api_key: str = None, model: str = "claude-4-6-opus"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else None
            user_messages = messages[1:] if system_msg else messages
            
            response = client.messages.create(
                model=self.model,
                system=system_msg,
                messages=[{"role": m["role"], "content": m["content"]} for m in user_messages],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0}
            )
        except Exception as e:
            logger.error(f"Anthropic client error: {e}")
            return LLMResponse(content="Anthropic error", model=self.model)


class LLMClient:
    """统一 LLM 客户端 - 自动选择最优模型"""
    
    # 模型优先级配置 (参考 TradingAgents v0.2.5)
    MODEL_PRIORITY = {
        "primary": "omlx",      # 本地 omlx (Qwen3.6-35B)
        "fallback": ["openai", "anthropic", "ollama"]
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.clients = {}
        self._init_clients()
        
    def _init_clients(self):
        """初始化所有客户端"""
        # omlx (本地)
        self.clients["omlx"] = OmlxClient(
            base_url=self.config.get("omlx_url", "http://127.0.0.1:8080"),
            model=self.config.get("omlx_model", "default")
        )
        
        # OpenAI
        if self.config.get("openai_key"):
            self.clients["openai"] = OpenAIClient(
                api_key=self.config["openai_key"],
                model=self.config.get("openai_model", "gpt-5.5")
            )
        
        # Anthropic
        if self.config.get("anthropic_key"):
            self.clients["anthropic"] = AnthropicClient(
                api_key=self.config["anthropic_key"],
                model=self.config.get("anthropic_model", "claude-4-6-opus")
            )
            
    def get_response(self, messages: List[Dict], temperature: float = 0.7, 
                     max_tokens: int = 2048, model: str = None) -> LLMResponse:
        """获取 LLM 响应，自动故障转移"""
        target_model = model or self.MODEL_PRIORITY["primary"]
        
        # 尝试主模型
        try:
            if target_model in self.clients:
                return self.clients[target_model].chat(messages, temperature, max_tokens)
        except Exception as e:
            logger.warning(f"Primary model {target_model} failed: {e}")
        
        # 故障转移到 fallback
        for fallback in self.MODEL_PRIORITY["fallback"]:
            if fallback in self.clients:
                logger.info(f"Fallback to {fallback}")
                return self.clients[fallback].chat(messages, temperature, max_tokens)
        
        return LLMResponse(content="All models unavailable", model="unknown")
    
    def get_client_for_model(self, model_name: str):
        """根据模型名称获取对应客户端"""
        if model_name in self.clients:
            return self.clients[model_name]
        return self.clients.get("omlx")
