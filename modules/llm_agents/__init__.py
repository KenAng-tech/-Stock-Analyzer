"""
LLM Multi-Agent Layer - 基于 TradingAgents 的多 Agent 决策层

架构:
    Market Data ──→ Analyst Team ──→ Research Team ──→ Trader ──→ Risk Manager ──→ Portfolio Manager
                    │                    │                  │              │
                  Fundamental        Bull/Bear          Entry/Exit     Position Sizing
                  Sentiment          Debate             Timing         Exposure Limits
                  Technical          Synthesis        Decision         Risk Limits
                  News               Manager

SOTA Reference:
    - TradingAgents v0.2.5 (Tauric Research, 2026-05) - 44,000+ Stars
    - AlphaCrafter (NJU, 2026-05) - Full-Stack Multi-Agent Framework
    - QuantAgent (SBU, 2025-09) - Price-Driven HFT

模型支持:
    - Qwen3.6-35B (本地 omlx 服务)
    - GPT-5.5 / GPT-5.4 (OpenAI)
    - Claude 4.6 (Anthropic)
    - Gemini 3.1 (Google)
    - 本地 Ollama
"""

from .analyst_team import FundAnalyst, SentimentAnalyst, TechnicalAnalyst, NewsAnalyst
from .research_team import BullResearcher, BearResearcher, ResearchManager
from .trader_agent import TraderAgent
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .agent_coordinator import AgentCoordinator

__all__ = [
    'FundAnalyst', 'SentimentAnalyst', 'TechnicalAnalyst', 'NewsAnalyst',
    'BullResearcher', 'BearResearcher', 'ResearchManager',
    'TraderAgent', 'RiskManager', 'PortfolioManager',
    'AgentCoordinator'
]
