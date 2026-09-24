#!/usr/bin/env python3
"""
Factor Mining Module - 基于 AlphaCrafter 和 QuantaAlpha 的因子挖掘

SOTA Reference:
- AlphaCrafter (NJU, 2026-05): 连续自适应因子发现
- QuantaAlpha (SUFE, 2026-02): LLM 驱动的自进化因子挖掘
- FactorMiner (THU, 2026-02): 自进化 Agent 金融 Alpha 发现
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modules.logger import logger

# LLMClient 可能不存在（llm_agents/ 目录可能缺失）
try:
    from modules.llm_agents.llm_client import LLMClient, extract_json
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    LLMClient = None  # type: ignore
    extract_json = None  # type: ignore
    logger.warning("[FactorMining] LLMClient 不可用，将使用规则引擎降级")


@dataclass
class Factor:
    """因子数据结构"""
    name: str
    description: str
    value: float
    ic: float  # Information Coefficient
    icir: float  # IC Information Ratio
    efficacy: float  # 因子有效性 0-1
    market_regime: str  # 适用市场状态
    timestamp: float = field(default_factory=time.time)


class FactorMiner:
    """因子挖掘器 - 基于 LLM 的因子发现"""
    
    PROMPT_TEMPLATE = """你是专业的量化因子挖掘专家。基于市场数据和历史表现，发现新的 alpha 因子。

【当前市场数据】
股票: {stock_name} ({stock_code})
市场状态: {market_regime}
波动率: {volatility}
趋势: {trend}

【现有因子表现】
{existing_factors}

【数据挖掘要求】
1. 分析价格、成交量、技术指标的潜在模式
2. 发现新的因子组合或改进现有因子
3. 评估因子的市场状态适应性
4. 给出因子有效性预测

请以 JSON 格式输出:
{{
    "new_factors": [
        {{"name": "因子名称", "description": "描述", "formula": "计算公式"}}
    ],
    "improved_factors": [
        {{"name": "因子名称", "improvement": "改进点", "expected_ic_improvement": 0.0-1.0}}
    ],
    "regime_dependency": "牛市/熊市/震荡",
    "confidence": 0.0-1.0
}}"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.factors = {}

    def mine_factors(self, stock_data: Dict, existing_factors: List[str]) -> List[Factor]:
        """挖掘新因子 — LLM 或规则引擎降级"""
        # 无 LLM → 规则引擎降级
        if self.llm is None:
            return self._rule_mine_factors(stock_data, existing_factors)

        try:
            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", ""),
                market_regime=stock_data.get("market_regime", "neutral"),
                volatility=stock_data.get("volatility", "--"),
                trend=stock_data.get("trend", "neutral"),
                existing_factors="\n".join(existing_factors) if existing_factors else "无"
            )

            response = self.llm.get_response([{"role": "user", "content": prompt}])

            try:
                # 2026-09-09: extract_json (剥 ``` 围栏 + 语言标签 + 大括号提取) 替代手写
                # split 剥壳 — 旧逻辑剥壳后残留 "json" 语言标签 → json.loads 恒失败
                data = extract_json(response.content)
                # 如果 JSON 解析成功但返回的是交易决策 (RuleEngine 降级)，使用默认因子
                if not data or "new_factors" not in data:
                    data = {
                        "new_factors": [
                            {"name": "量价动量因子 (VPM)", "description": "结合价格动量与成交量变化", "formula": "VPM = Return * Volume_Ratio"},
                            {"name": "多周期趋势因子 (MPTC)", "description": "多周期趋势一致性", "formula": "MPTC = Sum(Trend_i) / N"},
                            {"name": "波动率调整动量 (VAM)", "description": "波动率调整后的动量", "formula": "VAM = Return / Volatility"},
                        ],
                        "improved_factors": [],
                        "regime_dependency": stock_data.get("market_regime", "neutral"),
                        "confidence": 0.6
                    }
            except Exception:
                # JSON 解析失败，使用默认因子
                data = {
                    "new_factors": [
                        {"name": "量价动量因子 (VPM)", "description": "结合价格动量与成交量变化", "formula": "VPM = Return * Volume_Ratio"},
                        {"name": "多周期趋势因子 (MPTC)", "description": "多周期趋势一致性", "formula": "MPTC = Sum(Trend_i) / N"},
                        {"name": "波动率调整动量 (VAM)", "description": "波动率调整后的动量", "formula": "VAM = Return / Volatility"},
                    ],
                    "improved_factors": [],
                    "regime_dependency": stock_data.get("market_regime", "neutral"),
                    "confidence": 0.6
                }

            factors = []
            for f in data.get("new_factors", []):
                factor = Factor(
                    name=f.get("name", ""),
                    description=f.get("description", ""),
                    value=0.0,
                    ic=0.0,
                    icir=0.0,
                    efficacy=data.get("confidence", 0.5),
                    market_regime=data.get("regime_dependency", "neutral")
                )
                factors.append(factor)

            return factors
        except Exception as e:
            logger.error(f"FactorMiner error: {e}")
            return []

    def _rule_mine_factors(self, stock_data: Dict, existing_factors: List[str]) -> List[Factor]:
        """规则引擎降级 — 基于启发式规则生成因子"""
        logger.debug("[FactorMining] 使用规则引擎生成因子 (LLM 不可用)")
        regime = stock_data.get("market_regime", "neutral")
        volatility = stock_data.get("volatility", "medium")

        # 根据市场状态和波动率选择因子
        if regime == "bull" and volatility == "low":
            candidates = [
                Factor(name="动量突破因子", description="趋势跟踪动量", value=0.7, ic=0.15, icir=1.2, efficacy=0.7, market_regime="bull"),
                Factor(name="成交量放大因子", description="量价齐升", value=0.6, ic=0.12, icir=1.0, efficacy=0.6, market_regime="bull"),
            ]
        elif regime == "bear":
            candidates = [
                Factor(name="低波动防御因子", description="低波防御", value=0.5, ic=0.08, icir=0.8, efficacy=0.5, market_regime="bear"),
                Factor(name="高股息因子", description="防御性选股", value=0.4, ic=0.06, icir=0.6, efficacy=0.4, market_regime="bear"),
            ]
        else:
            # 中性/震荡市
            candidates = [
                Factor(name="均值回归因子", description="震荡市均值回归", value=0.6, ic=0.10, icir=0.9, efficacy=0.6, market_regime="sideways"),
                Factor(name="波动率突破因子", description="波动率扩张突破", value=0.5, ic=0.09, icir=0.8, efficacy=0.5, market_regime="sideways"),
            ]

        return candidates[:3]


class FactorEvaluator:
    """因子评估器 - 评估因子有效性"""
    
    def calculate_ic(self, factor_values: np.ndarray, returns: np.ndarray) -> float:
        """计算 Information Coefficient"""
        if len(factor_values) != len(returns):
            return 0.0
        return np.corrcoef(factor_values, returns)[0, 1]
    
    def calculate_icir(self, factor_values: np.ndarray, returns: np.ndarray) -> float:
        """计算 IC Information Ratio"""
        ic = self.calculate_ic(factor_values, returns)
        if np.std(returns) == 0:
            return 0.0
        return ic / np.std(returns)
    
    def evaluate_factor(self, factor_name: str, factor_values: np.ndarray, 
                       returns: np.ndarray) -> Factor:
        """评估单个因子"""
        ic = self.calculate_ic(factor_values, returns)
        icir = self.calculate_icir(factor_values, returns)
        
        return Factor(
            name=factor_name,
            description=f"因子: {factor_name}",
            value=float(np.mean(factor_values)),
            ic=float(ic),
            icir=float(icir),
            efficacy=abs(ic),
            market_regime="neutral"
        )


class FactorMiningEngine:
    """因子挖掘引擎 - 整合 LLM 挖掘和传统评估"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.factor_miner = FactorMiner(llm_client)
        self.factor_evaluator = FactorEvaluator()
        self.factor_history = {}
    
    def mine_and_evaluate(self, stock_data: Dict, factor_data: Dict) -> List[Factor]:
        """挖掘并评估因子"""
        # LLM 因子挖掘
        llm_factors = self.factor_miner.mine_factors(stock_data, [])
        
        # 传统因子评估
        if "factor_values" in factor_data and "returns" in factor_data:
            for name, values in factor_data["factor_values"].items():
                factor = self.factor_evaluator.evaluate_factor(
                    name, np.array(values), np.array(factor_data["returns"])
                )
                llm_factors.append(factor)
        
        return llm_factors
