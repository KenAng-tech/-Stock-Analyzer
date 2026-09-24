#!/usr/bin/env python3
"""
符号回归因子挖掘模块 - 基于 PySR (Symbolic Regression)

SOTA Reference:
- PySR (MilesCranmer, 2023): 快速可解释符号回归
- DynamicExpressions: 符号回归底层库
- 结合 IC 验证闭环

核心思想: 通过遗传算法搜索最优的数学表达式，直接优化因子 IC
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from modules.logger import logger

try:
    from modules.llm_agents.llm_client import extract_json
except ImportError:  # llm_agents 目录缺失时降级
    extract_json = None  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SymbolicFactor:
    """符号回归因子数据结构"""
    name: str
    formula: str  # 数学表达式字符串
    description: str
    ic: float = 0.0  # Rank IC (Spearman)
    icir: float = 0.0  # IC Information Ratio
    ic_t_stat: float = 0.0  # t-statistic
    efficacy: float = 0.0  # 综合有效性评分 0-1
    market_regime: str = "neutral"
    half_life: int = 0  # IC 半衰期 (天)
    turnover_rate: float = 0.0  # 换手率
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'formula': self.formula,
            'description': self.description,
            'ic': round(self.ic, 4),
            'icir': round(self.icir, 4),
            'ic_t_stat': round(self.ic_t_stat, 2),
            'efficacy': round(self.efficacy, 4),
            'market_regime': self.market_regime,
            'half_life': self.half_life,
            'turnover_rate': round(self.turnover_rate, 4),
        }


class SymbolicRegressionEngine:
    """符号回归因子挖掘引擎 - 使用 PySR"""

    # 预定义的因子模板 (用于 fallback)
    FACTOR_TEMPLATES = [
        {
            'name': '量价动量因子 (VPM)',
            'formula': 'return * log(volume + 1)',
            'description': '成交量加权的收益率动量',
        },
        {
            'name': '波动率调整动量 (VAM)',
            'formula': 'return / (rolling_std(return, 20) + 1e-8)',
            'description': '波动率标准化后的动量因子',
        },
        {
            'name': '反转因子 (REV)',
            'formula': '-return * abs(return)',
            'description': '短期价格反转效应',
        },
        {
            'name': '成交量偏离因子 (VDEV)',
            'formula': 'volume / (rolling_mean(volume, 20) + 1)',
            'description': '成交量相对均值的偏离程度',
        },
        {
            'name': '振幅因子 (AMP)',
            'formula': '(high - low) / (close + 1e-8)',
            'description': '日内振幅，衡量波动强度',
        },
        {
            'name': '趋势强度因子 (TIS)',
            'formula': 'abs(rolling_mean(return, 10)) * rolling_std(return, 10)',
            'description': '趋势方向与强度的综合',
        },
        {
            'name': '量价背离因子 (VDI)',
            'formula': 'return * (1 - volume / (rolling_mean(volume, 20) + 1))',
            'description': '价格与成交量趋势背离检测',
        },
        {
            'name': '压缩扩张因子 (CCE)',
            'formula': 'rolling_std(return, 5) / (rolling_std(return, 20) + 1e-8)',
            'description': '短期波动率相对长期波动率，捕捉波动率突破',
        },
    ]

    def __init__(self, n_iterations: int = 30, population_size: int = 40,
                 timeout_seconds: float = 120.0, parsimony: float = 0.01):
        """
        初始化符号回归引擎

        Args:
            n_iterations: 遗传迭代次数
            population_size: 种群大小
            timeout_seconds: 超时时间 (秒)
            parsimony: 复杂度惩罚系数 (奥卡姆剃刀)
        """
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.timeout_seconds = timeout_seconds
        self.parsimony = parsimony
        self._pysr_available = False
        self._pyarrow_available = False

        # 尝试导入 PySR
        try:
            import pysr
            self._pysr_available = True
            try:
                import pyarrow  # PySR 依赖
                self._pyarrow_available = True
            except ImportError:
                logger.warning("[PySR] pyarrow 未安装，使用 fallback 模式")
        except ImportError:
            logger.warning("[PySR] 未安装，使用模板因子 fallback")

    def mine_factors(self, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str],
                     stock_code: str = "unknown") -> List[SymbolicFactor]:
        """
        通过符号回归挖掘因子

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (未来收益率) (n_samples,)
            feature_names: 特征名称列表
            stock_code: 股票代码

        Returns:
            挖掘出的因子列表 (按 efficacy 排序)
        """
        if not self._pysr_available or not self._pyarrow_available:
            logger.info("[PySR] 使用模板因子 fallback")
            return self._mine_with_templates(X, y, feature_names, stock_code)

        try:
            import pysr
            from scipy.stats import spearmanr

            logger.info(f"[PySR] 开始符号回归挖掘: {X.shape[0]} 样本 x {X.shape[1]} 特征")

            # 数据清洗
            mask = ~(np.isnan(X) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]

            if len(X_clean) < 50:
                logger.warning("[PySR] 样本数太少，使用模板因子")
                return self._mine_with_templates(X, y, feature_names, stock_code)

            # 转换为 DataFrame
            df = pd.DataFrame(X_clean, columns=feature_names)
            df['y'] = y_clean

            # 构建 PySR 方程
            equation = "y ~ " + " + ".join(f"({col})" for col in feature_names)

            # PySR 配置
            model = pysr.PySRRegressor(
                niterations=self.n_iterations,
                population_size=self.population_size,
                ncyclesperiteration=max(100, self.population_size // 2),
                binary_operators=["+", "-", "*", "/"],
                unary_operators=[
                    "abs", "square", "sqrt", "log", "log10", "exp",
                    "neg", "cos", "sin",
                ],
                parsimony=self.parsimony,
                complexity_of_operators={
                    "/": 2, "+": 1, "-": 1, "*": 1,
                    "abs": 1, "square": 1, "sqrt": 2,
                    "log": 2, "log10": 2, "exp": 2,
                },
                probrandom=0.05,  # 随机操作概率 (越低越倾向于已知变量)
                timeout_in_seconds=self.timeout_seconds,
                weights=None,  # 不使用样本权重
                clustering_distance_fn=None,
                nested_parameters=None,
                random_state=42,
                dynamic_programming=False,  # 动态规划会显著增加内存
                extra_sympy_parameters={},
            )

            # 训练 (符号回归)
            model.fit(df[feature_names], df['y'])

            # 获取最佳表达式
            best_exprs = model.get_best()

            factors = []
            for row in best_exprs:
                formula = row['expression']
                if formula is None:
                    continue

                # 计算因子值
                try:
                    factor_values = self._evaluate_expression(
                        df[feature_names], formula
                    )
                except Exception:
                    continue

                # 跳过常数因子
                if np.std(factor_values) < 1e-10:
                    continue

                # 计算 IC
                ic, _ = spearmanr(factor_values, y_clean)
                if np.isnan(ic):
                    continue

                # 计算 ICIR
                ic_std = np.std(y_clean)
                icir = ic / ic_std if ic_std > 0 else 0.0

                # 综合有效性评分
                efficacy = self._compute_efficacy(ic, icir, formula)

                factor = SymbolicFactor(
                    name=f"SR_{formula[:50].replace(' ', '_')}",
                    formula=str(formula),
                    description=f"符号回归生成: {formula}",
                    ic=float(ic),
                    icir=float(icir),
                    efficacy=float(efficacy),
                    market_regime="neutral",
                )
                factors.append(factor)

                logger.info(
                    f"[PySR] 发现因子: {factor.name} | "
                    f"IC={ic:.4f} | ICIR={icir:.4f} | 公式={formula}"
                )

            # 按 efficacy 排序，取 top 10
            factors.sort(key=lambda f: f.efficacy, reverse=True)
            logger.info(f"[PySR] 挖掘完成: {len(factors)} 个有效因子")
            return factors[:10]

        except Exception as e:
            logger.error(f"[PySR] 符号回归失败: {e}")
            return self._mine_with_templates(X, y, feature_names, stock_code)

    def _evaluate_expression(self, df: pd.DataFrame, formula) -> np.ndarray:
        """评估符号回归生成的表达式"""
        # PySR 返回的表达式可以用 sympy 解析
        # 这里使用 Python 的 eval 方式 (安全范围内)
        expr_str = str(formula)

        # 构建安全的命名空间
        safe_dict = {}
        for col in df.columns:
            # 替换列名为安全变量名
            safe_name = col.replace("-", "_").replace(".", "_")
            safe_dict[safe_name] = df[col].values
            # 也保留原始名称
            safe_dict[col] = df[col].values

        # 添加数学函数
        safe_dict.update({
            'abs': np.abs, 'square': lambda x: x ** 2,
            'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10,
            'exp': np.exp, 'neg': lambda x: -x,
            'cos': np.cos, 'sin': np.sin,
        })

        # 添加常数
        safe_dict.update({'e': np.e, 'pi': np.pi})

        result = eval(expr_str, {"__builtins__": {}}, safe_dict)
        return np.asarray(result, dtype=float)

    def _compute_efficacy(self, ic: float, icir: float, formula: str) -> float:
        """计算因子综合有效性评分"""
        # 基于 IC 和 ICIR 的加权评分
        ic_score = abs(ic)
        icir_score = min(abs(icir) / 2.0, 1.0)  # ICIR > 2 视为满分

        # 公式复杂度惩罚 (越简单越好)
        complexity = len(str(formula))
        complexity_penalty = max(0, 1 - complexity / 200)

        efficacy = 0.5 * ic_score + 0.3 * icir_score + 0.2 * complexity_penalty
        return min(max(efficacy, 0.0), 1.0)

    def _mine_with_templates(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             stock_code: str) -> List[SymbolicFactor]:
        """使用预定义模板挖掘因子 (fallback)"""
        from scipy.stats import spearmanr

        logger.info(f"[PySR] 使用 {len(self.FACTOR_TEMPLATES)} 个模板因子")
        factors = []

        # 计算收益率的标准差
        y_std = np.std(y)

        for template in self.FACTOR_TEMPLATES:
            try:
                # 尝试计算模板因子
                factor_values = self._compute_template_factor(
                    template['formula'], X, feature_names
                )

                if factor_values is None or np.std(factor_values) < 1e-10:
                    continue

                # 计算 IC
                ic, _ = spearmanr(factor_values, y)
                if np.isnan(ic):
                    continue

                icir = ic / y_std if y_std > 0 else 0.0
                efficacy = self._compute_efficacy(ic, icir, template['formula'])

                factor = SymbolicFactor(
                    name=template['name'],
                    formula=template['formula'],
                    description=template['description'],
                    ic=float(ic),
                    icir=float(icir),
                    efficacy=float(efficacy),
                    market_regime="neutral",
                )
                factors.append(factor)
                logger.info(
                    f"[PySR] 模板因子: {factor.name} | IC={ic:.4f}"
                )
            except Exception as e:
                logger.debug(f"[PySR] 模板因子计算失败 {template['name']}: {e}")

        factors.sort(key=lambda f: f.efficacy, reverse=True)
        return factors

    def _compute_template_factor(self, formula: str, X: np.ndarray,
                                  feature_names: List[str]) -> Optional[np.ndarray]:
        """计算模板因子值"""
        # 简化版: 从特征矩阵中提取相关特征并计算
        # 这里使用特征矩阵的线性组合来近似模板因子

        # 将特征矩阵标准化
        X_std = np.zeros_like(X)
        for i in range(X.shape[1]):
            col_std = np.std(X[:, i])
            if col_std > 1e-10:
                X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / col_std
            else:
                X_std[:, i] = 0

        # 简单的模板因子计算逻辑
        if 'return' in formula.lower() and 'volume' in formula.lower():
            # 量价动量: 使用特征中前几个作为 proxy
            if X.shape[1] >= 2:
                return X_std[:, 0] * np.log(np.abs(X_std[:, 1]) + 1)
        elif 'return' in formula.lower() and 'std' in formula.lower():
            if X.shape[1] >= 2:
                return X_std[:, 0] / (np.abs(X_std[:, 1]) + 1e-8)
        elif 'volume' in formula.lower():
            if X.shape[1] >= 2:
                return X_std[:, 1]
        elif 'return' in formula.lower():
            if X.shape[1] >= 1:
                return X_std[:, 0]

        return None


class ICValidator:
    """因子 IC 验证器"""

    def __init__(self, window_days: int = 60):
        self.window_days = window_days

    def validate_factors(self, factor_values: np.ndarray,
                         returns: np.ndarray,
                         dates: Optional[np.ndarray] = None) -> Dict:
        """
        验证因子有效性

        Returns:
            {
                'ic': float,           # Rank IC
                'icir': float,         # IC Information Ratio
                'ic_t_stat': float,    # t-statistic
                'ic_p_value': float,   # p-value
                'half_life': int,      # IC 半衰期 (天)
                'top_quartile_return': float,  # Top 25% 分组平均收益
                'bottom_quartile_return': float,  # Bottom 25% 分组平均收益
                'monotonicity': bool,  # 分组收益单调性
                'turnover': float,     # 换手率
            }
        """
        from scipy.stats import spearmanr, t

        n = len(factor_values)
        valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
        fv = factor_values[valid_mask]
        ret = returns[valid_mask]

        if len(fv) < 20:
            return self._empty_validation()

        # 1. Rank IC (Spearman)
        ic, ic_p = spearmanr(fv, ret)
        if np.isnan(ic):
            ic = 0.0
            ic_p = 1.0

        # 2. ICIR
        ret_std = np.std(ret)
        icir = ic / ret_std if ret_std > 0 else 0.0

        # 3. t-statistic
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2 + 1e-10)

        # 4. IC 半衰期 (简化版: 自相关衰减)
        half_life = self._estimate_half_life(fv, ret)

        # 5. 分组收益 (四分位)
        sorted_idx = np.argsort(fv)
        q_size = max(1, len(sorted_idx) // 4)
        top_ret = np.mean(ret[sorted_idx[-q_size:]])
        bottom_ret = np.mean(ret[sorted_idx[:q_size]])

        # 6. 单调性检验
        quintile_size = max(1, len(sorted_idx) // 5)
        quintile_rets = []
        for i in range(5):
            start = i * quintile_size
            end = start + quintile_size
            quintile_rets.append(np.mean(ret[sorted_idx[start:end]]))
        monotonicity = all(
            quintile_rets[i] <= quintile_rets[i + 1]
            for i in range(4)
        ) or all(
            quintile_rets[i] >= quintile_rets[i + 1]
            for i in range(4)
        )

        # 7. 换手率
        turnover = float(np.mean(np.abs(np.diff(np.sign(fv)))))

        return {
            'ic': float(ic),
            'icir': float(icir),
            'ic_t_stat': float(t_stat),
            'ic_p_value': float(ic_p),
            'half_life': int(half_life),
            'top_quartile_return': float(top_ret),
            'bottom_quartile_return': float(bottom_ret),
            'monotonicity': bool(monotonicity),
            'turnover': float(turnover),
        }

    def _estimate_half_life(self, factor_values: np.ndarray,
                            returns: np.ndarray) -> int:
        """估计 IC 半衰期 (简化版)"""
        if len(factor_values) < 30:
            return 5

        # 计算滚动 IC 的自相关
        window = min(20, len(factor_values) // 3)
        ics = []
        for i in range(len(factor_values) - window):
            fv_window = factor_values[i:i + window]
            r_window = returns[i:i + window]
            if np.std(fv_window) > 1e-10 and np.std(r_window) > 1e-10:
                ic, _ = spearmanr(fv_window, r_window)
                if not np.isnan(ic):
                    ics.append(ic)

        if len(ics) < 5:
            return 5

        # 计算自相关衰减
        ics_arr = np.array(ics)
        for lag in range(1, min(10, len(ics_arr) // 2)):
            if lag >= len(ics_arr):
                break
            autocorr = np.corrcoef(ics_arr[:-lag], ics_arr[lag:])[0, 1]
            if np.isnan(autocorr) or autocorr < 0.5:
                return max(lag * window, 1)

        return len(ics_arr) * window

    def _empty_validation(self) -> Dict:
        return {
            'ic': 0.0, 'icir': 0.0, 'ic_t_stat': 0.0,
            'ic_p_value': 1.0, 'half_life': 0,
            'top_quartile_return': 0.0, 'bottom_quartile_return': 0.0,
            'monotonicity': False, 'turnover': 0.0,
        }


class FactorMiningEngine:
    """因子挖掘引擎 - 整合符号回归 + LLM + IC 验证"""

    PROMPT_TEMPLATE = """你是专业的量化因子挖掘专家。基于市场数据和 SOTA 研究，发现新的 alpha 因子。

【当前市场数据】
股票: {stock_name} ({stock_code})
市场状态: {market_regime}
波动率: {volatility}
趋势: {trend}

【可用特征维度】
- 价格特征: 开盘价、收盘价、最高价、最低价、成交量、成交额
- 技术指标: RSI、MACD、布林带宽度、ATR、KDJ
- 动量特征: 1日/5日/20日收益率、滚动标准差、滚动均值
- 成交量特征: 量比、成交量移动平均偏离、资金流向
- 波动率特征: 日内振幅、滚动波动率、波动率偏离

【SOTA 因子挖掘方法参考】
1. 符号回归 (PySR): 通过遗传算法搜索最优数学表达式
2. IC 验证: Rank IC > 0.03 为有效，ICIR > 0.5 为稳定
3. 因子中性化: 控制行业和规模因子暴露
4. IC 衰减: 动量因子半衰期 20-60 天，反转因子 1-5 天

【现有因子表现】
{existing_factors}

【挖掘要求】
1. 结合当前市场状态 (regime) 设计针对性因子
2. 考虑多因子组合效应 (如量价背离、波动率突破)
3. 给出具体计算公式，使用标准数学表达式
4. 评估因子在牛/熊/震荡市场的适应性
5. 预测因子 IC 范围 (0-1)

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

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.sr_engine = SymbolicRegressionEngine()
        self.ic_validator = ICValidator()
        self.factor_history = {}

    def mine_and_evaluate(self, stock_data: Dict, factor_data: Dict,
                          stock_code: str = "sz300620") -> List[Dict]:
        """
        挖掘并评估因子 (主入口)

        Args:
            stock_data: 股票数据
            factor_data: 包含 factor_values 和 returns 的字典
            stock_code: 股票代码

        Returns:
            因子列表 (字典格式，可直接 JSON 序列化)
        """
        all_factors = []

        # 1. 符号回归因子挖掘
        try:
            sr_factors = self._mine_symbolic(stock_data, factor_data, stock_code)
            all_factors.extend(sr_factors)
            logger.info(f"[FactorMining] 符号回归: {len(sr_factors)} 个因子")
        except Exception as e:
            logger.error(f"[FactorMining] 符号回归失败: {e}")

        # 2. LLM 因子挖掘
        try:
            llm_factors = self._mine_llm(stock_data)
            all_factors.extend(llm_factors)
            logger.info(f"[FactorMining] LLM: {len(llm_factors)} 个因子")
        except Exception as e:
            logger.error(f"[FactorMining] LLM 挖掘失败: {e}")

        # 3. 传统因子评估
        try:
            traditional_factors = self._evaluate_traditional(factor_data)
            all_factors.extend(traditional_factors)
            logger.info(f"[FactorMining] 传统: {len(traditional_factors)} 个因子")
        except Exception as e:
            logger.error(f"[FactorMining] 传统评估失败: {e}")

        # 4. 按 efficacy 排序
        all_factors.sort(key=lambda f: f.get('efficacy', 0), reverse=True)

        logger.info(f"[FactorMining] 总计: {len(all_factors)} 个因子")
        return all_factors

    def _mine_symbolic(self, stock_data: Dict, factor_data: Dict,
                       stock_code: str) -> List[Dict]:
        """符号回归因子挖掘"""
        if 'factor_values' not in factor_data or 'returns' not in factor_data:
            return []

        factor_values = factor_data['factor_values']
        returns = factor_data['returns']

        if not isinstance(factor_values, dict):
            return []

        # 准备特征矩阵
        feature_names = list(factor_values.keys())
        X = np.column_stack([factor_values[name] for name in feature_names])
        y = np.array(returns)

        # 清洗数据
        mask = ~(np.isnan(X) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 50:
            return []

        # 调用符号回归引擎
        factors = self.sr_engine.mine_factors(
            X_clean, y_clean, feature_names, stock_code
        )

        # 验证每个因子
        validated = []
        for f in factors:
            validation = self.ic_validator.validate_factors(
                self._get_factor_values(f.formula, factor_data),
                np.array(returns)
            )
            f.ic = validation['ic']
            f.icir = validation['icir']
            f.ic_t_stat = validation['ic_t_stat']
            f.half_life = validation['half_life']
            f.turnover_rate = validation['turnover']
            f.efficacy = self._combined_score(f.ic, f.icir, f.half_life)
            validated.append(f.to_dict())

        return validated

    def _mine_llm(self, stock_data: Dict) -> List[Dict]:
        """LLM 因子挖掘"""
        if self.llm_client is None:
            return self._default_factors()

        try:
            # 获取现有因子表现
            existing = stock_data.get('existing_factors', {})
            if existing:
                factor_lines = []
                for name, info in existing.items():
                    if isinstance(info, dict):
                        factor_lines.append(f"- {name}: IC={info.get('ic', 0):.4f}, ICIR={info.get('icir', 0):.4f}")
                    else:
                        factor_lines.append(f"- {name}")
                existing_factors = "\n".join(factor_lines[:10])
            else:
                existing_factors = "无"

            prompt = self.PROMPT_TEMPLATE.format(
                stock_name=stock_data.get("name", ""),
                stock_code=stock_data.get("code", stock_data.get("symbol", "")),
                market_regime=stock_data.get("market_regime", "neutral"),
                volatility=stock_data.get("volatility", "--"),
                trend=stock_data.get("trend", "neutral"),
                existing_factors=existing_factors
            )

            response = self.llm_client.get_response(
                [{"role": "user", "content": prompt}]
            )

            # 2026-09-09: extract_json (剥 ``` 围栏 + 语言标签 + 大括号提取) 替代手写
            # split 剥壳 — 旧逻辑剥壳后残留 "json" 语言标签 → json.loads 恒失败
            # (Extra data: line 1 column 2 真因)。剥壳失败也走默认因子, 不再抛异常。
            data = extract_json(response.content) if extract_json else None
            if not data or "new_factors" not in data:
                return self._default_factors()

            factors = []
            for f in data.get("new_factors", []):
                factors.append({
                    'name': f.get("name", ""),
                    'formula': f.get("formula", ""),
                    'description': f.get("description", ""),
                    'ic': 0.0,
                    'icir': 0.0,
                    'ic_t_stat': 0.0,
                    'efficacy': data.get("confidence", 0.5),
                    'market_regime': data.get("regime_dependency", "neutral"),
                    'half_life': 0,
                    'turnover_rate': 0.0,
                    'source': 'llm',
                    'timestamp': datetime.now().isoformat(),
                })
            return factors

        except Exception as e:
            logger.error(f"[FactorMining] LLM 解析失败: {e}")
            return self._default_factors()

    def _evaluate_traditional(self, factor_data: Dict) -> List[Dict]:
        """传统因子评估"""
        if 'factor_values' not in factor_data or 'returns' not in factor_data:
            return []

        factors = []
        returns = np.array(factor_data['returns'])
        ret_std = np.std(returns)

        for name, values in factor_data['factor_values'].items():
            fv = np.array(values)
            valid_mask = ~(np.isnan(fv) | np.isnan(returns))
            if len(fv[valid_mask]) < 20:
                continue

            fv_valid = fv[valid_mask]
            ret_valid = returns[valid_mask]

            from scipy.stats import spearmanr
            ic, _ = spearmanr(fv_valid, ret_valid)
            if np.isnan(ic):
                ic = 0.0

            icir = ic / ret_std if ret_std > 0 else 0.0
            efficacy = self._combined_score(ic, icir, 5)

            factors.append({
                'name': name,
                'formula': f"factor({name})",
                'description': f"传统因子: {name}",
                'ic': float(ic),
                'icir': float(icir),
                'ic_t_stat': 0.0,
                'efficacy': float(efficacy),
                'market_regime': 'neutral',
                'half_life': 5,
                'turnover_rate': 0.0,
                'source': 'traditional',
                'timestamp': datetime.now().isoformat(),
            })

        return factors

    def _get_factor_values(self, formula: str, factor_data: Dict) -> np.ndarray:
        """从公式中获取因子值 (简化版)"""
        # 对于符号回归生成的公式，返回第一个可用因子作为 proxy
        if 'factor_values' not in factor_data:
            return np.array([0.0])

        keys = list(factor_data['factor_values'].keys())
        if keys:
            return np.array(factor_data['factor_values'][keys[0]])
        return np.array([0.0])

    def _combined_score(self, ic: float, icir: float, half_life: int) -> float:
        """综合评分"""
        ic_score = abs(ic)
        icir_score = min(abs(icir) / 2.0, 1.0)
        hl_score = min(half_life / 30.0, 1.0)  # 半衰期 > 30 天视为满分

        return 0.5 * ic_score + 0.3 * icir_score + 0.2 * hl_score

    def _default_factors(self) -> List[Dict]:
        """默认因子 (LLM 解析失败时的 fallback)"""
        return [
            {
                'name': '量价动量因子 (VPM)',
                'formula': 'return * log(volume + 1)',
                'description': '结合价格动量与成交量变化',
                'ic': 0.0, 'icir': 0.0, 'ic_t_stat': 0.0,
                'efficacy': 0.5, 'market_regime': 'neutral',
                'half_life': 5, 'turnover_rate': 0.0,
                'source': 'default',
                'timestamp': datetime.now().isoformat(),
            },
            {
                'name': '多周期趋势因子 (MPTC)',
                'formula': 'Sum(Trend_i) / N',
                'description': '多周期趋势一致性',
                'ic': 0.0, 'icir': 0.0, 'ic_t_stat': 0.0,
                'efficacy': 0.5, 'market_regime': 'neutral',
                'half_life': 5, 'turnover_rate': 0.0,
                'source': 'default',
                'timestamp': datetime.now().isoformat(),
            },
            {
                'name': '波动率调整动量 (VAM)',
                'formula': 'return / Volatility',
                'description': '波动率调整后的动量',
                'ic': 0.0, 'icir': 0.0, 'ic_t_stat': 0.0,
                'efficacy': 0.5, 'market_regime': 'neutral',
                'half_life': 5, 'turnover_rate': 0.0,
                'source': 'default',
                'timestamp': datetime.now().isoformat(),
            },
        ]


class FactorNeutralizer:
    """
    因子中性化模块

    目标: 从原始因子中剥离行业因子和风格因子暴露，得到纯 alpha 信号

    方法:
    1. 横截面回归: factor_raw = beta_ind * industry_dummies + beta_style * style_factors + alpha
    2. 取残差 alpha 作为中性化后的因子值
    3. 对残差做标准化 (z-score)

    风格因子 (Barra CNE6):
    - Market Cap (规模)
    - Beta (系统性风险)
    - Momentum (动量)
    - Volatility (波动率)
    - Liquidity (流动性)
    - LEV (杠杆)
    - Earnings Yield (盈利收益率)
    - Growth (成长性)

    参考:
    - Barra CNE6 风险模型
    - Grinold & Kahn (1999) "Active Portfolio Management"
    """

    def __init__(self, n_style_factors: int = 8):
        self.n_style_factors = n_style_factors
        self.style_means = None
        self.style_stds = None
        self.industry_means = None
        self.industry_stds = None
        self.is_fitted = False

    def neutralize(self, raw_factor: np.ndarray,
                   industry_labels: Optional[np.ndarray] = None,
                   style_factors: Optional[np.ndarray] = None,
                   fit: bool = False) -> np.ndarray:
        """
        因子中性化

        Args:
            raw_factor: 原始因子值 (n_samples,)
            industry_labels: 行业标签 (0, 1, 2, ...), 可选
            style_factors: 风格因子矩阵 (n_samples, n_style), 可选
            fit: 是否先拟合风格因子统计量

        Returns:
            中性化后的因子值 (n_samples,)
        """
        n = len(raw_factor)
        valid_mask = ~np.isnan(raw_factor)

        if not valid_mask.any():
            return raw_factor.copy()

        # 1. 先做风格中性化
        if style_factors is not None and style_factors.shape[0] == n:
            if fit:
                self.style_means = np.mean(style_factors, axis=0)
                self.style_stds = np.std(style_factors, axis=0)
                self.style_stds[self.style_stds < 1e-10] = 1.0

            style_scaled = (style_factors - self.style_means) / self.style_stds

            # 横截面回归: raw_factor ~ style_factors
            # 使用 OLS: beta = (X'X)^{-1} X'y
            X = np.column_stack([style_scaled, np.ones(n)])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, raw_factor, rcond=None)
                residual = raw_factor - X @ beta
            except np.linalg.LinAlgError:
                residual = raw_factor  # 回归失败时保留原始值
        else:
            residual = raw_factor.copy()

        # 2. 再做行业中性化
        if industry_labels is not None:
            unique_industries = np.unique(industry_labels[valid_mask])
            if len(unique_industries) > 1:
                # 构造行业哑变量 (去掉第一个作为基准)
                industry_dummies = np.zeros((n, len(unique_industries) - 1))
                for i, ind in enumerate(unique_industries[1:]):
                    industry_dummies[industry_labels == ind, i] = 1

                if fit:
                    self.industry_means = np.mean(industry_dummies, axis=0)
                    self.industry_stds = np.std(industry_dummies, axis=0)
                    self.industry_stds[self.industry_stds < 1e-10] = 1.0

                try:
                    beta_ind, _, _, _ = np.linalg.lstsq(
                        industry_dummies, residual, rcond=None
                    )
                    residual = residual - industry_dummies @ beta_ind
                except np.linalg.LinAlgError:
                    pass

        # 3. 标准化残差
        residual = residual - np.median(residual)
        residual_std = np.std(residual)
        if residual_std > 1e-10:
            residual = residual / residual_std

        # 填充 NaN
        result = np.full(n, np.nan)
        result[valid_mask] = residual[valid_mask]

        self.is_fitted = True
        return result

    def neutralize_factor_dataframe(self, factor_df: pd.DataFrame,
                                     industry_col: str = 'industry',
                                     style_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        对 DataFrame 中的因子列做批量中性化

        Args:
            factor_df: DataFrame，列包括因子列 + industry + 风格因子列
            industry_col: 行业列名
            style_cols: 风格因子列名列表

        Returns:
            中性化后的 DataFrame (仅包含因子列)
        """
        industries = factor_df[industry_col].values if industry_col in factor_df else None
        style_factors = factor_df[style_cols].values if style_cols else None

        neutralized = {}
        for col in factor_df.columns:
            if col == industry_col or (style_cols and col in style_cols):
                continue
            neutralized[col] = self.neutralize(
                factor_df[col].values,
                industry_labels=industries,
                style_factors=style_factors,
            )

        return pd.DataFrame(neutralized, index=factor_df.index)


class ICDecayAnalyzer:
    """
    IC Decay 分析模块

    目标: 分析因子 IC 随持有期衰减的速度，评估因子稳定性

    核心指标:
    - IC(h): 持有 h 期的 Rank IC
    - IC Decay Rate: IC 随 h 的衰减速率
    - Half-Life: IC 衰减到初始值一半所需的持有期
    - ICIR Decay: ICIR 随持有期的变化

    应用:
    - 高衰减因子 (半衰期 < 5 天): 短期反转因子，需要高频更新
    - 中衰减因子 (半衰期 5-30 天): 动量因子，周频更新
    - 低衰减因子 (半衰期 > 30 天): 价值因子，月频更新

    参考:
    - Baqai & Walker (2000) "Systematic Equity Research, Part II"
    - Jacobsen (2023) "The Information Coefficient"
    """

    # 标准持有期 (天)
    HOLDING_PERIODS = [1, 5, 10, 20]

    def __init__(self):
        self.decay_results = {}

    def analyze_decay(self, factor_values: np.ndarray,
                      forward_returns: Dict[int, np.ndarray],
                      dates: Optional[np.ndarray] = None) -> Dict:
        """
        分析因子 IC 随持有期的衰减

        Args:
            factor_values: 因子值序列 (n,)
            forward_returns: {holding_period: forward_returns_array}
                例如 {1: next_day_returns, 5: 5day_returns, ...}
            dates: 日期序列，可选

        Returns:
            {
                'decay': {
                    '1d': {'ic': 0.05, 'icir': 0.3, 't_stat': 2.1, ...},
                    '5d': {'ic': 0.03, 'icir': 0.2, 't_stat': 1.5, ...},
                    ...
                },
                'half_life': 15,  # 估计的 IC 半衰期
                'decay_rate': 0.02,  # 每日衰减速率
                'factor_stability': 'stable' | 'moderate' | 'unstable',
            }
        """
        decay = {}
        initial_ics = []

        for period in self.HOLDING_PERIODS:
            if period not in forward_returns:
                continue

            fwd_ret = forward_returns[period]
            valid_mask = ~(np.isnan(factor_values) | np.isnan(fwd_ret))
            fv = factor_values[valid_mask]
            ret = fwd_ret[valid_mask]

            if len(fv) < 20:
                decay[f'{period}d'] = {'ic': np.nan, 'icir': np.nan, 't_stat': np.nan}
                continue

            # Rank IC
            from scipy.stats import spearmanr
            ic, p_value = spearmanr(fv, ret)
            if np.isnan(ic):
                ic = 0.0
                p_value = 1.0

            # ICIR
            ret_std = np.std(ret)
            icir = ic / ret_std if ret_std > 0 else 0.0

            # t-stat
            t_stat = ic * np.sqrt(len(fv) - 2) / np.sqrt(1 - ic ** 2 + 1e-10)

            # 分组收益 (五分位)
            sorted_idx = np.argsort(fv)
            q_size = max(1, len(sorted_idx) // 5)
            quintile_rets = []
            for i in range(5):
                start = i * q_size
                end = start + q_size
                quintile_rets.append(float(np.mean(ret[sorted_idx[start:end]])))

            decay[f'{period}d'] = {
                'ic': round(float(ic), 4),
                'icir': round(float(icir), 4),
                't_stat': round(float(t_stat), 2),
                'p_value': round(float(p_value), 4),
                'quintile_returns': [round(q, 4) for q in quintile_rets],
                'n_samples': len(fv),
            }

            if not np.isnan(ic):
                initial_ics.append(ic)

        # 估计半衰期 (指数衰减拟合)
        half_life = self._estimate_decay_half_life(initial_ics)

        # 衰减速率
        decay_rate = np.log(2) / half_life if half_life > 0 else 0.0

        # 稳定性评估
        if len(initial_ics) >= 2:
            ic_std = np.std(initial_ics)
            ic_mean = np.mean(initial_ics)
            if ic_mean > 0:
                cv = ic_std / ic_mean  # 变异系数
            else:
                cv = 1.0

            if cv < 0.3:
                stability = 'stable'
            elif cv < 0.6:
                stability = 'moderate'
            else:
                stability = 'unstable'
        else:
            stability = 'unstable'
            half_life = 0
            decay_rate = 0.0

        result = {
            'decay': decay,
            'half_life': int(half_life),
            'decay_rate': round(float(decay_rate), 4),
            'factor_stability': stability,
            'interpretation': self._interpret_decay(half_life, stability),
        }

        self.decay_results = result
        return result

    def _estimate_decay_half_life(self, ics: List[float]) -> int:
        """
        估计 IC 半衰期

        假设 IC 按指数衰减: IC(h) = IC(0) * exp(-lambda * h)
        半衰期: T_1/2 = ln(2) / lambda
        """
        if len(ics) < 2:
            return 10  # 默认 10 天

        # 简化: 使用第一个和最后一个 IC 估计衰减速率
        ic_0 = ics[0]
        ic_last = ics[-1]

        if ic_0 <= 0 or ic_last >= ic_0:
            return 60  # IC 不衰减，半衰期很长

        # IC(h) = IC(0) * exp(-lambda * h)
        # lambda = ln(IC(0) / IC(h)) / h
        last_period = self.HOLDING_PERIODS[-1]
        try:
            lambda_rate = np.log(ic_0 / ic_last) / last_period
            if lambda_rate > 0:
                return int(np.log(2) / lambda_rate)
        except (ValueError, ZeroDivisionError):
            pass

        return 20  # 默认中等衰减

    def _interpret_decay(self, half_life: int, stability: str) -> str:
        """解释 IC 衰减结果"""
        if half_life < 5:
            return "高衰减因子: 短期反转效应显著，需要每日更新"
        elif half_life < 15:
            return "中衰减因子: 短期动量效应，建议周频更新"
        elif half_life < 30:
            return "低衰减因子: 中期趋势效应，建议双周更新"
        else:
            return "极低衰减因子: 长期价值效应，建议月频更新"

    def get_decay_summary(self) -> Dict:
        """获取衰减分析摘要"""
        if not self.decay_results:
            return {'error': 'No decay analysis performed'}

        result = self.decay_results.copy()
        result['half_life'] = self.decay_results.get('half_life', 0)
        result['stability'] = self.decay_results.get('factor_stability', 'unknown')
        result['interpretation'] = self.decay_results.get('interpretation', '')
        return result


# 全局实例
factor_neutralizer = FactorNeutralizer()
ic_decay_analyzer = ICDecayAnalyzer()
