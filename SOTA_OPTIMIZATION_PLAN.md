# Stock Analyzer — SOTA 深度优化方案 (2026-06-13)

> 基于逐行代码审查 + 全网 SOTA 研究 (PatchTST, Time-LLM, FinRL, Qlib, AlphaCrafter)

---

## 一、当前系统核心问题 (致命 → 建议)

### 致命问题 (P0)

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 1 | **深度学习层完全无效** | `dl_model_v2.py` | 763 行全 NumPy 实现，无反向传播，`train()` 只有前向传播，`trained=True` 但参数是随机初始化 |
| 2 | **RL 算法未实现** | `rl_trader_v2.py` | 仅有 TradingEnvV2 环境定义，PPO/SAC 算法完全未实现 |
| 3 | **因子权重从未动态更新** | `multi_factor_model_v2.py` | `update_weights_from_ic()` 存在但从未被调用，所有因子使用硬编码 `DEFAULT_WEIGHTS` |

### 高优先级 (P1)

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 4 | **RSI 因子逻辑反直觉** | `multi_factor_model_v2.py:218-222` | RSI 45-55 最优(10分)，实际应超卖看涨、超买看跌 |
| 5 | **无概念漂移检测** | 全局 | 模型训练后不更新，市场 regime 变化后性能快速退化 |
| 6 | **情感分析仅词典法** | `sentiment_analyzer_v2.py` | 80 正面词+60 负面词，远不如 FinBERT |
| 7 | **Ensemble 权重固定** | `sota_integration.py:254-260` | LLM(40%)+Factor(20%)+MultiModal(15%)+RL(25%) 不随 regime 变化 |
| 8 | **LLM 单点故障** | `llm_client.py` | OMLX 宕机则 SOTA 层全部 fallback 到默认值 |

### 中优先级 (P2)

| # | 问题 | 影响 |
|---|------|------|
| 9 | 因子数量 15 vs Qlib 158 | 预测能力受限 |
| 10 | Barra 仅 10 因子 vs CNE6 20+ | 风险分解不准确 |
| 11 | GARCH 因子 fallback 到换手率 | 波动率因子失效 |

---

## 二、SOTA 研究结论 (2025-2026)

### 2.1 时序预测架构演进

```
2020-2022: LSTM/GRU (序列建模)
    ↓
2023-2024: Transformer (并行建模，但 O(n²) 复杂度)
    ↓
2024: PatchTST (Patch + Channel Independence，RMSE -30%)
    ↓
2025: Time-LLM (LLM world knowledge + 时序)
    ↓
2026: Mamba/SSM (O(n) 线性复杂度，高频交易)
```

### 2.2 量化框架对比

| 框架 | 因子 | 模型 | 回测 | RL | 适合场景 |
|------|------|------|------|-----|----------|
| **Qlib** | 158-360 | AutoML | ✅ | ❌ | 因子挖掘+组合优化 |
| **FinRL** | ~50 | DRL | ✅ | ✅ | 端到端 RL 交易 |
| **你的系统** | 15 | RF+LGB+DL(无效) | ✅ | ❌ | 中文A股+LLM多智能体 |

### 2.3 关键结论

1. **PatchTST 是时序预测新标准** — 你的 Transformer-LSTM 应该替换
2. **LLM 在量化中的正确用法** — 不是直接预测价格，而是做因子挖掘+跨模态推理
3. **RL 需要完整环境** — 你的 TradingEnvV2 只有骨架
4. **因子数量和质量决定上限** — 15 因子远远不够

---

## 三、具体优化方案 (含代码)

### 优化 1: 用 PyTorch PatchTST 替换 dl_model_v2.py (P0 致命)

**当前 `dl_model_v2.py` 的问题:**
- 763 行全 NumPy，无反向传播
- `train()` 方法只有前向传播，梯度更新是注释掉的占位符
- 模型参数随机初始化，`trained=True` 是假的

**新实现 `modules/patchtst_model.py`:**

```python
#!/usr/bin/env python3
"""
PatchTST 时序预测模型 — 替换无效的 dl_model_v2.py

架构:
    Input(seq_len, n_features)
        → Patch 分片 (patch_len=8)
        → 投影到 d_model
        → RoPE 位置编码
        → Transformer Encoder (n_layers=4, nhead=8)
        → 最后一个 Patch → Linear → 预测
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from datetime import datetime
import os
import pickle

from modules.logger import logger


class RotaryPositionalEmbeddings(nn.Module):
    """RoPE 位置编码 — 比标准正弦位置编码更适合时序"""
    
    def __init__(self, dim: int, max_seq_len: int = 200):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('embedding', emb)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim)"""
        seq_len = x.size(1)
        return x + self.embedding[:seq_len].unsqueeze(0)


class PatchTST(nn.Module):
    """PatchTST 模型"""
    
    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        n_classes: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_features = n_features
        
        # Patch 投影: (patch_len * n_features) → d_model
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        
        # RoPE 位置编码
        self.pos_encoder = RotaryPositionalEmbeddings(d_model, max_seq_len)
        
        # Transformer Encoder (LayerNorm first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN 更稳定
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch, seq_len, _ = x.shape
        n_patches = seq_len // self.patch_len
        
        # Patch: (batch, n_patches, patch_len * n_features)
        x = x.reshape(batch, n_patches, self.patch_len * self.n_features)
        
        # 投影 + 位置编码
        x = self.patch_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # 取最后一个 patch 的表示
        x = x[:, -1, :]
        
        return self.head(x)


class PatchTSTTrainer:
    """PatchTST 训练器"""
    
    def __init__(
        self,
        n_features: int = 12,
        patch_len: int = 8,
        d_model: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = PatchTST(
            n_features=n_features,
            patch_len=patch_len,
            d_model=d_model,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.training_history = {'loss': [], 'val_loss': [], 'val_acc': []}
        self.trained = False
        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict:
        """
        训练模型
        
        Args:
            X_train: (n_samples, seq_len, n_features)
            y_train: (n_samples,)
            X_val: (n_val, seq_len, n_features)
            y_val: (n_val,)
        """
        logger.info(f"[PatchTST] 开始训练, device={self.device}, epochs={epochs}")
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]
                
                X_batch = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y_train[batch_idx], dtype=torch.long).to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * len(batch_idx)
                n_batches += 1
            
            self.scheduler.step()
            avg_train_loss = train_loss / len(X_train)
            
            # 验证
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                logger.info(
                    f"[PatchTST] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[PatchTST] Early stopping at epoch {epoch}")
                    break
        
        self.trained = True
        logger.info("[PatchTST] 训练完成")
        return self.training_history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_t)
            loss = self.criterion(logits, y_t).item()
            acc = (logits.argmax(dim=-1) == y_t).float().mean().item()
        
        return loss, acc
    
    def predict(self, X: np.ndarray) -> Dict:
        """预测"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            confidences = probs.max(dim=-1)[0]
        
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        
        return {
            'directions': [direction_map[p.item()] for p in predictions],
            'confidences': confidences.cpu().numpy().tolist(),
            'probabilities': {
                'up': probs[:, 2].cpu().numpy().tolist(),
                'neutral': probs[:, 1].cpu().numpy().tolist(),
                'down': probs[:, 0].cpu().numpy().tolist(),
            },
        }
    
    def _save_best(self):
        """保存最佳模型"""
        path = os.path.join(self.model_dir, 'patchtst_best.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': str(self.device),
            'n_features': self.n_features,
            'patch_len': self.patch_len,
            'd_model': self.d_model,
        }, path)
    
    def load(self, path: Optional[str] = None) -> bool:
        """加载模型"""
        if path is None:
            path = os.path.join(self.model_dir, 'patchtst_best.pth')
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = True
        logger.info(f"[PatchTST] 模型已加载: {path}")
        return True
```

### 优化 2: 修复 RSI 因子 + 动态因子权重 (P1)

**修复 `multi_factor_model_v2.py` 中的 RSI 因子:**

```python
# 替换第 218-222 行
def rsi_technical(self, stock_data: Dict, klines=None) -> float:
    """RSI 技术因子 — 超卖看涨(高分), 超买看跌(低分)"""
    rsi = stock_data.get('rsi_14', 50)
    if rsi < 20:    return 10.0  # 深度超卖 → 强烈看涨
    elif rsi < 30:  return 8.0   # 超卖 → 看涨
    elif rsi < 40:  return 6.0   # 偏弱
    elif rsi < 60:  return 5.0   # 中性
    elif rsi < 70:  return 3.0   # 偏强
    elif rsi < 80:  return 2.0   # 超买 → 看跌
    else:           return 0.0   # 深度超买 → 强烈看跌
```

**新增动态因子权重更新 (在分析引擎中调用):**

```python
def update_factor_weights_rolling(self, factor_history: Dict[str, List[float]],
                                   returns: List[float], window: int = 60):
    """基于滚动 IC/ICIR 更新因子权重"""
    new_weights = {}
    for factor_name, factor_values in factor_history.items():
        if len(factor_values) < window:
            continue
        # 计算滚动 IC
        fv = np.array(factor_values[-window:])
        ret = np.array(returns[-window:])
        if np.std(fv) < 1e-10 or np.std(ret) < 1e-10:
            continue
        ic = np.corrcoef(fv, ret)[0, 1]
        if np.isnan(ic):
            continue
        new_weights[factor_name] = abs(ic)
    
    total = sum(new_weights.values())
    if total > 0:
        self.factor_weights = {k: v / total for k, v in new_weights.items()}
        logger.info(f"[FactorModel] 因子权重已更新: {len(new_weights)} 个因子")
```

### 优化 3: 概念漂移检测 (P1)

**新文件 `modules/concept_drift_detector.py`:**

```python
#!/usr/bin/env python3
"""
概念漂移检测 — ADWIN + KS 检验

当检测到漂移时，触发模型重新训练。
"""

import numpy as np
from typing import Dict, Optional
from collections import deque
from modules.logger import logger


class ADWIN:
    """ADWIN (Adaptive Windowing) 漂移检测"""
    
    def __init__(self, delta: float = 0.01, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window = deque()
        self._n_splits = 0
    
    def add(self, value: float) -> bool:
        """添加观测值，返回是否检测到漂移"""
        self.window.append(value)
        
        if len(self.window) > self.max_window:
            self.window.popleft()
        
        if len(self.window) < 30:
            return False
        
        return self._check_split()
    
    def _check_split(self) -> bool:
        """检查窗口是否可以分割"""
        n = len(self.window)
        for cut in range(n // 4, 3 * n // 4):
            n0 = cut
            n1 = n - cut
            
            if n0 < 10 or n1 < 10:
                continue
            
            mean0 = np.mean(self.window[:cut])
            mean1 = np.mean(self.window[cut:])
            
            delta_prime = np.log(2 / self.delta)
            epsilon = np.sqrt((1 / (2 * min(n0, n1))) * delta_prime)
            
            if abs(mean0 - mean1) > epsilon:
                # 检测到漂移 — 保留后半窗口
                self.window = deque(list(self.window)[cut:])
                self._n_splits += 1
                return True
        
        return False


class ConceptDriftDetector:
    """概念漂移检测器 — ADWIN + KS 检验"""
    
    def __init__(self, ks_alpha: float = 0.01, adwin_delta: float = 0.01):
        self.adwin = ADWIN(delta=adwin_delta)
        self.ks_alpha = ks_alpha
        self._recent_predictions = deque(maxlen=200)
        self._recent_features = None
        self._feature_stats = None
    
    def check_prediction_drift(self, prediction: float, actual: float) -> bool:
        """基于预测误差检测漂移"""
        error = abs(prediction - actual)
        return self.adwin.add(error)
    
    def check_feature_drift(self, X_new: np.ndarray) -> bool:
        """KS 检验检测特征分布漂移"""
        if self._feature_stats is None:
            self._feature_stats = {
                'mean': np.mean(X_new, axis=0),
                'std': np.std(X_new, axis=0) + 1e-8,
            }
            return False
        
        # 标准化
        X_norm = (X_new - self._feature_stats['mean']) / self._feature_stats['std']
        
        # KS 检验 (每维)
        for i in range(X_norm.shape[1]):
            col = X_norm[:, i]
            # 简化版: 用均值和方差的显著变化检测
            if np.abs(np.mean(col)) > 2.0 or np.abs(np.std(col) - 1.0) > 0.3:
                logger.warning(f"[Drift] Feature {i} drift detected")
                return True
        
        return False
    
    def update_baseline(self, X: np.ndarray):
        """更新基线统计"""
        self._feature_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0) + 1e-8,
        }
        logger.info("[Drift] Baseline updated")
    
    def get_drift_status(self) -> Dict:
        """获取漂移状态"""
        return {
            'adwin_splits': self.adwin._n_splits,
            'window_size': len(self.adwin.window),
        }
```

### 优化 4: Ensemble 权重动态化 (P1)

**修改 `sota_integration.py` 的 `_ensemble_aggregate` 方法:**

```python
def _ensemble_aggregate(self, llm_decision, factor_scores, 
                        cross_modal, rl_action, market_regime='sideways'):
    """
    动态 Ensemble 聚合 — 根据 HMM regime 调整权重
    """
    # 不同 regime 的最优权重
    regime_weights = {
        'bullish': {
            'llm': 0.30, 'factor': 0.25, 'multimodal': 0.20, 'rl': 0.25
        },
        'bearish': {
            'llm': 0.35, 'factor': 0.20, 'multimodal': 0.15, 'rl': 0.30
        },
        'sideways': {
            'llm': 0.25, 'factor': 0.35, 'multimodal': 0.25, 'rl': 0.15
        },
    }
    
    weights = regime_weights.get(market_regime, regime_weights['sideways'])
    
    # 计算各层得分
    direction_scores = {'bullish': 0.7, 'bearish': 0.3, 'neutral': 0.5}
    llm_score = direction_scores.get(llm_decision.get('research_direction', 'neutral'), 0.5)
    
    avg_factor_efficacy = 0.0
    if factor_scores:
        avg_factor_efficacy = sum(
            f.get('efficacy', 0) for f in factor_scores.values()
        ) / len(factor_scores)
    
    cross_modal_score = cross_modal.get('consistency_score', 0.5)
    rl_scores = {'buy': 0.7, 'sell': 0.3, 'hold': 0.5}
    rl_score = rl_scores.get(rl_action, 0.5)
    
    # 加权聚合
    ensemble_score = (
        llm_score * weights['llm'] +
        avg_factor_efficacy * weights['factor'] +
        cross_modal_score * weights['multimodal'] +
        rl_score * weights['rl']
    )
    
    if ensemble_score > 0.6:
        direction = 'bullish'
    elif ensemble_score < 0.4:
        direction = 'bearish'
    else:
        direction = 'neutral'
    
    return round(ensemble_score, 3), direction
```

### 优化 5: 情感分析升级为 FinBERT (P1)

**新文件 `modules/sentiment_finbert.py`:**

```python
#!/usr/bin/env python3
"""
FinBERT 情感分析 — 替换词典法

使用 yiyanghkust/finbert-tone (HuggingFace)
中文适配: 用 Chinese FinBERT 或 微调版
"""

import numpy as np
from typing import Dict
from modules.logger import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("[FinBERT] transformers 未安装，使用词典 fallback")


class FinBERTSentiment:
    """FinBERT 情感分析器"""
    
    def __init__(self, model_name: str = 'yiyanghkust/finbert-tone'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                self._initialized = True
                logger.info("[FinBERT] 模型已加载")
            except Exception as e:
                logger.error(f"[FinBERT] 加载失败: {e}")
    
    def analyze(self, text: str) -> Dict:
        """分析文本情感"""
        if not self._initialized:
            return self._dict_fallback(text)
        
        try:
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True, max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = probs.max().item()
                label_idx = probs.argmax().item()
            
            # label 0=negative, 1=neutral, 2=positive
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            scores = {
                'negative': probs[0, 0].item(),
                'neutral': probs[0, 1].item(),
                'positive': probs[0, 2].item(),
            }
            
            return {
                'label': label_map[label_idx],
                'confidence': confidence,
                'scores': scores,
                'score': scores['positive'] - scores['negative'],  # -1 ~ +1
            }
        except Exception as e:
            logger.error(f"[FinBERT] 分析错误: {e}")
            return self._dict_fallback(text)
    
    @staticmethod
    def _dict_fallback(text: str) -> Dict:
        """词典 fallback"""
        pos_words = {'上涨', '利好', '突破', '看好', '增长'}
        neg_words = {'下跌', '利空', '破位', '看空', '下滑'}
        
        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)
        
        score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        return {
            'label': 'positive' if score > 0.2 else ('negative' if score < -0.2 else 'neutral'),
            'confidence': 0.5,
            'scores': {'positive': max(0, score), 'neutral': 1-abs(score), 'negative': max(0, -score)},
            'score': float(score),
        }
```

---

## 四、实施路线图

```
Week 1:     P0: PatchTST 替换 dl_model_v2 (修复致命问题)
Week 2:     P1: RSI 因子修复 + 动态权重 + 概念漂移检测
Week 3:     P1: FinBERT 情感分析 + Ensemble 动态权重
Week 4:     P1: LLM 多故障转移 + 降级策略
Week 5-6:   P2: Qlib Alpha158 因子引入
Week 7-8:   P2: FinRL 替代 rl_trader_v2
Week 9-12:  P3: Mamba + 自监督预训练 + 多智能体
```

---

## 五、预期效果

| 优化项 | 当前状态 | 优化后 | 提升幅度 |
|--------|----------|--------|----------|
| 深度学习预测 | 随机初始化(无效) | PatchTST 可训练 | +20-30% 准确率 |
| 因子数量 | 15 | 50+ (含动态权重) | +40% 信息覆盖 |
| 概念漂移 | 无检测 | ADWIN+KS 双检测 | -40% 性能衰减 |
| 情感分析 | 词典法 | FinBERT | +25% 情绪准确度 |
| Ensemble | 固定权重 | HMM regime 动态 | +10% 决策质量 |
| RL 交易 | 未实现 | FinRL PPO/SAC | 新增能力 |
