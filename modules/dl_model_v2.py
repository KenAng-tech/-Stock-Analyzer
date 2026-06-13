#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深度学习模型 V2 — PatchTST 时序预测模型 (2026-06-13 升级)

升级内容:
1. [NEW] PatchTST (ICML 2023) — 基于 Patch 的 Transformer，O(n/patch_len) 复杂度
2. [NEW] RoPE 位置编码 — 旋转位置编码，适合长序列外推
3. [NEW] 完整 PyTorch 训练管线 — 反向传播 + AdamW + Cosine Annealing
4. [DEPRECATED] NumPy Transformer-LSTM (无反向传播，保留为 fallback)

架构:
    Input(seq_len, features)
        → Patch 分片 (patch_len=8)
        → 投影到 d_model=128
        → RoPE 位置编码
        → Transformer Encoder (4 layers, 8 heads)
        → 最后一个 Patch → Linear Head
        → 输出 (up/neutral/down)

依赖:
    - PyTorch ≥ 2.0 (必需)

向后兼容:
    - 保留 NumPy 实现作为 fallback (当 PyTorch 不可用时)
    - DeepLearningEnsemble 接口不变
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import pickle

from modules.logger import logger

# ── 优先使用 PatchTST (PyTorch) ──────────────────────────────

try:
    from modules.patchtst_model import (
        PatchTST,
        PatchTSTTrainer,
        DeepLearningEnsemble as _PatchTSTEnsemble,
    )
    USE_PATCHTST = True
    logger.info("[DLModelV2] PatchTST (PyTorch) 已加载 — 默认使用")
except ImportError as e:
    USE_PATCHTST = False
    logger.warning(f"[DLModelV2] PatchTST 加载失败 ({e})，尝试 NumPy fallback")


# ── NumPy 实现的深度学习组件 ──────────────────────────────────

class NumPyAttention:
    """
    自注意力机制 (NumPy 实现)

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Q, K, V 投影
        self.w_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.w_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.w_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.w_o = np.random.randn(embed_dim, embed_dim) * 0.02

        self._cache = {}

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch, seq_len, embed_dim = x.shape

        # 投影 Q, K, V
        Q = x @ self.w_q
        K = x @ self.w_k
        V = x @ self.w_v

        # 缩放点积注意力
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        # Softmax
        scores = self._softmax(scores, axis=-1)

        if self.dropout > 0:
            scores = scores * (np.random.rand(*scores.shape) > self.dropout) / (1 - self.dropout)

        # 加权求和
        output = scores @ V

        # 输出投影
        output = output @ self.w_o

        return output

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class NumPyTransformerEncoder:
    """
    Transformer Encoder 层 (NumPy 实现)

    结构:
        Input → Multi-Head Attention → Add&Norm → FFN → Add&Norm → Output
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ff_dim: int = 256, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.attention = NumPyAttention(embed_dim, num_heads, dropout)

        # FFN
        self.w1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.w2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

        # LayerNorm 参数
        self.ln1_gamma = np.ones(embed_dim)
        self.ln1_beta = np.zeros(embed_dim)
        self.ln2_gamma = np.ones(embed_dim)
        self.ln2_beta = np.zeros(embed_dim)

        self.dropout = dropout

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Multi-Head Attention + Residual + LayerNorm
        attn_out = self.attention(x, mask)
        if self.dropout > 0:
            attn_out = attn_out * (np.random.rand(*attn_out.shape) > self.dropout) / (1 - self.dropout)
        x = self._layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)

        # FFN + Residual + LayerNorm
        ffn_out = self._ffn(x)
        if self.dropout > 0:
            ffn_out = ffn_out * (np.random.rand(*ffn_out.shape) > self.dropout) / (1 - self.dropout)
        x = self._layer_norm(x + ffn_out, self.ln2_gamma, self.ln2_beta)

        return x

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + eps)
        return x * gamma + beta

    def _ffn(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2


class NumPyLSTM:
    """
    LSTM 单元 (NumPy 实现)

    结构:
        f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # 遗忘门
        i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # 输入门
        o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # 输出门
        c_t = f_t * c_{t-1} + i_t * tanh(W_c * [h_{t-1}, x_t] + b_c)
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 合并权重矩阵 [W_h, W_x]
        concat_dim = hidden_dim + input_dim

        # 遗忘门
        self.W_f = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_f = np.zeros(hidden_dim)

        # 输入门
        self.W_i = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_i = np.zeros(hidden_dim)

        # 输出门
        self.W_o = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_o = np.zeros(hidden_dim)

        # 候选细胞状态
        self.W_c = np.random.randn(concat_dim, hidden_dim) * 0.1
        self.b_c = np.zeros(hidden_dim)

    def __call__(self, x_seq: np.ndarray,
                 h0: Optional[np.ndarray] = None,
                 c0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
            h0: (batch, hidden_dim)
            c0: (batch, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
            h_final: (batch, hidden_dim)
            c_final: (batch, hidden_dim)
        """
        batch, seq_len, input_dim = x_seq.shape

        if h0 is None:
            h = np.zeros((batch, self.hidden_dim))
        else:
            h = h0

        if c0 is None:
            c = np.zeros((batch, self.hidden_dim))
        else:
            c = c0

        outputs = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]

            # 拼接 [h_{t-1}, x_t]
            concat = np.concatenate([h, x_t], axis=-1)

            # 门控计算
            f_t = self._sigmoid(concat @ self.W_f + self.b_f)
            i_t = self._sigmoid(concat @ self.W_i + self.b_i)
            o_t = self._sigmoid(concat @ self.W_o + self.b_o)
            c_tilde = np.tanh(concat @ self.W_c + self.b_c)

            # 细胞状态和隐藏状态更新
            c = f_t * c + i_t * c_tilde
            h = o_t * np.tanh(c)

            outputs.append(h)

        output = np.stack(outputs, axis=1)
        return output, h, c

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SelfAttentionGRU:
    """
    Self-Attention GRU — 在 GRU 中引入注意力机制

    架构:
        Input → GRU → Hidden States → Self-Attention → Context Vector → Output

    优势:
        - GRU 捕捉时序依赖
        - Self-Attention 捕捉关键时间步
        - 可解释性：注意力权重显示哪些时间步最重要
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_classes: int = 3, dropout: float = 0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # GRU 单元 (简化版：两个 GRU 层)
        self.gru1 = NumPyLSTM(input_dim, hidden_dim)
        self.gru2 = NumPyLSTM(hidden_dim, hidden_dim // 2)

        # Self-Attention 层
        self.attn_w = np.random.randn(hidden_dim // 2, 1) * 0.02

        # 输出层
        self.fc1 = np.random.randn(hidden_dim // 2, 32) * 0.02
        self.fc2 = np.random.randn(32, num_classes) * 0.02

    def _attention(self, hidden_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算自注意力权重

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, seq_len)
        """
        batch, seq_len, hidden_dim = hidden_states.shape

        # 计算注意力分数
        attn_scores = np.squeeze(hidden_states @ self.attn_w, axis=-1)  # (batch, seq_len)

        # Softmax 归一化
        attn_weights = self._softmax(attn_scores, axis=-1)

        # 加权求和
        context = np.sum(hidden_states * attn_weights[:, :, np.newaxis], axis=1)

        return context, attn_weights

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def __call__(self, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, seq_len)
        """
        # GRU 层
        h1, _, _ = self.gru1(x_seq)
        h2, _, _ = self.gru2(h1)

        # Self-Attention
        context, attn_weights = self._attention(h2)

        # 前馈网络
        h = np.maximum(0, context @ self.fc1)  # ReLU
        if self.dropout > 0:
            h = h * (np.random.rand(*h.shape) > self.dropout) / (1 - self.dropout)
        logits = h @ self.fc2

        return logits, attn_weights

    def predict_proba(self, x_seq: np.ndarray) -> np.ndarray:
        """预测概率"""
        logits, _ = self(x_seq)
        return self._softmax(logits, axis=-1)

    def save(self, path: str):
        """保存模型参数"""
        data = {
            'gru1_W_f': self.gru1.W_f, 'gru1_b_f': self.gru1.b_f,
            'gru1_W_i': self.gru1.W_i, 'gru1_b_i': self.gru1.b_i,
            'gru1_W_o': self.gru1.W_o, 'gru1_b_o': self.gru1.b_o,
            'gru1_W_c': self.gru1.W_c, 'gru1_b_c': self.gru1.b_c,
            'gru2_W_f': self.gru2.W_f, 'gru2_b_f': self.gru2.b_f,
            'gru2_W_i': self.gru2.W_i, 'gru2_b_i': self.gru2.b_i,
            'gru2_W_o': self.gru2.W_o, 'gru2_b_o': self.gru2.b_o,
            'gru2_W_c': self.gru2.W_c, 'gru2_b_c': self.gru2.b_c,
            'attn_w': self.attn_w,
            'fc1': self.fc1, 'fc2': self.fc2,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[SelfAttentionGRU] 模型已保存：{path}")

    @classmethod
    def load(cls, path: str) -> 'SelfAttentionGRU':
        """加载模型参数"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            input_dim=data['gru1_W_f'].shape[0] - data['gru1_W_f'].shape[1],
            hidden_dim=data['gru1_W_f'].shape[1],
        )

        # 恢复参数
        for key, value in data.items():
            if key.startswith('gru1_'):
                setattr(model.gru1, key.replace('gru1_', ''), value)
            elif key.startswith('gru2_'):
                setattr(model.gru2, key.replace('gru2_', ''), value)
            else:
                setattr(model, key, value)

        return model


# ── Transformer-LSTM 混合模型 ─────────────────────────────────

class TransformerLSTMModel:
    """
    Transformer-LSTM 混合架构

    架构:
        Input → Positional Encoding
              → Transformer Encoder (n_layers)
              → LSTM
              → Self-Attention Pooling
              → FFN Head

    优势:
        - Transformer: 全局依赖，并行计算
        - LSTM: 时序建模，状态传递
        - 结合两者优势
    """

    def __init__(self, input_dim: int = 12, d_model: int = 64,
                 num_heads: int = 8, n_layers: int = 2,
                 lstm_hidden: int = 128, num_classes: int = 3,
                 dropout: float = 0.1, max_seq_len: int = 100):
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # 输入投影
        self.input_proj = np.random.randn(input_dim, d_model) * 0.02

        # Transformer Encoder 层
        self.transformer_layers = [
            NumPyTransformerEncoder(d_model, num_heads, dropout=dropout)
            for _ in range(n_layers)
        ]

        # LSTM 层
        self.lstm = NumPyLSTM(d_model, lstm_hidden)

        # Self-Attention Pooling
        self.pooling_attn = np.random.randn(lstm_hidden, 1) * 0.02

        # 输出头
        self.fc1 = np.random.randn(lstm_hidden, 64) * 0.02
        self.fc2 = np.random.randn(64, num_classes) * 0.02

        # 位置编码
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """创建位置编码"""
        pe = np.zeros((max_len, d_model))

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        return pe

    def _attention_pooling(self, hidden_states: np.ndarray) -> np.ndarray:
        """Self-Attention Pooling"""
        batch, seq_len, hidden_dim = hidden_states.shape

        attn_scores = np.squeeze(hidden_states @ self.pooling_attn, axis=-1)
        attn_weights = self._softmax(attn_scores, axis=-1)

        context = np.sum(hidden_states * attn_weights[:, :, np.newaxis], axis=1)
        return context

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def __call__(self, x_seq: np.ndarray) -> np.ndarray:
        """
        Args:
            x_seq: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        batch, seq_len, input_dim = x_seq.shape

        # 输入投影 + 位置编码
        x = x_seq @ self.input_proj

        # 添加位置编码
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:seq_len, :]

        # Transformer Encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # LSTM
        lstm_out, _, _ = self.lstm(x)

        # Self-Attention Pooling
        context = self._attention_pooling(lstm_out)

        # 输出头
        h = np.maximum(0, context @ self.fc1)
        if self.dropout > 0:
            h = h * (np.random.rand(*h.shape) > self.dropout) / (1 - self.dropout)
        logits = h @ self.fc2

        return logits

    def predict_proba(self, x_seq: np.ndarray) -> np.ndarray:
        """预测概率"""
        logits = self(x_seq)
        return self._softmax(logits, axis=-1)

    def save(self, path: str):
        """保存模型"""
        data = {
            'input_proj': self.input_proj,
            'transformer_layers': [
                {
                    'w_q': layer.attention.w_q,
                    'w_k': layer.attention.w_k,
                    'w_v': layer.attention.w_v,
                    'w_o': layer.attention.w_o,
                    'w1': layer.w1, 'b1': layer.b1,
                    'w2': layer.w2, 'b2': layer.b2,
                    'ln1_gamma': layer.ln1_gamma, 'ln1_beta': layer.ln1_beta,
                    'ln2_gamma': layer.ln2_gamma, 'ln2_beta': layer.ln2_beta,
                }
                for layer in self.transformer_layers
            ],
            'lstm_W_f': self.lstm.W_f, 'lstm_b_f': self.lstm.b_f,
            'lstm_W_i': self.lstm.W_i, 'lstm_b_i': self.lstm.b_i,
            'lstm_W_o': self.lstm.W_o, 'lstm_b_o': self.lstm.b_o,
            'lstm_W_c': self.lstm.W_c, 'lstm_b_c': self.lstm.b_c,
            'pooling_attn': self.pooling_attn,
            'fc1': self.fc1, 'fc2': self.fc2,
            'config': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'n_layers': self.n_layers,
                'lstm_hidden': self.lstm_hidden,
                'num_classes': self.num_classes,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[TransformerLSTMModel] 模型已保存：{path}")

    @classmethod
    def load(cls, path: str) -> 'TransformerLSTMModel':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        model = cls(**config)

        model.input_proj = data['input_proj']
        model.pooling_attn = data['pooling_attn']
        model.fc1 = data['fc1']
        model.fc2 = data['fc2']

        # 恢复 Transformer 层
        for i, layer_data in enumerate(data['transformer_layers']):
            layer = model.transformer_layers[i]
            layer.attention.w_q = layer_data['w_q']
            layer.attention.w_k = layer_data['w_k']
            layer.attention.w_v = layer_data['w_v']
            layer.attention.w_o = layer_data['w_o']
            layer.w1 = layer_data['w1']
            layer.b1 = layer_data['b1']
            layer.w2 = layer_data['w2']
            layer.b2 = layer_data['b2']
            layer.ln1_gamma = layer_data['ln1_gamma']
            layer.ln1_beta = layer_data['ln1_beta']
            layer.ln2_gamma = layer_data['ln2_gamma']
            layer.ln2_beta = layer_data['ln2_beta']

        # 恢复 LSTM
        model.lstm.W_f = data['lstm_W_f']
        model.lstm.b_f = data['lstm_b_f']
        model.lstm.W_i = data['lstm_W_i']
        model.lstm.b_i = data['lstm_b_i']
        model.lstm.W_o = data['lstm_W_o']
        model.lstm.b_o = data['lstm_b_o']
        model.lstm.W_c = data['lstm_W_c']
        model.lstm.b_c = data['lstm_b_c']

        return model


# ── 集成模型 ──────────────────────────────────────────────

class DeepLearningEnsemble:
    """
    深度学习集成模型

    整合:
    1. Transformer-LSTM 混合模型
    2. Self-Attention GRU
    3. FinBERT 情感分析 (见 sentiment_bert.py)

    输出：综合预测 + 置信度
    """

    def __init__(self, input_dim: int = 12, sequence_length: int = 20):
        self.input_dim = input_dim
        self.sequence_length = sequence_length

        # 初始化子模型
        self.transformer_lstm = TransformerLSTMModel(
            input_dim=input_dim,
            d_model=64,
            num_heads=8,
            n_layers=2,
            lstm_hidden=128,
            num_classes=3,
        )

        self.attention_gru = SelfAttentionGRU(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=3,
        )

        self.trained = False
        self.model_dir = os.path.join(os.path.dirname(__file__), 'dl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def predict(self, x_seq: np.ndarray) -> Dict:
        """
        集成预测

        Args:
            x_seq: (batch, seq_len, input_dim) 或 (seq_len, input_dim)
        Returns:
            预测结果字典
        """
        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, :, :]

        batch = x_seq.shape[0]

        # Transformer-LSTM 预测
        tl_probs = self.transformer_lstm.predict_proba(x_seq)

        # Attention-GRU 预测
        gru_probs = self.attention_gru.predict_proba(x_seq)

        # 集成 (等权平均，可扩展为学习权重)
        fused_probs = (tl_probs + gru_probs) / 2

        # 预测结果
        predictions = np.argmax(fused_probs, axis=1)
        confidences = np.max(fused_probs, axis=1)

        # 映射到方向
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        directions = [direction_map[p] for p in predictions]

        return {
            'directions': directions,
            'confidences': confidences.tolist(),
            'probabilities': {
                'up': fused_probs[:, 2].tolist(),
                'neutral': fused_probs[:, 1].tolist(),
                'down': fused_probs[:, 0].tolist(),
            },
            'model_details': {
                'transformer_lstm_probs': tl_probs.tolist(),
                'attention_gru_probs': gru_probs.tolist(),
            }
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """
        训练集成模型

        注意：当前实现使用 NumPy 回退模式
        完整训练需要 MLX 或 PyTorch 支持
        """
        logger.info(f"[DeepLearningEnsemble] 开始训练，epochs={epochs}, batch_size={batch_size}")

        # 简化训练：使用随机梯度下降更新
        # 完整实现需要反向传播和更复杂的优化器

        n_samples = len(X_train)

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = X_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                # 前向传播
                logits = self.transformer_lstm(batch_X)

                # 计算交叉熵损失
                probs = self._softmax(logits, axis=-1)
                probs = np.clip(probs, 1e-10, 1.0)

                # 一个热编码
                y_onehot = np.zeros_like(probs)
                y_onehot[np.arange(len(batch_y)), batch_y] = 1

                loss = -np.mean(np.sum(y_onehot * np.log(probs), axis=1))
                epoch_loss += loss * len(batch_X)
                n_batches += 1

                # 简化梯度更新 (需要完整反向传播)
                # 这里仅做占位，实际训练需要完整梯度计算

            avg_loss = epoch_loss / n_samples

            if epoch % 10 == 0:
                logger.info(f"[DeepLearningEnsemble] Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        self.trained = True
        logger.info("[DeepLearningEnsemble] 训练完成")

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def save(self, path: Optional[str] = None):
        """保存模型"""
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.model_dir, f'dl_ensemble_{timestamp}.pkl')

        data = {
            'transformer_lstm': self.transformer_lstm,
            'attention_gru': self.attention_gru,
            'trained': self.trained,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"[DeepLearningEnsemble] 模型已保存：{path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'DeepLearningEnsemble':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            input_dim=data['input_dim'],
            sequence_length=data['sequence_length'],
        )

        model.transformer_lstm = data['transformer_lstm']
        model.attention_gru = data['attention_gru']
        model.trained = data['trained']

        return model


# ── 全局实例 ──────────────────────────────────────────────────

if USE_PATCHTST:
    dl_ensemble = _PatchTSTEnsemble()
    logger.info("[DLModelV2] 全局实例: PatchTST DeepLearningEnsemble")
else:
    # NumPy fallback — 旧实现 (无实际训练能力)
    dl_ensemble = DeepLearningEnsemble()
    logger.warning("[DLModelV2] 全局实例: NumPy DeepLearningEnsemble (无训练能力)")
