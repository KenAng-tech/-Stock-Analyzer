"""
Multi-Modal Module - 跨模态推理

SOTA Reference:
- FCMR (Hanyang University, ACL 2025): Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning
- FINMME (PKU, 25-05): Benchmark Dataset for Financial Multi-Modal Reasoning
- MM-DREX (ZJU, CityU, 25-09): Multimodal-Driven Dynamic Routing of LLM Experts
"""

from .cross_modal import MultiModalEngine, CrossModalReasoner, MultiModalInsight

__all__ = ['MultiModalEngine', 'CrossModalReasoner', 'MultiModalInsight']
