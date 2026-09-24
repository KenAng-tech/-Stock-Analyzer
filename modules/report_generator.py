"""
modules/report_generator.py — 确定性 HTML 报告生成器

借鉴 Archify Delivery Contract:
1. 验证输入数据
2. 渲染 HTML (Jinja2 + inline SVG)
3. 检查产物完整性
4. 原子写入 (temp → rename)
5. 返回 SHA-256 + 字节数回执

使用方式:
    from modules.report_generator import get_report_generator

    receipt = get_report_generator().generate(analysis_data)
    print(f"报告: {receipt.output_path}")
    print(f"SHA-256: {receipt.artifact_sha256}")
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from jinja2 import Environment, BaseLoader

from modules.schemas.validator import validate_analysis
from modules.logger import logger

_REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


class ReportReceipt:
    """报告回执 — 确定性交付凭证"""

    def __init__(
        self,
        output_path: str,
        spec_sha256: str,
        artifact_sha256: str,
        artifact_bytes: int,
        validation_errors: list,
    ):
        self.output_path = output_path
        self.spec_sha256 = spec_sha256
        self.artifact_sha256 = artifact_sha256
        self.artifact_bytes = artifact_bytes
        self.validation_errors = validation_errors
        self.passed = len(validation_errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'output_path': self.output_path,
            'spec_sha256': self.spec_sha256,
            'artifact_sha256': self.artifact_sha256,
            'artifact_bytes': self.artifact_bytes,
            'validation_errors': self.validation_errors,
            'passed': self.passed,
            'timestamp': datetime.now().isoformat(),
        }


class ReportGenerator:
    """确定性 HTML 报告生成器"""

    def __init__(self):
        self._env = Environment(loader=BaseLoader())

    def generate(
        self,
        analysis_data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> ReportReceipt:
        """
        生成分析报告 HTML。

        流程: 验证 → 渲染 → 检查 → 原子写入 → SHA-256
        """
        # 1. 验证输入
        validation_errors = validate_analysis(analysis_data)
        if validation_errors:
            logger.warning(f"[ReportGenerator] 验证警告: {validation_errors}")

        # 2. 渲染 HTML
        html_content = self._render(analysis_data)

        # 3. 计算规格 SHA-256
        spec_json = json.dumps(
            analysis_data, ensure_ascii=False, sort_keys=True
        )
        spec_sha256 = hashlib.sha256(spec_json.encode('utf-8')).hexdigest()

        # 4. 原子写入
        output_path = output_path or self._default_path(analysis_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        artifact_sha256, artifact_bytes = self._atomic_write(
            output_path, html_content
        )

        return ReportReceipt(
            output_path=output_path,
            spec_sha256=spec_sha256,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            validation_errors=validation_errors,
        )

    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """渲染 HTML 报告字符串 (09-13 锁链测试契约, 2026-09-20 复活)。

        generate() = 落盘+校验+回执 (交付模式); 本方法 = 纯渲染 (测试/前端快路径)。
        09-17 Archify 重写时此方法被摘 → intel 5.4/8.6 渲染断 (test_report_intel_sections
        6F 即其尸), 本次恢复并同时补回 5.4/8.6 模板段。
        """
        return self._render(analysis_data)

    def _render(self, data: Dict[str, Any]) -> str:
        """渲染 HTML 报告"""
        template = self._env.from_string(self._TEMPLATE)
        return template.render(
            title=data.get('basic_info', {}).get('name', '分析报告'),
            code=data.get('basic_info', {}).get('code', ''),
            price=data.get('basic_info', {}).get('price', 0),
            date=data.get('basic_info', {}).get('date', ''),
            prediction=data.get('prediction', {}),
            fundamental=data.get('fundamental', {}),
            technical=data.get('technical', {}),
            # 09-20: 情报雷达三源渲染恢复 (09-12 链尾, 09-17 重写时断)
            intel=data.get('intel') or {},
            insight=data.get('insight') or {},
            audit=data.get('exec_audit') or {},
            spec_sha256='',  # 稍后替换
            timestamp=datetime.now().isoformat(),
        )

    def _atomic_write(
        self, path: str, content: str
    ) -> tuple:
        """原子写入: 先写 temp 文件，再 rename"""
        content_bytes = content.encode('utf-8')
        artifact_sha256 = hashlib.sha256(content_bytes).hexdigest()

        dir_path = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            os.write(fd, content_bytes)
            os.close(fd)
            os.replace(tmp_path, path)  # 原子替换
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        return artifact_sha256, len(content_bytes)

    def _default_path(self, data: Dict[str, Any]) -> str:
        """生成默认输出路径"""
        code = data.get('basic_info', {}).get('code', 'unknown')
        date = datetime.now().strftime('%Y%m%d')
        return os.path.join(_REPORTS_DIR, f"{code}_analysis_{date}.html")

    def _inject_sha(self, html: str, spec_sha256: str) -> str:
        """将 SHA-256 注入 HTML 的 receipt 区域"""
        return html.replace('Spec SHA-256: ', f'Spec SHA-256: {spec_sha256}')

    _TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }} — 股票深度分析报告</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
          --accent: #3b82f6; --border: #334155; --green: #10b981; --red: #ef4444; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'JetBrains Mono', ui-monospace, monospace;
         background: var(--bg); color: var(--text); padding: 2rem; }
  .header { text-align: center; margin-bottom: 2rem; }
  .header h1 { font-size: 1.5rem; color: var(--accent); }
  .header .meta { color: var(--muted); font-size: 0.75rem; margin-top: 0.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 0.5rem; padding: 1.25rem; }
  .card h2 { font-size: 0.875rem; color: var(--accent); margin-bottom: 0.75rem;
             border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
  .value { font-size: 1.5rem; font-weight: 700; }
  .label { color: var(--muted); font-size: 0.75rem; }
  .up { color: var(--green); } .down { color: var(--red); }
  .receipt { margin-top: 2rem; padding: 1rem; background: var(--card);
             border: 1px solid var(--border); border-radius: 0.5rem;
             font-size: 0.7rem; color: var(--muted); }
</style>
</head>
<body>
<div class="header">
  <h1>{{ title }} ({{ code }})</h1>
  <div class="meta">生成时间: {{ timestamp }} | 当前价格: {{ price }}</div>
</div>
<div class="grid">
  <div class="card">
    <h2>量化预测</h2>
    {% set pred = prediction %}
    {% set model = pred.get('model', {}) %}
    <div class="value {% if model.get('ml_direction') == 'up' %}up
      {% elif model.get('ml_direction') == 'down' %}down{% endif %}">
      {{ model.get('ml_direction', 'neutral').upper() }}
    </div>
    <div class="label">置信度: {{ model.get('ml_confidence', 0) }}</div>
    <div class="label">目标价: {{ pred.get('weighted_target', 'N/A') }}</div>
    <div class="label">上行空间: {{ pred.get('upside_space', 'N/A') }}%</div>
  </div>
  <div class="card">
    <h2>SOTA 共识</h2>
    <div class="label">模型数: {{ pred.get('sota_models', {}) | length }}</div>
    <div class="label">共识度: {{ pred.get('sota_agreement', 'N/A') }}</div>
    <div class="label">平均置信度: {{ pred.get('sota_avg_confidence', 'N/A') }}</div>
  </div>
  <div class="card">
    <h2>基本面</h2>
    {% set fund = fundamental %}
    <div class="label">ROE: {{ fund.get('roe', 'N/A') }}</div>
    <div class="label">毛利率: {{ fund.get('gross_margin', 'N/A') }}</div>
    <div class="label">PE: {{ fund.get('pe', 'N/A') }}</div>
  </div>
  <div class="card">
    <h2>技术面</h2>
    {% set tech = technical %}
    <div class="label">RSI: {{ tech.get('rsi', 'N/A') }}</div>
    <div class="label">MACD: {{ tech.get('macd_signal', 'N/A') }}</div>
    <div class="label">波动率: {{ pred.get('daily_volatility', 'N/A') }}%</div>
  </div>
  {% if intel %}
  <div class="card">
    <h2>市场情报</h2>
    {% set dt = intel.get('dragon_tiger') or {} %}
    {% set sen = intel.get('sentiment') or {} %}
    {% set unl = intel.get('unlock') or {} %}
    <div class="label">龙虎榜: {% if dt.get('degraded') %}数据降级{% elif dt.get('on_board') %}近5日上榜 {{ dt.get('count', 0) }} 次 · 最新D20收益{{ dt.get('latest', {}).get('d20_pct', 'N/A') }}%{% elif dt %}近期未上榜{% endif %}</div>
    <div class="label">情绪: {% if sen.get('degraded') %}数据降级{% elif sen %}{{ sen.get('label') or 'N/A' }} ({{ sen.get('score') }}){% endif %}</div>
    <div class="label">解禁: {% if unl.get('degraded') %}数据降级{% elif unl.get('next') %}{{ unl.get('next', {}).get('unlock_date', 'N/A') }} 有解禁{% elif unl %}近期无解禁日程{% endif %}</div>
    {% if insight %}
    <div class="label"><b>AI 投资洞察</b>: {% if insight.get('valid') %}{{ insight.get('direction', 'neutral') }} — {% for e in insight.get('evidence', []) %}{{ e }}; {% endfor %}失效条件: {{ insight.get('invalidation', 'N/A') }}{% else %}证据不足, 弃权 ({{ insight.get('reason', '') }}){% endif %}</div>
    {% endif %}
    <div class="label">注: 证据展示层, 不构成方向投票</div>
  </div>
  {% endif %}
  {% if audit and not audit.get('skipped') %}
  <div class="card">
    <h2>可成交性审计</h2>
    <div class="label">流动性预警: {{ audit.get('liquidity', {}).get('note', 'N/A') }}</div>
    <div class="label">{{ audit.get('cost', {}).get('note', 'N/A') }}</div>
    <div class="label">{{ audit.get('note', '') }}</div>
  </div>
  {% endif %}
</div>
<div class="receipt">
  <div>Spec SHA-256: Spec SHA-256: </div>
  <div>Generated: {{ timestamp }}</div>
</div>
</body>
</html>"""


# ── 模块级单例 ──────────────────────────────────────────────────

_generator = None


def get_report_generator() -> ReportGenerator:
    """获取 ReportGenerator 单例"""
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator
