"""
Static Routes - /, /webgui.html, /sota_dashboard.html, /dl_dashboard.html

Extracted from app.py (lines 657-666, 3092-3101, 4863-4872).
"""

from flask import Blueprint, send_file

bp = Blueprint('static', __name__)


@bp.route('/')
def index():
    """Main dashboard page"""
    from flask import render_template
    return render_template('index.html')


@bp.route('/webgui.html')
def webgui():
    """Quant webgui page"""
    return send_file("webgui.html")


@bp.route('/dl_dashboard.html')
def dl_dashboard():
    """Deep Learning Dashboard page"""
    return send_file("templates/dl_dashboard.html")


@bp.route('/sota_dashboard.html')
def sota_dashboard():
    """SOTA Quantitative Model Dashboard page"""
    return send_file("sota_dashboard.html")
