"""
SOTA Routes — Dispatcher (thin wrapper that registers all sub-blueprints)

Previously contained 1,811 lines and 55 functions.
Now delegates to 4 focused blueprint modules.
"""

from flask import Blueprint

# Import all sub-blueprints (side-effect: registers routes)
from .sota_predict_routes import bp as sota_predict_bp
from .sota_status_routes import bp as sota_status_bp
from .sota_advanced_routes import bp as sota_advanced_bp
from .sota_llm_routes import bp as sota_llm_bp

# Register blueprints under /api/sota prefix
# Note: Each sub-blueprint already defines its own bp with the full prefix.
# To avoid conflicts, we register them without prefix and let each handle its own URL.
# Actually, since each sub-bp uses @bp.route('/api/sota/...'), they register absolute paths.
# We just need to collect them.

def register_sota_routes(app):
    """Register all SOTA sub-blueprints with the Flask app"""
    app.register_blueprint(sota_predict_bp)
    app.register_blueprint(sota_status_bp)
    app.register_blueprint(sota_advanced_bp)
    app.register_blueprint(sota_llm_bp)

