#!/usr/bin/env python3
"""
Stock Analyzer - Threading Mode Startup
Uses threading instead of eventlet for better macOS compatibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch to disable eventlet
os.environ['EVENTLET_NO_GREENDNS'] = '1'

# Import and patch before app initialization
import flask_socketio
original_socketio_init = flask_socketio.SocketIO.__init__

def patched_socketio_init(self, app=None, *args, **kwargs):
    # Force threading mode
    kwargs['async_mode'] = 'threading'
    return original_socketio_init(self, app, *args, **kwargs)

flask_socketio.SocketIO.__init__ = patched_socketio_init

# Now import the app
from app import app, socketio

# Reinitialize socketio with threading
socketio = flask_socketio.SocketIO(app, cors_allowed_origins="*", async_mode='threading')

print("=" * 60)
print("  Stock Analyzer System (Threading Mode)")
print("=" * 60)
print(f"  Server: http://127.0.0.1:5002")
print(f"  API: http://127.0.0.1:5002/api/stock/sz300620")
print("=" * 60)

# Start with socketio for WebSocket support
socketio.run(app, host='0.0.0.0', port=5002, debug=False, allow_unsafe_werkzeug=True)
