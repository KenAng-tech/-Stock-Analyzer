#!/bin/bash
set -e

# Change to the stock_analyzer directory
cd /Users/claw/stock_analyzer

# Activate virtual environment
source venv/bin/activate

# Kill any existing processes on port 5002
lsof -ti :5002 | xargs kill -9 2>/dev/null || true

# Start the server
exec python3 server_final.py
