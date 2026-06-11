#!/bin/bash
# Stock Analyzer - Startup Script

echo "=========================================="
echo "  Stock Analyzer System"
echo "=========================================="
echo ""

# Activate virtual environment
cd "$(dirname "$0")"
source venv/bin/activate

# Create necessary directories
mkdir -p data logs

# Start the application
echo "Starting Stock Analyzer server..."
echo "URL: http://localhost:5000"
echo "API: http://localhost:5000/api/stock/sz300620"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python app.py
