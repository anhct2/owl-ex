#!/bin/bash

echo "======================================"
echo "Testing DeepSeek Model with Owl"
echo "======================================"

# Kiểm tra xem file có tồn tại không
if [ ! -f "examples/run_deepseek.py" ]; then
    echo "❌ Error: examples/run_deepseek.py not found"
    exit 1
fi

# Kiểm tra DEEPSEEK_API_KEY
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️ Warning: DEEPSEEK_API_KEY environment variable is not set"
    echo "You may need to set it in your .env file or export it manually:"
    echo "export DEEPSEEK_API_KEY=your_api_key_here"
    # Cố gắng load từ .env nếu có
    if [ -f ".env" ]; then
        source .env
        echo "Loaded environment from .env file"
    fi
    
    if [ -f "owl/.env" ]; then
        source owl/.env
        echo "Loaded environment from owl/.env file"
    fi
fi

echo "Running DeepSeek model test..."
echo ""

# Gọi script với prompt test
python examples/run_deepseek.py "Search for information about quantum computing and provide a short explanation of quantum bits (qubits). Then create a simple Python code example that simulates a basic quantum operation."

echo ""
echo "======================================"
echo "Test completed"
echo "======================================" 