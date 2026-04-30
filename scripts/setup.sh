#!/bin/bash
# Create setup.sh script
echo "🔧 Setting up Environment..."

# Check Python version
python3 --version || { echo "❌ Python 3.12+ required"; exit 1; }

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
source .venv/bin/activate

# Install the packages
set -e
.venv/bin/pip install -r requirements.txt

echo ""
echo "✅ Setup complete! Make sure to activate virtual environment by running source .venv/bin/activate."
