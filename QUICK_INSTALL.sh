#!/bin/bash
# Quick installation script for Python 3.11

echo "=========================================="
echo "Python 3.11 Installation for NLP Backend"
echo "=========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Step 1: Installing Homebrew..."
    echo "   (This will ask for your password)"
    echo ""
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH (for Apple Silicon Macs)
    if [ -f /opt/homebrew/bin/brew ]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    # For Intel Macs
    elif [ -f /usr/local/bin/brew ]; then
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew is already installed!"
fi

echo ""
echo "📦 Step 2: Installing Python 3.11..."
brew install python@3.11

echo ""
echo "✅ Step 3: Verifying installation..."
python3.11 --version

echo ""
echo "📦 Step 4: Installing backend dependencies..."
cd "$(dirname "$0")/backend"
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt
python3.11 -m spacy download en_core_web_sm

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To start the backend, run:"
echo "  cd backend"
echo "  python3.11 start_server.py"
echo ""

