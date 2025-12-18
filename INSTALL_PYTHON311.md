# Install Python 3.11 - Step by Step Guide

## Option 1: Install Homebrew + Python 3.11 (Recommended)

### Step 1: Install Homebrew
Open Terminal and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Follow the prompts and enter your password when asked.

### Step 2: Install Python 3.11
After Homebrew is installed, run:
```bash
brew install python@3.11
```

### Step 3: Verify Installation
```bash
python3.11 --version
```
Should show: `Python 3.11.x`

### Step 4: Install Dependencies
```bash
cd backend
python3.11 -m pip install -r requirements.txt
python3.11 -m spacy download en_core_web_sm
```

### Step 5: Start Backend
```bash
python3.11 start_server.py
```

---

## Option 2: Download Python 3.11 from python.org

1. Visit: https://www.python.org/downloads/
2. Download Python 3.11.x for macOS
3. Run the installer
4. After installation, use: `python3.11` instead of `python3`

---

## Option 3: Use pyenv (Advanced)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Install Python 3.11
pyenv install 3.11.9
pyenv global 3.11.9
```

---

## Quick Test After Installation

Once Python 3.11 is installed, test the backend:

```bash
cd /Users/yaser/Desktop/NLP_FULL-STACK_project/backend
python3.11 start_server.py
```

The mutex lock error should be resolved! ✅

