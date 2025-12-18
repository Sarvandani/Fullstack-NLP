# Backend Startup Issue - Mutex Lock Error

## Problem
On macOS with Python 3.9, you may encounter: `mutex lock failed: Invalid argument`

This is a known compatibility issue between PyTorch/transformers and Python 3.9 on macOS.

## Solutions

### Option 1: Use Python 3.11+ (Recommended)
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Use it to start the backend
cd backend
python3.11 start_server.py
```

### Option 2: Use the wrapper script
```bash
cd backend
./start_backend.sh
```

### Option 3: Manual start with environment variables
```bash
cd backend
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 start_server.py
```

### Option 4: If still failing, try this workaround
The backend may work if you start it and wait - sometimes it recovers after the initial error.

## Current Status
- Backend code is ready
- Models are configured correctly  
- Issue is system-level compatibility

Try Option 1 first (Python 3.11+) - it usually resolves the issue completely.

