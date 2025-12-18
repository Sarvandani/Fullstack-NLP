#!/bin/bash
# Wrapper script to start backend with proper environment variables
# This prevents mutex lock errors on macOS

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

cd "$(dirname "$0")"
python3 start_server.py

