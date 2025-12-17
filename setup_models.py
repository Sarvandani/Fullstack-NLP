#!/usr/bin/env python3
"""
Setup script to download all NLP models locally
Run this before starting the application
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import subprocess

def download_models():
    """Download all required models"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("Downloading NLP Models")
    print("=" * 60)
    
    # 1. Sentiment Analysis Model
    print("\n[1/4] Downloading Sentiment Analysis Model...")
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            cache_dir=models_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            cache_dir=models_dir
        )
        print("✅ Sentiment model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading sentiment model: {e}")
    
    # 2. NER Model (spaCy)
    print("\n[2/4] Downloading NER Model (spaCy)...")
    print("Model: en_core_web_sm")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ spaCy NER model downloaded successfully!")
        else:
            print(f"⚠️  spaCy download output: {result.stdout}")
            print(f"⚠️  Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")
    
    # 3. Classification Model
    print("\n[3/4] Downloading Text Classification Model...")
    print("Model: distilbert-base-uncased")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased",
            cache_dir=models_dir
        )
        print("✅ Classification model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading classification model: {e}")
    
    # 4. Summarization Model
    print("\n[4/4] Downloading Summarization Model...")
    print("Model: sshleifer/distilbart-cnn-12-6")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=models_dir
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            cache_dir=models_dir
        )
        print("✅ Summarization model downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading summarization model: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All models downloaded! You can now start the application.")
    print("=" * 60)
    print(f"\nModels are cached in: {models_dir}")

if __name__ == "__main__":
    download_models()

