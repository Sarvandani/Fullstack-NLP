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
    
    # 3. Classification Model (Zero-shot) - Try multiple options
    print("\n[3/4] Downloading Text Classification Model (Zero-shot)...")
    print("Trying models in order: BART-large (fastest) -> DistilBERT (lightweight) -> DeBERTa-v3 (best accuracy)")
    
    models_to_try = [
        ("typeform/distilbert-base-uncased-mnli", "DistilBERT (~250MB, fastest, good accuracy)"),
        ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "DeBERTa-v3 (~500MB, best accuracy)"),
        ("facebook/bart-large-mnli", "BART-large (~1.6GB, very good accuracy)")
    ]
    
    downloaded = False
    for model_name, description in models_to_try:
        try:
            print(f"\n  Trying: {model_name}")
            print(f"  {description}")
            from transformers import pipeline
            # Pre-download the zero-shot classification model
            classification_pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                cache_dir=models_dir,
                device=-1  # CPU
            )
            print(f"  ✅ {model_name} downloaded successfully!")
            downloaded = True
            break  # Success, stop trying other models
        except Exception as e:
            print(f"  ⚠️  Error downloading {model_name}: {e}")
            continue
    
    if not downloaded:
        print("  ❌ All classification models failed to download")
        print("  Will use keyword-based classification as fallback")
    
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
    print("\n💡 Tip: Models are cached locally, so they won't download again on next startup.")
    print("   The first request may still take a moment to load models into memory.")

if __name__ == "__main__":
    download_models()

