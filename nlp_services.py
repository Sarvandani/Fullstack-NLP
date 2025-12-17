import os
import re
from typing import Dict, List, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForSequenceClassification as AutoModel
)
import spacy
from PyPDF2 import PdfReader

class NLPService:
    def __init__(self):
        """Initialize all NLP models"""
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Lazy loading - models will be loaded on first use
        self.sentiment_pipeline = None
        self.nlp = None
        self.classification_pipeline = None
        self.summarizer_pipeline = None
        self.models_loaded = False
        
        print("NLP Service initialized (models will load on first request)")
    
    def _load_sentiment_model(self):
        """Load DistilBERT sentiment analysis model"""
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        try:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.models_dir
            )
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.models_dir
            )
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.sentiment_tokenizer,
                device=-1,  # CPU
                truncation=True,
                max_length=512
            )
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            raise
    
    def _load_ner_model(self):
        """Load spaCy NER model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
    
    def _load_classification_model(self):
        """Load zero-shot classification model for better accuracy"""
        # Using a lightweight zero-shot classification model
        # This can classify text into any categories without fine-tuning
        model_name = "facebook/bart-large-mnli"
        try:
            # Use a smaller, faster model for zero-shot classification
            # BART-large-mnli is good but heavy, let's use a smaller alternative
            # Actually, let's use a more lightweight zero-shot model
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                cache_dir=self.models_dir,
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Error loading zero-shot classification model, falling back to keyword-based: {e}")
            # Fallback: will use keyword-based approach
            self.classification_pipeline = None
    
    def _load_summarization_model(self):
        """Load DistilBART for summarization"""
        model_name = "sshleifer/distilbart-cnn-12-6"
        try:
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.models_dir
            )
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.models_dir
            )
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.summarizer_model,
                tokenizer=self.summarizer_tokenizer,
                device=-1,  # CPU
                truncation=True
            )
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            raise
    
    def _ensure_models_loaded(self):
        """Load models if not already loaded"""
        if self.models_loaded:
            return
        
        print("Loading NLP models (first request - this may take a minute)...")
        try:
            self._load_sentiment_model()
            self._load_ner_model()
            self._load_classification_model()
            self._load_summarization_model()
            self.models_loaded = True
            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        self._ensure_models_loaded()
        
        try:
            # Truncate text to safe character length before processing
            # DistilBERT max is 512 tokens, average ~4 chars per token = ~2000 chars
            # Be conservative: use 1800 chars to leave margin
            max_chars = 1800
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(' ', 1)[0] + "..."  # Cut at word boundary
            
            # Pipeline should handle tokenization and truncation, but we pre-truncate for safety
            result = self.sentiment_pipeline(text)[0]
            
            return {
                "label": result["label"],
                "score": round(result["score"], 4),
                "sentiment": "positive" if result["label"] == "POSITIVE" else "negative"
            }
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}"}
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        self._ensure_models_loaded()
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            # Group by entity type
            entity_groups = {}
            for ent in entities:
                label = ent["label"]
                if label not in entity_groups:
                    entity_groups[label] = []
                entity_groups[label].append(ent["text"])
            
            return {
                "entities": entities,
                "entity_groups": entity_groups,
                "total_entities": len(entities)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text into categories using zero-shot classification"""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        self._ensure_models_loaded()
        
        try:
            # Define categories with better descriptions for zero-shot classification
            candidate_labels = [
                "technology",
                "business",
                "science",
                "health",
                "sports",
                "politics",
                "entertainment",
                "education",
                "general"
            ]
            
            # Truncate text if too long (zero-shot models have token limits)
            max_chars = 1000
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(' ', 1)[0] + "..."
            
            # Use zero-shot classification if available
            if self.classification_pipeline:
                try:
                    result = self.classification_pipeline(text, candidate_labels, multi_label=False)
                    
                    # Get top category and confidence
                    top_category = result["labels"][0] if result["labels"] else "general"
                    confidence = result["scores"][0] if result["scores"] else 0.0
                    
                    # Create category scores dictionary
                    category_scores = {}
                    for label, score in zip(result["labels"], result["scores"]):
                        category_scores[label] = round(score, 4)
                    
                    # Get sentiment for additional context
                    sentiment_result = self.analyze_sentiment(text)
                    
                    return {
                        "category": top_category,
                        "confidence": round(confidence, 4),
                        "category_scores": category_scores,
                        "sentiment": sentiment_result.get("sentiment", "neutral")
                    }
                except Exception as e:
                    print(f"Zero-shot classification failed, using fallback: {e}")
                    # Fall through to keyword-based approach
            
            # Fallback: Improved keyword-based classification
            categories = {
                "technology": ["computer", "software", "tech", "digital", "internet", "code", "programming", "app", "device", "system", "network", "algorithm", "data", "server", "cloud", "ai", "artificial intelligence", "machine learning"],
                "business": ["company", "business", "market", "sales", "revenue", "profit", "corporate", "enterprise", "finance", "investment", "stock", "trade", "commerce", "customer", "client"],
                "science": ["research", "study", "scientific", "experiment", "hypothesis", "theory", "discovery", "laboratory", "scientist", "physics", "chemistry", "biology", "mathematics"],
                "health": ["health", "medical", "doctor", "patient", "treatment", "disease", "medicine", "hospital", "clinic", "symptom", "diagnosis", "therapy", "wellness"],
                "sports": ["game", "player", "team", "sport", "match", "championship", "athlete", "coach", "tournament", "competition", "football", "basketball", "soccer"],
                "politics": ["government", "political", "election", "policy", "law", "president", "minister", "parliament", "democracy", "vote", "campaign", "party"],
                "entertainment": ["movie", "film", "music", "show", "actor", "singer", "celebrity", "entertainment", "theater", "concert", "performance"],
                "education": ["school", "university", "student", "teacher", "education", "learn", "study", "course", "degree", "academic"]
            }
            
            text_lower = text.lower()
            scores = {}
            
            # Improved scoring: count keyword matches and use word boundaries
            import re
            for category, keywords in categories.items():
                score = 0
                for keyword in keywords:
                    # Use word boundaries for better matching
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = len(re.findall(pattern, text_lower))
                    score += matches
                scores[category] = score
            
            # Get top category
            top_category = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
            max_score = scores[top_category] if top_category in scores else 0
            
            # Calculate confidence based on score ratio
            total_score = sum(scores.values())
            confidence = round(max_score / total_score, 4) if total_score > 0 else 0.0
            
            # Also get sentiment for additional context
            sentiment_result = self.analyze_sentiment(text)
            
            return {
                "category": top_category,
                "confidence": confidence,
                "category_scores": {k: round(v / total_score, 4) if total_score > 0 else 0 for k, v in scores.items()},
                "sentiment": sentiment_result.get("sentiment", "neutral")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def summarize_text(self, text: str) -> Dict[str, Any]:
        """Summarize the text"""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        self._ensure_models_loaded()
        
        try:
            # DistilBART max input length is 1024 tokens
            # Average ~4 chars per token, so 1024 tokens ≈ 4000 chars
            # Be conservative: use 3500 chars to leave margin
            max_input_chars = 3500
            min_length = 30
            max_output_length = 150
            
            # Truncate text to safe length before processing
            if len(text) > max_input_chars:
                text = text[:max_input_chars].rsplit(' ', 1)[0] + "..."  # Cut at word boundary
            
            # For very short texts, adjust min_length
            if len(text) < 100:
                min_length = max(10, len(text) // 10)
                max_output_length = min(50, len(text) // 2)
            
            # Use pipeline - it will handle tokenization
            result = self.summarizer_pipeline(
                text,
                max_length=max_output_length,
                min_length=min_length,
                do_sample=False
            )
            
            # Check if result is valid
            if not result or len(result) == 0:
                return {"error": "Summarization failed - no output generated"}
            
            summary = result[0].get("summary_text", "")
            if not summary:
                return {"error": "Summarization failed - empty summary"}
            
        except (IndexError, KeyError) as e:
            # Handle index out of range or key errors
            return {"error": f"Summarization failed - text may be too short or invalid: {str(e)}"}
        except Exception as e:
            return {"error": f"Summarization failed: {str(e)}"}
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text), 4) if len(text) > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

