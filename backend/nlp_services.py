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
        """Load high-performance zero-shot classification model"""
        # Using DeBERTa-v3-base-mnli - much better accuracy than DistilBERT
        # Trained on multiple NLI datasets (MNLI, FEVER, ANLI) for superior zero-shot performance
        # Free and open-source model with excellent classification accuracy
        try:
            print("Loading high-performance zero-shot classification model (DeBERTa-v3)...")
            self.classification_pipeline = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                cache_dir=self.models_dir,
                device=-1  # CPU
            )
            print("✅ High-performance classification model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading DeBERTa-v3 model: {e}")
            print("Falling back to DistilBERT model...")
            try:
                # Fallback to DistilBERT if DeBERTa fails
                self.classification_pipeline = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    cache_dir=self.models_dir,
                    device=-1  # CPU
                )
                print("✅ Fallback model (DistilBERT) loaded successfully")
            except Exception as e2:
                print(f"❌ Error loading fallback model: {e2}")
                print("Will use keyword-based classification")
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
        """Classify text into categories using automatic zero-shot classification with comprehensive categories"""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        
        self._ensure_models_loaded()
        
        try:
            # Comprehensive set of categories for automatic recognition
            candidate_labels = [
                "technology and computing",
                "artificial intelligence and machine learning",
                "software development and programming",
                "business and finance",
                "economics and markets",
                "startups and entrepreneurship",
                "science and research",
                "medicine and healthcare",
                "mental health and wellness",
                "education and learning",
                "politics and government",
                "law and legal",
                "news and current events",
                "sports and athletics",
                "entertainment and media",
                "music and arts",
                "movies and television",
                "literature and books",
                "travel and tourism",
                "food and cooking",
                "fashion and style",
                "environment and climate",
                "energy and sustainability",
                "automotive and transportation",
                "real estate and property",
                "personal finance and investing",
                "career and employment",
                "relationships and social",
                "philosophy and religion",
                "history and culture",
                "gaming and esports",
                "social media and internet culture",
                "science fiction and fantasy",
                "general discussion"
            ]
            
            # Use more text for better classification (DeBERTa can handle longer texts)
            # Truncate if too long, but keep more context for better accuracy
            max_chars = 2000  # Increased from 1500 for better context
            if len(text) > max_chars:
                # Keep the beginning (most important) and truncate at word boundary
                text = text[:max_chars].rsplit(' ', 1)[0] + "..."
            
            # Use zero-shot classification with multi-label to detect multiple topics
            if self.classification_pipeline:
                try:
                    # Enable multi-label to detect multiple relevant categories
                    # DeBERTa-v3 is much better at understanding context and relationships
                    result = self.classification_pipeline(
                        text, 
                        candidate_labels, 
                        multi_label=True,
                        hypothesis_template="This text is about {}."  # Better template for classification
                    )
                    
                    # Get all categories with confidence > 0.08 (lower threshold for better detection)
                    # DeBERTa-v3 is more accurate, so we can use a lower threshold
                    relevant_categories = []
                    category_scores = {}
                    
                    for label, score in zip(result["labels"], result["scores"]):
                        category_scores[label] = round(score, 4)
                        if score > 0.08:  # Lower threshold for better multi-label detection
                            relevant_categories.append({
                                "category": label,
                                "confidence": round(score, 4)
                            })
                    
                    # Sort by confidence
                    relevant_categories.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    # Get top 3-5 categories (more for better coverage)
                    top_categories = relevant_categories[:5] if len(relevant_categories) >= 5 else relevant_categories
                    
                    # Primary category is the top one
                    primary_category = top_categories[0]["category"] if top_categories else "general discussion"
                    primary_confidence = top_categories[0]["confidence"] if top_categories else 0.0
                    
                    # Get sentiment for additional context
                    sentiment_result = self.analyze_sentiment(text)
                    
                    return {
                        "category": primary_category,
                        "confidence": primary_confidence,
                        "top_categories": top_categories,  # Top 3-5 categories
                        "all_category_scores": category_scores,  # All scores for reference
                        "sentiment": sentiment_result.get("sentiment", "neutral"),
                        "multi_label": len(top_categories) > 1  # Indicates multiple topics detected
                    }
                except Exception as e:
                    print(f"Zero-shot classification failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall through to improved keyword-based approach
            
            # Enhanced fallback: Comprehensive keyword-based classification
            categories = {
                "technology and computing": ["computer", "software", "tech", "digital", "internet", "code", "programming", "app", "device", "system", "network", "algorithm", "data", "server", "cloud", "hardware", "software", "platform", "application", "database", "cybersecurity"],
                "artificial intelligence and machine learning": ["ai", "artificial intelligence", "machine learning", "deep learning", "neural network", "algorithm", "model", "training", "dataset", "nlp", "natural language", "computer vision", "robotics", "automation"],
                "software development and programming": ["code", "programming", "developer", "coding", "software", "application", "api", "framework", "library", "debug", "git", "repository", "deployment", "backend", "frontend"],
                "business and finance": ["company", "business", "market", "sales", "revenue", "profit", "corporate", "enterprise", "finance", "investment", "stock", "trade", "commerce", "customer", "client", "revenue", "growth", "strategy"],
                "economics and markets": ["economy", "economic", "market", "trading", "stock", "currency", "inflation", "gdp", "unemployment", "financial", "banking", "investment", "portfolio"],
                "startups and entrepreneurship": ["startup", "entrepreneur", "founder", "venture", "funding", "investor", "pitch", "business plan", "innovation", "disrupt"],
                "science and research": ["research", "study", "scientific", "experiment", "hypothesis", "theory", "discovery", "laboratory", "scientist", "physics", "chemistry", "biology", "mathematics", "astronomy", "geology"],
                "medicine and healthcare": ["health", "medical", "doctor", "patient", "treatment", "disease", "medicine", "hospital", "clinic", "symptom", "diagnosis", "therapy", "surgery", "medication", "clinical"],
                "mental health and wellness": ["mental health", "wellness", "therapy", "counseling", "anxiety", "depression", "stress", "mindfulness", "meditation", "psychology", "psychiatrist"],
                "education and learning": ["school", "university", "student", "teacher", "education", "learn", "study", "course", "degree", "academic", "curriculum", "teaching", "learning"],
                "politics and government": ["government", "political", "election", "policy", "law", "president", "minister", "parliament", "democracy", "vote", "campaign", "party", "senate", "congress"],
                "law and legal": ["law", "legal", "attorney", "lawyer", "court", "judge", "lawsuit", "legal case", "jurisdiction", "legislation", "constitution"],
                "news and current events": ["news", "breaking", "report", "journalism", "article", "headline", "current events", "latest", "update"],
                "sports and athletics": ["game", "player", "team", "sport", "match", "championship", "athlete", "coach", "tournament", "competition", "football", "basketball", "soccer", "baseball", "tennis"],
                "entertainment and media": ["entertainment", "media", "celebrity", "show", "television", "tv", "broadcast", "journalism"],
                "music and arts": ["music", "song", "artist", "album", "concert", "performance", "art", "painting", "sculpture", "gallery", "exhibition"],
                "movies and television": ["movie", "film", "cinema", "actor", "director", "series", "episode", "netflix", "hollywood"],
                "literature and books": ["book", "novel", "author", "writing", "literature", "poetry", "publisher", "reading"],
                "travel and tourism": ["travel", "trip", "vacation", "tourist", "destination", "hotel", "flight", "journey", "explore"],
                "food and cooking": ["food", "cooking", "recipe", "restaurant", "cuisine", "chef", "meal", "dining", "kitchen"],
                "fashion and style": ["fashion", "style", "clothing", "designer", "outfit", "trend", "wardrobe"],
                "environment and climate": ["environment", "climate", "global warming", "pollution", "sustainability", "green", "renewable", "carbon", "emissions"],
                "energy and sustainability": ["energy", "solar", "wind", "renewable", "sustainability", "green energy", "power", "electricity"],
                "automotive and transportation": ["car", "vehicle", "automotive", "driving", "transportation", "highway", "traffic"],
                "real estate and property": ["real estate", "property", "house", "home", "apartment", "mortgage", "rent", "buying"],
                "personal finance and investing": ["investment", "savings", "retirement", "portfolio", "stocks", "bonds", "financial planning"],
                "career and employment": ["career", "job", "employment", "work", "profession", "resume", "interview", "salary"],
                "relationships and social": ["relationship", "friendship", "social", "dating", "marriage", "family", "community"],
                "philosophy and religion": ["philosophy", "religion", "spiritual", "belief", "faith", "theology", "ethics", "moral"],
                "history and culture": ["history", "historical", "culture", "tradition", "heritage", "ancient", "civilization"],
                "gaming and esports": ["game", "gaming", "video game", "esports", "gamer", "console", "pc gaming"],
                "social media and internet culture": ["social media", "facebook", "twitter", "instagram", "tiktok", "viral", "meme", "online"],
                "science fiction and fantasy": ["sci-fi", "science fiction", "fantasy", "futuristic", "space", "alien", "dystopian"]
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
            
            # Get top 3 categories
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_categories[:3]
            
            # Primary category
            top_category = top_3[0][0] if top_3 and top_3[0][1] > 0 else "general discussion"
            max_score = top_3[0][1] if top_3 else 0
            
            # Calculate confidence based on score ratio
            total_score = sum(scores.values())
            confidence = round(max_score / total_score, 4) if total_score > 0 else 0.0
            
            # Get top 3 categories with confidence scores
            top_categories = []
            for cat, score in top_3:
                if score > 0:
                    cat_confidence = round(score / total_score, 4) if total_score > 0 else 0.0
                    top_categories.append({
                        "category": cat,
                        "confidence": cat_confidence
                    })
            
            # Normalize all scores
            normalized_scores = {k: round(v / total_score, 4) if total_score > 0 else 0 for k, v in scores.items()}
            
            # Also get sentiment for additional context
            sentiment_result = self.analyze_sentiment(text)
            
            return {
                "category": top_category,
                "confidence": confidence,
                "top_categories": top_categories,  # Top 3 categories
                "all_category_scores": normalized_scores,  # All scores for reference
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "multi_label": len(top_categories) > 1  # Indicates multiple topics detected
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

