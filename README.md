# 📊 SARVANDANI - Text and PDF analysis

A comprehensive full-stack application for Natural Language Processing tasks including Sentiment Analysis, Named Entity Recognition, Text Classification, and Summarization.

## 🌐 Live Application

**🔗 [Try it now: https://fullstack-nlp-sarvandani.onrender.com/](https://fullstack-nlp-sarvandani.onrender.com/)**

## 🚀 Tech Stack

### Backend

- **FastAPI** (0.104.1) - Modern, fast Python web framework for building APIs
- **Uvicorn** (0.24.0) - ASGI server for running FastAPI
- **Transformers** (4.35.2) - Hugging Face library for NLP models
- **PyTorch** (2.1.1) - Deep learning framework for model inference
- **spaCy** (3.7.2) - NLP library for Named Entity Recognition
- **PyPDF2** (3.0.1) - PDF text extraction
- **Python 3.9+** - Programming language

### Frontend

- **HTML5** - Markup language
- **CSS3** - Styling with modern features (gradients, animations)
- **Vanilla JavaScript** - No framework dependencies
- **Fetch API** - For HTTP requests

### NLP Models

- **DistilBERT** - Sentiment Analysis (`distilbert-base-uncased-finetuned-sst-2-english`)
- **spaCy Small** - Named Entity Recognition (`en_core_web_sm`)
- **DistilBART** - Text Summarization (`sshleifer/distilbart-cnn-12-6`)
- **DistilBERT-MNLI** - Text Classification (Primary) (`typeform/distilbert-base-uncased-mnli`)
  - Fast and lightweight (~250MB)
  - Good accuracy for zero-shot classification
  - Automatic recognition of 30+ categories
  - Multi-label classification (detects multiple topics)
- **DeBERTa-v3** - Text Classification (Fallback) (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
  - Best accuracy, trained on MNLI + FEVER + ANLI datasets
  - Larger model (~500MB)
- **BART-large** - Text Classification (Fallback) (`facebook/bart-large-mnli`)

### Development Tools

- **Git** - Version control
- **Docker** (optional) - Containerization for deployment

## 📋 Features

- 📊 **Sentiment Analysis** - Analyze text sentiment (positive/negative)
- 🏷️ **Named Entity Recognition** - Extract entities like names, locations, organizations
- 📝 **Text Classification** - Classify text into categories (technology, business, science, etc.)
- ✂️ **Summarization** - Generate concise summaries of text
- 📄 **PDF Support** - Upload and process PDF files
- ⚡ **Lightweight Models** - Fast local inference using optimized models

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the project**

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLP models (optional - they download automatically on first use):**
   ```bash
   python setup_models.py
   ```

## 🚀 Running the Application

### Start Backend

```bash
cd backend
python main.py
```

The backend will start on an available port (default: 5000-5100).

### Start Frontend

```bash
cd frontend
python -m http.server 4000
```

Then open `http://localhost:4000` in your browser.

## 📦 Project Structure

```
NLP_FULL-STACK_project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── nlp_services.py      # NLP model services
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html          # Single-page frontend
├── models/                 # Cached models (auto-created)
├── setup_models.py         # Model download script
└── README.md              # This file
```

## 🌐 Deployment

**Key Points:**
- Don't upload the `models/` folder (too large, ~2-3GB)
- Models download automatically on first request
- Update frontend API URL to your backend URL
- Deployed on Render.com (free tier)

## 📝 API Endpoints

- `GET /health` - Health check
- `POST /api/sentiment` - Analyze sentiment
- `POST /api/ner` - Extract named entities
- `POST /api/classify` - Classify text
- `POST /api/summarize` - Summarize text
- `POST /api/process-all` - Process all tasks at once
- `POST /api/process-pdf` - Process PDF file

## ⚙️ Configuration

- **Text Length Limits:**
  - Sentiment Analysis: ~1,800 characters (~450 words)
  - Summarization: ~3,500 characters (~875 words)
  - Longer texts are automatically truncated

- **Ports:**
  - Frontend: 4000 (fixed)
  - Backend: 5000-5100 (auto-detected)

## 📄 License

MIT License

## ⚠️ Disclaimer

This application is provided "as is" without any warranties or guarantees. The NLP models used are pre-trained and may not always produce 100% accurate results. Results are for informational purposes only and should not be used as the sole basis for important decisions. The author is not responsible for any consequences arising from the use of this application.

## 👨‍💻 Author

**SARVANDANI**

---

*For questions, issues, or contributions, please open an issue on GitHub.*
