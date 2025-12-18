import os
# Set environment variables FIRST to prevent threading conflicts
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import tempfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp_services import NLPService

app = FastAPI(title="NLP Full-Stack API")

# CORS middleware - allow all localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP service
nlp_service = NLPService()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "NLP API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/sentiment")
async def analyze_sentiment(input_data: TextInput):
    try:
        result = nlp_service.analyze_sentiment(input_data.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ner")
async def extract_entities(input_data: TextInput):
    try:
        result = nlp_service.extract_entities(input_data.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify")
async def classify_text(input_data: TextInput):
    try:
        result = nlp_service.classify_text(input_data.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
async def summarize_text(input_data: TextInput):
    try:
        result = nlp_service.summarize_text(input_data.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        with os.fdopen(temp_fd, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Extract text from PDF
        text = nlp_service.extract_text_from_pdf(temp_path)
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Process all NLP tasks
        results = {
            "text": text,
            "sentiment": nlp_service.analyze_sentiment(text),
            "ner": nlp_service.extract_entities(text),
            "classification": nlp_service.classify_text(text),
            "summary": nlp_service.summarize_text(text)
        }
        
        return JSONResponse(content=results)
    except Exception as e:
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-all")
async def process_all(input_data: TextInput):
    try:
        text = input_data.text
        results = {
            "text": text,
            "sentiment": nlp_service.analyze_sentiment(text),
            "ner": nlp_service.extract_entities(text),
            "classification": nlp_service.classify_text(text),
            "summary": nlp_service.summarize_text(text)
        }
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_available_port(start_port=5000, max_port=5100):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise Exception(f"No available ports found between {start_port} and {max_port}")

if __name__ == "__main__":
    # Check for PORT environment variable (for production hosting)
    import os
    env_port = os.environ.get("PORT")
    
    if env_port:
        # Production: use provided port
        port = int(env_port)
        print(f"Starting backend server on port {port} (from PORT env)")
    else:
        # Development: find available port
        port = find_available_port(5000, 5100)
        print(f"Starting backend server on port {port}")
        print(f"Frontend should connect to: http://localhost:{port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

