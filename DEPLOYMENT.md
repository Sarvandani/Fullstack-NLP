# 🚀 Deployment Guide - Free Hosting

This guide explains how to deploy your NLP Full-Stack application to free hosting services.

## 📦 What Files to Deploy

### ✅ Files to Include (Upload to Host)

**Backend:**
- `backend/main.py`
- `backend/nlp_services.py`
- `backend/requirements.txt`
- `backend/start_server.py` (optional)

**Frontend:**
- `frontend/index.html`

**Root:**
- `setup_models.py` (optional, for manual setup)

### ❌ Files to EXCLUDE (Don't Upload)

- `models/` directory (too large, ~2-3GB)
- `backend.log` / `frontend.log`
- `__pycache__/` directories
- `.git/` directory
- Any local test files

**Models will be downloaded automatically on first run!**

---

## 🌐 Recommended Free Hosting Options

### Option 1: Render (Recommended) ⭐

**Pros:** Easy setup, free tier, automatic deployments

**Steps:**

1. **Create accounts:**
   - Backend: [render.com](https://render.com) - Create a Web Service
   - Frontend: [render.com](https://render.com) - Create a Static Site

2. **Deploy Backend:**
   ```bash
   # Connect your GitHub repo
   # Build Command: (leave empty or: pip install -r backend/requirements.txt)
   # Start Command: cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

3. **Deploy Frontend:**
   - Upload `frontend/index.html`
   - Update API URL in `index.html` to your backend URL
   - Or use environment variable

4. **Environment Variables (Backend):**
   ```
   PORT=10000
   ```

**Free Tier Limits:**
- 750 hours/month
- Spins down after 15 min inactivity
- Models download on first request (may take 2-3 min)

---

### Option 2: Railway

**Pros:** Good free tier, easy deployment

**Steps:**

1. Sign up at [railway.app](https://railway.app)
2. Create new project from GitHub
3. Add service for backend
4. Set start command: `cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Railway auto-detects Python and installs requirements

**Free Tier:**
- $5 credit/month
- Models download on first run

---

### Option 3: PythonAnywhere

**Pros:** Free tier, good for Python apps

**Steps:**

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files via web interface
3. Create web app (Flask/WSGI)
4. Configure to run FastAPI with uvicorn

**Free Tier:**
- Limited CPU time
- Models download on first request

---

### Option 4: Fly.io

**Pros:** Good free tier, Docker support

**Steps:**

1. Install Fly CLI
2. Create `Dockerfile` (see below)
3. Run: `fly launch`
4. Deploy: `fly deploy`

---

## 📝 Important Configuration

### 1. Update Backend for Production

Create `backend/main.py` with production settings:

```python
# At the end of main.py, replace the if __name__ block:
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 2. Update Frontend API URL

In `frontend/index.html`, update the API URL:

```javascript
// Change this line:
let API_URL = 'http://localhost:5000';

// To your backend URL:
let API_URL = 'https://your-backend-url.onrender.com';
```

Or use environment detection:

```javascript
// Auto-detect based on hostname
let API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000' 
    : 'https://your-backend-url.onrender.com';
```

### 3. CORS Configuration

Make sure CORS allows your frontend domain:

```python
# In backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐳 Docker Deployment (Optional)

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY backend/ ./backend/
COPY setup_models.py .

# Pre-download models (optional, takes time but faster first request)
# RUN python setup_models.py

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📋 Deployment Checklist

- [ ] Backend files uploaded
- [ ] Frontend files uploaded
- [ ] `requirements.txt` includes all dependencies
- [ ] API URL updated in frontend
- [ ] CORS configured for frontend domain
- [ ] Port configuration set (use `$PORT` env variable)
- [ ] Models directory in `.gitignore` (already done)
- [ ] Test backend health endpoint
- [ ] Test frontend can connect to backend

---

## ⚡ Quick Deploy Script

Create `deploy.sh`:

```bash
#!/bin/bash

# Clean up
rm -rf models/__pycache__ backend/__pycache__

# Create deployment package
tar -czf deploy.tar.gz \
    backend/ \
    frontend/ \
    setup_models.py \
    README.md \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='models'

echo "✅ Deployment package created: deploy.tar.gz"
echo "📦 Upload this to your hosting service"
```

---

## 🔧 Environment Variables

Set these in your hosting platform:

**Backend:**
- `PORT` - Server port (usually auto-set by host)
- `PYTHONUNBUFFERED=1` - For better logging

**Optional:**
- `MODELS_DIR` - Custom models directory (default: `./models`)

---

## ⚠️ Important Notes

1. **Models Download:**
   - Models are NOT uploaded (too large ~2-3GB)
   - They download automatically on first request
   - First request may take 2-5 minutes
   - Subsequent requests are fast

2. **Free Tier Limitations:**
   - Services may spin down after inactivity
   - First request after spin-down will be slow (model download)
   - Consider keeping service "awake" with a ping service

3. **Memory Requirements:**
   - Models need ~2-3GB RAM
   - Some free tiers may not have enough
   - Check hosting provider limits

4. **Cold Start:**
   - First request downloads models
   - Can take 2-5 minutes
   - Consider pre-warming the service

---

## 🚀 Recommended Setup

**Best for Free Hosting:**
1. **Backend:** Render or Railway
2. **Frontend:** Render Static Site or Netlify
3. **Models:** Download on first run (automatic)

**Alternative:**
- Use a single service that hosts both (like Railway with two services)

---

## 📞 Troubleshooting

**Models not downloading?**
- Check internet connection on host
- Verify Hugging Face access
- Check logs for errors

**CORS errors?**
- Update CORS origins in `backend/main.py`
- Include your frontend domain

**Port errors?**
- Use `$PORT` environment variable
- Don't hardcode port numbers

**Slow first request?**
- Normal! Models are downloading
- Consider pre-downloading in Dockerfile

---

## 🎯 Quick Start (Render Example)

1. Push code to GitHub
2. Connect GitHub to Render
3. Create Web Service (backend)
4. Create Static Site (frontend)
5. Update frontend API URL
6. Deploy!

**That's it!** Models download automatically on first use.

