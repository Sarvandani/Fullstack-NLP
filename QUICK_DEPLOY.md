# ⚡ Quick Deployment Guide

## TL;DR - What to Upload

### ✅ Upload These Files:

```
backend/
  ├── main.py
  ├── nlp_services.py
  └── requirements.txt

frontend/
  └── index.html
```

### ❌ DON'T Upload:

- `models/` folder (too large, ~2-3GB)
- Log files
- `__pycache__/` folders

**Models download automatically on first run!**

---

## 🚀 Render.com (Easiest - Recommended)

### Backend Setup:

1. Go to [render.com](https://render.com) → Sign up
2. Click "New" → "Web Service"
3. Connect your GitHub repo OR upload files
4. Settings:
   - **Name:** nlp-backend
   - **Environment:** Python 3
   - **Build Command:** `pip install -r backend/requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command:** `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

5. Click "Create Web Service"
6. Copy your backend URL (e.g., `https://nlp-backend.onrender.com`)

### Frontend Setup:

1. In Render, click "New" → "Static Site"
2. Upload `frontend/index.html` OR connect GitHub
3. **Before deploying:** Update API URL in `index.html`:
   ```javascript
   // Change this line:
   let API_URL = 'http://localhost:5000';
   
   // To your backend URL:
   let API_URL = 'https://nlp-backend.onrender.com';
   ```
4. Deploy!

**Done!** First request will download models (takes 2-5 min).

---

## 🚂 Railway.app

1. Go to [railway.app](https://railway.app) → Sign up
2. "New Project" → "Deploy from GitHub repo"
3. Add service → Select your repo
4. Railway auto-detects Python
5. Set start command: `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Update frontend API URL to Railway backend URL

---

## 📝 Important: Update Frontend

**Before deploying frontend, update the API URL:**

In `frontend/index.html`, find:
```javascript
let API_URL = 'http://localhost:5000';
```

Change to your backend URL:
```javascript
let API_URL = 'https://your-backend-url.onrender.com';
```

Or use auto-detection:
```javascript
// Auto-detect based on hostname
const isLocalhost = window.location.hostname === 'localhost' || 
                    window.location.hostname === '127.0.0.1';
let API_URL = isLocalhost 
    ? 'http://localhost:5000' 
    : 'https://your-backend-url.onrender.com';
```

---

## ⚠️ Important Notes

1. **Models:** Don't upload models folder. They download automatically (~2-3GB, takes 2-5 min first time)

2. **First Request:** Will be slow (downloading models). This is normal!

3. **Free Tier Limits:**
   - Services may sleep after inactivity
   - First request after sleep = slow (re-downloading models)
   - Consider using a "keep-alive" service

4. **Memory:** Need ~2-3GB RAM for models. Check hosting limits.

---

## 🎯 Recommended Setup

**Best Free Option:**
- **Backend:** Render.com (Web Service)
- **Frontend:** Render.com (Static Site) or Netlify
- **Models:** Auto-download (don't upload)

**Alternative:**
- Railway.app (can host both backend and frontend)

---

## ✅ Deployment Checklist

- [ ] Backend deployed and running
- [ ] Frontend API URL updated
- [ ] CORS allows frontend domain
- [ ] Test backend: `https://your-backend.onrender.com/health`
- [ ] Test frontend can connect
- [ ] First request works (may take 2-5 min for models)

---

## 🆘 Troubleshooting

**"Models not downloading"**
- Check hosting service has internet access
- Check logs for errors
- Verify Hugging Face is accessible

**"CORS error"**
- Update `backend/main.py` CORS settings
- Add your frontend domain to allowed origins

**"Port error"**
- Use `$PORT` environment variable
- Don't hardcode port numbers

---

## 📦 Using the Deploy Script

Run this to create a clean package:

```bash
./deploy.sh
```

This creates a `deploy_package/` folder with only necessary files.

---

**That's it! Your app should be live! 🎉**

