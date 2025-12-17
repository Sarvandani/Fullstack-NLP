# 🚀 How to Deploy to Render.com (Free Hosting)

Complete step-by-step guide to deploy your NLP app on Render.com for FREE.

---

## 📋 Overview

**What you'll do:**
1. ✅ Upload code to GitHub (without models)
2. ✅ Deploy backend on Render
3. ✅ Deploy frontend on Render
4. ✅ Models download automatically (no manual upload needed!)

**Time needed:** 15-20 minutes

---

## Part 1: GitHub Setup (Upload Your Code)

### Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon → **"New repository"**
3. Fill in:
   - **Repository name:** `nlp-fullstack-app` (or any name)
   - **Visibility:** **Public** (required for free hosting)
   - **DO NOT** initialize with README
4. Click **"Create repository"**

---

### Step 2: Prepare Files for GitHub

**✅ Files to upload to GitHub:**
```
✅ backend/
   ├── main.py
   ├── nlp_services.py
   └── requirements.txt
✅ frontend/
   └── index.html
✅ README.md
✅ .gitignore (hidden file - see note below)
✅ setup_models.py (optional)
```

**📝 Note about .gitignore:**
- `.gitignore` is a **hidden file** (starts with a dot)
- On Mac: Press `Cmd + Shift + .` in Finder to show hidden files
- On Windows: Enable "Show hidden files" in File Explorer
- **OR** just use command line (git will find it automatically)
- **OR** create it manually if missing (see below)

**❌ Files to EXCLUDE (don't upload):**
```
❌ models/ folder (too large, ~2-3GB)
❌ *.log files
❌ __pycache__/ folders
❌ Any cached files
```

**✅ Good news:** Your `.gitignore` already excludes these!

---

### Step 3: Push Code to GitHub

**Option A: Using Command Line**

```bash
# Navigate to your project folder
cd /Users/yaser/Desktop/NLP_FULL-STACK_project

# Initialize git (if not already done)
git init

# Add files (models/ is automatically excluded by .gitignore)
git add backend/ frontend/ README.md .gitignore setup_models.py

# Commit
git commit -m "Initial commit: NLP Full-Stack App"

# Connect to GitHub
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nlp-fullstack-app.git
git push -u origin main
```

**Option B: Using GitHub Desktop**
1. Download [GitHub Desktop](https://desktop.github.com)
2. File → Add Local Repository
3. Select your project folder
4. Commit and push

**Option C: Upload via GitHub Website**
1. In your GitHub repo, click "uploading an existing file"
2. Drag and drop: `backend/`, `frontend/`, `README.md`, `.gitignore`
3. Commit changes

---

### ✅ GitHub Checklist

- [ ] Repository created on GitHub
- [ ] Code files uploaded (backend/, frontend/, README.md, .gitignore)
- [ ] Models folder NOT uploaded (check .gitignore)
- [ ] Code is on GitHub and visible

---

## Part 2: Render Setup (Deploy Your App)

### Step 4: Create Render Account

1. Go to [render.com](https://render.com)
2. Click **"Get Started for Free"**
3. Sign up with **GitHub** (easiest option)
4. Authorize Render to access your GitHub

---

### Step 5: Deploy Backend on Render

1. **In Render dashboard, click "New +"**
2. **Select "Web Service"**

3. **Connect Repository:**
   - Select your GitHub account
   - Choose repository: `nlp-fullstack-app`
   - Click **"Connect"**

4. **Configure Settings:**
   ```
   Name: nlp-backend
   Region: Choose closest to you
   Branch: main
   Root Directory: (leave empty)
   Environment: Python 3
   Plan: Free
   ```

5. **Build & Deploy Settings:**
   ```
   Build Command:
   pip install -r backend/requirements.txt && python -m spacy download en_core_web_sm
   
   Start Command:
   cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

6. **Click "Create Web Service"**

7. **Wait for deployment** (5-10 minutes first time)
   - Watch the logs
   - Wait for "Your service is live" message

8. **Copy your backend URL:**
   - Example: `https://nlp-backend.onrender.com`
   - **Save this URL!** You'll need it next.

---

### Step 6: Update Frontend API URL

**Before deploying frontend, connect it to your backend:**

1. **On your computer, open:** `frontend/index.html`

2. **Find this line (around line 336):**
   ```javascript
   let API_URL = 'https://your-backend-url.onrender.com';
   ```

3. **Replace with YOUR actual backend URL:**
   ```javascript
   let API_URL = 'https://nlp-backend.onrender.com';
   ```
   (Use the URL you copied from Step 5)

4. **Save the file**

5. **Push to GitHub:**
   ```bash
   git add frontend/index.html
   git commit -m "Update API URL for production"
   git push
   ```

---

### Step 7: Deploy Frontend on Render

1. **In Render dashboard, click "New +"**
2. **Select "Static Site"**

3. **Connect Repository:**
   - Select your GitHub account
   - Choose repository: `nlp-fullstack-app`
   - Click **"Connect"**

4. **Configure Settings:**
   ```
   Name: nlp-frontend
   Branch: main
   Root Directory: frontend
   Build Command: (leave empty)
   Publish Directory: frontend
   Plan: Free
   ```

5. **Click "Create Static Site"**

6. **Wait for deployment** (1-2 minutes)

7. **Copy your frontend URL:**
   - Example: `https://nlp-frontend.onrender.com`
   - **This is your live app URL!**

---

### ✅ Render Checklist

- [ ] Backend deployed on Render
- [ ] Backend URL copied
- [ ] Frontend API URL updated in code
- [ ] Frontend code pushed to GitHub
- [ ] Frontend deployed on Render
- [ ] Frontend URL copied

---

## Part 3: Test Your App

### Step 8: Test Everything

1. **Open your frontend URL:**
   - `https://nlp-frontend.onrender.com`

2. **Test the app:**
   - Enter some text
   - Click "Process Text"
   - **⚠️ First request takes 2-5 minutes** (downloading models - this is normal!)
   - Wait patiently
   - Subsequent requests will be fast

3. **Verify it works:**
   - Sentiment analysis shows results
   - Summarization works
   - NER extracts entities
   - Classification works

---

## 📦 About Models (Important!)

### ❌ Models DON'T Go to GitHub
- Models are **~2-3GB** (too large for GitHub)
- Already excluded by `.gitignore`
- **Never upload models to GitHub!**

### ✅ Models Download Automatically on Render
- When your app runs for the first time
- Code automatically downloads from Hugging Face
- Takes 2-5 minutes (first time only)
- Models are cached on Render's server
- **You don't need to do anything!**

### 📍 Where Models Are:
- **GitHub:** ❌ Not stored (too large)
- **Your Computer:** ✅ In `models/` folder (local only)
- **Render:** ✅ Auto-downloaded on first run

---

## ⚠️ Free Tier Limitations

### Sleep Mode
- Backend sleeps after **15 minutes** of inactivity
- First request after sleep = slow (re-downloading models)
- **Solution:** Use [UptimeRobot](https://uptimerobot.com) to ping your backend every 5 minutes
  - Ping URL: `https://your-backend.onrender.com/health`

### First Request
- Takes **2-5 minutes** (downloading models)
- This is **normal** - be patient!
- Models are ~2-3GB total

### Monthly Limits
- **750 hours/month** free
- Usually enough for personal projects

---

## 🔧 Troubleshooting

### Backend Not Working?

1. **Check Render Logs:**
   - Render dashboard → Your backend → "Logs" tab
   - Look for errors

2. **Common Issues:**
   - Models downloading (wait 2-5 min - normal!)
   - Port error (should use `$PORT` - already configured)
   - Memory error (free tier limits)

3. **Test Backend:**
   - Visit: `https://your-backend.onrender.com/health`
   - Should return: `{"status":"healthy"}`

### Frontend Can't Connect?

1. **Check API URL:**
   - Open `frontend/index.html`
   - Verify API URL matches your backend URL exactly

2. **Check CORS:**
   - Backend allows all origins (already configured)
   - If issues, check `backend/main.py` CORS settings

3. **Check Browser Console:**
   - Press F12 → Console tab
   - Look for error messages

### Models Not Downloading?

- Check Render logs for errors
- Verify internet access (should work automatically)
- Models download on first request automatically
- Be patient - takes 2-5 minutes!

---

## 📝 Complete Checklist

### GitHub:
- [ ] Repository created
- [ ] Code files uploaded
- [ ] Models folder NOT uploaded
- [ ] Code visible on GitHub

### Render Backend:
- [ ] Account created
- [ ] Backend service created
- [ ] Repository connected
- [ ] Build command set
- [ ] Start command set
- [ ] Backend deployed
- [ ] Backend URL copied

### Frontend:
- [ ] API URL updated in code
- [ ] Updated code pushed to GitHub
- [ ] Frontend service created on Render
- [ ] Frontend deployed
- [ ] Frontend URL copied

### Testing:
- [ ] Frontend opens
- [ ] Backend connects
- [ ] First request completed (models downloaded)
- [ ] All features working

---

## 🎉 You're Done!

Your app is now live at:
- **Frontend:** `https://your-frontend.onrender.com`
- **Backend:** `https://your-backend.onrender.com`

**Remember:**
- ✅ First request = slow (downloading models) - **This is normal!**
- ✅ Service may sleep after inactivity (free tier)
- ✅ Models download automatically - **No manual upload needed!**

---

## 🆘 Need Help?

1. **Check Render logs** for errors
2. **Verify files are in GitHub** (check repository)
3. **Make sure API URL is correct** in frontend
4. **First request always takes time** (downloading models - normal!)
5. **Check browser console** (F12) for frontend errors

**That's it! Your NLP app is now live on the internet! 🚀**
