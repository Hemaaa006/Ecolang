# ECOLANG - Quick Start Guide

This is a simplified, step-by-step guide to get your ECOLANG application running in under 1 hour.

---

## What You're Building

A web application that displays sign language videos side-by-side with their 3D mesh renderings:

- **Left side**: Original sign language video
- **Right side**: 3D SMPL-X mesh rendering
- **User action**: Select video from dropdown → automatic mesh generation

---

## Prerequisites Checklist

Before starting, make sure you have:

- [ ] Google Account with Google Drive
- [ ] GitHub Account
- [ ] 4 sign language video files (MP4, ~60 seconds each)
- [ ] 1800 NPZ parameter files per video (stored in `Extracted_parameters/` folders)
- [ ] SMPL-X model file (`SMPLX_NEUTRAL.npz`)
- [ ] ngrok account (free: https://ngrok.com/signup)

---

## Step 1: Setup Google Drive (15 min)

### 1.1 Create Folder Structure

In Google Drive, create:

```
/MyDrive/ecolang/
├── Extracted_parameters/
│   ├── video1_speaking/      (1800 .npz files)
│   ├── video2_gestures/      (1800 .npz files)
│   ├── video3_conversation/  (1800 .npz files)
│   └── video4_demonstration/ (1800 .npz files)
├── models/
│   └── SMPLX_NEUTRAL.npz
└── videos/
    ├── video1_speaking.mp4
    ├── video2_gestures.mp4
    ├── video3_conversation.mp4
    └── video4_demonstration.mp4
```

### 1.2 Share Videos Publicly

For each video in `/MyDrive/ecolang/videos/`:
1. Right-click → Share
2. Change to: "Anyone with the link" can view
3. Copy the share link
4. Save all 4 links in a text file

### 1.3 Get Direct Download URLs

**Option A: Use Helper Script**
```bash
python convert_drive_links.py
```
Follow prompts, paste share links, copy output.

**Option B: Manual**
Extract FILE_ID from each link:
```
From: https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
To:   https://drive.google.com/uc?export=download&id=1ABC123xyz
```

### 1.4 Update config.py

Open `signmesh/config.py` and replace each `PASTE_FILE_ID_HERE` with actual FILE_IDs.

---

## Step 2: Setup Colab Backend (20 min)

### 2.1 Get ngrok Token

1. Go to: https://ngrok.com/signup
2. Sign up (free)
3. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
4. Copy your authtoken

### 2.2 Create Colab Notebook

1. Open: https://colab.research.google.com/
2. Create new notebook
3. Change runtime: **Runtime → Change runtime type → GPU**
4. Copy code from `ecolang_mesh_api_colab.py` into 6 separate cells

### 2.3 Configure & Run

1. In **Cell 6**, replace `YOUR_NGROK_TOKEN_HERE` with your actual token
2. Run cells in order: 1 → 2 → 3 → 4 → 5 → 6
3. Wait for Cell 6 to show ngrok URL
4. **Copy the URL** (e.g., `https://abc123.ngrok-free.app`)
5. **Keep Cell 6 running!**

---

## Step 3: Deploy Streamlit (15 min)

### 3.1 Update & Push Code

```bash
git add signmesh/config.py
git commit -m "Update video FILE_IDs for Google Drive"
git push origin main
```

### 3.2 Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Click "New app"
3. Select your repository
4. Main file: `streamlit_app.py`
5. **Before deploying**, click "Advanced settings" → "Secrets"
6. Add:
   ```toml
   COLAB_API_URL = "https://your-ngrok-url.ngrok-free.app"
   ```
7. Click "Deploy"
8. Wait 2-3 minutes

---

## Step 4: Test (10 min)

### 4.1 Open Your App

Go to your Streamlit Cloud URL (e.g., `https://yourapp.streamlit.app`)

### 4.2 Select a Video

1. Click dropdown
2. Select "Video 1 - Speaking"
3. Watch progress bar: "Generating frame X of 1800"
4. Wait 5-10 minutes for first generation
5. View result: original video (left) + mesh video (right)

### 4.3 Try Other Videos

Select other videos from dropdown. Each takes ~10 minutes on first generation.

---

## Architecture Overview

```
┌─────────────────────┐
│   Your Browser      │
│   (Streamlit UI)    │
└──────────┬──────────┘
           │
           ↓ HTTPS
┌──────────────────────┐
│  Google Colab        │
│  (FastAPI + ngrok)   │
│  - Renders meshes    │
│  - Uses GPU          │
└──────────┬───────────┘
           │
           ↓ Reads from
┌──────────────────────┐
│  Google Drive        │
│  - Videos            │
│  - NPZ parameters    │
│  - SMPL-X model      │
└──────────────────────┘
```

---

## How It Works

1. **User selects video** from dropdown
2. **Streamlit sends request** to Colab API: "Render frame 1"
3. **Colab loads** NPZ parameters from Google Drive
4. **Colab generates** 3D mesh using SMPL-X
5. **Colab renders** mesh to 720x720 image
6. **Colab sends back** base64-encoded image
7. **Repeat** for all 1800 frames
8. **Streamlit compiles** frames into video
9. **User watches** side-by-side comparison

---

## Common Issues

### "Cannot connect to Colab API"

**Fix:**
1. Check Colab notebook is running (Cell 6)
2. Check ngrok URL matches Streamlit secret
3. Restart Cell 6 if needed, update Streamlit secret with new URL

### "Frame not found"

**Fix:**
1. Verify 1800 NPZ files exist in Google Drive
2. Check folder names match: `video1_speaking` in both places
3. Check file names: `frame_0001_params.npz` through `frame_1800_params.npz`

### Video doesn't load (left column)

**Fix:**
1. Check video is shared publicly in Google Drive
2. Test direct URL in browser (should download video)
3. Verify FILE_ID is correct in config.py

### Generation is slow

**Fix:**
1. Check Colab is using GPU (Cell 3 should show "cuda")
2. Change runtime: Runtime → Change runtime type → T4 GPU
3. Restart runtime if needed

---

## Maintenance

### Daily Operation

1. **Start Colab**: Run all cells (1-6)
2. **Copy ngrok URL**: From Cell 6 output
3. **Update Streamlit secret** (if URL changed)
4. **Keep Colab tab open** (prevents timeout)

### Free Tier Limitations

- **Colab**: 12-hour sessions, disconnects after 90min in background
- **ngrok**: 2-hour sessions, need to restart and update URL
- **Solution**: Check every 1-2 hours, restart if needed

### Upgrade Options

- **Colab Pro** ($10/month): 24-hour sessions, better GPUs
- **ngrok Pro** ($10/month): Longer sessions, custom domains
- **Total**: $20/month for reliable operation

---

## File Structure

```
Project/
├── streamlit_app.py          ← Main web app
├── requirements.txt          ← Python dependencies
├── convert_drive_links.py    ← Helper script
├── DEPLOYMENT_GUIDE.md       ← Full documentation
├── QUICK_START.md           ← This file
├── ecolang_mesh_api_colab.py ← Colab notebook code
└── signmesh/
    └── config.py            ← Configuration (update FILE_IDs here)
```

---

## Next Steps

After successful deployment:

1. **Generate all videos**: Select each video once to generate cache
2. **Share your app**: Send Streamlit URL to users
3. **Monitor usage**: Check Streamlit Cloud dashboard for traffic
4. **Keep Colab running**: Restart every 12 hours (free tier)

---

## Getting Help

- **Deployment Guide**: See `DEPLOYMENT_GUIDE.md` for detailed instructions
- **Streamlit Docs**: https://docs.streamlit.io/
- **Colab Help**: https://research.google.com/colaboratory/faq.html
- **ngrok Docs**: https://ngrok.com/docs

---

## Timeline

| Phase | Time | Task |
|-------|------|------|
| Phase 1 | 15 min | Setup Google Drive |
| Phase 2 | 20 min | Create Colab backend |
| Phase 3 | 15 min | Deploy Streamlit |
| Phase 4 | 10 min | Test application |
| **Total** | **60 min** | **Complete setup** |
| Per video | 10 min | First-time generation |

---

## Success Criteria

You're done when:

- ✅ Streamlit app loads at your URL
- ✅ Dropdown shows 4 videos
- ✅ Selecting video triggers generation
- ✅ Progress bar shows frame count
- ✅ Original video displays in left column
- ✅ Mesh video displays in right column after generation
- ✅ Both videos play simultaneously

---

**Ready to start? Follow Step 1 above!**

For detailed troubleshooting and advanced configuration, see `DEPLOYMENT_GUIDE.md`.
