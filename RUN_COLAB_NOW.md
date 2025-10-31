# ✅ READY TO RUN COLAB!

Everything is configured! Your config.py has the correct FILE_IDs.

---

## 🚀 STEP-BY-STEP: Run Google Colab Backend

### 1. Open Google Colab
- Go to: **https://colab.research.google.com/**
- Sign in with your Google account

### 2. Create New Notebook
- Click: **File → New notebook**
- Click: **Runtime → Change runtime type → GPU → Save**

### 3. Copy the 6 Code Blocks

Open the file: **[colab_mesh_api_CUSTOM.py](colab_mesh_api_CUSTOM.py)**

You'll see 6 sections labeled:
- CELL 1: Install Dependencies
- CELL 2: Mount Google Drive
- CELL 3: Load SMPL-X Model
- CELL 4: Create Rendering Function
- CELL 5: Create API Server
- CELL 6: Start Server with ngrok

---

### 4. Paste Each Block into Separate Cells

**For CELL 1:**
1. In Colab, click in the empty cell
2. Copy everything under "CELL 1: Install Dependencies" from the custom file
3. Paste into Colab
4. Click the ▶ PLAY button
5. Wait ~2 minutes

**For CELL 2:**
1. Click "+ Code" to create new cell
2. Copy everything under "CELL 2: Mount Google Drive"
3. Paste into Colab
4. Click ▶ PLAY
5. **A popup appears** → Click "Connect to Google Drive" → Allow
6. Should show: "✓ ALL CHECKS PASSED"

**For CELL 3:**
1. Create new cell
2. Copy "CELL 3: Load SMPL-X Model"
3. Paste and RUN
4. Should show: "✓ SMPL-X model loaded successfully!"

**For CELL 4:**
1. Create new cell
2. Copy "CELL 4: Create Rendering Function"
3. Paste and RUN
4. Should show: "✓ Render function works!"

**For CELL 5:**
1. Create new cell
2. Copy "CELL 5: Create API Server"
3. Paste and RUN
4. Should show: "✓ API server configured"

**For CELL 6:**
1. Create new cell
2. Copy "CELL 6: Start Server with ngrok"
3. **IMPORTANT:** Before pasting, you need your ngrok token!

---

### 5. Get ngrok Token (if you don't have it)

1. Go to: **https://ngrok.com/signup**
2. Sign up (free)
3. Go to: **https://dashboard.ngrok.com/get-started/your-authtoken**
4. Copy the token (looks like: `2abc123XYZ_long_string`)

---

### 6. Add Your ngrok Token to CELL 6

In the CELL 6 code, find this line:
```python
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"
```

Replace `YOUR_NGROK_TOKEN_HERE` with your actual token:
```python
NGROK_AUTH_TOKEN = "2abc123XYZ_your_actual_token"
```

Then paste the entire CELL 6 code into Colab and click ▶ PLAY.

---

### 7. Copy the ngrok URL

After CELL 6 runs, you'll see output like:
```
======================================================================
🚀 ECOLANG MESH API IS NOW RUNNING!
======================================================================

📡 Public URL: https://abc123-45-67-89-10.ngrok-free.app

======================================================================
📋 COPY THIS URL FOR STREAMLIT CLOUD:
======================================================================

https://abc123-45-67-89-10.ngrok-free.app

======================================================================
⚠️  KEEP THIS CELL RUNNING - DON'T STOP IT!
======================================================================
```

**COPY THE URL:** `https://abc123-45-67-89-10.ngrok-free.app`

**IMPORTANT:** Leave this cell running! Don't stop it or close the browser tab.

---

## ✅ Success Checklist

After running all 6 cells:

- [ ] Cell 1: Shows "✓ All dependencies installed"
- [ ] Cell 2: Shows "✓ ALL CHECKS PASSED"
- [ ] Cell 3: Shows "✓ SMPL-X model loaded successfully!" and "Using device: cuda"
- [ ] Cell 4: Shows "✓ Render function works!"
- [ ] Cell 5: Shows "✓ API server configured"
- [ ] Cell 6: Shows ngrok URL (like https://abc123.ngrok-free.app)
- [ ] Cell 6 is still running (spinning icon or "Running" status)

---

## 🎯 Next Step

Once you have the ngrok URL, you're ready to deploy to Streamlit!

**Continue to:** Deploy to Streamlit Cloud (I'll give you instructions once Colab is running)

---

## ❓ Troubleshooting

### Cell 2 fails with "Base folder not found"
- Check: `/MyDrive/ecolang/` exists in Google Drive
- Create it if missing

### Cell 3 shows "Using device: cpu" instead of "cuda"
- Go to: Runtime → Change runtime type → GPU → Save
- Rerun all cells

### Cell 4 fails "Test file not found"
- Check: Your NPZ files are in `/MyDrive/ecolang/Extracted_parameters/ch07_speakerview_012_parameters/`
- Files should be named: `frame_0001_params.npz`, `frame_0002_params.npz`, etc.

### Cell 6 fails "Invalid token"
- Check you copied the entire ngrok token
- Make sure it's between the quotes: `"your_token_here"`

---

**Ready? Open [colab_mesh_api_CUSTOM.py](colab_mesh_api_CUSTOM.py) and start with CELL 1!**
