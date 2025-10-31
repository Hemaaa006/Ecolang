# How to Run convert_drive_links.py

## Method 1: Run the Script Interactively

1. **Open Command Prompt**
   - Press `Win + R`
   - Type `cmd` and press Enter

2. **Navigate to your project folder**
   ```bash
   cd "c:\Users\dell\Desktop\Streamlit_render app"
   ```

3. **Run the script**
   ```bash
   python convert_drive_links.py
   ```

4. **Follow the prompts**
   - The script will ask for each video's Google Drive share link
   - Paste each link when prompted
   - Press Enter after each link
   - The script will extract FILE_IDs automatically
   - At the end, it will show you the complete VIDEO_LIBRARY configuration

5. **Copy the output**
   - Select and copy the entire VIDEO_LIBRARY section from the output
   - Open `signmesh/config.py`
   - Replace the existing VIDEO_LIBRARY section (lines 45-78)
   - Save the file

---

## Method 2: Manual Conversion (No Script)

If you don't want to run the script, follow [MANUAL_SETUP.md](MANUAL_SETUP.md) for step-by-step manual instructions.

**Quick steps:**
1. Get Google Drive share links for your 4 videos
2. Extract FILE_ID from each link (the part between `/d/` and `/view`)
3. Open `signmesh/config.py`
4. Replace `PASTE_FILE_ID_HERE` with your actual FILE_IDs
5. Save the file

---

## What You Need Before Running

- [ ] 4 videos uploaded to Google Drive: `/MyDrive/ecolang/videos/`
- [ ] Each video shared publicly: "Anyone with the link" can view
- [ ] Share links copied and ready to paste

---

## Example: What the Script Does

**Input (you provide):**
```
Share link: https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
```

**Output (script extracts):**
```
FILE_ID: 1ABC123xyz
Direct URL: https://drive.google.com/uc?export=download&id=1ABC123xyz
```

**Final output (ready to paste):**
```python
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1ABC123xyz',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    # ... all 4 videos
}
```

---

## Troubleshooting

### Script shows "Error: EOF when reading a line"
This happens when running in background. Open a Command Prompt window and run it directly (see Method 1 above).

### Script shows Unicode error
This is fixed in the latest version. Make sure you have the updated `convert_drive_links.py` file.

### Can't find Python
Make sure Python is installed. Test by running:
```bash
python --version
```

If not installed, download from: https://www.python.org/downloads/

---

## Alternative: Use the Manual Setup Guide

For a completely script-free approach, see [MANUAL_SETUP.md](MANUAL_SETUP.md) which provides step-by-step instructions with examples.

Total time: ~10 minutes for manual setup.
