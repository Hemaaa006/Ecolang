# ECOLANG Manual Setup Guide

If you prefer step-by-step instructions without running scripts, follow this guide.

---

## Step 1: Upload Videos to Google Drive

1. Open Google Drive in your browser
2. Navigate to `/MyDrive/`
3. Create folder: `ecolang`
4. Inside `ecolang`, create folder: `videos`
5. Upload your 4 video files to `/MyDrive/ecolang/videos/`

**Expected structure:**
```
/MyDrive/ecolang/
└── videos/
    ├── video1_speaking.mp4
    ├── video2_gestures.mp4
    ├── video3_conversation.mp4
    └── video4_demonstration.mp4
```

---

## Step 2: Make Videos Public & Get Share Links

**For each video (repeat 4 times):**

1. Right-click the video in Google Drive
2. Click **"Share"**
3. Change permission to: **"Anyone with the link"** can **view**
4. Click **"Copy link"**
5. Paste the link into a text file

**You should have 4 links like:**
```
https://drive.google.com/file/d/1ABC123xyz456/view?usp=sharing
https://drive.google.com/file/d/1DEF456abc789/view?usp=sharing
https://drive.google.com/file/d/1GHI789def012/view?usp=sharing
https://drive.google.com/file/d/1JKL012ghi345/view?usp=sharing
```

---

## Step 3: Extract FILE_IDs

From each share link, extract the FILE_ID (the part between `/d/` and `/view`):

**Example:**
```
Share link: https://drive.google.com/file/d/1ABC123xyz456/view?usp=sharing
                                            ↑              ↑
                                          /d/            /view
FILE_ID: 1ABC123xyz456
```

**Extract all 4 FILE_IDs:**

| Video | FILE_ID |
|-------|---------|
| video1_speaking.mp4 | (your FILE_ID here) |
| video2_gestures.mp4 | (your FILE_ID here) |
| video3_conversation.mp4 | (your FILE_ID here) |
| video4_demonstration.mp4 | (your FILE_ID here) |

---

## Step 4: Update config.py

1. Open: `c:\Users\dell\Desktop\Streamlit_render app\signmesh\config.py`
2. Find the `VIDEO_LIBRARY` section (around line 45)
3. Replace each `PASTE_FILE_ID_HERE` with your actual FILE_IDs

**Before:**
```python
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=PASTE_FILE_ID_HERE',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    # ... more videos
}
```

**After (example with fake FILE_IDs):**
```python
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1ABC123xyz456',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video2_gestures': {
        'title': 'Video 2 - Gestures',
        'filename': 'video2_gestures.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1DEF456abc789',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    # ... and so on for all 4 videos
}
```

4. Save the file

---

## Step 5: Verify Your Setup

**Checklist:**
- [ ] 4 videos uploaded to `/MyDrive/ecolang/videos/` on Google Drive
- [ ] Each video has NPZ parameters in `/MyDrive/ecolang/Extracted_parameters/{video_name}/`
- [ ] Each NPZ folder has 1800 files: `frame_0001_params.npz` through `frame_1800_params.npz`
- [ ] SMPL-X model exists: `/MyDrive/ecolang/models/SMPLX_NEUTRAL.npz`
- [ ] All 4 videos are shared publicly in Google Drive
- [ ] `signmesh/config.py` updated with all 4 real FILE_IDs (no `PASTE_FILE_ID_HERE` remaining)

---

## Step 6: Test Video URLs

Before deploying, test that your videos are accessible:

1. Open `signmesh/config.py`
2. Copy one of your direct URLs, e.g.:
   ```
   https://drive.google.com/uc?export=download&id=1ABC123xyz456
   ```
3. Paste it into your browser address bar
4. Press Enter
5. **Expected result:** Video should start downloading

If you see "Access Denied" or "Need permission":
- Go back to Google Drive
- Right-click the video → Share
- Make sure it's set to "Anyone with the link" can view
- Try the URL again

---

## Next Steps

Once Step 6 works for all 4 videos, you're ready to:

1. **Create Colab Backend** - Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Phase 2
2. **Deploy Streamlit** - Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Phase 3
3. **Test Application** - Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) Phase 4

---

## Quick Reference: Direct URL Format

The direct download URL format is:
```
https://drive.google.com/uc?export=download&id=FILE_ID
```

Always use this format in `config.py`, NOT the share link format.

**Wrong:**
```python
'github_url': 'https://drive.google.com/file/d/1ABC123xyz456/view?usp=sharing',
```

**Correct:**
```python
'github_url': 'https://drive.google.com/uc?export=download&id=1ABC123xyz456',
```

---

## Troubleshooting

### Problem: Can't find FILE_ID in share link

**Solution:** Make sure you copied the full share link. It should contain `/file/d/` in it.

### Problem: Video won't download from direct URL

**Solution:**
1. Check video is shared publicly in Google Drive
2. Try opening the original share link first - if that works, the FILE_ID is correct
3. Make sure you're using the `uc?export=download&id=` format

### Problem: Don't have 4 videos

**Solution:**
- If you have fewer videos, skip some entries or duplicate one video for testing
- Update the video count in the script or manually edit config.py

---

## Example: Complete Manual Conversion

**Given these share links:**
```
Video 1: https://drive.google.com/file/d/1aB2cD3eF4gH5i6J/view?usp=sharing
Video 2: https://drive.google.com/file/d/7kL8mN9oP0qR1sT/view?usp=sharing
Video 3: https://drive.google.com/file/d/2uV3wX4yZ5aB6cD/view?usp=sharing
Video 4: https://drive.google.com/file/d/7eF8gH9iJ0kL1mN/view?usp=sharing
```

**Extract FILE_IDs:**
```
Video 1: 1aB2cD3eF4gH5i6J
Video 2: 7kL8mN9oP0qR1sT
Video 3: 2uV3wX4yZ5aB6cD
Video 4: 7eF8gH9iJ0kL1mN
```

**Update config.py VIDEO_LIBRARY:**
```python
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=1aB2cD3eF4gH5i6J',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video2_gestures': {
        'title': 'Video 2 - Gestures',
        'filename': 'video2_gestures.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=7kL8mN9oP0qR1sT',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video3_conversation': {
        'title': 'Video 3 - Conversation',
        'filename': 'video3_conversation.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=2uV3wX4yZ5aB6cD',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    'video4_demonstration': {
        'title': 'Video 4 - Demonstration',
        'filename': 'video4_demonstration.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=7eF8gH9iJ0kL1mN',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    }
}
```

Done! Save the file and proceed to Colab setup.

---

This manual approach takes about 10 minutes and doesn't require running any scripts.
