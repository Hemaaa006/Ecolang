# ECOLANG Videos

## Upload Your Videos Here

This folder should contain your sign language videos in MP4 format.

### Expected Videos

According to `signmesh/config.py`, these videos are expected:

1. **video1_speaking.mp4** - Video 1 - Speaking
2. **video2_gestures.mp4** - Video 2 - Gestures
3. **video3_conversation.mp4** - Video 3 - Conversation
4. **video4_demonstration.mp4** - Video 4 - Demonstration

### Video Requirements

- Format: MP4
- Duration: 60 seconds (1800 frames at 30fps)
- Must have corresponding NPZ parameters in Google Drive

### How to Upload

1. Place your MP4 files in this folder
2. Commit and push to GitHub
3. Videos will be accessible at: `https://raw.githubusercontent.com/Hemaaa006/Ecolang/main/videos/{filename}.mp4`

### Google Drive Structure

Each video must have corresponding NPZ parameters in:
```
/MyDrive/ecolang/Extracted_parameters/{video_name}/
├── frame_0001_params.npz
├── frame_0002_params.npz
└── ... (1800 files total)
```

---

**Note:** You can also upload videos directly through GitHub web interface:
1. Navigate to: https://github.com/Hemaaa006/Ecolang/tree/main/videos
2. Click "Add file" → "Upload files"
3. Drag and drop your MP4 files
4. Commit the changes
