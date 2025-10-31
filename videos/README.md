# ECOLANG Videos

## Upload Instructions

Place your sign language videos in this folder. Each video should:
- Be in MP4 format
- Duration: 60 seconds (1800 frames at 30fps)
- Have corresponding NPZ parameters in Google Drive

## Expected Videos

According to `config.py`, the following videos are expected:

1. **video1_speaking.mp4** - Video 1 - Speaking
2. **video2_gestures.mp4** - Video 2 - Gestures
3. **video3_conversation.mp4** - Video 3 - Conversation
4. **video4_demonstration.mp4** - Video 4 - Demonstration

## NPZ Parameters

Each video must have corresponding NPZ parameters stored in Google Drive at:
```
/MyDrive/ecolang/Extracted_parameters/{video_name}/
├── frame_0001_params.npz
├── frame_0002_params.npz
└── ... (1800 files total)
```

## After Uploading

After uploading videos to GitHub, the raw URLs will be:
```
https://raw.githubusercontent.com/Hemaaa006/Ecolang/main/videos/{filename}.mp4
```

These URLs are already configured in `signmesh/config.py`.
