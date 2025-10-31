# ECOLANG - Sign Language 3D Mesh Renderer

A web application for rendering 3D SMPL-X meshes from sign language videos, featuring side-by-side comparison of original videos and generated mesh animations.

## Overview

**ECOLANG** allows users to view sign language videos alongside their 3D mesh representations in real-time. The application uses a split architecture with a lightweight Streamlit frontend and GPU-accelerated Google Colab backend for mesh rendering.

### Key Features

- **Clean Interface**: Minimalist UI with video dropdown selector
- **Side-by-Side Display**: Original video (left) + 3D mesh (right)
- **On-Demand Generation**: Meshes generated when video is selected
- **Progress Tracking**: Real-time progress bar for 1800-frame videos
- **GPU Acceleration**: Powered by Google Colab's free GPU

---

## Quick Links

- **[QUICK_START.md](QUICK_START.md)** - Get started in 60 minutes
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Comprehensive deployment documentation
- **[ecolang_mesh_api_colab.py](ecolang_mesh_api_colab.py)** - Colab notebook code template
- **[convert_drive_links.py](convert_drive_links.py)** - Helper script for Google Drive setup

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      USER'S BROWSER                          │
│                  (Streamlit Web Interface)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ↓ HTTPS/REST API
┌───────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB BACKEND                             │
│              (FastAPI + ngrok tunnel)                         │
│                                                               │
│  - Loads SMPL-X model                                        │
│  - Renders 3D meshes from NPZ parameters                     │
│  - Returns base64-encoded images                             │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓ Reads from
┌───────────────────────────────────────────────────────────────┐
│                   GOOGLE DRIVE STORAGE                        │
│                   /MyDrive/ecolang/                           │
│                                                               │
│  ├── videos/              (4 MP4 files)                      │
│  ├── Extracted_parameters/ (7200 NPZ files, 1800 per video)  │
│  └── models/              (SMPLX_NEUTRAL.npz)                │
└───────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Frontend (Streamlit Cloud)
- **Streamlit**: Web UI framework
- **httpx**: Async HTTP client for API calls
- **OpenCV**: Video compilation
- **Pillow**: Image processing

### Backend (Google Colab)
- **FastAPI**: REST API framework
- **PyTorch**: Deep learning framework
- **SMPL-X**: 3D human body model
- **PyRender**: 3D mesh rendering
- **Trimesh**: Mesh processing
- **ngrok**: Public tunnel for Colab

### Storage (Google Drive)
- Videos (MP4)
- NPZ parameter files
- SMPL-X model files

---

## Getting Started

### Prerequisites

- Google Account (for Drive and Colab)
- GitHub Account (for hosting Streamlit app)
- ngrok Account (free tier: https://ngrok.com/signup)
- Sign language video files (MP4, ~60 seconds each)
- Extracted SMPL-X parameters (NPZ files)
- SMPL-X model file

### Installation

Follow either:
- **[QUICK_START.md](QUICK_START.md)** - Streamlined 60-minute guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Detailed step-by-step instructions

### Basic Setup Steps

1. **Setup Google Drive** (15 min)
   - Create `/MyDrive/ecolang/` folder structure
   - Upload videos, NPZ files, and SMPL-X model
   - Get Google Drive FILE_IDs for videos

2. **Create Colab Backend** (20 min)
   - Create new Colab notebook with GPU
   - Copy code from `ecolang_mesh_api_colab.py`
   - Start API server with ngrok

3. **Deploy Streamlit Frontend** (15 min)
   - Update `signmesh/config.py` with FILE_IDs
   - Deploy to Streamlit Cloud
   - Add Colab ngrok URL to secrets

4. **Test Application** (10 min)
   - Open Streamlit app URL
   - Select video from dropdown
   - Watch mesh generation progress

---

## Project Structure

```
ecolang/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies (frontend)
├── convert_drive_links.py        # Google Drive URL converter
├── ecolang_mesh_api_colab.py     # Colab backend code template
│
├── signmesh/
│   └── config.py                # Configuration (VIDEO_LIBRARY, API URLs)
│
├── videos/
│   └── README.md                # Video upload instructions
│
├── DEPLOYMENT_GUIDE.md          # Comprehensive deployment guide
├── QUICK_START.md               # Quick start guide (60 min)
└── README.md                    # This file
```

---

## Configuration

### Video Library

Edit `signmesh/config.py` to configure your videos:

```python
VIDEO_LIBRARY = {
    'video1_speaking': {
        'title': 'Video 1 - Speaking',
        'filename': 'video1_speaking.mp4',
        'github_url': 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID',
        'frames': 1800,
        'fps': 30,
        'duration': '1:00'
    },
    # ... more videos
}
```

### Streamlit Secrets

Add to Streamlit Cloud secrets:

```toml
COLAB_API_URL = "https://your-ngrok-url.ngrok-free.app"
```

---

## Usage

### For End Users

1. Open the Streamlit app URL
2. Select a video from the dropdown menu
3. Wait for mesh generation (5-10 minutes first time)
4. Watch original video and 3D mesh side-by-side
5. Select other videos to view different animations

### For Administrators

**Daily Operations:**
1. Start Google Colab notebook
2. Run all cells (1-6)
3. Copy ngrok URL from Cell 6
4. Update Streamlit Cloud secrets if URL changed
5. Keep Colab tab open to prevent timeout

**Maintenance:**
- Restart Colab every 12 hours (free tier)
- Update ngrok URL in Streamlit secrets when changed
- Monitor Streamlit Cloud logs for errors

---

## API Documentation

### Endpoints

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "message": "Colab API is running",
  "device": "cuda",
  "model_loaded": true
}
```

#### POST `/render_frame`
Render a single frame from NPZ parameters

**Request:**
```json
{
  "video_id": "video1_speaking",
  "frame_number": 123,
  "person_id": 0
}
```

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_png_image",
  "frame_number": 123,
  "error": null
}
```

---

## Performance

### Typical Metrics

- **Frame render time**: 0.3-0.5 seconds (Colab T4 GPU)
- **Full video generation**: 5-10 minutes (1800 frames)
- **API latency**: 50-100ms (ngrok overhead)
- **Image size**: ~100KB per frame (720x720 PNG)

### Optimization Options

- **Batch rendering**: Process multiple frames in parallel (50-70% faster)
- **Lower resolution**: Render 512x512 instead of 720x720 (40% faster)
- **Frame skipping**: Render every 2nd frame (50% faster, still smooth at 15fps)
- **Pre-generation**: Generate all videos once, store in Drive (instant playback)

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Cannot connect to Colab API" | Verify Colab Cell 6 is running, check ngrok URL |
| "Frame not found" | Verify 1800 NPZ files exist in Google Drive |
| "SMPL-X model not found" | Check `SMPLX_NEUTRAL.npz` exists in `/models/` |
| Video doesn't load | Verify Google Drive sharing is public |
| Slow generation | Ensure Colab is using GPU (not CPU) |
| ngrok URL stops working | Restart Cell 6, update Streamlit secrets |

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed troubleshooting.

---

## Cost Analysis

### Free Tier (Current Setup)

| Service | Plan | Cost | Limitations |
|---------|------|------|-------------|
| Google Drive | Free | $0/month | 15GB storage |
| Google Colab | Free | $0/month | 12hr sessions, 90min background |
| ngrok | Free | $0/month | 2hr sessions |
| Streamlit Cloud | Community | $0/month | Public apps only |
| **Total** | | **$0/month** | Requires manual restarts |

### Upgraded Setup

| Service | Plan | Cost | Benefits |
|---------|------|------|----------|
| Google Colab | Pro | $10/month | 24hr sessions, better GPUs |
| ngrok | Pro | $10/month | Longer sessions, custom domains |
| **Total** | | **$20/month** | More reliable, less maintenance |

---

## Development

### Local Testing

Run Streamlit app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

Set environment variable for Colab API:
```bash
export COLAB_API_URL="https://your-ngrok-url.ngrok-free.app"
streamlit run streamlit_app.py
```

### Adding New Videos

1. Upload video to Google Drive `/MyDrive/ecolang/videos/`
2. Create NPZ parameters folder: `/MyDrive/ecolang/Extracted_parameters/{video_id}/`
3. Add 1800 NPZ files: `frame_0001_params.npz` through `frame_1800_params.npz`
4. Get Google Drive FILE_ID (share publicly)
5. Add entry to `VIDEO_LIBRARY` in `signmesh/config.py`
6. Commit and push changes

---

## Future Enhancements

### Planned Features

- [ ] Download button for generated mesh videos
- [ ] Interactive 3D viewer (Three.js integration)
- [ ] Frame-by-frame navigation
- [ ] Multiple camera angles
- [ ] Real-time video upload and processing
- [ ] Comparison mode (two videos side-by-side)
- [ ] Export mesh as OBJ/FBX format
- [ ] Usage analytics dashboard

### Advanced Integrations

- [ ] Integrate pose estimation pipeline (end-to-end processing)
- [ ] Support for multiple signers in one video
- [ ] Real-time streaming (WebRTC)
- [ ] Mobile app version
- [ ] VR/AR viewing mode

---

## Contributing

This is a research/educational project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project uses:
- SMPL-X model (requires license from https://smpl-x.is.tue.mpg.de/)
- Other dependencies have their respective licenses

---

## Support

For issues and questions:

- **Setup Help**: See [QUICK_START.md](QUICK_START.md)
- **Detailed Docs**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Streamlit Help**: https://docs.streamlit.io/
- **Colab Help**: https://research.google.com/colaboratory/faq.html

---

## Acknowledgments

- **SMPL-X**: Body model from Max Planck Institute
- **Streamlit**: Web framework
- **Google Colab**: Free GPU compute
- **ngrok**: Public tunneling service

---

## Version History

- **v1.0.0** (Current)
  - Initial release
  - Clean ECOLANG interface
  - Google Drive video hosting
  - Colab backend with ngrok
  - Side-by-side video display
  - Progress tracking for generation

---

**Ready to deploy?** Start with [QUICK_START.md](QUICK_START.md)!
