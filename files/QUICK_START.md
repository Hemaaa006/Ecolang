# SignMesh - Quick Start for Claude Code

## One Command to Build Everything

```bash
# Give Claude Code this prompt:
"Build the SignMesh application according to BUILD_APP.md specifications"
```

## What You Get

A production-ready Streamlit app that:
- ✅ Displays sign language videos
- ✅ Renders synchronized 3D SMPL-X meshes  
- ✅ Handles missing frames gracefully (5-15% of frames)
- ✅ Provides statistics and error reporting
- ✅ Works on Google Colab or cloud servers

## File Structure

```
signmesh/
├── app.py                 # Main Streamlit UI
├── mesh_renderer.py       # SMPL-X rendering engine
├── file_manager.py        # Path management
├── batch_render.py        # Batch processing
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Key Features

### 1. Smart Error Handling
```python
# Problem: Some NPZ files don't have person detection
# Solution: Use previous frame's mesh as fallback
if current_frame_missing_person:
    use_last_valid_mesh()
```

### 2. Two PCA Modes
```python
# Try without PCA first (full hand poses)
# If model missing hand components → fallback to PCA mode
# Automatic detection and adaptation
```

### 3. Flexible Deployment
```python
# Automatically detects environment:
# - Google Colab: /content/drive/MyDrive/SignMesh
# - Production: /app/data  
# - Local: ./data
```

## Your Data Structure (Google Drive)

```
SignMesh/
├── videos/
│   ├── video1_speaking.mp4
│   ├── video2_gestures.mp4
│   ├── video3_conversation.mp4
│   └── video4_demonstration.mp4
│
├── npz_files/
│   ├── video1_speaking/
│   │   ├── frame_0001_params.npz
│   │   ├── frame_0002_params.npz
│   │   └── ... (1800 files)
│   ├── video2_gestures/
│   └── ...
│
└── models/
    └── SMPLX_NEUTRAL.npz
```

## NPZ File Format (Critical!)

Each NPZ contains:
```python
# Metadata
'person_ids'  # Array - CHECK LENGTH FIRST!

# SMPL-X Parameters (only exist if person detected)
'person_0_smplx_root_pose'   # (3,)
'person_0_smplx_body_pose'   # (63,)
'person_0_smplx_lhand_pose'  # (45,)
'person_0_smplx_rhand_pose'  # (45,)
'person_0_smplx_jaw_pose'    # (3,)
'person_0_smplx_shape'       # (10,)
'person_0_smplx_expr'        # (10,)
'person_0_cam_trans'         # (3,)
```

## Usage

### Run Streamlit App
```bash
streamlit run app.py
```

### Batch Render All Frames
```bash
python batch_render.py \
  --video_id video1_speaking \
  --output_dir ./rendered \
  --device cuda \
  --use_fallback
```

## The Missing Frame Problem (SOLVED!)

**Issue**: 5-15% of frames don't have person detections
```python
# This will crash:
data['person_0_smplx_root_pose']  # KeyError!
```

**Solution**: Check first, fallback if missing
```python
# Always do this:
person_ids = data.get('person_ids', np.array([]))
if len(person_ids) == 0:
    use_previous_frame_mesh()  # Fallback
else:
    render_current_frame()
```

## Expected Results

For 1800 frames (60 sec video):
- **90-95% frames**: Valid detection → render successfully
- **5-10% frames**: No detection → use fallback
- **<1% frames**: Complete failure → show error

With fallback enabled: **100% rendered frames**

## UI Features

1. **Dropdown Video Selector** (matches your HTML prototype)
2. **Side-by-side Display**: Video left, Mesh right
3. **Frame Scrubber**: Jump to any frame
4. **Statistics Dashboard**: Success rate, fallback usage
5. **Error Reporting**: Clear status messages

## Performance

- **Render Speed**: ~100-200ms per frame (GPU)
- **Memory**: ~2GB for model + cache
- **Storage**: ~360MB per video (NPZ files)

## Common Issues & Solutions

### "No person detected"
✅ **Normal** - Enable fallback to use previous frame

### "Missing hand components"  
✅ **Automatic** - Falls back to PCA mode (12D hands)

### "KeyError: person_0_smplx_*"
✅ **Handled** - Checks `person_ids` first, uses fallback

### Slow rendering
✅ **Solution** - Use GPU (`device='cuda'`)

## Production Deployment

### Option 1: Pre-render Everything
```bash
# Render all 1800 frames per video
python batch_render.py --video_id video1 --output_dir frames

# Serve pre-rendered images (fast!)
# ~30-60 minutes per video
```

### Option 2: Real-time Rendering
```bash
# Render on-demand (more flexible)
streamlit run app.py
# Works well with GPU
```

## Testing Checklist

Before deployment:
- [ ] Test with valid frame (person detected)
- [ ] Test with missing frame (no person)
- [ ] Test first frame (no previous fallback)
- [ ] Test video playback sync
- [ ] Check statistics accuracy
- [ ] Verify all 4 videos load

## What Makes This Production-Ready

✅ **Comprehensive error handling** - Never crashes  
✅ **Fallback mechanism** - Smooth output even with missing data  
✅ **Automatic PCA detection** - Works with any SMPL-X model  
✅ **Clear user feedback** - Status messages for every action  
✅ **Statistics tracking** - Know exactly what's happening  
✅ **Batch processing** - Pre-render for production  
✅ **Flexible deployment** - Colab, local, or cloud  

## Next Steps

1. **Copy BUILD_APP.md to Claude Code**
2. **Run the build command**
3. **Upload data to Google Drive**
4. **Configure paths in config.py**
5. **Run `streamlit run app.py`**
6. **Enjoy SignMesh! 🎬**

---

**All specifications in BUILD_APP.md**  
Built to handle real-world messy data gracefully!
