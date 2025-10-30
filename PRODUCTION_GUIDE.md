# Production Deployment Guide - Handling Missing Frames

## Problem: KeyError for Some Frames

**Root Cause**: OSX doesn't detect a person in every frame, so those frames don't have `person_0_smplx_*` keys.

## Solutions

### Option 1: Use Fallback (Recommended)
**Strategy**: When a frame has no person, repeat the last valid frame's mesh

**Pros**:
- Smooth video output
- No gaps in rendering
- Better user experience

**Cons**:
- Mesh won't match video for those frames
- Slightly misleading for frames without people

### Option 2: Skip Frames
**Strategy**: Only render frames with valid detections

**Pros**:
- Accurate (only shows what was detected)
- Honest about missing data

**Cons**:
- Video will have gaps
- Requires post-processing to sync

### Option 3: Placeholder Image
**Strategy**: Show "No Detection" placeholder for missing frames

**Pros**:
- Clear indication of missing data
- Maintains frame count

**Cons**:
- Disrupts visual flow
- Less professional looking

## Implementation

### 1. Streamlit App (streamlit_app.py)

**Features**:
- Real-time frame rendering
- Fallback to last valid frame
- Statistics dashboard
- Error handling UI

**Usage**:
```bash
streamlit run streamlit_app.py
```

**Configuration**:
- Set SMPL-X model path
- Set NPZ directory
- Enable/disable fallback mode
- Adjust image size

### 2. Batch Processing (batch_render.py)

**For processing all 1800 frames**:

```bash
python batch_render.py \
  --npz_dir /path/to/npz_frames \
  --model_path /path/to/models \
  --output_dir /path/to/output \
  --img_size 720 \
  --use_fallback \
  --device cuda
```

**Output**:
- PNG images for each frame
- `render_report.json` with statistics
- Error breakdown by type

### 3. Colab Integration

For Google Colab, use the error-handling wrapper:

```python
def render_frame_safe(npz_file, model, device, last_valid=None):
    """
    Render with fallback
    
    Returns: (image, is_fallback, error)
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Check if person exists
        if len(data.get('person_ids', [])) == 0:
            if last_valid is not None:
                return last_valid, True, "no_person"
            return None, False, "no_person"
        
        # Load and render...
        # [rest of rendering code]
        
        return image, False, None
        
    except KeyError as e:
        if last_valid is not None:
            return last_valid, True, f"missing_key:{e}"
        return None, False, f"missing_key:{e}"
```

## Error Types You'll Encounter

### 1. No Person Detected
```
Error: 'person_0_smplx_root_pose is not a file in the archive'
```
**Meaning**: OSX didn't detect anyone in this frame

**Solutions**:
- Use fallback frame
- Skip frame
- Show placeholder

### 2. Missing Keys
```
Error: Missing parameter: 'person_0_smplx_body_pose'
```
**Meaning**: Person detected but parameters incomplete

**Solutions**:
- Check OSX output settings
- Use fallback frame
- Skip frame

### 3. Corrupt NPZ
```
Error: Error loading NPZ: [various]
```
**Meaning**: File is corrupted or unreadable

**Solutions**:
- Re-run OSX for that frame
- Skip frame
- Use fallback

## Expected Failure Rate

For typical videos:
- **Stationary subjects**: 1-5% missing frames
- **Moving subjects**: 5-15% missing frames  
- **Occlusions**: 10-30% missing frames
- **Multiple people**: 5-20% missing frames

## Production Checklist

### Pre-Processing
- [ ] Verify all NPZ files exist (1800 files)
- [ ] Run quick validation scan
- [ ] Check first/last frames have detections
- [ ] Estimate failure rate

### Processing
- [ ] Enable fallback mode
- [ ] Set appropriate image size
- [ ] Use GPU if available
- [ ] Monitor progress
- [ ] Save error log

### Post-Processing
- [ ] Review error report
- [ ] Check output frame count
- [ ] Verify video sync
- [ ] Handle completely failed frames

## Streamlit Integration

### Backend Structure

```python
# app.py
import streamlit as st
from mesh_renderer import MeshRenderer

# Initialize once
@st.cache_resource
def get_renderer(model_path):
    return MeshRenderer(model_path)

def main():
    renderer = get_renderer("/path/to/models")
    
    # Video selector
    video = st.selectbox("Video", ["Video-1", "Video-2", ...])
    
    # Frame sync
    frame_time = st.session_state.get('video_time', 0)
    frame_num = int(frame_time * 30)  # 30 FPS
    
    # Render
    npz_file = f"frame_{frame_num:04d}_params.npz"
    img, error = renderer.render_frame(npz_file, use_fallback=True)
    
    if img is not None:
        st.image(img)
    if error:
        st.caption(f"⚠️ {error}")
```

### Real-Time Sync

```javascript
// In Streamlit component
video.ontimeupdate = () => {
    const frameNum = Math.floor(video.currentTime * 30);
    Streamlit.setComponentValue({
        video_time: video.currentTime,
        frame_num: frameNum
    });
};
```

## Performance Optimization

### 1. Pre-render All Frames
```bash
# Render all frames beforehand
python batch_render.py --npz_dir /path/to/npz --output_dir /path/to/frames

# Then serve pre-rendered frames
# Much faster than real-time rendering
```

### 2. Cache Recent Frames
```python
from functools import lru_cache

@lru_cache(maxsize=30)
def get_frame(frame_num):
    return render_frame(frame_num)
```

### 3. Use Smaller Images for Preview
```python
# Preview: 512x512
# Full render: 1024x1024
img_size = 512 if preview_mode else 1024
```

## Error Handling Best Practices

### 1. Graceful Degradation
```python
try:
    img = render_frame(npz_file)
except KeyError:
    img = last_valid_frame  # Fallback
except Exception:
    img = placeholder_image  # Ultimate fallback
```

### 2. User Communication
```python
if error == "no_person":
    st.info("⚠️ No person detected in this frame (showing previous frame)")
elif error:
    st.warning(f"⚠️ {error}")
```

### 3. Logging
```python
import logging

logging.info(f"Frame {num}: {'success' if img else 'failed'}")
if error:
    logging.warning(f"Frame {num}: {error}")
```

## Testing Strategy

### 1. Unit Tests
```python
def test_missing_person():
    """Test fallback when person missing"""
    renderer = MeshRenderer(model_path)
    img1 = renderer.render_frame("frame_0001.npz")  # Valid
    img2 = renderer.render_frame("frame_0050.npz")  # Missing person
    assert img2 is not None  # Should use fallback
```

### 2. Integration Tests
```python
def test_full_video():
    """Test rendering full video"""
    renderer = MeshRenderer(model_path)
    frames = []
    for i in range(1, 1801):
        img, err = renderer.render_frame(f"frame_{i:04d}.npz")
        frames.append(img)
    
    # Check we got 1800 frames
    assert len(frames) == 1800
    assert all(f is not None for f in frames)
```

### 3. Load Tests
```python
# Test concurrent requests
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(render_frame, i) for i in range(100)]
    results = [f.result() for f in futures]
```

## Monitoring & Analytics

### Key Metrics
- **Success Rate**: % of frames rendered successfully
- **Fallback Rate**: % of frames using fallback
- **Render Time**: Average time per frame
- **Error Distribution**: Types of errors encountered

### Dashboard Example
```python
st.metric("Success Rate", f"{success_rate:.1f}%")
st.metric("Fallback Used", fallback_count)
st.metric("Avg Render Time", f"{avg_time:.2f}s")

# Error breakdown
st.bar_chart(error_distribution)
```

## Troubleshooting

### High Failure Rate (>30%)
**Possible causes**:
- Poor video quality
- Heavy occlusions
- Fast movement
- Multiple people

**Solutions**:
- Adjust OSX detection threshold
- Use different person tracker
- Increase temporal smoothing

### Slow Rendering (<1 FPS)
**Possible causes**:
- CPU rendering
- High image size
- No caching

**Solutions**:
- Use GPU: `--device cuda`
- Reduce size: `--img_size 512`
- Pre-render all frames

### Memory Issues
**Possible causes**:
- Loading all frames at once
- No garbage collection
- Large meshes

**Solutions**:
- Process in batches
- Clear cache periodically
- Use lower resolution

## Next Steps

1. **Deploy Streamlit app** with error handling
2. **Pre-render all frames** using batch script
3. **Create error report** to analyze patterns
4. **Optimize based on metrics**
5. **Add user feedback** for edge cases

## Files Included

- `streamlit_app.py` - Full Streamlit application
- `batch_render.py` - Batch processing script
- `PRODUCTION_GUIDE.md` - This guide

Both scripts include comprehensive error handling for missing frames!
