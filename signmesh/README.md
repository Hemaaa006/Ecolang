# SignMesh - Video to Mesh Rendering

SMPL-X mesh reconstruction from sign language videos with real-time rendering.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Data Paths

Edit `config.py` or set environment variables:
```python
BASE_PATH = "/path/to/your/data"
```

### 3. Data Structure
```
SignMesh/
├── videos/
├── npz_files/
│   ├── video1_speaking/
│   ├── video2_gestures/
│   └── ...
└── models/
    └── SMPLX_NEUTRAL.npz
```

### 4. Run Application
```bash
streamlit run app.py
```

## Usage

1. Initialize renderer in sidebar
2. Select video from dropdown
3. Choose frame number
4. Click "Render Frame"

## Batch Processing

Pre-render all frames:
```bash
python batch_render.py \
  --video_id video1_speaking \
  --output_dir ./rendered \
  --device cuda \
  --use_fallback
```

## Troubleshooting

### "No person detected"
- Frame has no person detection
- Enable fallback to use previous frame

### "Missing hand components"
- SMPL-X model incomplete
- App automatically falls back to PCA mode

## License
MIT
