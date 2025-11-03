# ECOLANG Streamlit / Colab Rendering Suite

Minimal two-part application that surfaces SMPL-X mesh renders from Google Drive.
The Streamlit front-end lists every source video stored in Drive, shows render
status, and lets you kick off new renders through the Colab FastAPI backend.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `streamlit_app.py` | Streamlit Cloud/CLI entry point – imports `ecolang.app.main()` |
| `ecolang/app.py` | Main Streamlit UI (video browser, twin players, render controls) |
| `ecolang/api_client.py` | Thin client used by the UI to talk to the Colab API |
| `ecolang/.streamlit/config.toml` | Optional Streamlit theme & server settings |
| `COLAB_BACKEND_AUTOMATED.py` | Notebook-friendly FastAPI server for Colab |
| `requirements.txt` | Python dependency list for the Streamlit app |
| `videos/` | Placeholder for sample videos (kept for structure only) |

Everything else has been removed to keep the project surface area small and
reduce ambiguity about the active code path.

## Requirements

- Python 3.10+
- pip
- (Linux) system packages for OpenGL/FFmpeg playback might be required when
  running locally: `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`,
  `libxrender-dev`, `libgomp1`.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Colab Backend

1. Open a fresh Google Colab notebook and upload
   `COLAB_BACKEND_AUTOMATED.py` (or copy/paste the cells).
2. Fill in your ngrok auth token inside the script (`ngrok.set_auth_token`).
3. Execute the cells top-to-bottom. The script will:
   - Install system/py dependencies.
   - Mount `MyDrive/ecolang/`.
   - Auto-discover source videos (`videos/`), NPZ parameter folders
     (`Extracted_parameters/`), and past renders (`rendered_videos/`).
   - Launch a FastAPI server with endpoints exposed via ngrok.
4. Copy the ngrok HTTPS URL reported at the end; you will use this in the
   Streamlit secrets under the key `COLAB_API_URL`.

> Renders run one at a time. Starting a new job while another is active will
> prompt the UI to cancel the previous job unless forced.

## Running the Streamlit Front-End

1. Set the Colab backend URL in your local environment or Streamlit secrets:

   ```bash
   streamlit secrets set COLAB_API_URL "https://<your-ngrok-subdomain>.ngrok-free.app"
   ```

   (For Streamlit Cloud, add the same key/value in the online secrets manager.)

2. Launch the app:

   ```bash
   streamlit run streamlit_app.py
   ```

The UI will fetch the video library, display original vs. rendered status,
and render two synchronized HTML5 video players for the selected item.

## Drive Folder Expectations

The backend assumes the following structure inside Google Drive (`MyDrive/ecolang`):

```
ecolang/
├─ videos/                     # Raw MP4s (names become video IDs)
├─ Extracted_parameters/
│  ├─ <video-id>_parameters/   # frame_0001_params.npz, etc.
└─ rendered_videos/            # Output MP4s written/uploaded by the backend
```

The backend caches Drive folder IDs and render manifests in
`/content/drive/MyDrive/ecolang/.cache/`. These are generated automatically and
can be deleted if you ever need a clean slate.

## Development Tips

- When modifying the Streamlit experience, work inside `ecolang/app.py`. The
  top-level `streamlit_app.py` intentionally stays tiny to keep Streamlit Cloud
  deployments simple.
- The Colab backend exposes helper endpoints:
  - `GET /health`
  - `GET /video_library`
  - `POST /render_video`
  - `GET /render_progress/<video_id>`
  - `GET /rendered_file/<video_id>` / `GET /original_file/<video_id>`
- Use `fastapi.testclient` or curl against the ngrok tunnel to debug without
  touching the UI.

## Maintenance

- Render metadata is tracked in `.cache/render_manifest.json` on Drive. If
  you manually remove rendered videos from Drive, call
  `GET /render_manifest?refresh=true` to rebuild the manifest.
- The backend automatically uploads completed renders to Drive and makes them
  publicly readable. If you prefer private files, adjust
  `upload_to_drive_auto` in `COLAB_BACKEND_AUTOMATED.py`.
- Keep ngrok tokens secret. Regenerate if the URL is ever compromised.

Enjoy the cleaner project structure!

