# ECOLANG Sample Videos

This directory is only a placeholder to document the expected layout. The
Streamlit UI discovers *actual* videos from Google Drive via the Colab backend,
so you do **not** need to commit production MP4s into the repo.

## Using Local Samples (Optional)

1. Drop one or more MP4 files in this folder. The filename (without extension)
   becomes the `video_id`.
2. Ensure Drive contains matching parameter folders at  
   `/MyDrive/ecolang/Extracted_parameters/<video_id>_parameters/`.
3. Trigger a library refresh in the UI or call
   `GET /render_manifest?refresh=true` on the backend.

## Tips

- Use H.264 MP4 for the smoothest playback in browsers.
- Frame rate and duration are detected automatically; 30 fps is recommended.
- Keep the repository light—treat this folder as scratch space when you need
  local demos.
