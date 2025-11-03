"""
ECOLANG - Main Streamlit Application
Minimal control panel for Colab rendering backend
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import streamlit as st

from api_client import get_api_client

st.set_page_config(
    page_title="ECOLANG - 3D Mesh Rendering",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal styling for clean layout
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px !important;
    }

    .page-title {
        text-align: center;
        font-size: 2.75rem;
        font-weight: 700;
        color: #1f2933;
        letter-spacing: 0.04em;
        margin-bottom: 1.5rem;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.75rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-pill.status-ok {
        background: #d1e7dd;
        color: #0f5132;
    }
    .status-pill.status-warn {
        background: #f8d7da;
        color: #842029;
    }
    .status-pill.status-info {
        background: #dbeafe;
        color: #1d4ed8;
    }

    .video-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0 1.2rem 0;
    }
    .video-table th {
        text-align: left;
        font-size: 0.85rem;
        font-weight: 600;
        color: #4b5563;
        padding: 0.45rem 0.65rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .video-table td {
        padding: 0.55rem 0.65rem;
        font-size: 0.85rem;
        border-bottom: 1px solid #f3f4f6;
        color: #1f2933;
    }
    .video-table tr.selected {
        background: #f0f7ff;
    }
    .video-table tr:hover {
        background: #f8fafc;
    }
    .video-name {
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .video-wrapper {
        position: relative;
        width: 100%;
        padding-bottom: 56.25%;
        background: #000;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.25);
    }
    .video-wrapper video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        border: none;
    }

    .placeholder-card {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 320px;
        background: #f8fafc;
        border: 1px dashed #cbd5f5;
        border-radius: 14px;
        color: #64748b;
        font-size: 0.95rem;
        text-align: center;
        padding: 1.5rem;
    }

    .sync-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(15, 118, 228, 0.92);
        color: #fff;
        padding: 0.75rem 1.6rem;
        border-radius: 999px;
        font-weight: 600;
        letter-spacing: 0.04em;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.35);
        cursor: pointer;
        display: none;
        z-index: 20;
    }

    .message-card {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        border: 1px solid #dbeafe;
        background: #eff6ff;
        color: #1d4ed8;
        margin-bottom: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def status_pill(text: str, variant: str) -> str:
    return f'<span class="status-pill status-{variant}">{text}</span>'


def build_video_table(library: Dict[str, dict], selected_video: Optional[str]) -> str:
    rows = [
        "<table class='video-table'>",
        "<thead><tr><th>Video</th><th>Frames</th><th>Original</th><th>NPZ</th><th>Rendered</th></tr></thead>",
        "<tbody>",
    ]
    for video_id, meta in library.items():
        is_selected = "selected" if video_id == selected_video else ""
        frames = meta.get("frames")
        frames_text = f"{int(frames):,}" if isinstance(frames, (int, float)) and frames else "–"
        original_ok = bool(meta.get("original_path"))
        npz_ok = bool(meta.get("npz_exists"))
        rendered_ok = meta.get("status") == "ready"
        rows.append(
            "<tr class='{cls}'>"
            "<td class='video-name'>{name}</td>"
            "<td>{frames}</td>"
            "<td>{original}</td>"
            "<td>{npz}</td>"
            "<td>{rendered}</td>"
            "</tr>".format(
                cls=is_selected,
                name=meta.get("title") or video_id,
                frames=frames_text,
                original=status_pill("Available", "ok" if original_ok else "warn"),
                npz=status_pill("Ready" if npz_ok else "Missing", "ok" if npz_ok else "warn"),
                rendered=status_pill(
                    "Ready" if rendered_ok else "Pending",
                    "ok" if rendered_ok else "warn",
                ),
            )
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def load_video_library(api_client, refresh: bool = False) -> Tuple[Dict[str, dict], Optional[str], Optional[str]]:
    ok, payload = api_client.get_video_library(refresh=refresh)
    if not ok:
        return {}, None, payload.get("error")
    videos: Dict[str, dict] = {}
    for item in payload.get("videos", []):
        video_id = item.get("video_id")
        if not video_id:
            continue
        videos[video_id] = item
    return videos, payload.get("active_job"), None


def resolve_original_url(api_client, video_id: str, entry: dict) -> Optional[str]:
    if entry.get("original_url"):
        return entry["original_url"]
    if entry.get("original_endpoint"):
        return f"{api_client.api_url}{entry['original_endpoint']}"
    return api_client.get_original_stream_url(video_id)


def resolve_rendered_url(api_client, video_id: str, entry: dict) -> Optional[str]:
    if entry.get("file_url"):
        return entry["file_url"]
    if entry.get("rendered_url"):
        return entry["rendered_url"]
    if entry.get("rendered_endpoint"):
        return f"{api_client.api_url}{entry['rendered_endpoint']}"
    if entry.get("status") == "ready":
        return api_client.get_rendered_stream_url(video_id)
    if entry.get("drive_download_url"):
        return entry["drive_download_url"]
    if entry.get("drive_preview_url"):
        return entry["drive_preview_url"]
    return None


def render_video_player_html(source_url: str, element_id: str, include_overlay: bool = False) -> str:
    overlay_html = ""
    if include_overlay:
        overlay_html = """
        <div id="sync-overlay" class="sync-overlay">Play Both Videos</div>
        <script>
            function playSyncedVideos() {
                const overlay = document.getElementById('sync-overlay');
                if (overlay) {
                    overlay.style.display = 'none';
                }
                const renderedVideo = document.getElementById('rendered-video');
                if (renderedVideo) {
                    renderedVideo.play().catch(function(){});
                }
                const originalVideo = document.getElementById('original-video');
                if (originalVideo) {
                    originalVideo.currentTime = renderedVideo ? renderedVideo.currentTime : originalVideo.currentTime;
                    originalVideo.play().catch(function(){});
                }
            }
            const renderedVideoElement = document.getElementById('rendered-video');
            if (renderedVideoElement) {
                renderedVideoElement.addEventListener('loadedmetadata', function() {
                    const overlay = document.getElementById('sync-overlay');
                    if (overlay) {
                        overlay.style.display = 'flex';
                    }
                });
                renderedVideoElement.addEventListener('play', function() {
                    const overlay = document.getElementById('sync-overlay');
                    if (overlay) {
                        setTimeout(function(){ overlay.style.display = 'none'; }, 400);
                    }
                });
            }
        </script>
        """
    return (
        f"<div class='video-wrapper'>"
        f"<video id='{element_id}' src='{source_url}' controls playsinline preload='metadata'></video>"
        f"{overlay_html}"
        f"</div>"
    )


def main():
    st.markdown('<div class="page-title">ECOLANG</div>', unsafe_allow_html=True)

    api_client = get_api_client()

    if "api_healthy" not in st.session_state:
        with st.spinner("Connecting to backend..."):
            healthy, info = api_client.health_check()
            st.session_state.api_healthy = healthy
            st.session_state.api_info = info

    if not st.session_state.api_healthy:
        st.error(f"Backend connection failed: {st.session_state.api_info}")
        st.info("Ensure the Colab backend is running and the Streamlit secret COLAB_API_URL is set.")
        return

    defaults = {
        "video_library": {},
        "active_job_id": None,
        "selected_video_id": None,
        "rendered_video_url": None,
        "rendering_in_progress": False,
        "force_confirm_video": None,
        "force_confirm_active": None,
        "library_needs_refresh": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    refresh_requested = st.button("Refresh Library", key="refresh_library_button")
    if refresh_requested:
        st.session_state.library_needs_refresh = True

    if st.session_state.library_needs_refresh or not st.session_state.video_library:
        with st.spinner("Loading video library..."):
            library, active_job, error = load_video_library(api_client, refresh=True)
        if error:
            st.error(f"Failed to load video library: {error}")
            return
        st.session_state.video_library = library
        st.session_state.active_job_id = active_job
        st.session_state.library_needs_refresh = False
    else:
        library = st.session_state.video_library

    library = st.session_state.video_library
    active_job = st.session_state.active_job_id

    if not library:
        st.warning("No videos were found in /content/drive/MyDrive/ecolang/videos.")
        return

    video_ids = list(library.keys())
    if st.session_state.selected_video_id not in video_ids:
        st.session_state.selected_video_id = video_ids[0]

    selection_col, info_col = st.columns([0.7, 0.3], gap="large")
    selected_video = selection_col.selectbox(
        "Select a video",
        options=video_ids,
        index=video_ids.index(st.session_state.selected_video_id),
        format_func=lambda vid: library[vid].get("title") or vid,
    )

    if selected_video != st.session_state.selected_video_id:
        st.session_state.selected_video_id = selected_video
        st.session_state.rendered_video_url = None
        st.session_state.rendering_in_progress = False
        st.session_state.force_confirm_video = None
        st.session_state.force_confirm_active = None

    selected_entry = library[st.session_state.selected_video_id]

    # Display library table with current selection highlighted
    table_html = build_video_table(library, st.session_state.selected_video_id)
    st.markdown(table_html, unsafe_allow_html=True)

    original_url = resolve_original_url(api_client, st.session_state.selected_video_id, selected_entry)
    rendered_ready = selected_entry.get("status") == "ready"
    if rendered_ready:
        st.session_state.rendered_video_url = resolve_rendered_url(api_client, st.session_state.selected_video_id, selected_entry)
    elif not st.session_state.rendering_in_progress:
        st.session_state.rendered_video_url = None

    st.session_state.rendering_in_progress = (
        active_job == st.session_state.selected_video_id and not rendered_ready
    )

    status_badges = [
        status_pill("Original" if selected_entry.get("original_path") else "Original Missing",
                    "ok" if selected_entry.get("original_path") else "warn"),
        status_pill("NPZ Ready" if selected_entry.get("npz_exists") else "NPZ Missing",
                    "ok" if selected_entry.get("npz_exists") else "warn"),
    ]
    if rendered_ready:
        status_badges.append(status_pill("Rendered Available", "ok"))
    else:
        status_badges.append(status_pill("Render Pending", "warn"))

    st.markdown(f"<div class='status-row'>{''.join(status_badges)}</div>", unsafe_allow_html=True)

    if active_job and active_job != st.session_state.selected_video_id:
        active_label = library.get(active_job, {}).get("title") or active_job
        st.markdown(
            f"<div class='message-card'>Rendering in progress for <strong>{active_label}</strong>. "
            "Starting a new render will stop the current job.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.force_confirm_video and st.session_state.force_confirm_video != st.session_state.selected_video_id:
        st.session_state.force_confirm_video = None
        st.session_state.force_confirm_active = None

    render_disabled = not selected_entry.get("npz_exists") or (
        st.session_state.rendering_in_progress and active_job == st.session_state.selected_video_id
    )
    render_label = "Render Video" if not st.session_state.rendering_in_progress else "Rendering..."

    render_clicked = st.button(
        render_label,
        key="render_button",
        disabled=render_disabled,
        use_container_width=True,
        type="primary",
    )

    if render_clicked:
        video_url, status, payload = api_client.render_video(st.session_state.selected_video_id)
        if payload.get("success"):
            if payload.get("already_exists"):
                st.session_state.rendered_video_url = resolve_rendered_url(
                    api_client, st.session_state.selected_video_id, selected_entry
                )
                st.session_state.rendering_in_progress = False
                st.session_state.library_needs_refresh = True
                st.info("Rendered video already exists. Loaded latest version.")
                st.rerun()
            else:
                st.session_state.rendering_in_progress = True
                st.session_state.active_job_id = payload.get("job_id", st.session_state.selected_video_id)
                st.session_state.rendered_video_url = None
                st.session_state.library_needs_refresh = True
                st.success("Rendering started. Progress will update below.")
                st.rerun()
        else:
            error_code = payload.get("error") or status.split(":", 1)[-1]
            if error_code == "render_in_progress":
                st.session_state.force_confirm_video = st.session_state.selected_video_id
                st.session_state.force_confirm_active = payload.get("active_job")
                st.warning("Another rendering job is active. Confirm to stop it and start this one.")
            elif error_code == "npz_missing":
                st.error("No NPZ parameters were found for this video. Rendering is disabled.")
            elif error_code == "unable_to_cancel":
                st.error("Unable to stop the current render. Please wait for it to finish.")
            else:
                st.error(f"Failed to start rendering: {error_code}")

    if (
        st.session_state.force_confirm_video == st.session_state.selected_video_id
        and st.session_state.force_confirm_active
    ):
        active_label = library.get(st.session_state.force_confirm_active, {}).get("title") or st.session_state.force_confirm_active
        confirm = st.button(
            f"Stop '{active_label}' and render '{library[st.session_state.selected_video_id].get('title') or st.session_state.selected_video_id}'",
            key="force_render_button",
            type="primary",
            use_container_width=True,
        )
        if confirm:
            _, status, payload = api_client.render_video(st.session_state.selected_video_id, force=True)
            if payload.get("success"):
                st.session_state.rendering_in_progress = True
                st.session_state.active_job_id = payload.get("job_id", st.session_state.selected_video_id)
                st.session_state.rendered_video_url = None
                st.session_state.force_confirm_video = None
                st.session_state.force_confirm_active = None
                st.session_state.library_needs_refresh = True
                st.success("Previous render stopped. New rendering started.")
                st.rerun()
            else:
                st.error(f"Unable to force start: {payload.get('error', status)}")

    if not selected_entry.get("npz_exists"):
        st.warning("NPZ parameter folder not found. Rendering is disabled until the parameters are uploaded.")
    elif not rendered_ready and not st.session_state.rendering_in_progress:
        st.info("Parameters found. Click 'Render Video' to generate the mesh render.")

    col_original, col_rendered = st.columns(2, gap="large")

    with col_original:
        col_original.subheader("Original Video", divider=True)
        if original_url:
            col_original.markdown(
                render_video_player_html(original_url, "original-video"),
                unsafe_allow_html=True,
            )
        else:
            col_original.markdown(
                "<div class='placeholder-card'>Original video is not available in Drive.</div>",
                unsafe_allow_html=True,
            )

    with col_rendered:
        col_rendered.subheader("Rendered Video", divider=True)
        if st.session_state.rendering_in_progress and active_job == st.session_state.selected_video_id:
            col_rendered.markdown(
                "<div class='placeholder-card'>Rendering in progress...</div>",
                unsafe_allow_html=True,
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
            max_frames = selected_entry.get("frames") or 0

            while st.session_state.rendering_in_progress:
                progress = api_client.get_render_progress(st.session_state.selected_video_id)
                if not progress:
                    break

                current = progress.get("current", 0)
                total = progress.get("total", max_frames) or 1
                pct = int(min(100, (current / total) * 100))
                progress_bar.progress(pct)
                status_text.markdown(f"**Frame {current} of {total}** ({pct}%)")

                job_status = progress.get("status")
                if job_status == "complete":
                    st.session_state.rendered_video_url = resolve_rendered_url(
                        api_client, st.session_state.selected_video_id, selected_entry
                    )
                    st.session_state.rendering_in_progress = False
                    st.session_state.library_needs_refresh = True
                    time.sleep(0.5)
                    st.rerun()
                elif job_status in {"error", "cancelled"}:
                    error_msg = progress.get("error", job_status)
                    st.error(f"Render {job_status}: {error_msg}")
                    st.session_state.rendering_in_progress = False
                    st.session_state.library_needs_refresh = True
                    time.sleep(0.5)
                    st.rerun()

                time.sleep(2)
        elif st.session_state.rendered_video_url:
            col_rendered.markdown(
                render_video_player_html(st.session_state.rendered_video_url, "rendered-video", include_overlay=True),
                unsafe_allow_html=True,
            )
        else:
            col_rendered.markdown(
                "<div class='placeholder-card'>No rendered output yet. Start a render to see the result here.</div>",
                unsafe_allow_html=True,
            )


# Streamlit runs main() when executing the script
main()
