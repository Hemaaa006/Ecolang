"""
Batch render all frames for a video
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import os

# Must have mesh_renderer, file_manager, config accessible
from mesh_renderer import MeshRenderer
from file_manager import FileManager
import config
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Batch render all frames')
    parser.add_argument('--video_id', required=True, help='Video ID to render')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--use_fallback', action='store_true', help='Use fallback for missing frames')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_mgr = FileManager()
    renderer = MeshRenderer(config.MODEL_PATH, device=args.device)

    # Get video info
    if args.video_id not in config.VIDEO_LIBRARY:
        print(f"Error: Video ID '{args.video_id}' not found")
        return

    video_info = config.VIDEO_LIBRARY[args.video_id]
    total_frames = video_info['frames']

    print(f"Rendering {total_frames} frames for {args.video_id}")
    print(f"Output: {output_dir}")

    # Statistics
    stats = {
        'total': total_frames,
        'success': 0,
        'fallback': 0,
        'failed': 0,
        'errors': []
    }

    # Process frames
    for frame_num in tqdm(range(1, total_frames + 1)):
        npz_path = file_mgr.get_npz_path(args.video_id, frame_num)
        output_path = output_dir / f"mesh_{frame_num:04d}.png"

        if not os.path.exists(npz_path):
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_num,
                'error': 'npz_not_found'
            })
            continue

        # Render
        img, status = renderer.render_frame(npz_path, use_fallback=args.use_fallback)

        if img is None:
            stats['failed'] += 1
            stats['errors'].append({
                'frame': frame_num,
                'error': status
            })
            continue

        # Save
        Image.fromarray(img).save(output_path)

        if status == "success":
            stats['success'] += 1
        elif status.startswith("fallback"):
            stats['fallback'] += 1
        else:
            stats['failed'] += 1

    # Save report
    report_path = output_dir / 'render_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("RENDERING COMPLETE")
    print("="*70)
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Fallback: {stats['fallback']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nReport: {report_path}")

if __name__ == '__main__':
    main()
