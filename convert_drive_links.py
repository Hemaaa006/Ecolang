#!/usr/bin/env python3
"""
Google Drive Share Link to Direct Download URL Converter
=========================================================

This script helps you convert Google Drive share links to direct download URLs
for use in the ECOLANG config.py file.

Usage:
    python convert_drive_links.py

Then paste your Google Drive share links when prompted.
"""

import re


def extract_file_id(share_link):
    """Extract FILE_ID from Google Drive share link"""
    # Pattern: https://drive.google.com/file/d/FILE_ID/view...
    pattern = r'/d/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, share_link)

    if match:
        return match.group(1)
    else:
        return None


def convert_to_direct_url(file_id):
    """Convert FILE_ID to direct download URL"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def main():
    print("="*70)
    print("  Google Drive Link Converter for ECOLANG")
    print("="*70)
    print("\nThis script will help you convert Google Drive share links")
    print("to direct download URLs for your config.py file.\n")

    print("Instructions:")
    print("1. Upload your videos to Google Drive: /MyDrive/ecolang/videos/")
    print("2. Right-click each video → Share → 'Anyone with the link' can view")
    print("3. Copy the share link")
    print("4. Paste it below when prompted\n")
    print("="*70)

    videos = [
        ('video1_speaking.mp4', 'Video 1 - Speaking'),
        ('video2_gestures.mp4', 'Video 2 - Gestures'),
        ('video3_conversation.mp4', 'Video 3 - Conversation'),
        ('video4_demonstration.mp4', 'Video 4 - Demonstration')
    ]

    results = {}

    for filename, title in videos:
        print(f"\n📹 {title} ({filename})")
        print("-" * 70)

        while True:
            share_link = input("Paste Google Drive share link (or 'skip'): ").strip()

            if share_link.lower() == 'skip':
                print("  ⚠️  Skipped")
                results[filename] = "PASTE_FILE_ID_HERE"
                break

            file_id = extract_file_id(share_link)

            if file_id:
                direct_url = convert_to_direct_url(file_id)
                print(f"  ✓ FILE_ID extracted: {file_id}")
                print(f"  ✓ Direct URL: {direct_url}")
                results[filename] = file_id
                break
            else:
                print("  ❌ Invalid link format. Please try again.")
                print("     Expected format: https://drive.google.com/file/d/FILE_ID/view...")

    # Generate config.py snippet
    print("\n" + "="*70)
    print("  COPY THIS TO YOUR config.py FILE")
    print("="*70)
    print("\nVIDEO_LIBRARY = {")

    for filename, title in videos:
        video_id = filename.replace('.mp4', '')
        file_id = results[filename]
        print(f"    '{video_id}': {{")
        print(f"        'title': '{title}',")
        print(f"        'filename': '{filename}',")
        print(f"        'github_url': 'https://drive.google.com/uc?export=download&id={file_id}',")
        print(f"        'frames': 1800,")
        print(f"        'fps': 30,")
        print(f"        'duration': '1:00'")
        print(f"    }},")

    print("}")
    print("\n" + "="*70)
    print("✅ Done! Copy the above configuration to signmesh/config.py")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
