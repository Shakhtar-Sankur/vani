"""Prints results/m0/<lang>/<system>.json as one table.

    python scripts/summarize_m0.py [results/m0]
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0")
for lang_dir in sorted(glob.glob(os.path.join(root, "*", ""))):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(lang_dir, "*.json")))]
    if not rows:
        continue
    print(f"== {os.path.basename(os.path.dirname(lang_dir))}")
    print(f"{'system':30s} {'WER':>7s} {'CER':>7s} {'script':>7s} {'s/audio s':>10s} {'tokens':>7s} {'>448':>5s}  detected language")
    for r in sorted(rows, key=lambda r: r["cer"]):
        print(f"{r['system']:30s} {100 * r['wer']:6.1f}% {100 * r['cer']:6.1f}% {100 * r['script_share']:6.1f}% "
              f"{r['seconds_per_audio_second']:10.4f} {r.get('tokens_per_transcript', ''):>7} {r.get('hit_448_limit', ''):>5}  "
              f"{r.get('detected_languages', '')}")
