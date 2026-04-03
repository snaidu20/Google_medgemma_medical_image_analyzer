#!/usr/bin/env python3
"""
CLI entry point for MedGemma Chest X-Ray Analyzer.

Usage examples:
    python run_analysis.py --image path/to/xray.png
    python run_analysis.py --image https://upload.wikimedia.org/.../Chest_Xray.png
    python run_analysis.py --image xray.png --type findings_only --output report.txt
    python run_analysis.py --image xray.png --prompt "Focus on the left lung apex"
    python run_analysis.py --image xray.png --mode quantized --type detailed_report
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make project root importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv()

from configs.model_config import DISCLAIMER, PROMPT_LABELS, PROMPTS
from src.analyzer import ChestXRayAnalyzer
from src.image_utils import load_image, validate_xray_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="MedGemma X-Ray CLI",
        description="Analyze a chest X-ray image with MedGemma and generate a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis types:
  detailed_report  Full structured radiology report (default)
  describe         General description of the X-ray
  findings_only    Key findings as bullet points
  compare          Comparison to a normal chest X-ray
  simple           Patient-friendly plain language explanation

Examples:
  python run_analysis.py --image chest.png
  python run_analysis.py --image chest.png --type findings_only
  python run_analysis.py --image chest.png --mode quantized --output report.txt
        """,
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to local X-ray image file or HTTP/HTTPS URL",
    )
    parser.add_argument(
        "--type",
        dest="analysis_type",
        choices=list(PROMPTS.keys()),
        default="detailed_report",
        metavar="TYPE",
        help=f"Analysis type. Choices: {', '.join(PROMPTS.keys())} (default: detailed_report)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt text (overrides --type if provided)",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "quantized", "full"],
        default="quantized",
        help="Model loading mode (default: quantized)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save the report to this file path",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (overrides HF_TOKEN env var)",
    )
    return parser.parse_args()


def print_header() -> None:
    print("\n" + "═" * 64)
    print("  🫁  MedGemma Chest X-Ray Analyzer")
    print("  Powered by google/medgemma-1.5-4b-it")
    print("═" * 64 + "\n")


def print_section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


def main() -> int:
    args = parse_args()
    print_header()

    # ── Step 1: Load Image ────────────────────────────────────────────────────
    print_section("Step 1/3 — Loading image")
    print(f"  Source : {args.image}")
    try:
        image = load_image(args.image)
    except (FileNotFoundError, ConnectionError, ValueError) as exc:
        print(f"  ❌ {exc}")
        return 1
    print(f"  ✅ Loaded: {image.size[0]}x{image.size[1]}px ({image.mode})")

    # ── Validate ──────────────────────────────────────────────────────────────
    validation = validate_xray_image(image)
    for w in validation["warnings"]:
        print(f"  ⚠️  {w}")

    # ── Step 2: Load Model ────────────────────────────────────────────────────
    print_section("Step 2/3 — Loading model")
    print(f"  Mode   : {args.mode}")
    hf_token = args.token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("  ⚠️  HF_TOKEN not set. Model download may fail for gated repos.")

    analyzer = ChestXRayAnalyzer(mode=args.mode, hf_token=hf_token)
    t0 = time.time()
    try:
        analyzer.load_model()
    except Exception as exc:
        print(f"  ❌ Model load failed: {exc}")
        return 1
    load_time = time.time() - t0
    print(f"  ✅ Model ready in {load_time:.1f}s")

    # ── Step 3: Analyze ───────────────────────────────────────────────────────
    prompt_label = PROMPT_LABELS.get(args.analysis_type, args.analysis_type)
    print_section("Step 3/3 — Running analysis")
    print(f"  Type   : {prompt_label}")
    if args.prompt:
        print(f"  Prompt : {args.prompt}")

    t0 = time.time()
    try:
        result = analyzer.analyze(
            image=image,
            prompt_key=args.analysis_type,
            custom_prompt=args.prompt,
        )
    except Exception as exc:
        print(f"  ❌ Analysis failed: {exc}")
        return 1
    analysis_time = time.time() - t0
    print(f"  ✅ Analysis completed in {analysis_time:.1f}s")

    # ── Format & Display Report ───────────────────────────────────────────────
    sep = "═" * 64
    report_lines = [
        "",
        sep,
        "  RADIOLOGY REPORT",
        sep,
        f"  Model  : {result['model']}",
        f"  Prompt : {result['prompt_used']}",
        f"  Timing : model_load={load_time:.1f}s  analysis={analysis_time:.1f}s",
        "",
        result["report"],
        "",
        sep,
        f"  {DISCLAIMER}",
        sep,
        "",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    # ── Save to File ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report_text, encoding="utf-8")
        print(f"  💾 Report saved to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
