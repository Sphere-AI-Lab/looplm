"""Export the paper's original figures for the static project page.

Usage: python3 scripts/export_figures.py ../iclr2027_looplm_analysis
Requires PyMuPDF: python3 -m pip install pymupdf
"""
import argparse
import json
import shutil
from pathlib import Path

import fitz

FIGURES = {
    "recurrence": "motivation/llama_motivation_trends_final.pdf",
    "architecture": "baseloop_coreloop/llama_baseloop_coreloop_compact.pdf",
    "initial-state": "input_injection/qwen_all_shapes_compact.pdf",
    "history-state": "input_injection/qwen_4x7_history_windows_compact.pdf",
    "timestep": "timestep/qwen_family_compact.pdf",
    "combined": "combine/combine_final_results.pdf",
    "geometry": "geometry/four_questions_main.pdf",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the paper's figs directory")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "docs/assets/figures"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, relative in FIGURES.items():
        source = args.source / "figs" / relative
        with fitz.open(source) as document:
            page = document[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2400 / page.rect.width, 2400 / page.rect.width), alpha=False)
            pixmap.save(destination / f"{name}.png")
            manifest[name] = {"source": f"figs/{relative}", "width": pixmap.width, "height": pixmap.height}
        shutil.copyfile(source, destination / f"{name}.pdf")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
