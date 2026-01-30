#!/usr/bin/env python3
"""
Transcribe all Bertsekas section PDFs to markdown using docling.
"""

import json
from pathlib import Path
from docling.document_converter import DocumentConverter

# Configuration
INPUT_DIR = Path("/Users/pranjal/Code/rl/ch02_planning_learning/papers/bertsekas_rl_2ed")
OUTPUT_DIR = INPUT_DIR / "md"


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load index to get section list
    index_path = INPUT_DIR / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    sections = index["sections"]
    total = len(sections)

    print(f"Transcribing {total} PDFs to markdown...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Initialize converter once
    converter = DocumentConverter()

    for i, section in enumerate(sections, 1):
        pdf_path = INPUT_DIR / section["file"]
        md_filename = section["id"] + ".md"
        md_path = OUTPUT_DIR / md_filename

        # Skip if already transcribed
        if md_path.exists():
            print(f"[{i}/{total}] Skipping {section['id']} (already exists)")
            continue

        print(f"[{i}/{total}] Transcribing {section['file']}...")

        try:
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()

            # Add header with metadata
            header = f"""# {section['title']}

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** {section['pages']}
**Topics:** {', '.join(section['topics'])}

---

"""
            md_path.write_text(header + markdown)
            print(f"    -> Created {md_filename} ({len(markdown):,} chars)")

        except Exception as e:
            print(f"    ERROR: {e}")

    # Update index with markdown file paths
    for section in sections:
        section["markdown_file"] = "md/" + section["id"] + ".md"

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone! Markdown files saved to: {OUTPUT_DIR}")
    print("Updated index.json with markdown_file paths.")


if __name__ == "__main__":
    main()
