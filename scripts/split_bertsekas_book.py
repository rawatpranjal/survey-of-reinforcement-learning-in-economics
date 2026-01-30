"""
Split Bertsekas RL Course 2nd Edition into chapter PDFs.
Uses PyMuPDF (fitz) to extract page ranges.
"""

import fitz  # PyMuPDF
from pathlib import Path

# Configuration
INPUT_PDF = Path("ch02_planning_learning/papers/RLCOURSECOMPLETE 2ndEDITION.pdf")
OUTPUT_DIR = Path("ch02_planning_learning/papers")

# Page ranges (1-indexed, inclusive) - PyMuPDF uses 0-indexed
SECTIONS = [
    ("bertsekas2025_rl_course_2ed_frontmatter.pdf", 1, 20),
    ("bertsekas2025_rl_course_2ed_ch01.pdf", 21, 182),
    ("bertsekas2025_rl_course_2ed_ch02.pdf", 183, 368),
    ("bertsekas2025_rl_course_2ed_ch03.pdf", 369, 491),
    ("bertsekas2025_rl_course_2ed_refs.pdf", 492, 521),
]


def split_pdf():
    """Split the input PDF into sections based on page ranges."""
    if not INPUT_PDF.exists():
        print(f"Error: Input PDF not found at {INPUT_PDF}")
        return False

    print(f"Opening {INPUT_PDF}...")
    doc = fitz.open(INPUT_PDF)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    for output_name, start_page, end_page in SECTIONS:
        output_path = OUTPUT_DIR / output_name

        # Convert to 0-indexed for PyMuPDF
        start_idx = start_page - 1
        end_idx = end_page  # fitz.select uses exclusive end

        print(f"Extracting pages {start_page}-{end_page} to {output_name}...")

        # Create new PDF with selected pages
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_page - 1)
        new_doc.save(output_path)
        new_doc.close()

        print(f"  Created {output_path} ({end_page - start_page + 1} pages)")

    doc.close()
    print("\nDone! All sections extracted successfully.")
    return True


if __name__ == "__main__":
    split_pdf()
