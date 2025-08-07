import fitz  # PyMuPDF - library to read and work with PDFs
import re
import json
from pathlib import Path
from statistics import mean

# =========================
# Configuration
# =========================
# Common section names that usually appear in research papers.
# We use these to help guess if a block is a section heading.
KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methodology", "approach", "experiment", "evaluation",
    "results", "analysis", "discussion", "conclusion", "future work", "references"
}

# Patterns to catch numbered headings like:
# "1. Introduction", "1.1 Background", or "I. Related Work"
NUMBERING_REGEXES = [
    r'^\s*\d+(\.\d+)*\s+[A-Z]',  # Matches decimal numbering
    r'^\s*[IVXLC]+\.\s+[A-Z]',   # Matches Roman numeral numbering
]

# =========================
# Step 1: Read PDF and extract text with layout info
# =========================
def extract_blocks_with_layout(pdf_path):
    """
    Reads a PDF and pulls out chunks of text (blocks) along with
    font size, position, and page number — so we can later guess
    which ones are section titles.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = "".join([span["text"] for span in line["spans"]]).strip()
                if not text:
                    continue
                font_sizes = [span["size"] for span in line["spans"]]
                avg_font = mean(font_sizes)
                all_blocks.append({
                    "text": text,
                    "font_size": avg_font,
                    "y0": block["bbox"][1],  # Top Y-coordinate of the block
                    "page": page_num
                })
    return all_blocks

# =========================
# Step 2: Helpers to check if a block looks like a heading
# =========================
def is_all_caps(text):
    return text.isupper() and len(text) > 3

def is_title_case(text):
    return text.istitle()

def matches_numbering(text):
    return any(re.match(rgx, text) for rgx in NUMBERING_REGEXES)

# =========================
# Step 3: Score blocks to guess headings
# =========================
def compute_scores(blocks):
    """
    Gives each text block a 'heading-likelihood' score
    based on font size, capitalization, numbering, and spacing.
    """
    font_sizes = [b["font_size"] for b in blocks]
    avg_font = mean(font_sizes)

    for i, block in enumerate(blocks):
        score = 0

        # Bigger font than average → more likely a heading
        if block["font_size"] >= avg_font * 1.15:
            score += 3

        # ALL CAPS or Title Case gives extra points
        if is_all_caps(block["text"]):
            score += 1
        if is_title_case(block["text"]):
            score += 1

        # Matches something like "1. Introduction" or "II. Background"
        if matches_numbering(block["text"]):
            score += 2

        # Extra vertical gap before this block → might be a heading
        if i > 0:
            y_gap = block["y0"] - blocks[i - 1]["y0"]
            if y_gap > 20:  # tweak as needed per document style
                score += 1

        block["score"] = score
    return blocks

# =========================
# Step 4: Turn scored blocks into structured sections
# =========================
def process_pdf_to_sections(pdf_path, threshold=4):
    """
    Goes through the PDF, figures out where headings are,
    and groups the text under each heading into a clean JSON structure.
    """
    blocks = extract_blocks_with_layout(pdf_path)
    scored_blocks = compute_scores(blocks)

    sections = []
    current_section = None

    for block in scored_blocks:
        text = block["text"]
        score = block["score"]

        if score >= threshold:
            # Found a new heading → save previous section if it exists
            if current_section and current_section["text"].strip():
                sections.append(current_section)

            # Make the heading look nice (title case if it’s known)
            title = text.strip()
            title_lower = title.lower()
            if any(known in title_lower for known in KNOWN_HEADINGS):
                section_title = title.title()
            else:
                section_title = title

            # Start a fresh section
            current_section = {"section": section_title, "text": ""}

        else:
            # Not a heading → add this text to the current section
            if current_section:
                current_section["text"] += text + "\n"
            else:
                # If no section started yet, put it under "Unknown"
                current_section = {"section": "Unknown", "text": text + "\n"}

    # Add the very last section to the list
    if current_section and current_section["text"].strip():
        sections.append(current_section)

    return sections

# =========================
# Step 5: Save the structured sections as JSON
# =========================
def save_sections_as_json(sections, output_path):
    """
    Saves the list of sections to a JSON file so other tools can read it.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

# =========================
# Step 6: Run from the command line
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunker.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = Path(pdf_path).with_suffix(".chunks.json")

    sections = process_pdf_to_sections(pdf_path)
    save_sections_as_json(sections, output_path)

    print(f" Done! Sections saved to {output_path}")
