import os
import re
from dotenv import load_dotenv
from agentic_doc.parse import parse

# Load .env explicitly (your file is in D:\Capstone_Project_SDAIA\src\.env)
load_dotenv(r"D:\Capstone_Project_SDAIA\src\.env", override=True)


def _normalize_vendor_name(input_path: str) -> str:
    """
    Convert "Vendor C.pdf" -> "vendor_c"
    """
    name = os.path.splitext(os.path.basename(input_path))[0].strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name or "vendor"


def strip_agentic_comments(md: str) -> str:
    """
    Remove ONLY metadata comments inserted by agentic_doc:
    <!-- text, from page ... with ID ... -->
    """
    return re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)


def normalize_newlines(md: str) -> str:
    """
    Keep newlines, but normalize them safely.
    - Convert CRLF to LF
    - Remove trailing spaces per line
    - Collapse huge blank gaps (but preserve paragraphs)
    """
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = "\n".join(line.rstrip() for line in md.split("\n"))
    md = re.sub(r"\n{5,}", "\n\n\n", md)  # keep structure
    return md


def fix_hyphenation(md: str) -> str:
    """
    Fix broken words across line breaks:
    cap-\nstone -> capstone
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", md)


def ensure_markdown_headings(md: str) -> str:
    """
    Make headings appear as real headings with newlines.
    This is LOSSLESS: it only adds '\n' and '## ' prefixes when safe.
    """

    # 1) If you have inline headings like "## 23 - ...", ensure new line before them
    md = re.sub(r"(?<!\n)\s*(#{1,6}\s+)", r"\n\1", md)

    # 2) Promote numbered Arabic section titles to headings if they appear as standalone lines
    # Examples: "23 - التأهيل اللاحق" or "23- التأهيل اللاحق"
    md = re.sub(
        r"(?m)^(?P<num>\d{1,3}\s*[-–]\s*.+?)\s*$",
        r"## \g<num>",
        md
    )

    # 3) Whitelist some common Arabic section titles (standalone lines) -> headings
    # (You can add more here later)
    common_headers = [
        "مقدمة", "نطاق العمل", "الأهداف", "المنهجية", "الجدول الزمني",
        "خطة العمل", "إدارة المشروع", "الحوكمة", "الأمن السيبراني",
        "إدارة المخاطر", "ضمان الجودة", "المخرجات", "الافتراضات", "القيود"
    ]
    for h in common_headers:
        md = re.sub(rf"(?m)^{re.escape(h)}\s*$", rf"## {h}", md)

    return md


def insert_paragraph_breaks(md: str) -> str:
    """
    The key to making it look like your RFP file:
    Insert line breaks after sentence endings, but do it safely.

    IMPORTANT:
    - We do NOT delete text.
    - We only add '\n' after punctuation when there isn't already a newline.
    """

    # Add newline after Arabic sentence ends: "؟" "。" "!" "."
    md = re.sub(r"(؟)(?!\n)\s+", r"\1\n", md)
    md = re.sub(r"([.!])(?!\n)\s+", r"\1\n", md)

    # For Arabic commas/semicolons, insert softer breaks (optional but helps readability)
    md = re.sub(r"(،)(?!\n)\s+", r"\1\n", md)
    md = re.sub(r"(؛)(?!\n)\s+", r"\1\n", md)

    # Ensure lists start on a new line if stuck inline:
    # "1) ..." or "1. ..." or "- ..."
    md = re.sub(r"(?<!\n)\s(\d+\s*[\)\.])\s+", r"\n\1 ", md)
    md = re.sub(r"(?<!\n)\s-\s+", r"\n- ", md)

    # Collapse too many consecutive single-line breaks into paragraphs nicely
    md = re.sub(r"\n{4,}", "\n\n\n", md)

    return md


def cleanup_spacing_without_destroying_lines(md: str) -> str:
    """
    Normalize spacing BUT preserve newlines.
    Avoid clean_extra_whitespace() because it tends to squash layout.
    """
    # Replace tabs with spaces
    md = md.replace("\t", " ")

    # Reduce multiple spaces inside lines (not across newlines)
    md = "\n".join(re.sub(r"[ ]{2,}", " ", line) for line in md.split("\n"))

    # Remove stray spaces before punctuation
    md = re.sub(r"\s+([،؛:!?.؟])", r"\1", md)

    return md


def run_pipeline(input_path: str, output_folder: str) -> str:
    vendor_slug = _normalize_vendor_name(input_path)
    final_md_path = os.path.join(output_folder, f"{vendor_slug}_cleaned.md")

    print(f"🚀 Parsing PDF to Markdown: {os.path.basename(input_path)}")
    results = parse([input_path])

    if not results or not getattr(results[0], "markdown", ""):
        raise ValueError("Parsed Markdown is empty. Check agentic_doc config or PDF.")

    md = results[0].markdown

    print("Removing agentic_doc comments (<!-- ... -->)")
    md = strip_agentic_comments(md)

    print("Normalizing newlines (preserve structure)")
    md = normalize_newlines(md)

    print("Fixing broken hyphenated words")
    md = fix_hyphenation(md)

    print("Enforcing headings and section structure")
    md = ensure_markdown_headings(md)

    print("Inserting paragraph breaks (lossless)")
    md = insert_paragraph_breaks(md)

    print("Final spacing cleanup (without destroying layout)")
    md = cleanup_spacing_without_destroying_lines(md)

    # Final guarantee: file ends with newline
    if not md.endswith("\n"):
        md += "\n"

    os.makedirs(output_folder, exist_ok=True)
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Saved ONE clean file: {final_md_path}")
    return final_md_path



