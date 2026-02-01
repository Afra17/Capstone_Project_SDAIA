import fitz  # PyMuPDF
import re
import os
from unstructured.cleaners.core import clean_extra_whitespace


def clean_vendor_text(text: str) -> str:
    """
    Standardized cleaning for Vendor proposals.
    """
    # 1) Remove noise: remove backslashes "\" (سبب الـ SyntaxError كان هنا)
    text = text.replace("\\", "")

    # 2) Strip Page markers like: --- PAGE 14 ---
    text = re.sub(r'---\s*PAGE\s*\d+\s*---', '', text)

    # 3) Standardize bullet points (•, O, o, -) to Markdown '*'
    text = re.sub(r'^[•Oo-]\s*', '* ', text, flags=re.MULTILINE)

    # 4) Map Arabic Sections (القسم ...) to H2 headers
    text = re.sub(r'(القسم\s+[\u0621-\u064A]+:.*)', r'\n## \1', text)

    # 5) Map Sub-sections (e.g., 2.1, 3.2) to H3 headers
    text = re.sub(r'^(\d+\.\d+.*)', r'\n### \1', text, flags=re.MULTILINE)

    # 6) Final white-space optimization for token efficiency
    text = clean_extra_whitespace(text)

    return text


def batch_process_vendors(input_folder: str, output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            md_filename = filename.rsplit(".", 1)[0] + ".md"
            md_path = os.path.join(output_folder, md_filename)

            print(f"📄 Processing: {filename}...")

            try:
                # Open PDF
                doc = fitz.open(pdf_path)

                # Collect text from all pages
                pages_text = []
                for page in doc:
                    pages_text.append(page.get_text())

                doc.close()

                full_text = "\n".join(pages_text)

                # Clean
                cleaned_md = clean_vendor_text(full_text)

                # Save as markdown
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_md)

                print(f"✅ Created: {md_filename}")

            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")


# --- EXECUTION ---
# Put all your vendor PDFs in a folder named 'vendors'
batch_process_vendors(input_folder="vendors", output_folder="cleaned_markdowns")
