import os
from typing import List, Dict

import pdfplumber
import camelot
import pytesseract
from PIL import Image

from utils import PDF_DIR


# -------------------------------------------------------------
# 1. TEXT EXTRACTION WITH OCR FALLBACK
# -------------------------------------------------------------
def extract_text_with_ocr(pdf_path: str,
                          min_chars_for_text: int = 40) -> List[Dict]:
    """
    Extracts text page-by-page using pdfplumber.
    Falls back to OCR if needed.
    Returns list: {page_num, text}
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Primary text extraction
            text = (page.extract_text() or "").strip()

            # OCR fallback if too little text
            if len(text) < min_chars_for_text:
                try:
                    img = page.to_image(resolution=300).original
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        text = ocr_text
                except Exception as e:
                    print(f"[WARN] OCR failed on p.{page_num}: {e}")

            if text:
                pages.append({"page_num": page_num, "text": text})

    return pages


# -------------------------------------------------------------
# 2. TABLE EXTRACTION
# -------------------------------------------------------------
def extract_tables_as_text(pdf_path: str) -> List[Dict]:
    """
    Extract tables using Camelot.
    Converts tables into README-style text blocks.
    """
    tables_text = []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all")
    except Exception as e:
        print(f"[WARN] Camelot failed: {e}")
        return tables_text

    for t in tables:
        df = t.df
        table_str = df.to_csv(index=False)

        tables_text.append({
            "page_num": t.page,
            "text": f"Extracted Table (Page {t.page}):\n{table_str}"
        })

    return tables_text


# -------------------------------------------------------------
# 3. DOCUMENT UNIT COLLECTION
# -------------------------------------------------------------
def extract_document_units(pdf_path: str) -> List[Dict]:
    """
    Collects text units and table units.
    Returns a unified list:
    {type: 'text'|'table', page_num, text}
    """
    units = []

    text_pages = extract_text_with_ocr(pdf_path)
    table_pages = extract_tables_as_text(pdf_path)

    for item in text_pages:
        units.append({
            "type": "text",
            "page_num": item["page_num"],
            "text": item["text"],
        })

    for item in table_pages:
        units.append({
            "type": "table",
            "page_num": item["page_num"],
            "text": item["text"],
        })

    return units


# -------------------------------------------------------------
# 4. NOTEBOOKLM-STYLE CHUNKING
# -------------------------------------------------------------
def chunk_text_notebooklm(text: str,
                          max_chars: int = 1200) -> List[str]:
    """
    Small natural-language chunks inspired by NotebookLM.
    NotebookLM uses page-first + paragraph-based chunking.

    We break:
    - by line
    - merge lines until < 1200 chars
    - flush frequently (fast, memory-safe)
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    buf = ""

    for line in lines:
        if len(buf) + len(line) + 1 > max_chars:
            chunks.append(buf.strip())
            buf = ""

        buf += " " + line

    if buf.strip():
        chunks.append(buf.strip())

    return chunks


# -------------------------------------------------------------
# 5. CREATE CHUNKS FROM A SINGLE PDF
# -------------------------------------------------------------
def create_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract units → chunk them NotebookLM-style.
    Returns unified RAG-ready chunks:

    {
        "pdf_name": "...",
        "page_num": 3,
        "unit_type": "text",
        "chunk_index": 1,
        "chunk_text": "...."
    }
    """
    pdf_name = os.path.basename(pdf_path)
    units = extract_document_units(pdf_path)

    all_chunks = []

    for u in units:
        chunks = chunk_text_notebooklm(u["text"], max_chars=1200)

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "pdf_name": pdf_name,
                "page_num": u["page_num"],
                "unit_type": u["type"],
                "chunk_index": i + 1,
                "chunk_text": ch,
            })

    return all_chunks


# -------------------------------------------------------------
# 6. COLLECT CHUNKS FROM PDF DIRECTORY
# -------------------------------------------------------------
def collect_all_chunks(pdf_dir: str = PDF_DIR) -> List[Dict]:
    """
    Grab all PDFs → extract → chunk → combine.
    """
    pdf_paths = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    print(f"Found {len(pdf_paths)} PDFs in {pdf_dir}")

    all_chunks = []

    for pdf in pdf_paths:
        print(f"[PDF] {pdf}")
        chunks = create_chunks_from_pdf(pdf)
        print(f"  → {len(chunks)} chunks created")
        all_chunks.extend(chunks)

    print(f"Total chunks from all PDFs: {len(all_chunks)}")
    return all_chunks
