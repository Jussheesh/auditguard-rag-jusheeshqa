import io
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from pypdf import PdfReader
import pypdfium2 as pdfium
from app.config import PREVIEWS_DIR
from app.classifier import classify_document

def get_or_render_page_image(doc_prefix: str, page_num: int) -> Path | None:
    """
    On-demand single-page renderer for large documents (e.g. 300-page manuals).
    Renders just the requested page in ~30ms and caches it to disk in compressed WebP format (80% quality).
    """
    file_name = f"{doc_prefix}_page_{page_num}.webp"
    file_path = PREVIEWS_DIR / file_name
    if file_path.exists():
        return file_path

    png_fallback = PREVIEWS_DIR / f"{doc_prefix}_page_{page_num}.png"
    if png_fallback.exists():
        return png_fallback

    from app.config import DOCS_DIR
    pdf_path = DOCS_DIR / f"{doc_prefix}.pdf"
    if not pdf_path.exists():
        return None

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if 1 <= page_num <= len(pdf):
                page = pdf[page_num - 1]
                image = page.render(scale=1.25).to_pil()
                image.save(file_path, "WEBP", quality=80)
                return file_path
        finally:
            pdf.close()
    except Exception as e:
        print(f"Error on-demand rendering page {page_num}: {e}")
    return None

def render_page_previews(pdf_bytes: bytes, doc_prefix: str) -> Dict[int, str]:
    """
    Smart Page Preview Renderer:
    - Pre-renders up to the first 3 pages at fast 1.2x scale for instant UI thumbs,
      and routes the rest to the on-demand /api/preview endpoint (sub-30ms lazy render).
    """
    from app.config import DOCS_DIR
    pdf_save_path = DOCS_DIR / f"{doc_prefix}.pdf"
    try:
        pdf_save_path.write_bytes(pdf_bytes)
    except Exception as e:
        print(f"Warning saving master PDF: {e}")

    page_map = {}
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            total_pages = len(pdf)
            pages_to_prerender = min(total_pages, 3)

            for idx in range(total_pages):
                page_num = idx + 1
                file_name = f"{doc_prefix}_page_{page_num}.webp"
                file_path = PREVIEWS_DIR / file_name

                if idx < pages_to_prerender:
                    page = pdf[idx]
                    image = page.render(scale=1.2).to_pil()
                    image.save(file_path, "WEBP", quality=80)
                    page_map[page_num] = f"/static/previews/{file_name}"
                else:
                    page_map[page_num] = f"/api/preview/{doc_prefix}/{page_num}"
        finally:
            pdf.close()
    except Exception as e:
        print(f"Warning: Could not render page preview with pypdfium2: {e}")
    return page_map

def clean_text_to_markdown(raw_text: str) -> str:
    """
    Converts raw extracted document text into clean, structured Markdown
    for fast AI comprehension, retaining headings, lists, and tables.
    """
    lines = raw_text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if re.match(r'^\d+\.\s+[A-Z\s]{3,}', stripped):
            cleaned.append(f"## {stripped}")
        elif re.match(r'^\d+\.\d+\s+', stripped):
            cleaned.append(f"### {stripped}")
        elif re.match(r'^[•\*\-]\s+', stripped):
            cleaned.append(f"- {stripped.lstrip('•*- ')}")
        else:
            cleaned.append(stripped)
    return "\n".join(cleaned)

def ocr_extract_page_image(page_image: Any) -> Optional[str]:
    """
    Attempts optical character recognition (OCR) on a rendered page image using pytesseract.
    """
    try:
        import pytesseract
        text = pytesseract.image_to_string(page_image)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return None

def extract_pages_from_pdf(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    pages = []
    total_pdf_pages = 0
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pdf_pages = len(reader.pages)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                md_text = clean_text_to_markdown(text)
                pages.append({
                    "filename": filename,
                    "page": idx + 1,
                    "text": md_text
                })
    except Exception as e:
        print(f"pypdf extraction notice: {e}")

    if not pages or len(pages) < total_pdf_pages:
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
            try:
                pdfium_total = len(pdf)
                if total_pdf_pages == 0:
                    total_pdf_pages = pdfium_total
                for idx in range(pdfium_total):
                    page_num = idx + 1
                    if not any(p["page"] == page_num for p in pages):
                        page = pdf[idx]
                        tp = None
                        t = ""
                        try:
                            tp = page.get_textpage()
                            t = tp.get_text_range() or ""
                        except Exception:
                            pass
                        finally:
                            if tp and hasattr(tp, "close"):
                                try:
                                    tp.close()
                                except Exception:
                                    pass
                            if hasattr(page, "close"):
                                try:
                                    page.close()
                                except Exception:
                                    pass
                        if t.strip():
                            pages.append({
                                "filename": filename,
                                "page": page_num,
                                "text": clean_text_to_markdown(t)
                            })
            finally:
                pdf.close()
        except Exception as e:
            print(f"pypdfium2 extraction notice: {e}")

    if total_pdf_pages > 0 and len(pages) < total_pdf_pages:
        existing_page_nums = {p["page"] for p in pages}
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
            try:
                for idx in range(total_pdf_pages):
                    page_num = idx + 1
                    if page_num not in existing_page_nums:
                        page = pdf[idx]
                        ocr_text = None
                        try:
                            pil_img = page.render(scale=1.5).to_pil()
                            ocr_text = ocr_extract_page_image(pil_img)
                        except Exception:
                            ocr_text = None
                        finally:
                            if hasattr(page, "close"):
                                try:
                                    page.close()
                                except Exception:
                                    pass

                        if ocr_text:
                            pages.append({
                                "filename": filename,
                                "page": page_num,
                                "text": clean_text_to_markdown(ocr_text),
                                "is_scanned": True,
                                "is_ocr": True
                            })
                        else:
                            pages.append({
                                "filename": filename,
                                "page": page_num,
                                "text": f"Page {page_num} of {filename} (Visual / Scanned Content - Multimodal Image Attached)",
                                "is_scanned": True,
                                "is_ocr": False
                            })
            finally:
                pdf.close()
        except Exception:
            for page_num in range(1, total_pdf_pages + 1):
                if page_num not in existing_page_nums:
                    pages.append({
                        "filename": filename,
                        "page": page_num,
                        "text": f"Page {page_num} of {filename} (Visual / Scanned Content)",
                        "is_scanned": True,
                        "is_ocr": False
                    })

    pages.sort(key=lambda x: x["page"])
    return pages

def extract_pages_from_text(raw_text: str, filename: str) -> List[Dict[str, Any]]:
    """Handles .txt, .md, .json, .py, .yaml, .xml, .log, and code files."""
    ext = Path(filename).suffix.lower()
    if ext == ".json":
        try:
            parsed = json.loads(raw_text)
            raw_text = f"## JSON Data ({filename})\n```json\n" + json.dumps(parsed, indent=2) + "\n```"
        except Exception:
            pass

    chunks = [c.strip() for c in raw_text.split("\n\n\n") if c.strip()]
    if not chunks:
        # Split large text files into ~3000-char logical pages
        chunks = [raw_text[i:i+3000] for i in range(0, max(1, len(raw_text)), 3000)]
    pages = []
    for idx, c in enumerate(chunks, 1):
        pages.append({
            "filename": filename,
            "page": idx,
            "text": c if c.strip() else f"Document {filename} (Page {idx})"
        })
    return pages

def extract_pages_from_docx(docx_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract sections, headings, paragraphs, and tables from Word documents (.docx)."""
    import docx
    doc = docx.Document(io.BytesIO(docx_bytes))
    sections = []
    current_section = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if p.style.name.startswith("Heading") or re.match(r"^\d+\.", t):
            if current_section:
                sections.append("\n\n".join(current_section))
                current_section = []
            current_section.append(f"## {t}")
        else:
            current_section.append(t)

    for table in doc.tables:
        rows_txt = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            rows_txt.append("| " + " | ".join(cells) + " |")
        if rows_txt:
            col_count = len(table.columns)
            rows_txt.insert(1, "| " + " | ".join(["---"] * max(1, col_count)) + " |")
            table_md = "\n".join(rows_txt)
            current_section.append(f"### Table Data ({filename}):\n{table_md}")

    if current_section:
        sections.append("\n\n".join(current_section))

    if not sections:
        sections = [f"Word Document ({filename})"]

    pages = []
    for idx, sec in enumerate(sections, 1):
        pages.append({
            "filename": filename,
            "page": idx,
            "text": clean_text_to_markdown(sec)
        })
    return pages

def extract_pages_from_pptx(pptx_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract slide-by-slide text, titles, speaker notes, and tables from PowerPoint (.pptx)."""
    from pptx import Presentation
    prs = Presentation(io.BytesIO(pptx_bytes))
    pages = []
    for idx, slide in enumerate(prs.slides, 1):
        slide_parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                for p in shape.text_frame.paragraphs:
                    t = p.text.strip()
                    if t:
                        slide_parts.append(t)
            elif getattr(shape, "has_table", False):
                table_rows = []
                for row in shape.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    table_rows.append("| " + " | ".join(cells) + " |")
                if table_rows:
                    col_count = len(shape.table.columns)
                    table_rows.insert(1, "| " + " | ".join(["---"] * max(1, col_count)) + " |")
                    slide_parts.append("### Slide Table:\n" + "\n".join(table_rows))

        # Include speaker notes if present
        if getattr(slide, "has_notes_slide", False) and slide.notes_slide.notes_text_frame:
            notes_txt = slide.notes_slide.notes_text_frame.text.strip()
            if notes_txt:
                slide_parts.append(f"**Speaker Notes (Slide {idx}):** {notes_txt}")

        slide_content = "\n\n".join(slide_parts).strip()
        if not slide_content:
            slide_content = f"Slide {idx} of {filename} (Visual Graphic / Title Slide)"

        pages.append({
            "filename": filename,
            "page": idx,
            "text": clean_text_to_markdown(f"## Slide {idx} ({filename})\n\n" + slide_content)
        })

    if not pages:
        pages.append({"filename": filename, "page": 1, "text": f"Presentation {filename}"})
    return pages

def extract_pages_from_excel(excel_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extracts all worksheets from Excel (.xlsx / .xlsm) into structured Markdown tables (1 page per sheet)."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), data_only=True)
    pages = []
    for idx, sheet_name in enumerate(wb.sheetnames, 1):
        ws = wb[sheet_name]
        rows_md = []
        text_summary = []
        for r_idx, row in enumerate(ws.iter_rows(values_only=True)):
            if any(cell is not None and str(cell).strip() != "" for cell in row):
                clean_cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
                row_line = "| " + " | ".join(clean_cells) + " |"
                rows_md.append(row_line)
                # Also create readable prose sentences per row so BM25 & dense search match easily
                non_empty = [c for c in clean_cells if c]
                if non_empty:
                    text_summary.append(f"Row {r_idx + 1} ({sheet_name}): " + " — ".join(non_empty) + ".")

        if rows_md:
            col_count = max(1, rows_md[0].count("|") - 1)
            rows_md.insert(1, "| " + " | ".join(["---"] * col_count) + " |")
            sheet_text = (
                f"## Excel Worksheet: {sheet_name} (File: {filename})\n\n"
                + "\n".join(rows_md)
                + "\n\n### Sheet Row Details:\n"
                + "\n".join(text_summary)
            )
            pages.append({
                "filename": filename,
                "page": idx,
                "text": sheet_text
            })

    if not pages:
        pages.append({"filename": filename, "page": 1, "text": f"Excel Workbook {filename} (No populated rows)"})
    return pages

def extract_pages_from_csv(csv_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extracts CSV data into a structured Markdown table + searchable row statements."""
    try:
        raw_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = csv_bytes.decode("latin-1", errors="replace")

    reader = list(csv.reader(io.StringIO(raw_text)))
    rows_md = []
    row_sentences = []
    headers = reader[0] if reader else []

    for r_idx, row in enumerate(reader):
        if any(c.strip() for c in row):
            clean_cells = [c.strip().replace("\n", " ") for c in row]
            rows_md.append("| " + " | ".join(clean_cells) + " |")
            if r_idx > 0 and headers:
                pairs = [f"{headers[i]}: {clean_cells[i]}" for i in range(min(len(headers), len(clean_cells))) if clean_cells[i]]
                if pairs:
                    row_sentences.append(f"Record {r_idx} ({filename}): " + ", ".join(pairs) + ".")
            else:
                row_sentences.append(" — ".join([c for c in clean_cells if c]) + ".")

    if rows_md:
        col_count = max(1, len(reader[0]))
        rows_md.insert(1, "| " + " | ".join(["---"] * col_count) + " |")

    full_md = (
        f"## CSV Dataset: {filename}\n\n"
        + "\n".join(rows_md)
        + "\n\n### Data Records:\n"
        + "\n".join(row_sentences)
    )
    return [{"filename": filename, "page": 1, "text": full_md}]

def extract_pages_from_image(image_bytes: bytes, filename: str, doc_id: str) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """
    Handles standalone images/charts/diagrams (.png, .jpg, .jpeg, .webp, .bmp, .gif).
    Saves a compressed WebP preview so Gemma 4 Vision receives the image directly during generation,
    and runs optional OCR if pytesseract is present.
    """
    file_name = f"{doc_id}_page_1.webp"
    file_path = PREVIEWS_DIR / file_name
    img_url = f"/static/previews/{file_name}"
    ocr_text = None

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.thumbnail((1400, 1400))
        pil_img.save(file_path, "WEBP", quality=82)
        ocr_text = ocr_extract_page_image(pil_img)
    except Exception as e:
        print(f"Image preview save notice: {e}")

    body_text = (
        f"## Visual Image / Chart Asset: {filename}\n\n"
        f"{ocr_text if ocr_text else f'Visual diagram, chart, or scanned image from {filename}. Multimodal image attached for inspection.'}"
    )
    pages = [{
        "filename": filename,
        "page": 1,
        "text": body_text,
        "is_scanned": True,
        "is_ocr": bool(ocr_text)
    }]
    return pages, {1: img_url}

def hierarchical_chunk(pages: List[Dict[str, Any]], page_images: Dict[int, str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Implements Small-to-Big (Parent-Child) chunking.
    Child sentences are indexed for precise vector matching.
    Parent paragraphs are passed to Gemma 4 for rich, coherent context.
    """
    page_images = page_images or {}
    chunks = []
    chunk_id = 0

    combined_header = " ".join([p["text"][:1000] for p in pages[:2]])
    classification = classify_document(combined_header)

    for page in pages:
        page_num = page["page"]
        img_url = page_images.get(page_num)

        paragraphs = [p.strip() for p in page["text"].split("\n\n") if len(p.strip()) > 25]
        if not paragraphs:
            paragraphs = [page["text"].strip()]

        for p_idx, para in enumerate(paragraphs):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', para) if len(s.strip()) > 12]
            if not sentences:
                sentences = [para]

            for sent in sentences:
                chunks.append({
                    "id": chunk_id,
                    "filename": page["filename"],
                    "page": page_num,
                    "parent_text": para,
                    "child_text": sent,
                    "image_url": img_url,
                    "has_visual": bool(img_url)
                })
                chunk_id += 1

    return chunks, classification