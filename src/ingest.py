"""
src/ingest.py — Document Processing Pipeline
=============================================
Handles PDF upload → text extraction → chunking → embedding → vector store.
Uses a local HuggingFace embedding model (no API rate limits).
"""

import os
import time
import uuid
import logging
import tempfile
import fitz
import base64

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# ── Constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # runs locally, no API calls
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

logger = logging.getLogger(__name__)


def process_documents(uploaded_files: list) -> Chroma:
    """
    Process multiple uploaded PDF files into a single combined vector store.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Chroma: A combined vector store with embeddings from all documents.
    """
    if not uploaded_files:
        raise ValueError("No files were uploaded.")

    all_chunks = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        if not file_bytes:
            logger.warning("Skipping empty file: %s", uploaded_file.name)
            continue

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()

            # ── Multimodal OCR Fallback (Image PDFs) ───────────────────
            # Only triggers if avg chars/page < 50 (image-based PDF).
            # Text PDFs skip OCR entirely.
            total_text_len = sum(len(p.page_content.strip()) for p in pages)
            avg_chars_per_page = total_text_len / max(1, len(pages))

            if avg_chars_per_page < 50:
                logger.info("'%s' is image-based (avg %d chars/page). Running OCR...",
                            uploaded_file.name, avg_chars_per_page)
                pdf_doc = fitz.open(tmp_path)
                ocr_images = []
                ocr_page_indices = []

                for i, page in enumerate(pages):
                    if len(page.page_content.strip()) <= 50:
                        ocr_page_indices.append(i)
                        pix = pdf_doc[i].get_pixmap(dpi=100)
                        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                        ocr_images.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })

                if ocr_images:
                    ocr_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)
                    batch_size = 15
                    total_batches = (len(ocr_images) + batch_size - 1) // batch_size

                    for batch_num, start_idx in enumerate(range(0, len(ocr_images), batch_size)):
                        batch_imgs = ocr_images[start_idx:start_idx + batch_size]
                        batch_indices = ocr_page_indices[start_idx:start_idx + batch_size]

                        logger.info("OCR batch %d/%d for '%s'...",
                                    batch_num + 1, total_batches, uploaded_file.name)

                        msg = HumanMessage(content=[
                            {"type": "text", "text": (
                                "Extract all text from these slide images verbatim. "
                                "Prefix each slide's text with 'Slide N:' where N is its "
                                "order (1, 2, 3...). Summarize charts and diagrams as text."
                            )}
                        ] + batch_imgs)

                        # Retry up to 3x with backoff on rate limit (429)
                        ocr_text = None
                        for attempt in range(3):
                            try:
                                ocr_text = ocr_llm.invoke([msg]).content
                                break
                            except Exception as e:
                                if "429" in str(e) and attempt < 2:
                                    wait = 35 * (attempt + 1)
                                    logger.warning("OCR rate limited, retrying in %ds...", wait)
                                    time.sleep(wait)
                                else:
                                    logger.warning("OCR failed for '%s': %s",
                                                   uploaded_file.name, e)
                                    break

                        if ocr_text:
                            # Distribute text: each page gets its own slide content
                            # (not all dumped into page 0 only)
                            slide_parts = ocr_text.split("Slide ")
                            slide_parts = [s.strip() for s in slide_parts if s.strip()]
                            for j, idx in enumerate(batch_indices):
                                if j < len(slide_parts):
                                    pages[idx].page_content = f"Slide {slide_parts[j]}"
                                else:
                                    pages[idx].page_content = ocr_text  # fallback
                            logger.info("OCR succeeded for batch %d of '%s'",
                                        batch_num + 1, uploaded_file.name)
                        else:
                            logger.warning("OCR failed for batch %d of '%s' after retries.",
                                           batch_num + 1, uploaded_file.name)

                pdf_doc.close()
            else:
                logger.info("'%s' is a text PDF (avg %d chars/page). Skipping OCR.",
                            uploaded_file.name, avg_chars_per_page)
            # ─────────────────────────────────────────────────────────────

            # Set source metadata for all pages
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = text_splitter.split_documents(pages)

            # ── CRITICAL: inject source label into EVERY chunk ──────────
            # Must happen AFTER splitting. Without this, only the first
            # chunk of each page carries the filename — subsequent chunks
            # are anonymous and retrieval can't identify their source.
            source_label = f"[Source Document: {uploaded_file.name}]"
            valid_chunks = []
            for chunk in chunks:
                content = chunk.page_content.strip()
                if len(content) < 20:
                    continue  # discard near-empty chunks (blank OCR pages)
                if not content.startswith("[Source Document:"):
                    chunk.page_content = f"{source_label}\n{content}"
                valid_chunks.append(chunk)
            # ─────────────────────────────────────────────────────────────

            all_chunks.extend(valid_chunks)
            logger.info("Loaded %d chunks from '%s'.", len(valid_chunks), uploaded_file.name)

        except Exception as exc:
            logger.error("Failed to process '%s': %s", uploaded_file.name, exc)

        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_chunks:
        raise RuntimeError("No text could be extracted from any of the uploaded files.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Use a UNIQUE collection name per session ────────────────────────
    # CRITICAL FIX: Without this, all sessions share the same default
    # "langchain" collection inside the same Streamlit process — data
    # from previous uploads (e.g. slide 18) bleeds into new sessions.
    collection_name = f"rag_session_{uuid.uuid4().hex[:12]}"

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    logger.info("Vector store '%s' created with %d vectors from %d file(s).",
                collection_name, len(all_chunks), len(uploaded_files))
    return vector_store
