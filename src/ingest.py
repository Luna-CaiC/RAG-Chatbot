"""
src/ingest.py — Document Processing Pipeline
=============================================
Handles PDF upload → text extraction → chunking → embedding → vector store.
Uses a local HuggingFace embedding model (no API rate limits).
"""

import os
import logging
import tempfile
import fitz
import base64

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ── Constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # runs locally, no API calls

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

            # Switch from PyPDFLoader to PyMuPDFLoader for better parsing
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()

            # ── Multimodal OCR Fallback (Batched for Rate Limits) ──────
            # Only trigger OCR if the ENTIRE document seems to be image-based 
            # (e.g. average < 50 chars per page). This prevents wasting API calls 
            # on blank/diagram pages inside normal text PDFs.
            total_text_len = sum(len(p.page_content.strip()) for p in pages)
            avg_chars_per_page = total_text_len / max(1, len(pages))
            
            if avg_chars_per_page < 50:
                logger.info("Document '%s' seems to be image-based (avg %d chars/page). Triggering OCR...", uploaded_file.name, avg_chars_per_page)
                pdf_doc = fitz.open(tmp_path)
                ocr_images = []
                ocr_page_indices = []
                
                for i, page in enumerate(pages):
                    # For image-based docs, OCR every page that has little text
                    if len(page.page_content.strip()) <= 50:
                        ocr_page_indices.append(i)
                        pix = pdf_doc[i].get_pixmap(dpi=100)
                        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                        ocr_images.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                
                if ocr_images:
                    ocr_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
                    
                    batch_size = 15
                    for start_idx in range(0, len(ocr_images), batch_size):
                        batch_imgs = ocr_images[start_idx:start_idx+batch_size]
                        batch_indices = ocr_page_indices[start_idx:start_idx+batch_size]
                        
                        msg = HumanMessage(content=[
                            {"type": "text", "text": "Extract all text from these slide images verbatim. Summarize charts if any. Prefix each slide's text with 'Slide: '."}
                        ] + batch_imgs)
                        
                        try:
                            ocr_text = ocr_llm.invoke([msg]).content
                            first_page = pages[batch_indices[0]]
                            first_page.page_content += f"\n\n[OCR Text from Slides]\n{ocr_text}"
                            logger.info("Successfully OCR'd batch of %d pages for %s", len(batch_imgs), uploaded_file.name)
                        except Exception as e:
                            logger.warning("Batch OCR failed for %s: %e", uploaded_file.name, e)
                
                pdf_doc.close()
            else:
                logger.info("Document '%s' is a normal text PDF (avg %d chars/page). Skipping OCR.", uploaded_file.name, avg_chars_per_page)
            # ───────────────────────────────────────────────────────────

            # Restore original filename in metadata AND inject it into the text
            # so the dense vector search can actually match the filename!
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.page_content = f"[Source Document: {uploaded_file.name}]\n{page.page_content}"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            logger.info("Loaded %d chunks from '%s'.", len(chunks), uploaded_file.name)

        except Exception as exc:
            logger.error("Failed to process '%s': %s", uploaded_file.name, exc)

        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_chunks:
        raise RuntimeError("No text could be extracted from any of the uploaded files.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
    )

    logger.info("Combined vector store created with %d vectors from %d file(s).",
                len(all_chunks), len(uploaded_files))
    return vector_store
