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

            # ── Multimodal OCR Fallback ────────────────────────────────
            # If a page is just an image (like exported PPT slides), PyMuPDF 
            # won't find text. We render those pages to PNG and ask Gemini to OCR.
            ocr_llm = None
            pdf_doc = fitz.open(tmp_path)
            
            for i, page in enumerate(pages):
                # If page has very little text, assume it's image-based
                if len(page.page_content.strip()) <= 50:
                    if ocr_llm is None:
                        ocr_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
                        logger.info("Initializing Gemini OCR fallback for image-based pages...")
                    
                    try:
                        # Render physical PDF page to image
                        pix = pdf_doc[i].get_pixmap(dpi=150)
                        img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                        
                        msg = HumanMessage(content=[
                            {"type": "text", "text": "Extract all text from this slide verbatim. Summarize charts if any."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ])
                        
                        ocr_text = ocr_llm.invoke([msg]).content
                        page.page_content += f"\n{ocr_text}"
                        logger.info(f"Successfully OCR'd page {i+1} of {uploaded_file.name}")
                    except Exception as e:
                        logger.warning(f"OCR failed for {uploaded_file.name} page {i+1}: {e}")
            
            pdf_doc.close()
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
