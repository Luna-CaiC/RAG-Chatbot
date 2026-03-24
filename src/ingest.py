"""
src/ingest.py — Document Processing Pipeline
=============================================
Handles PDF upload → text extraction → chunking → embedding → vector store.
Uses a local HuggingFace embedding model (no API rate limits).
"""

import os
import shutil
import tempfile
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # runs locally, no API calls

logger = logging.getLogger(__name__)


def process_document(uploaded_file) -> Chroma:
    """
    Process an uploaded PDF file and return a Chroma vector store.

    Steps:
        1. Save the uploaded file temporarily to disk.
        2. Load the PDF with PyPDFLoader.
        3. Split the text into chunks.
        4. Generate embeddings locally with HuggingFace.
        5. Store vectors in a local Chroma vector store.

    Args:
        uploaded_file: A Streamlit UploadedFile object (has .name and .read()).

    Returns:
        Chroma: A Chroma vector store populated with the document embeddings.

    Raises:
        ValueError: If the uploaded file is None or empty.
        RuntimeError: If any step in the pipeline fails.
    """

    # ── 0. Validate input ────────────────────────────────────────────
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    logger.info("Processing document: %s", uploaded_file.name)

    try:
        # ── 1. Save to a temporary file ──────────────────────────────
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        logger.info("Saved temp file: %s", tmp_path)

        # ── 2. Load PDF pages ────────────────────────────────────────
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            raise RuntimeError("PyPDFLoader returned no pages. Is the PDF valid?")

        logger.info("Loaded %d page(s) from PDF.", len(pages))

        # ── 3. Split into chunks ─────────────────────────────────────
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(pages)

        if not chunks:
            raise RuntimeError("Text splitting produced no chunks.")

        logger.info("Split into %d chunk(s).", len(chunks))

        # ── 4. Create embeddings (local — no API calls) ──────────────
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # ── 5. Build Chroma vector store (In-Memory) ─────────────────
        #    Using an in-memory store avoids SQLite file-locking issues
        #    (e.g., "readonly database") when users re-upload documents.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        logger.info(
            "Vector store created in-memory with %d vectors.",
            len(chunks)
        )

        return vector_store

    except (ValueError, RuntimeError):
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Document processing failed: {exc}"
        ) from exc

    finally:
        # ── Cleanup temp file ────────────────────────────────────────
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Removed temp file: %s", tmp_path)


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

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

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
