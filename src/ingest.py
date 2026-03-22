"""
src/ingest.py — Document Processing Pipeline
=============================================
Handles PDF upload → text extraction → chunking → embedding → vector store.
"""

import os
import shutil
import time
import tempfile
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_PERSIST_DIR = "chroma_db"
BATCH_SIZE = 10          # chunks per batch (~75 req/min, under 100/min limit)
BATCH_DELAY_SEC = 8      # seconds between batches
MAX_RETRIES = 10         # max retries on rate-limit errors
RETRY_DELAY_SEC = 65     # fixed retry wait (quota resets every 60s)

logger = logging.getLogger(__name__)


def _embed_batch_with_retry(vector_store, batch, batch_num, total):
    """Embed a batch of documents, retrying on 429 rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            vector_store.add_documents(batch)
            return
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                logger.warning(
                    "Rate limited on batch %d. Waiting %ds for quota reset (attempt %d/%d)...",
                    batch_num, RETRY_DELAY_SEC, attempt + 1, MAX_RETRIES,
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                raise
    raise RuntimeError(f"Batch {batch_num}/{total} failed after {MAX_RETRIES} retries.")


def process_document(uploaded_file) -> Chroma:
    """
    Process an uploaded PDF file and return a Chroma vector store.

    Steps:
        1. Save the uploaded file temporarily to disk.
        2. Load the PDF with PyPDFLoader.
        3. Split the text into chunks.
        4. Generate embeddings with Google Generative AI (in batches).
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

        # ── 4. Create embeddings ─────────────────────────────────────
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        # ── 5. Build Chroma vector store (in small batches) ──────────
        #    Clean any stale chroma_db first to avoid readonly errors.
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)

        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        # First batch — creates the store
        first_batch = chunks[:BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                vector_store = Chroma.from_documents(
                    documents=first_batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR,
                )
                break
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    logger.warning(
                        "Rate limited on first batch. Waiting %ds for quota reset (attempt %d/%d)...",
                        RETRY_DELAY_SEC, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    raise
        else:
            raise RuntimeError("First batch failed after max retries.")

        logger.info("Batch 1/%d embedded.", total_batches)

        # Remaining batches
        for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1

            time.sleep(BATCH_DELAY_SEC)  # respect rate limit
            _embed_batch_with_retry(vector_store, batch, batch_num, total_batches)
            logger.info("Batch %d/%d embedded.", batch_num, total_batches)

        logger.info(
            "Vector store created with %d vectors in '%s'.",
            len(chunks),
            CHROMA_PERSIST_DIR,
        )

        return vector_store

    except (ValueError, RuntimeError):
        # Re-raise known errors as-is
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
