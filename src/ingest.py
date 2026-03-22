"""
src/ingest.py — Document Processing Pipeline
=============================================
Handles PDF upload → text extraction → chunking → embedding → vector store.
"""

import os
import tempfile
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"
CHROMA_PERSIST_DIR = "chroma_db"

logger = logging.getLogger(__name__)


def process_document(uploaded_file) -> Chroma:
    """
    Process an uploaded PDF file and return a Chroma vector store.

    Steps:
        1. Save the uploaded file temporarily to disk.
        2. Load the PDF with PyPDFLoader.
        3. Split the text into chunks.
        4. Generate embeddings with Google Generative AI.
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

        # ── 5. Build Chroma vector store ─────────────────────────────
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

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
