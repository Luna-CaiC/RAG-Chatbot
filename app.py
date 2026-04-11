"""
RAG Chatbot — Streamlit Application
=====================================
Upload PDFs → ask questions → get citation-backed answers powered by
local HuggingFace embeddings and Google Gemini.
"""

import os
import re
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_community.vectorstores import Chroma

from src.ingest import process_documents
from src.history import ChatHistoryManager

load_dotenv(override=True)
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORS_BASE    = os.path.join("data", "session_vectors")

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — AI Document Assistant",
    page_icon="🤖",
    layout="centered",
)


# ── Cache the embedding model (loads once per server session) ─────────
@st.cache_resource(show_spinner="Loading embedding model (first time only)…")
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ── Persistent history manager ───────────────────────────────────────
history_mgr = ChatHistoryManager()


# ── FileWrapper: mimics Streamlit UploadedFile for on-disk files ─────
class FileWrapper:
    def __init__(self, name: str, content: bytes):
        self.name     = name
        self._content = content
        self.file_id  = f"disk_{name}"

    def read(self) -> bytes:
        return self._content


# ── Session state defaults ───────────────────────────────────────────
for key, default in {
    "chat_history":    [],
    "vector_store":    None,
    "session_id":      None,
    "doc_names":       [],
    "viewing_history": False,
    "pending_resume":  None,
    "last_file_ids":   None,
    "resume_no_create": False,
    "uploader_key":    0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Handle pending session resume (runs before sidebar) ──────────────
# FAST PATH: session_vectors/{id}/ exists → load Chroma from disk (seconds)
# READ-ONLY: no vectors → show history only, user can re-upload to resume
if st.session_state.pending_resume is not None:
    resume_id = st.session_state.pending_resume
    st.session_state.pending_resume = None  # Clear to avoid loops

    vector_dir = os.path.join(VECTORS_BASE, resume_id)

    if os.path.exists(vector_dir):
        # Fast path: load persisted Chroma — takes seconds, no re-embedding
        with st.spinner("🔄 Restoring session…"):
            try:
                embeddings = get_embeddings()
                vs = Chroma(
                    persist_directory=vector_dir,
                    embedding_function=embeddings,
                    collection_name="main",
                )
                st.session_state.vector_store    = vs
                st.session_state.viewing_history = False
                st.session_state.resume_no_create = True
            except Exception as e:
                st.warning(f"⚠️ Could not load session vectors: {e}")
                st.session_state.viewing_history = True
                st.session_state.vector_store    = None
    else:
        # No persisted vectors — instant read-only mode.
        # (Sessions before vector-persistence feature fall here.)
        st.session_state.viewing_history = True
        st.session_state.vector_store    = None

    st.rerun()


# ── LLM: Groq for chat, Gemini kept in ingest.py for OCR only ────────
# Groq free tier: no daily limit, fast (500+ tokens/s)
# Gemini free tier: 1500 RPD but shared with OCR → move chat to Groq
LLM = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.4,
)

SYSTEM_PROMPT = (
    "You are an expert document assistant. "
    "The user has uploaded the following document(s): {doc_names}.\n\n"
    "IMPORTANT CONTEXT NOTE:\n"
    "The context below is a SAMPLE of retrieved chunks from the documents — "
    "it is NOT the complete document. Large documents contain far more content "
    "than what is shown here. Do NOT imply that the examples you see are the "
    "only content in the document.\n\n"
    "INSTRUCTIONS:\n"
    "1. Read the retrieved context thoroughly before answering.\n"
    "2. For SUMMARY or OVERVIEW questions (e.g. 'what does this file contain?', "
    "'what is this document about?'):\n"
    "   - Describe the overall nature and scope of the document.\n"
    "   - Use phrases like 'for example', 'including but not limited to', "
    "'among other topics' — NOT 'it includes: [list]' which implies completeness.\n"
    "   - If the document clearly covers many topics, explicitly say so.\n"
    "   - Give a rich, multi-paragraph answer.\n"
    "3. For SPECIFIC questions, provide a complete and detailed answer ONLY using the provided context.\n"
    "4. If the provided context does NOT contain the answer to a specific question, you MUST "
    "explicitly state: 'I cannot answer this based on the provided documents.' Do NOT guess, "
    "and do NOT use your general knowledge to answer.\n"
    "5. Do NOT use any outside knowledge whatsoever. You are strictly restricted to the documents.\n"
    "6. Cite page numbers and source filenames when available.\n"
    "7. When the user refers to a document by filename, answer about its content.\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


# ────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── ✏️ New Chat button ───────────────────────────────────────────
    if st.button("✏️  New Chat", use_container_width=True, type="primary"):
        st.session_state.chat_history     = []
        st.session_state.vector_store     = None
        st.session_state.session_id       = None
        st.session_state.doc_names        = []
        st.session_state.viewing_history  = False
        st.session_state.last_file_ids    = None
        st.session_state.resume_no_create = False
        st.session_state.uploader_key    += 1   # clears the file uploader widget
        st.rerun()

    st.divider()

    # ── Upload section ───────────────────────────────────────────────
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more files (PDF, Word, or Image)",
        type=["pdf", "docx", "jpg", "jpeg", "png", "webp", "gif"],
        accept_multiple_files=True,
        help="Supported: PDF, Word (.docx), Images (.jpg, .jpeg, .png, .webp, .gif)",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        current_ids  = tuple(sorted(f.file_id for f in uploaded_files))
        previous_ids = st.session_state.last_file_ids

        if current_ids != previous_ids:
            if st.session_state.resume_no_create:
                # After resume: uploader still shows old files — sync IDs only
                st.session_state.last_file_ids    = current_ids
                st.session_state.resume_no_create  = False
            else:
                with st.spinner("Processing documents…"):
                    try:
                        # Read all bytes before any stream is consumed
                        file_data = [(f.name, f.read()) for f in uploaded_files]

                        st.session_state.vector_store    = None
                        st.session_state.chat_history    = []
                        st.session_state.viewing_history = False

                        session_id  = uuid.uuid4().hex[:8]
                        persist_dir = os.path.join(VECTORS_BASE, session_id)
                        embeddings  = get_embeddings()
                        wrappers    = [FileWrapper(n, d) for n, d in file_data]

                        vector_store = process_documents(
                            wrappers,
                            persist_directory=persist_dir,
                            embeddings=embeddings,
                        )
                        doc_names = [name for name, _ in file_data]

                        st.session_state.vector_store  = vector_store
                        st.session_state.last_file_ids = current_ids
                        st.session_state.doc_names     = doc_names
                        st.session_state.session_id    = session_id

                        history_mgr.create_session(session_id, doc_names)
                        history_mgr.save_session_files(session_id, file_data)

                        st.success(f"✅ {len(doc_names)} file(s) processed!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    if st.session_state.vector_store is not None:
        names = ", ".join(st.session_state.doc_names)
        st.info(f"📚 Active: **{names}**")

    st.divider()

    # ── Clear chat button ────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if st.session_state.session_id:
            history_mgr.delete_session(st.session_state.session_id)

        st.session_state.chat_history     = []
        st.session_state.viewing_history  = False
        st.session_state.vector_store     = None
        st.session_state.session_id       = None
        st.session_state.doc_names        = []
        st.session_state.last_file_ids    = None
        st.session_state.resume_no_create = False
        st.rerun()

    st.divider()

    # ── Past sessions list ───────────────────────────────────────────
    st.header("📜 Past Sessions")
    sessions  = history_mgr.load_sessions()
    active_id = st.session_state.session_id

    if not sessions:
        st.caption("No saved sessions yet.")
    else:
        for s in sessions[:20]:
            docs_label = ", ".join(s["documents"]) or "Unknown"
            ts         = s["timestamp"][:16].replace("T", " ")
            msg_count  = s["message_count"]
            label      = f"📝 {docs_label} ({msg_count} msgs)\n{ts}"
            btn_type   = "primary" if s["id"] == active_id else "secondary"

            if st.button(label, key=f"hist_{s['id']}",
                         use_container_width=True, type=btn_type):
                full_session = history_mgr.get_session(s["id"])
                if full_session:
                    st.session_state.chat_history = full_session["messages"]
                    st.session_state.doc_names    = full_session["documents"]
                    st.session_state.session_id   = s["id"]
                    st.session_state.pending_resume  = s["id"]
                    st.session_state.vector_store    = None
                    st.session_state.viewing_history = False
                    st.rerun()


# ────────────────────────────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.markdown(
    "**AI-powered document assistant** — Upload files in the sidebar, "
    "then ask questions. Answers are grounded in your documents with "
    "source citations. _Powered by HuggingFace Embeddings & Google Gemini._"
)

# Read-only banner
if st.session_state.viewing_history:
    st.warning(
        "📜 This is a read-only view of a past session "
        "(documents not available for this session). "
        "Upload files in the sidebar to continue chatting."
    )

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input — ALWAYS enabled (no disabled state)
user_query = st.chat_input("Ask a question about your document(s)…")

if user_query:
    st.session_state.viewing_history = False

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            # No documents uploaded yet — friendly prompt
            msg = (
                "📂 No documents loaded. "
                "Please upload a file in the sidebar first, "
                "then ask your question."
            )
            st.info(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("Thinking…"):
                try:
                    # ── MMR Retriever ────────────────────────────────────
                    num_docs  = len(st.session_state.doc_names)
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k":           max(8, num_docs * 6),
                            "fetch_k":     max(30, num_docs * 20),
                            "lambda_mult": 0.65,
                        },
                    )

                    # ── Rule-based query decomposition ───────────────────
                    def decompose_query(q: str) -> list[str]:
                        parts = re.split(r'[?？。]|\band\b|\bAND\b', q)
                        parts = [p.strip() for p in parts if len(p.strip()) > 5]
                        return parts if parts else [q]

                    sub_queries = decompose_query(user_query)
                    all_queries = list(dict.fromkeys([user_query] + sub_queries))

                    all_docs = []
                    for q in all_queries:
                        all_docs.extend(retriever.invoke(q))
                    unique_docs = list(
                        {doc.page_content: doc for doc in all_docs}.values()
                    )

                    # ── Chain ────────────────────────────────────────────
                    document_prompt = PromptTemplate.from_template(
                        "Source: {source}\nContent: {page_content}"
                    )
                    document_chain = create_stuff_documents_chain(
                        LLM, PROMPT, document_prompt=document_prompt
                    )
                    doc_names_str = ", ".join(st.session_state.doc_names) or "Unknown"
                    answer = document_chain.invoke({
                        "input":     user_query,
                        "doc_names": doc_names_str,
                        "context":   unique_docs,
                    })

                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    err = f"❌ Error generating response: {e}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )

    # Auto-save
    if st.session_state.session_id:
        history_mgr.save_messages(
            st.session_state.session_id,
            st.session_state.chat_history,
        )
