"""
RAG Chatbot — Streamlit Application
=====================================
Upload PDFs → ask questions → get citation-backed answers powered by
local HuggingFace embeddings and Google Gemini.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from src.ingest import process_documents
from src.history import ChatHistoryManager

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — AI Document Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Persistent history manager ───────────────────────────────────────
history_mgr = ChatHistoryManager()

# ── Session state defaults ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = False

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    # ── Upload section ───────────────────────────────────────────────
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload technical documents or research papers to query.",
    )

    if uploaded_files:
        # Build a fingerprint from all file IDs to detect changes
        current_ids = tuple(sorted(f.file_id for f in uploaded_files))
        previous_ids = st.session_state.get("last_file_ids", None)

        if current_ids != previous_ids:
            with st.spinner("Processing documents..."):
                try:
                    # Clear old state
                    st.session_state.vector_store = None
                    st.session_state.chat_history = []
                    st.session_state.viewing_history = False

                    vector_store = process_documents(uploaded_files)
                    doc_names = [f.name for f in uploaded_files]

                    st.session_state.vector_store = vector_store
                    st.session_state.last_file_ids = current_ids
                    st.session_state.doc_names = doc_names

                    # Create a new persistent session
                    session_id = history_mgr.create_session(doc_names)
                    st.session_state.session_id = session_id

                    st.success(f"✅ {len(doc_names)} file(s) processed!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if st.session_state.vector_store is not None:
        names = ", ".join(st.session_state.doc_names)
        st.info(f"📚 Active: **{names}**")

    st.divider()

    # ── Clear chat button ────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.viewing_history = False
        if st.session_state.session_id:
            history_mgr.save_messages(st.session_state.session_id, [])
        st.rerun()

    st.divider()

    # ── Past sessions ────────────────────────────────────────────────
    st.header("📜 Past Sessions")
    sessions = history_mgr.load_sessions()

    if not sessions:
        st.caption("No saved sessions yet.")
    else:
        for s in sessions[:10]:  # Show latest 10
            docs_label = ", ".join(s["documents"]) or "Unknown"
            ts = s["timestamp"][:16].replace("T", " ")  # "2026-03-24 00:37"
            msg_count = s["message_count"]
            label = f"📝 {docs_label} ({msg_count} msgs)\n{ts}"

            if st.button(label, key=f"hist_{s['id']}", use_container_width=True):
                full_session = history_mgr.get_session(s["id"])
                if full_session:
                    st.session_state.chat_history = full_session["messages"]
                    st.session_state.doc_names = full_session["documents"]
                    st.session_state.viewing_history = True
                    st.session_state.vector_store = None  # Vectors not available
                    st.session_state.session_id = s["id"]
                    st.rerun()

# ── LLM & chain setup ───────────────────────────────────────────────
LLM = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.3,
)

SYSTEM_PROMPT = (
    "You are an expert document assistant. "
    "The user has uploaded the following document(s): {doc_names}. "
    "Below is retrieved context from these documents.\n\n"
    "INSTRUCTIONS:\n"
    "1. Read the context thoroughly before answering.\n"
    "2. Base your answer on the provided context. "
    "If the context contains relevant information, use it to give a "
    "complete and detailed answer — do NOT say 'the document does not provide' "
    "when the context clearly contains the information.\n"
    "3. Only say 'I cannot answer this based on the provided document' if the "
    "context truly has NO relevant information at all.\n"
    "4. Do NOT use outside knowledge that contradicts or goes beyond the context.\n"
    "5. Cite page numbers and relevant sections when available.\n"
    "6. When the user refers to a document by filename (e.g. 'slide 18'), "
    "understand they mean the uploaded file and answer about its content.\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# ── Main chat interface ─────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.markdown(
    "**AI-powered document assistant** — Upload PDFs in the sidebar, "
    "then ask questions. Answers are grounded in your documents with "
    "source citations. _Powered by HuggingFace Embeddings & Google Gemini._"
)

# Show banner when viewing a past session
if st.session_state.viewing_history:
    st.warning(
        "📜 Viewing a past session. To ask new questions, "
        "upload the PDF(s) again in the sidebar."
    )

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input — disabled when no vector store (including history-only mode)
user_query = st.chat_input(
    "Ask a question about your document(s)...",
    disabled=st.session_state.vector_store is None,
)

if user_query:
    # Exit history-viewing mode when asking new questions
    st.session_state.viewing_history = False

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 8}
                )

                # ── Advanced RAG: Query Decomposition ──────────────────────
                # Break complex user queries into simpler sub-queries to ensure all 
                # parts of a compound question (e.g., across multiple docs) are retrieved.
                decomp_prompt = PromptTemplate(
                    template="Break this complex query into up to 3 concise, distinct search queries for a vector database. Return them ONLY as a comma-separated list.\nQuery: {query}",
                    input_variables=["query"]
                )
                decomp_chain = decomp_prompt | LLM | CommaSeparatedListOutputParser()
                sub_queries = decomp_chain.invoke({"query": user_query})
                # Always safely include the original query
                all_queries = [user_query] + sub_queries

                # Gather and deduplicate documents from all queries
                all_docs = []
                for q in all_queries:
                    all_docs.extend(retriever.invoke(q))
                
                unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
                # ───────────────────────────────────────────────────────────

                # Format each chunk so the LLM explicitly sees the source filename
                document_prompt = PromptTemplate.from_template(
                    "Source: {source}\nContent: {page_content}"
                )
                document_chain = create_stuff_documents_chain(
                    LLM, PROMPT, document_prompt=document_prompt
                )

                # Pass doc_names into the chain so the prompt knows which files are active
                doc_names_str = ", ".join(st.session_state.doc_names) or "Unknown"
                
                # Execute the chain directly with our globally retrieved unique docs
                response = document_chain.invoke({
                    "input": user_query,
                    "doc_names": doc_names_str,
                    "context": unique_docs,
                })
                answer = response

                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = f"❌ Error generating response: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

    # Auto-save conversation to persistent history
    if st.session_state.session_id:
        history_mgr.save_messages(
            st.session_state.session_id,
            st.session_state.chat_history,
        )
