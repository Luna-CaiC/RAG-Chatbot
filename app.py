"""
RAG Chatbot — Streamlit Application
=====================================
Upload a PDF → ask questions → get citation-backed answers from Gemini.
"""

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.ingest import process_document

# ── Load environment variables ───────────────────────────────────────
load_dotenv()

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Session state defaults ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ── Sidebar: PDF upload ─────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a technical document or research paper to query.",
    )

    if uploaded_file is not None:
        # Only process if we haven't already processed this file
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state.last_uploaded_file != uploaded_file.name
        ):
            with st.spinner("Processing document..."):
                try:
                    vector_store = process_document(uploaded_file)
                    st.session_state.vector_store = vector_store
                    st.session_state.last_uploaded_file = uploaded_file.name
                    # Clear chat history for the new document
                    st.session_state.chat_history = []
                    st.success(f"✅ *{uploaded_file.name}* processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing document: {e}")

    if st.session_state.vector_store is not None:
        st.info(f"📚 Active document: **{st.session_state.get('last_uploaded_file', 'N/A')}**")

# ── LLM & chain setup ───────────────────────────────────────────────
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on "
    "the provided context from the uploaded document. "
    "If the context does not contain the answer, say so honestly. "
    "Always cite the relevant parts of the context.\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

# ── Main chat interface ─────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Upload a PDF in the sidebar, then ask questions about it.")

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input(
    "Ask a question about your document...",
    disabled=st.session_state.vector_store is None,
)

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build the retrieval chain
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )
                document_chain = create_stuff_documents_chain(LLM, PROMPT)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Invoke the chain
                response = retrieval_chain.invoke({"input": user_query})
                answer = response["answer"]

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
