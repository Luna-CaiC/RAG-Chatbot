"""
RAG Chatbot — Streamlit Entry Point
====================================
A Retrieval-Augmented Generation system for domain-specific Q&A
powered by LangChain and Google Gemini.
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Main UI ─────────────────────────────────────────────────────────
st.title("🤖 Hello Gemini RAG")
st.markdown(
    "Welcome! This app will let you upload technical documents and "
    "ask questions answered by **Google Gemini** with retrieval-augmented context."
)
