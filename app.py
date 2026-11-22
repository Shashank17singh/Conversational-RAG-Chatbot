"""
Conversational RAG interface using Streamlit and LangChain.
Implements history-aware retrieval over PDF documents with an enterprise slate UI.
"""

import os
import tempfile
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_core import private_collection_name, safe_filename, uploads_fingerprint

CUSTOM_CSS = """
<style>
/* Hallmark & UI/UX Pro Max Design System · Modern Conversational Workspace
 * Theme: Deep Slate Neutral (#0d1117) with Electric Blue Accent (#3b82f6)
 */
:root {
  --bg-canvas: #0d1117;
  --bg-surface: #161b22;
  --bg-surface-elevated: #21262d;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-default: #30363d;
  --border-focus: #3b82f6;
  --text-primary: #f0f6fc;
  --text-secondary: #8b949e;
  --accent-primary: #3b82f6;
  --accent-hover: #2563eb;
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}

.stApp {
  background-color: var(--bg-canvas) !important;
  color: var(--text-primary) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

/* Chat bubble styling */
[data-testid="stChatMessage"] {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  padding: 1rem 1.25rem !important;
  margin-bottom: 0.75rem !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}

/* Inputs */
.stTextInput > div > div > input,
[data-baseweb="base-input"] > input {
  background-color: var(--bg-canvas) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}

.stTextInput > div > div > input:focus,
[data-baseweb="base-input"] > input:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}

/* Buttons */
.stButton > button {
  background-color: var(--bg-surface-elevated) !important;
  border: 1px solid var(--border-default) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
}

.stButton > button:hover {
  background-color: #30363d !important;
  border-color: var(--text-secondary) !important;
}

/* Cards & Badges */
.chat-header {
  margin-bottom: 1.5rem;
}

.badge-tag {
  display: inline-block;
  background-color: rgba(59, 130, 246, 0.12);
  border: 1px solid rgba(59, 130, 246, 0.3);
  color: #60a5fa;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.2rem 0.6rem;
  border-radius: 9999px;
  margin-bottom: 0.5rem;
}

.empty-state {
  text-align: center;
  padding: 3rem 1.5rem;
  background: var(--bg-surface);
  border: 1px dashed var(--border-default);
  border-radius: var(--radius-lg);
  margin: 1.5rem 0;
}

.empty-state h3 {
  color: var(--text-primary) !important;
  margin: 0.75rem 0 0.5rem 0 !important;
}

.empty-state p {
  color: var(--text-secondary) !important;
  font-size: 0.95rem;
  max-width: 480px;
  margin: 0 auto;
}
</style>
"""


def setup_environment() -> str:
    """
    Configures environment variables and retrieves API keys from Streamlit secrets or .env.
    Halts execution if the Groq API key is missing.
    """
    load_dotenv()
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    hf_key = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))

    if hf_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key

    if not groq_api_key:
        st.error(
            "No Groq API key configured. Add GROQ_API_KEY to a local .env file "
            "(for development) or to Streamlit's Secrets (for deployment)."
        )
        st.stop()

    return groq_api_key


def process_uploaded_pdfs(uploaded_files: list[Any]) -> list[Document]:
    """
    Saves uploaded Streamlit files to a secure temp directory, parses them,
    and returns a combined list of LangChain Document objects.
    """
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_pdf_path = os.path.join(temp_dir, safe_filename(uploaded_file.name))
            with open(temp_pdf_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf_path)
            documents.extend(loader.load())

    return documents


def get_session_history(session: str) -> BaseChatMessageHistory:
    """
    Retrieves or initializes a chat history object for the given session ID
    using Streamlit's session state to persist history across reruns.
    """
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


def get_session_vectorstore(uploaded_files: list[Any], embeddings: Any) -> Chroma:
    """
    Returns this session's in-memory index, rebuilding it only when the uploads change.
    Each index has its own collection, so one visitor's documents never reach another.
    """
    fingerprint = uploads_fingerprint(
        (file.name, file.getvalue()) for file in uploaded_files
    )
    cached = st.session_state.get("vector_index")
    if cached and cached[0] == fingerprint:
        return cached[1]

    documents = process_uploaded_pdfs(uploaded_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    if not splits:
        st.error("No content extracted from the PDF.")
        st.stop()

    vectorstore = Chroma(
        collection_name=private_collection_name(),
        embedding_function=embeddings,
    )
    vectorstore.add_documents(splits)
    st.session_state.vector_index = (fingerprint, vectorstore)
    return vectorstore


def main() -> None:
    """
    Main Streamlit application logic:
    1. Sets up the UI and collects inputs (session ID, PDFs).
    2. Processes PDFs into a Chroma vector store.
    3. Builds a history-aware RAG chain using LangChain.
    4. Handles user interaction and renders chat history.
    """
    st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    groq_api_key = setup_environment()

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

    st.markdown(
        """
        <div class="chat-header">
            <span class="badge-tag">Document Intelligence</span>
            <h1 style="margin: 0.25rem 0 0.5rem 0; font-size: 2rem; font-weight: 700;">Conversational RAG Assistant</h1>
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.95rem;">
                History-aware contextual document question answering powered by LangChain and Chroma.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Configuration")
        session_id = st.text_input("Session Identifier", value="default_session")

        st.subheader("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload Reference PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to index into Chroma vector storage.",
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} document(s) uploaded")

    if not uploaded_files:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No Documents Ingested</h3>
                <p>Upload one or more PDF documents in the left sidebar to generate vector embeddings and begin conversational exploration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    vectorstore = get_session_vectorstore(uploaded_files, embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Formulate a standalone question from chat history and input.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer based on context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    history = get_session_history(session_id)
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.spinner("Analyzing document context..."):
            response = conversational_rag_chain.invoke(
                {"input": user_prompt},
                config={"configurable": {"session_id": session_id}},
            )

        with st.chat_message("assistant"):
            st.markdown(response["answer"])


if __name__ == "__main__":
    main()
