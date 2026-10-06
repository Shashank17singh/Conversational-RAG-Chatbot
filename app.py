"""
Conversational RAG interface using Streamlit and LangChain.
Implements history-aware retrieval over PDF documents.
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
    # Avoid writing to the current directory; use a proper temp dir for concurrency safety
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
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


def main() -> None:
    """
    Main Streamlit application logic:
    1. Sets up the UI and collects inputs (session ID, PDFs).
    2. Processes PDFs into a Chroma vector store.
    3. Builds a history-aware RAG chain using LangChain.
    4. Handles user interaction and renders chat history.
    """
    st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
    groq_api_key = setup_environment()

    # We load models at the top to fail fast if keys/connectivity are missing
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

    st.title("Conversational RAG Chatbot")
    st.write("Upload PDF's and chat with their content")

    session_id = st.text_input("Session ID", value="default_session")
    uploaded_files = st.file_uploader(
        "Choose a PDF File", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        documents = process_uploaded_pdfs(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        if not splits:
            st.error("No content extracted from the PDF.")
            st.stop()

        # Rebuilding the vector store on every upload is inefficient for large files,
        # but acceptable for a simple MVP. In production, decouple ingestion from serving.
        vectorstore = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        vectorstore.add_documents(splits)
        retriever = vectorstore.as_retriever()

        # 1. Prompt to rewrite the question based on chat history
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

        # 2. Prompt to answer the standalone question using retrieved context
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

        # Wrap in history management
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.success("Response received!")
            st.write("**Assistant:**", response["answer"])

            with st.expander("View Chat History"):
                for msg in get_session_history(session_id).messages:
                    st.write(f"**{msg.type.capitalize()}:** {msg.content}")


if __name__ == "__main__":
    main()
