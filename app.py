import streamlit as st
import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

def setup_environment() -> str:
    """
    Loads environment variables and retrieves the Groq API key.
    
    Returns:
        str: The Groq API key.
    """
    load_dotenv()
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    hf_key = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))
    if hf_key:
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key

    if not groq_api_key:
        st.error(
            "No Groq API key configured. Add GROQ_API_KEY to a local .env file "
            "(for development) or to Streamlit's Secrets (for deployment)."
        )
        st.stop()
    return groq_api_key

def process_uploaded_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """
    Saves uploaded PDFs to a temporary file, extracts text using PyPDFLoader, 
    and cleans up the temporary files.
    
    Args:
        uploaded_files (List[Any]): List of Streamlit uploaded file objects.
        
    Returns:
        List[Document]: List of LangChain Document objects extracted from the PDFs.
    """
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp_{uploaded_file.name}"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
        os.remove(temppdf)
    return documents

def get_session_history(session: str) -> BaseChatMessageHistory:
    """
    Retrieves or initializes the chat message history for a given session ID.
    
    Args:
        session (str): The unique session identifier.
        
    Returns:
        BaseChatMessageHistory: The chat history object for the session.
    """
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

def main() -> None:
    """
    Main Streamlit application execution block.
    """
    groq_api_key = setup_environment()
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.title("Conversational RAG Chatbot")
    st.write("Upload PDF's and chat with their content")

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    session_id = st.text_input("Session ID", value="default_session")

    uploaded_files = st.file_uploader("Choose a PDF File", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = process_uploaded_pdfs(uploaded_files)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        if not splits:
            st.error("No content extracted from the PDF.")
            st.stop()

        vectorstore = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        vectorstore.add_documents(splits)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Formulate a standalone question from chat history and input."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")

        if user_input:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.success("Response received!")
            st.write("**Assistant:**", response['answer'])

            with st.expander("View Chat History"):
                for msg in get_session_history(session_id).messages:
                    st.write(f"**{msg.type.capitalize()}:** {msg.content}")

if __name__ == "__main__":
    main()
