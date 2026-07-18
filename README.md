<div align="center">

# 🤖 Conversational RAG Chatbot

**A Retrieval-Augmented Generation web app that lets you chat with your PDFs — with full conversational memory**

[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70b--versatile-F55036?style=for-the-badge&logoColor=white)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-6A5ACD?style=for-the-badge&logoColor=white)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

</div>

---

## 📖 Overview

A Retrieval-Augmented Generation (RAG) web app built with Streamlit, LangChain, and Groq. Upload any PDF and have a multi-turn conversation with its content — the app remembers your chat history so follow-up questions stay context-aware.

---

## ✨ Features

| | |
|---|---|
| 📄 **Multi-PDF Upload** | Upload one or more PDFs simultaneously |
| 🧠 **Context-Aware Conversations** | Chat history is used to reformulate follow-up questions |
| ⚡ **Groq-Powered LLM** | Ultra-fast inference using Llama 3.3 70B Versatile |
| 🔍 **Semantic Search** | HuggingFace embeddings + ChromaDB vector store |
| 🗂️ **Session Management** | Multiple isolated chat sessions supported |
| 🖥️ **Clean Streamlit UI** | Simple, browser-based interface |

---

## 🧠 Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
ChromaDB Vector Store
   │
   ├──────────────┬──────────────┐
   │              │              │
Retriever   Chat History         │
   │              │              │
   └──────┬───────┘              │
          ▼                      │
History-Aware Retriever (LangChain)
          │
          ▼
Groq LLM (Llama 3.3 70B Versatile)
          │
          ▼
     Final Answer 💬
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq — Llama 3.3 70B Versatile |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Loader | PyPDFLoader |

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository

```bash
git clone https://github.com/Shashank17singh/Chatbot_with_PDF_upload_and_History.git
cd Chatbot_with_PDF_upload_and_History
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

```bash
export GROQ_API_KEY="your_api_key_here"    # On Windows: set GROQ_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal, upload a PDF, and start chatting.

---
