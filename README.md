# ⚖️ Smart Client Intake and Legal Issue Identifier

Smart Client Intake and Legal Issue Identifier is an AI-powered Urdu legal assistant designed specifically for Pakistani law. It provides structured, context-aware legal guidance using Retrieval-Augmented Generation (RAG) with a FAISS vector database and HuggingFace LLM.

Built with Streamlit, it offers a professional chat interface, session management, and persistent chat history using SQLite.

# 🚀 Features

✅ Pure Urdu legal responses

✅ Structured answer format:

Legal Explanation

Relevant Laws / Sections / Articles

Legal Reasoning

Practical Advice

✅ FAISS vector-based knowledge retrieval (RAG)

✅ HuggingFace LLM integration

✅ Chat history saved in SQLite

✅ Auto chat title generation

✅ Session rename & delete

✅ Search chats

✅ Dark / Light theme toggle

✅ Professional UI with fixed bottom input

✅ Context-aware answers from local knowledge base

# 🏗️ Tech Stack

Frontend: Streamlit

LLM: HuggingFace Endpoint (Cerebras provider supported)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Database: FAISS

Database: SQLite

Language: Python

# 📂 Project Structure
LawGPT/
│
├── main.py              # Main application file
├── chats.db                 # SQLite database (auto-created)
├── .env                     # Environment variables
└── knowledge_base/
    └── vector_db/           # FAISS vector store

# 💬 How It Works

User enters a legal question in Urdu.

If it's the first message:

The system auto-generates a short Urdu chat title.

The system:

Searches relevant legal documents from FAISS.

Sends the question + retrieved context to HuggingFace LLM.

The response follows a structured legal format.

Chat is saved in SQLite database.

UI auto-scrolls to latest message.

# 🗃️ Database

The application automatically creates.

# 🎛️ Customization Options

From Sidebar:

Model ID (default: meta-llama/Llama-3.1-8B-Instruct)

Temperature

Max Tokens

Chat search

Rename / Delete chats

New Chat

Theme toggle
