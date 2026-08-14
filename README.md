# 📄 AI Document Chatbot

An AI-powered multilingual document chatbot that allows users to upload PDF or TXT documents and ask questions using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** with **FAISS vector search**, **HuggingFace embeddings**, and an **Ollama-hosted LLM** to generate answers based only on the uploaded document.

It also supports **voice input, multilingual conversations, and voice responses**.

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔎 Semantic search using FAISS
- 🤗 HuggingFace sentence embeddings
- 🦙 Ollama LLM integration
- 💬 Natural-language document Q&A
- 🎙️ Voice input using Faster-Whisper
- 🔊 Voice responses using Google Text-to-Speech
- 🌐 Multilingual support
- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Marathi
- 🇸🇦 Arabic
- 🔄 Automatic translation
- 💾 Persistent document indexing
- ⚡ Cached embeddings and indexes
- 🖥️ Interactive Streamlit interface

---

## 🏗️ Architecture

```text
                 PDF / TXT Document
                        │
                        ▼
              Document Text Extraction
                        │
                        ▼
                Text Chunking
                        │
                        ▼
             HuggingFace Embeddings
                        │
                        ▼
                  FAISS Vector DB
                        │
                        │
User Question ─────────┤
        │               │
        ▼               ▼
  Translation      Semantic Retrieval
        │               │
        └───────┬───────┘
                ▼
           Relevant Context
                │
                ▼
            Ollama LLM
                │
                ▼
          Generated Answer
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
   Text Response     Translation
                         │
                         ▼
                    gTTS Voice
