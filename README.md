# 📹 TubeMind: The Video Knowledge Engine

> **"Don't watch the whole video. Just ask what you need."**

TubeMind is a **Multimodal RAG (Retrieval-Augmented Generation)** application that allows users to "chat" with YouTube videos. It processes video content into text, indexes it using semantic search, and uses the **Llama 3.3** LLM to provide accurate answers with **timestamped citations**.

---

## 🚀 How It Works (The Architecture)

TubeMind allows you to bypass the "TL;DW" (Too Long; Didn't Watch) problem by treating video as a queryable database.

```mermaid
graph LR
    A[YouTube URL] --> B(Audio Extraction via yt-dlp)
    B --> C(Transcribe via OpenAI Whisper)
    C --> D(Chunking & Metadata Embedding)
    D --> E[(FAISS Vector Index)]
    E --> F{User Question}
    F --> G[Llama 3.3 via Groq]
    G --> H[Answer + Timestamp Link]
```
##  Tech Stack
* LLM Engine:** Llama 3.3 (70B) via Groq Cloud (Ultra-low latency)
* Transcription:** OpenAI Whisper (Local efficient base model)
* Vector Database:** FAISS (Facebook AI Similarity Search) - *Chosen for zero-dependency local performance.*
* Orchestration:** LangChain (Chains & Retrievers)
* Frontend:** Streamlit

##  Key Features
* Sub-Second Retrieval:** Uses FAISS in-memory indexing to find relevant video segments instantly.
* Timestamp Linking:** Every answer includes a clickable "jump-to" link (e.g., `[04:12]`) that takes you to the exact moment in the video.
* Intelligent Chunking:** Groups transcripts into context-aware blocks to prevent "context fragmentation" hallucinations.
* Cloud Native:** Deployed on Streamlit Community Cloud with auto-installing dependencies (`ffmpeg`).
