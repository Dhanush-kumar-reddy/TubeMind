import os
import shutil
import whisper
import yt_dlp
import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Load Models (Cached) ---
@st.cache_resource
def load_whisper_model():
    print("DEBUG: Loading Whisper Model...")
    return whisper.load_model("base")

@st.cache_resource
def load_embedding_model():
    print("DEBUG: Loading Embedding Model...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_video(video_url):
    print(f"DEBUG: Processing video {video_url}")
    
    # 1. Check FFmpeg (Essential for Audio)
    if not shutil.which("ffmpeg"):
        st.error("🚨 FFmpeg is not installed! Run 'brew install ffmpeg' in terminal.")
        st.stop()

    whisper_model = load_whisper_model()
    embeddings = load_embedding_model()
    
    # 2. Download Audio (Android Spoofing to bypass throttling)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    # Clean up old file
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        st.error(f"❌ Download Failed: {e}")
        st.stop()
    
    # 3. Transcribe with Whisper
    if not os.path.exists("temp_audio.mp3"):
        st.error("❌ Audio file not found. Download failed.")
        st.stop()

    st.toast("Transcribing audio... this takes a moment.")
    result = whisper_model.transcribe("temp_audio.mp3")
    
    # 4. Intelligent Chunking
    docs = []
    current_chunk_text = ""
    current_chunk_start = 0.0
    
    for segment in result['segments']:
        if current_chunk_text == "":
            current_chunk_start = segment['start']
        current_chunk_text += segment['text'] + " "
        
        # Chunk every ~1000 chars
        if len(current_chunk_text) >= 1000:
            doc = Document(
                page_content=current_chunk_text.strip(),
                metadata={"start_time": current_chunk_start}
            )
            docs.append(doc)
            current_chunk_text = "" 
            
    if current_chunk_text:
        doc = Document(
            page_content=current_chunk_text.strip(),
            metadata={"start_time": current_chunk_start}
        )
        docs.append(doc)
        
    print(f"DEBUG: Created {len(docs)} chunks.")
        
    # 5. Build Index
    print("DEBUG: Building FAISS Index...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever()

def get_answer_chain(retriever):
    # Check Secrets (Cloud) or Env (Local)
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
    if not api_key:
        st.error("🚨 Groq API Key not found! Add GROQ_API_KEY to your .env file.")
        st.stop()
        return None

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    except Exception as e:
        st.error(f"LLM Initialization Failed: {e}")
        st.stop()
        return None
    
    system_prompt = (
        "You are a video analysis assistant. Use the provided context to answer the question. "
        "The context includes transcripts with start times. "
        "Always explicitly cite the start time (e.g., 'at 54 seconds'). "
        "If you don't know, say you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    try:
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Chain Build Failed: {e}")
        st.stop()