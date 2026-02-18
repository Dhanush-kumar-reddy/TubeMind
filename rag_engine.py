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

# --- Cached Models ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_video(video_url):
    print(f"DEBUG: Processing video {video_url}")
    
    # 1. VERIFY FFMPEG
    if not shutil.which("ffmpeg"):
        st.error("🚨 FFmpeg is not installed! The app needs to be Rebooted.")
        st.stop()

    whisper_model = load_whisper_model()
    embeddings = load_embedding_model()
    
    # 2. Download Configuration (The Android Fix)
    ydl_opts = {
        'format': 'bestaudio/best',
        # ⬇️ THIS IS THE KEY FIX ⬇️
        # Pretend to be an Android device to bypass IP blocking
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': True 
    }
    
    # Clean up old file
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")

    # Attempt Download
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        # If Android spoofing fails, show the error
        st.error(f"Download Error: {e}")
        st.stop()
    
    # 3. Verify Download Success
    if not os.path.exists("temp_audio.mp3"):
        st.error("❌ Download failed. YouTube blocked the request. Try a different video or try again later.")
        st.stop()

    # 4. Transcribe
    result = whisper_model.transcribe("temp_audio.mp3")
    
    # --- INTELLIGENT CHUNKING ---
    docs = []
    current_chunk_text = ""
    current_chunk_start = 0
    
    for segment in result['segments']:
        if current_chunk_text == "":
            current_chunk_start = segment['start']
        
        current_chunk_text += segment['text'] + " "
        
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
        
    print(f"DEBUG: Created {len(docs)} large context chunks.")
        
    print("DEBUG: Building FAISS Index...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever()

def get_answer_chain(retriever):
    api_key = None
    
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    elif os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
        
    if not api_key:
        st.error("🚨 Groq API Key not found!")
        st.stop()
        return None

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    except Exception as e:
        st.error(f"LLM Initialization Failed: {e}")
        st.stop()
        return None
    
    system_prompt = (
        "You are a video analysis assistant. Use the following context to answer the question. "
        "The context includes transcripts with start times. "
        "Always explicitly cite the start time (e.g., 'at 54 seconds') if provided in the context metadata. "
        "If you don't know, say you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Chain Build Failed: {e}")
        st.stop()