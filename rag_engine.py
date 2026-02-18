import os
import shutil
import whisper
import yt_dlp
import tempfile
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
    
    # 1. VERIFY FFMPEG (Crucial Debug Step)
    if not shutil.which("ffmpeg"):
        st.error("🚨 FFmpeg is not installed! The app needs to be Rebooted.")
        st.stop()

    whisper_model = load_whisper_model()
    embeddings = load_embedding_model()
    
    # --- COOKIE HANDLING ---
    cookie_path = None
    temp_cookie_file = None
    
    try:
        # Create temp cookie file if secret exists
        if "YOUTUBE_COOKIES" in st.secrets:
            temp_cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt")
            temp_cookie_file.write(st.secrets["YOUTUBE_COOKIES"])
            temp_cookie_file.close()
            cookie_path = temp_cookie_file.name
        elif os.path.exists("cookies.txt"):
            cookie_path = "cookies.txt"

        # 2. Download Configuration (Robust)
        ydl_opts = {
            # Try audio-only first, then fallback to best video
            'format': 'bestaudio/best', 
            # Spoof a real browser to avoid "Bot" detection
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }],
            'outtmpl': 'temp_audio.%(ext)s',
            'quiet': False, # Show errors in logs
            'no_warnings': False,
            'cookiefile': cookie_path,
            'ignoreerrors': True # Try to keep going even if one format fails
        }
        
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
    except Exception as e:
        if temp_cookie_file and os.path.exists(temp_cookie_file.name):
            os.remove(temp_cookie_file.name)
        raise e 

    # Clean up cookies
    if temp_cookie_file and os.path.exists(temp_cookie_file.name):
        os.remove(temp_cookie_file.name)
    
    # 3. Verify Download Success
    if not os.path.exists("temp_audio.mp3"):
        st.error("❌ Download failed. YouTube blocked the request. Please check your Cookies.")
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