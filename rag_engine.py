import os
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
    print("DEBUG: Loading Whisper Model...")
    return whisper.load_model("base")

@st.cache_resource
def load_embedding_model():
    print("DEBUG: Loading Embedding Model...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_video(video_url):
    print(f"DEBUG: Processing video {video_url}")
    whisper_model = load_whisper_model()
    embeddings = load_embedding_model()
    
    # --- COOKIE HANDLING ---
    cookie_path = None
    temp_cookie_file = None
    
    try:
        # Create temp cookie file if secret exists (Fix for 403 Forbidden)
        if "YOUTUBE_COOKIES" in st.secrets:
            temp_cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt")
            temp_cookie_file.write(st.secrets["YOUTUBE_COOKIES"])
            temp_cookie_file.close()
            cookie_path = temp_cookie_file.name
        elif os.path.exists("cookies.txt"):
            cookie_path = "cookies.txt"

        # 1. Download Configuration
        ydl_opts = {
            # FIX: Use 'best' instead of 'bestaudio/best'
            # This allows yt-dlp to download a video file if audio-only is missing,
            # and the post-processor below will still convert it to MP3.
            'format': 'best', 
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }],
            'outtmpl': 'temp_audio.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'cookiefile': cookie_path
        }
        
        # Clean up old file
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")

        # Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
    except Exception as e:
        # Clean up temp file if error
        if temp_cookie_file and os.path.exists(temp_cookie_file.name):
            os.remove(temp_cookie_file.name)
        raise e 

    # Clean up cookies after download
    if temp_cookie_file and os.path.exists(temp_cookie_file.name):
        os.remove(temp_cookie_file.name)
    
    # 2. Transcribe
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
        
    # 3. Build FAISS Index
    print("DEBUG: Building FAISS Index...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever()

def get_answer_chain(retriever):
    api_key = None
    
    # Robust API Key Check
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