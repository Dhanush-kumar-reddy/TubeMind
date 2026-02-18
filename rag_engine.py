import os
import whisper
import yt_dlp
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

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
    
    # 1. Download
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True,
        'no_warnings': True
    }
    
    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    # 2. Transcribe
    result = whisper_model.transcribe("temp_audio.mp3")
    
    # --- INTELLIGENT CHUNKING (The Fix) ---
    docs = []
    current_chunk_text = ""
    current_chunk_start = 0
    
    # We group segments until they reach ~1000 characters (approx 30-60 seconds of speech)
    # This gives the LLM enough context to understand "What is RAG?"
    for segment in result['segments']:
        # If it's the start of a new chunk, record the timestamp
        if current_chunk_text == "":
            current_chunk_start = segment['start']
        
        current_chunk_text += segment['text'] + " "
        
        # If chunk is big enough, save it and reset
        if len(current_chunk_text) >= 1000:
            doc = Document(
                page_content=current_chunk_text.strip(),
                metadata={"start_time": current_chunk_start}
            )
            docs.append(doc)
            current_chunk_text = "" # Reset for next chunk
            
    # Don't forget the last leftover chunk!
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
    # Try getting key from Streamlit secrets (Cloud) or Environment (Local)
    api_key = None
    
    # Check Streamlit Cloud Secrets first (Standard way)
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    # Fallback to Environment Variable (Good for local testing)
    elif os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
        
    if not api_key:
        st.error("🚨 Groq API Key not found! Please add it to Streamlit Secrets.")
        st.stop()
        return None


    # 2. Initialize LLM (Rest of your code is fine...)
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    except Exception as e:
        st.error(f"LLM Initialization Failed: {e}")
        st.stop()
        return None
    
    # 3. Define Prompt
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
    
    # 4. Build Chain
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Chain Build Failed: {e}")
        st.stop()