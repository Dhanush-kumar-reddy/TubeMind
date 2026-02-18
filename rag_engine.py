import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Cached Models ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    Examples:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ -> dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ -> dQw4w9WgXcQ
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def process_video(video_url):
    print(f"DEBUG: Processing video {video_url}")
    embeddings = load_embedding_model()
    
    # 1. Extract Video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Could not find Video ID.")
        st.stop()

    # 2. Fetch Transcript (The "Lightweight" Way)
    transcript_text = []
    try:
        # Tries to get English transcripts (manual or auto-generated)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        
        # --- INTELLIGENT CHUNKING ---
        docs = []
        current_chunk_text = ""
        current_chunk_start = 0
        
        # Group segments into chunks of ~1000 characters
        for entry in transcript:
            if current_chunk_text == "":
                current_chunk_start = entry['start']
            
            current_chunk_text += entry['text'] + " "
            
            if len(current_chunk_text) >= 1000:
                doc = Document(
                    page_content=current_chunk_text.strip(),
                    metadata={"start_time": current_chunk_start}
                )
                docs.append(doc)
                current_chunk_text = "" 
                
        # Add the last chunk
        if current_chunk_text:
            doc = Document(
                page_content=current_chunk_text.strip(),
                metadata={"start_time": current_chunk_start}
            )
            docs.append(doc)
            
        print(f"DEBUG: Created {len(docs)} chunks from transcript.")

    except Exception as e:
        st.error(f"❌ Could not retrieve transcript. The video might not have captions enabled.\nError: {e}")
        st.stop()
        
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