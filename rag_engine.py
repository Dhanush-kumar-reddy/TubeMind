import os
import re
import base64
import tempfile
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
        st.error("❌ Invalid YouTube URL.")
        st.stop()
        
    print(f"DEBUG: Video ID extracted: {video_id}")

    # 2. Fetch Transcript with Cookies
    docs = []
    temp_cookie_file = None
    cookie_path = None

    try:
        # --- COOKIE SETUP ---
        # We need cookies to bypass the "Sign In" or "Bot" block
        if "YOUTUBE_COOKIES_B64" in st.secrets:
            # Decode the base64 string back to a file
            cookie_content = base64.b64decode(st.secrets["YOUTUBE_COOKIES_B64"]).decode('utf-8')
            
            temp_cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt")
            temp_cookie_file.write(cookie_content)
            temp_cookie_file.close()
            cookie_path = temp_cookie_file.name
            print(f"DEBUG: Using cookies from secrets at {cookie_path}")
        elif os.path.exists("cookies.txt"):
             cookie_path = "cookies.txt"

        # --- FETCH ---
        # Pass the cookie file to the API
        if cookie_path:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, cookies=cookie_path)
        else:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # --- CHUNK ---
        current_chunk_text = ""
        current_chunk_start = 0.0
        
        for entry in transcript:
            if current_chunk_text == "":
                current_chunk_start = entry['start']
            current_chunk_text += entry['text'] + " "
            if len(current_chunk_text) >= 1000:
                doc = Document(page_content=current_chunk_text.strip(), metadata={"start_time": current_chunk_start})
                docs.append(doc)
                current_chunk_text = "" 
                
        if current_chunk_text:
            doc = Document(page_content=current_chunk_text.strip(), metadata={"start_time": current_chunk_start})
            docs.append(doc)

    except Exception as e:
        # Clean up
        if temp_cookie_file and os.path.exists(temp_cookie_file.name):
            os.remove(temp_cookie_file.name)
            
        st.error(f"❌ Transcript Fetch Failed: {e}\n\nThis usually means YouTube blocked the server IP. Try running the app locally.")
        st.stop()

    # Clean up
    if temp_cookie_file and os.path.exists(temp_cookie_file.name):
        os.remove(temp_cookie_file.name)

    # 3. Build FAISS Index
    print("DEBUG: Building FAISS Index...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever()

def get_answer_chain(retriever):
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
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
        "You are a video analysis assistant. Answer using the context."
        "\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    
    try:
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Chain Build Failed: {e}")
        st.stop()