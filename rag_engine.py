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


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def extract_video_id(url: str):
    """
    Extracts the 11-char YouTube video ID from common URL formats.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&].*)?$"
    match = re.search(regex, url)
    return match.group(1) if match else None


def fetch_transcript_entries(video_id: str):
    """
    Supports newer youtube-transcript-api (fetch) and older (get_transcript).
    Returns list of dicts with keys: text, start
    """
    api = YouTubeTranscriptApi()

    # Newer API (v1+)
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id)
        return [{"text": snippet.text, "start": float(snippet.start)} for snippet in fetched]

    # Older API fallback
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(video_id)

    raise RuntimeError(
        "Unsupported youtube-transcript-api version. "
        "Please upgrade: pip install -U youtube-transcript-api"
    )


def process_video(video_url: str):
    print(f"DEBUG: Processing video {video_url}")
    embeddings = load_embedding_model()

    # 1. Extract Video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Could not find Video ID.")
        st.stop()

    print(f"DEBUG: Video ID extracted: {video_id}")

    # 2. Fetch Transcript + Chunk
    docs = []
    try:
        transcript_entries = fetch_transcript_entries(video_id)

        current_chunk_text = ""
        current_chunk_start = 0.0

        for entry in transcript_entries:
            if not current_chunk_text:
                current_chunk_start = float(entry["start"])

            current_chunk_text += entry["text"] + " "

            if len(current_chunk_text) >= 1000:
                docs.append(
                    Document(
                        page_content=current_chunk_text.strip(),
                        metadata={"start_time": current_chunk_start},
                    )
                )
                current_chunk_text = ""

        if current_chunk_text:
            docs.append(
                Document(
                    page_content=current_chunk_text.strip(),
                    metadata={"start_time": current_chunk_start},
                )
            )

        if not docs:
            st.error("❌ Transcript was fetched but no text chunks were created.")
            st.stop()

        print(f"DEBUG: Created {len(docs)} chunks from transcript.")

    except Exception as e:
        st.error(f"❌ Could not retrieve transcript.\n\nReason: {e}")
        st.stop()

    # 3. Build FAISS Index
    print("DEBUG: Building FAISS Index...")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever()


def get_answer_chain(retriever):
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("🚨 Groq API Key not found! Add GROQ_API_KEY to Streamlit secrets or env.")
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
        "If you don't know, say you don't know.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Chain Build Failed: {e}")
        st.stop()
        return None
