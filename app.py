import streamlit as st
import os
from rag_engine import process_video, get_answer_chain

st.set_page_config(page_title="TubeMind", page_icon="📹")

st.title("📹 TubeMind: Chat with Video")
st.caption("Powered by Groq, Whisper, and LangChain")

# Session State to hold the retriever so we don't re-process video on every chat
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for Video Input
with st.sidebar:
    video_url = st.text_input("Enter YouTube URL")
    if st.button("Analyze Video"):
        if video_url:
            with st.spinner("Watching video (Downloading & Transcribing)..."):
                try:
                    # Clean up old audio file if exists
                    if os.path.exists("temp_audio.mp3"):
                        os.remove("temp_audio.mp3")
                        
                    st.session_state.retriever = process_video(video_url)
                    st.success("Video Processed! You can now chat.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")

# Chat Interface
if st.session_state.retriever:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask something about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            rag_chain = get_answer_chain(st.session_state.retriever)
            response = rag_chain.invoke({"input": prompt})
            
            # The context usually contains the metadata we need
            answer = response['answer']
            
            # Extract timestamps from the source documents used
            sources = response['context']
            timestamps = sorted(list(set([int(doc.metadata['start_time']) for doc in sources])))
            
            # Format the output with a clickable link (approximate for YouTube)
            # YouTube format: t=120s
            timestamp_str = " | ".join([f"[{t}s]({video_url}&t={t}s)" for t in timestamps[:3]])
            
            final_response = f"{answer}\n\n**Relevant Segments:** {timestamp_str}"
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
else:
    st.info("Please process a video from the sidebar to start.")