# 1. Base Image: Official Python 3.11 (Slim version to keep it small)
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Dependencies (FFmpeg is critical for Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first (for better caching)
COPY requirements.txt .

# 5. Install Python Dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Expose the port Streamlit runs on
EXPOSE 8501

# 8. Command to run the app
# We also set the server address to 0.0.0.0 so AWS can find it
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]