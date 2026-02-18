import yt_dlp
import os

# URL to test (Use a long video, not a short)
VIDEO_URL = "https://youtu.be/W1D0puderfs?si=2PaBRZGL_-uvxdm8" 

def test_download():
    print("🚀 Starting Local Download Test...")
    
    # Check for cookies.txt
    cookie_path = "cookies.txt" if os.path.exists("cookies.txt") else None
    if cookie_path:
        print(f"✅ Found cookies.txt at: {cookie_path}")
    else:
        print("⚠️ No cookies.txt found. Trying anonymous download (might fail)...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'test_video.%(ext)s',
        'quiet': False,
        'no_warnings': False,
        'cookiefile': cookie_path,
        
        # --- Android Spoofing (The Trick) ---
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])
        print("\n🎉 SUCCESS! Download worked locally.")
        print("Check your folder for 'test_video.mp3'")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    # Clean up previous run
    if os.path.exists("test_video.mp3"):
        os.remove("test_video.mp3")
    test_download()