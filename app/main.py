from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import whisper
import requests
import os
import torch
import logging
import tempfile
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from google.cloud import secretmanager
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize FastAPI app
app = FastAPI()

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"🔥 Using device: {device}")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper Model
logging.info("🔄 Loading Whisper model...")
model = whisper.load_model("small")
logging.info("✅ Whisper model loaded successfully.")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    logging.info(f"📂 Received audio file: {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        file_location = temp_file.name

    try:
        logging.info("📝 Transcribing audio...")
        result = model.transcribe(file_location, language="en", word_timestamps=False)
        transcription_text = result["text"]
    except Exception as e:
        logging.error(f"❌ Error during transcription: {str(e)}")
        return {"error": "Transcription failed"}
    finally:
        os.remove(file_location)

    logging.info("✅ Transcription completed. Saving to Supabase...")

    try:
        supabase.table("transcriptions").insert({
            "filename": file.filename,
            "text": transcription_text
        }).execute()
    except Exception as e:
        logging.error(f"❌ Supabase insertion error: {str(e)}")

    return {"transcription": transcription_text}

@app.get("/transcriptions")
async def get_transcriptions():
    try:
        logging.info("📥 Fetching latest 10 transcriptions from Supabase...")
        response = (
            supabase
            .table("transcriptions")
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        transcriptions = response.data
        if not transcriptions:
            logging.warning("⚠️ No transcriptions found in the database.")
        return {"transcriptions": transcriptions}
    except Exception as e:
        logging.error(f"❌ Failed to fetch transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch transcriptions")

@app.get("/search")
async def search_google(q: str):
    logging.info(f"🔍 Searching Google for: {q}")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": q, "key": API_KEY, "cx": SEARCH_ENGINE_ID, "num": 3}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logging.error(f"❌ Google Search API failed: {response.status_code}")
        return {"error": "Search request failed"}

    data = response.json()
    results = [{"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")} for item in data.get("items", [])]
    return {"results": results}

@app.get("/image-search")
async def search_images(q: str):
    logging.info(f"🖼️ Searching for images: {q}")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": q,
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "searchType": "image",  # This enables image search
        "num": 5  # Number of images to return
    }

    response = requests.get(url, params=params)
    logging.info(response)
    if response.status_code != 200:
        logging.error(f"❌ Image Search API failed: {response.status_code}")
        logging.error(f"Response content: {response.text}")  # log raw HTML/error page
        return JSONResponse(
            status_code=response.status_code,
            content={"error": f"Image search failed: {response.status_code}"}
        )

    data = response.json()
    image_results = [
        {
            "title": item.get("title"),
            "image_link": item.get("link"),
            "thumbnail": item.get("image", {}).get("thumbnailLink"),
            "context_link": item.get("image", {}).get("contextLink")
        }
        for item in data.get("items", [])
    ]

    return {"images": image_results}

class ScrapeRequest(BaseModel):
    pages: List[str]

@app.post("/scrape/")
async def scrape_web(request: ScrapeRequest):
    pages = request.pages
    results = []
    for url in pages:
        results.append(scrape_page(url))
    return {"scraped_data": results}

def scrape_page(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        summary = " ".join([p.text for p in soup.find_all("p")[:5]]) or "No summary available"
        return {"url": url, "summary": summary}
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to fetch {url}: {e}")
        return {"url": url, "error": "Failed to fetch page"}

# Groq LLM API
client = Groq(api_key=GROQ_API_KEY)

class LLMRequest(BaseModel):
    prompt: str

@app.post("/llm/")
async def query_llm(request: LLMRequest):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": request.prompt}],
            model="llama-3.3-70b-versatile"
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"❌ Groq API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Groq request failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
