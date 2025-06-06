from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client
from google.cloud import speech
from bs4 import BeautifulSoup
from groq import Groq
import requests
import tempfile
import logging
import os

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate environment variables
required_vars = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
    "GOOGLE_SEARCH_API_KEY": API_KEY,
    "GOOGLE_SEARCH_ENGINE_ID": SEARCH_ENGINE_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}
for key, value in required_vars.items():
    if not value:
        raise EnvironmentError(f"❌ {key} is not set in the environment.")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
speech_client = speech.SpeechClient()
groq_client = Groq(api_key=GROQ_API_KEY)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ScrapeRequest(BaseModel):
    pages: List[str]

class LLMRequest(BaseModel):
    prompt: str

# Routes
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logging.info(f"📂 Received audio file: {file.filename}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            file_location = temp_file.name

        with open(file_location, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = speech_client.recognize(config=config, audio=audio)
        transcription_text = " ".join([result.alternatives[0].transcript for result in response.results])

        if background_tasks:
            background_tasks.add_task(save_transcription, file.filename, transcription_text)

        return {"transcription": transcription_text}

    except Exception as e:
        logging.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

def save_transcription(filename, text):
    try:
        supabase.table("transcriptions").insert({"filename": filename, "text": text}).execute()
        logging.info("✅ Transcription saved to Supabase.")
    except Exception as e:
        logging.error(f"❌ Supabase insertion error: {e}")

@app.get("/transcriptions")
async def get_transcriptions():
    try:
        response = (
            supabase
            .table("transcriptions")
            .select("*")
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        return {"transcriptions": response.data}
    except Exception as e:
        logging.error(f"❌ Fetch error: {e}")
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
        "searchType": "image",
        "num": 5
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        logging.error(f"❌ Image Search API failed: {response.status_code}")
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

@app.post("/scrape/")
async def scrape_web(request: ScrapeRequest):
    results = [scrape_page(url) for url in request.pages]
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

@app.post("/llm/")
async def query_llm(request: LLMRequest):
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": request.prompt}],
            model="llama-3.3-70b-versatile"
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"❌ Groq API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Groq request failed: {e}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
