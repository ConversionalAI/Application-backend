from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client
from google.cloud import speech
from google.oauth2 import service_account
from bs4 import BeautifulSoup
from groq import Groq
import requests
import tempfile
import logging
import os
import json

# Load .env file (local only)
load_dotenv()

# Get required environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

required_vars = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
    "GOOGLE_SEARCH_API_KEY": API_KEY,
    "SEARCH_ENGINE_ID": SEARCH_ENGINE_ID,
    "GROQ_API_KEY": GROQ_API_KEY,
}
for key, value in required_vars.items():
    if not value:
        raise RuntimeError(f"❌ Missing environment variable: {key}")

# Google credentials handling
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # file path (local)
google_creds_json = os.getenv("GOOGLE_CREDENTIALS")  # JSON string (Railway)

if google_creds_json:  # Railway style
    try:
        creds_info = json.loads(google_creds_json)
        speech_client = speech.SpeechClient(
            credentials=service_account.Credentials.from_service_account_info(creds_info)
        )
        logging.info("✅ Loaded Google credentials from GOOGLE_CREDENTIALS env var")
    except json.JSONDecodeError:
        raise RuntimeError("❌ GOOGLE_CREDENTIALS contains invalid JSON")
elif google_creds_path:  # Local file path style
    abs_path = os.path.abspath(os.path.normpath(google_creds_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"❌ Google credentials file not found at: {abs_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path
    speech_client = speech.SpeechClient()
    logging.info(f"✅ Loaded Google credentials from file: {abs_path}")
else:
    raise RuntimeError("❌ No Google Cloud credentials found. Set GOOGLE_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS")

# Initialize services
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ScrapeRequest(BaseModel):
    pages: List[str]

class LLMRequest(BaseModel):
    prompt: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logging.info("📥 /transcribe/ endpoint hit")
    file_location = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(await file.read())
            file_location = temp_file.name

        logging.info(f"🌀 Temporary file at: {file_location}")

        # Read audio content
        with open(file_location, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            language_code="en-US"
        )

        response = speech_client.recognize(config=config, audio=audio)
        transcription_text = " ".join([result.alternatives[0].transcript for result in response.results])
        logging.info("✅ Transcription completed")

        if background_tasks:
            background_tasks.add_task(save_transcription, file.filename, transcription_text)

        return {"transcription": transcription_text}

    except Exception as e:
        logging.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

    finally:
        if file_location and os.path.exists(file_location):
            os.remove(file_location)
            logging.info(f"🗑️ Deleted: {file_location}")

def save_transcription(filename, text):
    try:
        supabase.table("transcriptions").insert({"filename": filename, "text": text}).execute()
        logging.info("✅ Saved to Supabase")
    except Exception as e:
        logging.error(f"❌ Supabase insert error: {e}")

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
        raise HTTPException(status_code=500, detail="Fetch failed")

@app.get("/search")
async def search_google(q: str):
    logging.info(f"📥 /search → {q}")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": q, "key": API_KEY, "cx": SEARCH_ENGINE_ID, "num": 3}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return {"error": "Search failed"}

    data = response.json()
    results = [{"title": i["title"], "link": i["link"], "snippet": i["snippet"]} for i in data.get("items", [])]
    return {"results": results}

@app.get("/image-search")
async def search_images(q: str):
    logging.info(f"📥 /image-search → {q}")
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
        return JSONResponse(status_code=response.status_code, content={"error": "Image search failed"})

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
    logging.info(f"📥 /scrape/ with pages: {request.pages}")
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
        logging.error(f"❌ Scrape failed: {e}")
        return {"url": url, "error": "Failed to fetch"}

@app.post("/llm/")
async def query_llm(request: LLMRequest):
    logging.info(f"📥 /llm/ prompt: {request.prompt[:100]}...")
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": request.prompt}],
            model="llama3-70b-8192"
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        logging.error(f"❌ Groq failed: {e}")
        raise HTTPException(status_code=500, detail=f"Groq request failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
