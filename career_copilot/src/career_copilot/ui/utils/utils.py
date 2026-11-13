import os
from dotenv import load_dotenv
import uuid
from datetime import datetime, timezone

from career_copilot.llm.mistral_client import MistralClient

load_dotenv()

def build_llm_client():
    
    MISTRAL_API_KEY = os.getenv("MISTRALAI_API_KEY")
    MODEL = os.getenv("MISTRAL_MODEL")
    MISTRAL_API_URL = os.getenv("MISTRALAI_API_URL")

    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRALAI_API_KEY non défini")

    if not MISTRAL_API_URL:
        raise RuntimeError("MISTRAL_API_URL non défini")

    if not MODEL:
        raise RuntimeError("MODEL non défini")

    return MistralClient(model=MODEL, api_url=MISTRAL_API_URL, api_key=MISTRAL_API_KEY)

def create_run():
    run = {
        "run_id": str(uuid.uuid4()),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v1.0"
    }
    return run