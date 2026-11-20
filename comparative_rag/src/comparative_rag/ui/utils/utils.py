import os
from dotenv import load_dotenv
import uuid
from datetime import datetime, timezone
from langchain_mistralai import ChatMistralAI
from llama_index.llms.mistralai import MistralAI

from comparative_rag.llm.mistral_client import MistralClient

load_dotenv()

def create_run():
    try : 
        run = {
            "run_id": str(uuid.uuid4()),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "v1.0"
        }
        return run
    except Exception as e :
        print(f"Error when creating the Run : {e}.")

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

def chat_mistral_ai():
    MISTRAL_API_KEY = os.getenv("MISTRALAI_API_KEY")
    MODEL = os.getenv("MISTRAL_MODEL")
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRALAI_API_KEY non défini")
    if not MODEL:
        raise RuntimeError("MODEL non défini")
    return ChatMistralAI(model=MODEL,
                         api_key=MISTRAL_API_KEY,
                         temperature=0
                        )

def mistral_ai():
    MISTRAL_API_KEY = os.getenv("MISTRALAI_API_KEY")
    MODEL = os.getenv("MISTRAL_MODEL")
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRALAI_API_KEY non défini")
    if not MODEL:
        raise RuntimeError("MODEL non défini")
    return MistralAI(model=MODEL,
                         api_key=MISTRAL_API_KEY,
                         temperature=0
                        )