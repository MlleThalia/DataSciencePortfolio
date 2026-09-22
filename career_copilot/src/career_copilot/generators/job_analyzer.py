
import logging
from pydantic import ValidationError
from pathlib import Path
import json
import time
import os
from dotenv import load_dotenv

from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.models.job_offer import JobOffer
from career_copilot.models.tracing import Run, Trace
from career_copilot.llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_DIR = Path("traces/")

MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))

def save_json(data, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")

    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"✓ Saved {path}")

class JobAnalyzer:

    def __init__(self, client: MistralClient)-> None:
        self.client = client

    def analyze(self, job : JobOffer, run : Run, system_prompt : str)-> JobAnalysis | str | None:

        prompt = self._build_prompt(job)
        llm_calls = []
        job_analysis = None

        for attempt_number in range(1, MAX_RETRY+2):

            response = self.client.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                attempt_number=attempt_number,
                json_output=True,
            )

            logger.info(f"Job analysis | {attempt_number} attempt.")
            llm_response = response["llm_response"]
            llm_trace = response["trace"]

            if "http_error" in llm_trace.errors or "json_decode_error" in llm_trace.errors :
                llm_calls.append(llm_trace)
                time.sleep(1)
                continue
            else : 
                try :
                    job_analysis = JobAnalysis.model_validate(llm_response)
                    llm_trace.model_valid = True
                    llm_calls.append(llm_trace)
                    break

                except ValidationError as e :
                    llm_trace.model_valid = False
                    errors = llm_trace.errors
                    errors["validation_error"] = e.errors()
                    llm_trace.errors = errors
                    llm_calls.append(llm_trace)
                    time.sleep(1)
                    continue
        
        trace = Trace(run=run, llm_calls=llm_calls)
        save_json(trace, f"trace_{run.run_id}.json")
        return job_analysis
    
    def _build_prompt(self, job: JobOffer) -> str:
        return f"""
        Analyze the following job offer.

        Title: {job.title}
        Company: {job.company}
        Location: {job.city}, {job.country}

        Description:
        {job.description}
        """