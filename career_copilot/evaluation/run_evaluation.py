
import json
from typing import Mapping, Any
import os
import uuid
from datetime import datetime, timezone
import logging
from pathlib import Path
from dotenv import load_dotenv
import time

logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_DIR = Path("traces/")

MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.DEBUG)

from deterministic_evaluator import evaluate_job_analysis
from llm_judge import LlmAsJudge
from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.models.job_offer import JobOffer
from career_copilot.models.tracing import Run
from career_copilot.generators.job_analyzer import JobAnalyzer
from career_copilot.llm.mistral_client import MistralClient

# loading evaluation dataset and ground_truth
eval_dataset_path = "career_copilot/evaluation/eval_dataset.json" 
ground_truth_path = "career_copilot/evaluation/ground_truth.json" 

with open(eval_dataset_path, "r", encoding="utf-8") as file:
    eval_datasets = json.load(file)

with open(ground_truth_path, "r", encoding="utf-8") as file:
    ground_truth = json.load(file)

def create_run():
    run = {
        "run_id": str(uuid.uuid4()),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v1.0"
    }
    return run

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

mistral_client = MistralClient()

def predict(eval_datasets : list, system_prompt: str)->list[JobAnalysis]:
    "Runs predictions on evaluation datasets"
    job_analyzer = JobAnalyzer(client=mistral_client)
    predicted_job_analysis = []
    for dataset in eval_datasets :
        run = Run(**create_run())
        job_offer = JobOffer(**dataset)
        prediction = {"id" : job_offer.id} | job_analyzer.analyze(job_offer, run, system_prompt).model_dump()
        predicted_job_analysis.append(prediction)
        response = mistral_client.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
                attempt_number=1,
                json_output=True,
            )
        
        logger.debug(f"Job {job_offer.id} has been predicted with success!")
    
    return predicted_job_analysis

def run_evaluate(ground_truth: list[Mapping[str, Any]], predicted: list[JobAnalysis])->list[Mapping[str, Any]]:
    """Runs deterministic evaluation"""
    results = []
    llm_as_judge = LlmAsJudge(client=mistral_client)
    for ground_truth_item in ground_truth :
        job_id = ground_truth_item["job_id"]
        predicted_item = next(
                item for item in predicted
                if item["id"] == ground_truth_item["job_id"]
            )
        
        ground_truth_data = ground_truth_item["expected"].copy()
        predicted_data = predicted_item.copy()

        ground_truth_data.pop("job_id", None)
        predicted_data.pop("id", None)
        ground_truth_job_analysis = JobAnalysis(**ground_truth_data)
        predicted_job_analysis = JobAnalysis(**predicted_data)
        result = {"id" : job_id} | evaluate_job_analysis(ground_truth_job_analysis, predicted_job_analysis) | {"summary" : llm_as_judge.judge(ground_truth_job_analysis.summary, predicted_job_analysis.summary)}
        results.append(result)
        logger.debug(f"Job {job_id} has been evaluated with success!")

    return results

if __name__ == "__main__" :
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prompt_path', help="prompt_path")
    parser.add_argument('--file_name', help="evaluation results output path")

    args = parser.parse_args()
    prompt_path= args.prompt_path
    file_name = args.file_name

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    predictions = predict(eval_datasets, prompt)
    results = run_evaluate(ground_truth, predictions)

    if file_name : 
        with open(file_name, "w") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    print(json.dumps(results, ensure_ascii=False, indent=4))