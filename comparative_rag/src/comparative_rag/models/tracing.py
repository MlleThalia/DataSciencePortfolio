
from datetime import datetime
from pydantic import BaseModel
from typing import Mapping, Any

class Run(BaseModel):
    run_id : str
    start_time : datetime
    pipeline_version : str = "v1.0"

class LlmCall(BaseModel):

    attempt_number : int
    system_prompt : str
    user_prompt : str
    json_requested : bool = True
    prompt_tokens : int | None = None
    completion_tokens : int | None = None
    latency : float | None = None
    json_valid : bool | None = None
    model_valid : bool | None = None
    response : Any = None
    errors : Mapping[str, Any] | None = None


class Trace(BaseModel):
    run: Run
    llm_calls: list[LlmCall]