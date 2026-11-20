import os
import time
import requests
from typing import Mapping, Any

import logging

from comparative_rag.models.tracing import LlmCall


logger = logging.getLogger(__name__)

class MistralClient:

    def __init__(self, model: str = None, api_url: str = None, api_key: str = None):
        self.model = model or os.environ.get("MISTRAL_MODEL")
        self.api_url = api_url or os.environ.get("MISTRALAI_API_URL")
        self.api_key = api_key or os.environ.get("MISTRALAI_API_KEY")

        if not self.api_url or not self.api_key:
            raise RuntimeError("MISTRALAI_API_URL and MISTRALAI_API_KEY must be defined in environment")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        attempt_number: int = 1
    )->Mapping[str, Any]:
        """Builds the mistral client and returns the llm answer."""
        
        errors = {}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format" : {"type": "text"}
        }

        start = time.perf_counter()

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=500,
        )

        latency = time.perf_counter() - start

        llm_trace = LlmCall(
                        attempt_number=attempt_number,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        latency=latency,
                        errors = errors
                    )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            errors["http_error"] = str(e)
            llm_trace.response = response.text
            llm_trace.errors = errors
            return {
                    "llm_response": response.text,
                    "trace" : llm_trace
                }

        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        logger.info(
            "LLM call | model=%s | prompt=%d | completion=%d | total=%d | latency=%.2fs",
            self.model,
            prompt_tokens,
            completion_tokens,
            usage.get("total_tokens", 0),
            latency,
        )

        answer = data["choices"][0]["message"]["content"]

        llm_trace.prompt_tokens=prompt_tokens
        llm_trace.completion_tokens=completion_tokens

        return {
            "llm_response": answer,
            "trace" : llm_trace
        }