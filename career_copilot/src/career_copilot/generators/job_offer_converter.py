
import logging

from career_copilot.models.job_offer import JobOffer
from career_copilot.llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)

class JobOfferConverter:

    def __init__(self, client: MistralClient)-> None:
        self.client = client

    def convert(self, job: str, system_prompt: str)-> JobOffer:
        """Convert a str job offer in JobOffer."""
        prompt = self._build_prompt(job)

        response = self.client.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            json_output=True,
        )

        llm_response = response["llm_response"]

        logger.info(f"JobOffer generated with success.")
        return JobOffer.model_validate(llm_response)
    
    def _build_prompt(self, job: str) -> str:
        return f"""
        Convert the following job offer into a JobOffer Pydantic object.

        Job:
        {job}

        Return only valid JSON matching the JobOffer schema.
        """