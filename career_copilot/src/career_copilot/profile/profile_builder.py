
import logging

from career_copilot.models.candidate_profile import CandidateProfile
from career_copilot.llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)

class ProfileBuilder:

    def __init__(self, client: MistralClient)-> None:
        self.client = client

    def build_profile(self, candidate_profile: str, system_prompt: str)-> CandidateProfile:
        """Builds a profile."""
        prompt = self._build_prompt(candidate_profile)

        response = self.client.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            json_output=True,
        )

        logging.info(f"Candidate profile generated with success")
        return CandidateProfile.model_validate(response)
    
    def _build_prompt(self, candidate_profile: str) -> str:
        return f"""
        Analyze the following candidate resume.

        Extract the candidate's professional information and structure it
        according to the CandidateProfile model.

        Resume:
        {candidate_profile}
    """