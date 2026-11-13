
import logging

from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.models.candidate_profile import CandidateProfile
from career_copilot.models.score import CompatibilityScore
from career_copilot.llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)

class CompatibilityScorer:

    def __init__(self, client: MistralClient)-> None:
        self.client = client

    def score(self, job_analysis: JobAnalysis, candidate_profile : CandidateProfile, system_prompt: str)-> CompatibilityScore:
        """Generate a compatibility score."""
        prompt = self._build_prompt(job_analysis, candidate_profile)

        response = self.client.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            json_output=True,
        )

        llm_response = response["llm_response"]

        logger.info(f"Compatibility score generated with success.")
        return CompatibilityScore.model_validate(llm_response)
    
    def _build_prompt(self, job_analysis: JobAnalysis, candidate_profile: CandidateProfile) -> str:
        return f"""
        Evaluate the compatibility between the candidate and the job offer.

        Job analysis:
        {job_analysis.model_dump_json()}

        Candidate profile:
        {candidate_profile.model_dump_json()}

        Return only valid JSON matching the CompatibilityScore schema.
        """