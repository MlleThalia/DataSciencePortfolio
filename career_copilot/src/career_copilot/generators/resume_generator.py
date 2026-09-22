
import logging

from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.models.candidate_profile import CandidateProfile
from career_copilot.models.resume import Resume
from career_copilot.llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)

class ResumeGenerator:

    def __init__(self, client: MistralClient)-> None:
        self.client = client

    def generate(self, analysis: JobAnalysis, candidate_profile: CandidateProfile, system_prompt: str)-> Resume:

        prompt = self._build_prompt(analysis, candidate_profile)

        response = self.client.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            json_output=True,
        )

        logger.info(f"Resume generated with success.")
        return Resume.model_validate(response)
    
    def _build_prompt(
        self,
        analysis: JobAnalysis,
        candidate_profile: CandidateProfile,
    ) -> str:
        return f"""
        Create a tailored resume based on the following job analysis
        and candidate profile.

        Job analysis:
        {analysis.model_dump_json(indent=2)}

        Candidate profile:
        {candidate_profile.model_dump_json(indent=2)}
        """