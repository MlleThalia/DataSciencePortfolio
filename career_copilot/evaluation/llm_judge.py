
from career_copilot.llm.mistral_client import MistralClient

class LlmAsJudge():

    def __init__(self, client : MistralClient):
        self.client = client

    def judge(self, expected_summary : str, predicted_summary : str)-> dict | None:
        """Compares the expected summary to the predicted one and gives a score."""

        prompt = self._build_prompt(expected_summary, predicted_summary)

        response = self.client.generate(
            system_prompt="",
            user_prompt=prompt,
            json_output=True,
        )

        llm_response = response["llm_response"]
        llm_trace = response["trace"]

        if llm_trace.errors and("http_error" in llm_trace.errors or "json_decode_error" in llm_trace.errors) :
            return None
        else:
            return llm_response

    def _build_prompt(self, expected_summary : str, predicted_summary : str) -> str:
        return f"""
            Compare the expected summary with the predicted summary and evaluate how well the predicted summary matches the expected one.

            Expected summary:
            {expected_summary}

            Predicted summary:
            {predicted_summary}

            Give a score from 1 to 5, where:
            - 1 = very poor match
            - 2 = poor match
            - 3 = acceptable match
            - 4 = good match
            - 5 = excellent match

            The score should consider:
            - factual consistency with the expected summary
            - coverage of the main information
            - absence of unsupported or invented information
            - overall semantic similarity

            Return a valid JSON object with:
            - "score": an integer from 1 to 5
            - "reason": a concise explanation of why you gave this score

            Output:
            {{
                "score": 4,
                "reason": "The predicted summary covers the main information from the expected summary and does not introduce unsupported information."
            }}
            """