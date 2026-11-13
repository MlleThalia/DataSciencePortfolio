import gradio as gr
from pathlib import Path

from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.models.candidate_profile import CandidateProfile
from career_copilot.models.score import CompatibilityScore
from career_copilot.generators.compatibility_score import CompatibilityScorer
from career_copilot.ui.utils.utils import build_llm_client

BASE_DIR = Path(__file__).resolve().parents[4]
SCORING_PROMPT_PATH = BASE_DIR / "prompts" / "compatibility_score" / "prompt_v1.txt"

def render_compatibility_score(job_analysis_state, candidate_profile_state):

    gr.Markdown("## Compatibility Score")
    gr.Markdown(
    "Compare your profile with the job requirements to assess your overall compatibility. "
    "Identify your strengths, areas for improvement, and how well your qualifications match the position."
)


    scoring = gr.Button(
        "Score",
        variant="primary"
    )

    with gr.Group(visible = False) as score_group :
        gr.Markdown("## Score Results")

        overall_score= gr.Markdown()
        skills_score = gr.Markdown()
        experience_score = gr.Markdown()
        education_score = gr.Markdown()
        strengths = gr.Markdown()
        weaknesses = gr.Markdown()
        summary = gr.Markdown()

    scoring.click(
        fn=generate_score,
        inputs=[job_analysis_state, candidate_profile_state],
        outputs=[
            overall_score,
            skills_score,
            experience_score,
            education_score,
            strengths,
            weaknesses,
            summary,
            score_group
        ],
        show_progress="full"
    )

def display_compatibility_score(score: CompatibilityScore):

    return (
        f"**Overall compatibility:** {score.overall_score}/100",
        f"**Skills:** {score.skills_score}/100",
        f"**Experience:** {score.experience_score}/100",
        f"**Education:** {score.education_score}/100",
        f"**Strengths:**\n\n" + "\n".join(f"- {s}" for s in score.strengths),
        f"**Weaknesses:**\n\n" + "\n".join(f"- {w}" for w in score.weaknesses),
        f"**Summary:**\n\n{score.summary}",
        gr.update(visible=True)
    )

def generate_score(job_analysis, candidate_profile):

    if job_analysis is None:
        raise gr.Error(
            "Please paste a job offer description first."
        )
    
    if candidate_profile is None:
        raise gr.Error(
            "Please build your profile first."
        )
    
    try :
        mistral_client = build_llm_client()
        job_analysis = JobAnalysis.model_validate(job_analysis)
        candidate_profile = CandidateProfile.model_validate(candidate_profile)

        with open(SCORING_PROMPT_PATH, "r") as prompt_file :
            system_prompt = prompt_file.read()

        # Lancer l'analyse uniquement si les deux données existent
        scorer = CompatibilityScorer(mistral_client) 
        result = scorer.score(
            job_analysis,
            candidate_profile,
            system_prompt
        )

        return display_compatibility_score(result)

    except (FileNotFoundError, RuntimeError):
        raise gr.Error(
            "Application configuration error. Please contact support."
        )
    except gr.Error:
        raise
    except Exception:
        raise gr.Error(
            "Something went wrong during generation. Please try again."
        )