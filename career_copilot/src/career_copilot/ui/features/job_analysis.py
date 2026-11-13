import gradio as gr
from pathlib import Path

from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.generators.job_analyzer import JobAnalyzer
from career_copilot.generators.job_offer_converter import JobOfferConverter
from career_copilot.models.tracing import Run
from career_copilot.ui.utils.utils import build_llm_client, create_run

BASE_DIR = Path(__file__).resolve().parents[4]
JOB_ANALYZER_PROMPT_PATH = BASE_DIR / "prompts" / "job_analyzer" / "prompt_v1.txt"
JOB_CONVERTER_PROMPT_PATH = BASE_DIR / "prompts" / "job_offer_converter" / "prompt_v1.txt"

def render_job_analysis(job_offer_state, job_analysis_state):

    gr.Markdown("## Job Analysis")
    gr.Markdown("Paste a job offer description and generate an analysis.")

    job_text = gr.Textbox(
        label="Job Description",
        placeholder="Paste the complete job description here...",
        lines=15
    )

    job_analysis = gr.Button(
        "Analyze the job",
        variant="primary"
    )

    with gr.Group(visible = False) as job_analysis_group :
        gr.Markdown("## Job Analysis Results")

        summary = gr.Markdown()
        skills = gr.Markdown()
        technologies = gr.Markdown()
        experience = gr.Markdown()
        education = gr.Markdown()
        languages = gr.Markdown()
        keywords = gr.Markdown()

    job_analysis.click(
        fn=analyze_job,
        inputs=[job_text],
        outputs=[
            job_offer_state,
            job_analysis_state,
            summary,
            skills,
            technologies,
            experience,
            education,
            languages,
            keywords,
            job_analysis_group
        ],
        show_progress="full",
    )


def display_job_analysis(job_analysis: JobAnalysis):

    return (
        f"**Summary:**\n\n{job_analysis.summary}",
        f"**Skills:**\n\n{', '.join(job_analysis.skills)}",
        f"**Technologies:**\n\n{', '.join(job_analysis.technologies)}",
        f"**Experience:**\n\n{job_analysis.experience}",
        f"**Education:**\n\n{job_analysis.education}",
        f"**Languages:**\n\n{', '.join(job_analysis.languages)}",
        f"**Keywords:**\n\n{', '.join(job_analysis.keywords)}",
        gr.update(visible=True)
    )

def analyze_job(job_text):

    if job_text is "":
        raise gr.Error(
            "Please paste a job offer description first."
        )
    
    try :

        mistral_client = build_llm_client()
        run = Run(**create_run())
        job = convert_to_job_offer(job_text, mistral_client)

        with open(JOB_ANALYZER_PROMPT_PATH, "r") as prompt_file :
            system_prompt = prompt_file.read()

        # Lancer l'analyse uniquement si les deux données existent
        job_analyzer = JobAnalyzer(mistral_client)
        result = job_analyzer.analyze(
            job,
            run,
            system_prompt
        )

        return (job, result, *display_job_analysis(result))
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

def convert_to_job_offer(job_text, client): 
    try:
        job_offer_converter = JobOfferConverter(client)
        with open(JOB_CONVERTER_PROMPT_PATH, "r") as prompt_file :
            system_prompt = prompt_file.read()
        result = job_offer_converter.convert(
            job_text,
            system_prompt
        )
        return result

    except FileNotFoundError:
        raise gr.Error(
            "Application configuration error. Please contact support."
        )
    except gr.Error:
        raise
    except Exception:
        raise gr.Error(
            "Something went wrong during generation. Please try again."
        )