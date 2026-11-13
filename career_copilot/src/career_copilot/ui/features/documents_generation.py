import gradio as gr
import subprocess
import shutil
from pathlib import Path

TEMP_EXTENSIONS = [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"]

from career_copilot.models.candidate_profile import CandidateProfile
from career_copilot.models.job_analyser import JobAnalysis
from career_copilot.generators.resume_generator import ResumeGenerator
from career_copilot.generators.cover_letter_generator import CoverLetterGenerator
from career_copilot.llm.mistral_client import MistralClient
from career_copilot.renderers.resume_renderer import ResumeRenderer
from career_copilot.renderers.cover_letter_renderer import CoverLetterRenderer
from career_copilot.ui.utils.utils import build_llm_client

BASE_DIR = Path(__file__).resolve().parents[4]
RESUME_GENERATOR_PROMPT_PATH = BASE_DIR / "prompts" / "resume" / "prompt_v1.txt"
RESUME_TEMPLATE_PATH = BASE_DIR / "templates" / "resume.tex"
COVER_LETTER_GENERATOR_PROMPT_PATH = BASE_DIR / "prompts" / "cover_letter" / "prompt_v1.txt"
COVER_LETTER_TEMPLATE_PATH = BASE_DIR / "templates" / "cover_letter.tex"

def render_generated_documents(job_analysis_state, candidate_profile_state):
    
    gr.Markdown("## Resume and Cover Letter")

    gr.Markdown("Generate a suitable resume and cover letter based on the job analysis and your profile.")

    documents_generation_button = gr.Button(value='Generate Resume and Cover Letter', variant='primary')

    with gr.Group(visible=False) as generated_documents_group:
        gr.Markdown("## Generated documents")
        pdf_resume_download = gr.File(label="resume.pdf")
        latex_resume_download = gr.File(label="resume.tex")
        pdf_cover_letter_download = gr.File(label="cover_letter.pdf")
        latex_cover_letter_download = gr.File(label="cover.letter.tex")

    documents_generation_button.click(fn = generate_documents,
                                       inputs=[job_analysis_state, candidate_profile_state],
                                       outputs=[latex_resume_download, pdf_resume_download, latex_cover_letter_download, pdf_cover_letter_download, generated_documents_group],
                                       show_progress="full"
                                       )


def generate_documents(job_analysis, candidate_profile) :

    if job_analysis is None:
        raise gr.Error("Please paste a job description firts.")
    
    if candidate_profile is None : 
        raise gr.Error("Please upload a CV.")
    
    try : 
        mistral_client = build_llm_client()
    except RuntimeError : 
        raise gr.Error(
            "Application configuration error. Please contact support."
        )
    job_analysis = JobAnalysis.model_validate(job_analysis)
    candidate_profile = CandidateProfile.model_validate(candidate_profile)

    resume = generate_resume(job_analysis, candidate_profile, mistral_client)
    cover_letter = generate_cover_letter(job_analysis, candidate_profile, mistral_client)

    resume_renderer = ResumeRenderer(RESUME_TEMPLATE_PATH)
    cover_letter_renderer = CoverLetterRenderer(COVER_LETTER_TEMPLATE_PATH)

    latex_resume_path = BASE_DIR / "generated_documents" / "resume" /"resume.tex"
    if not latex_resume_path.exists():
        raise FileNotFoundError(latex_resume_path)
    
    latex_resume = resume_renderer.render(candidate_profile, resume)
    resume_renderer.save(latex_resume, latex_resume_path)
    pdf_resume_path = compile_latex(latex_resume_path)

    latex_cover_letter_path = BASE_DIR / "generated_documents" / "cover_letter" / "cover_letter.tex"
    if not latex_cover_letter_path.exists():
        raise FileNotFoundError(latex_cover_letter_path)
    
    latex_cover_letter = cover_letter_renderer.render(cover_letter)
    cover_letter_renderer.save(latex_cover_letter, latex_cover_letter_path)
    pdf_cover_letter_path = compile_latex(latex_cover_letter_path)

    return str(latex_resume_path), str(pdf_resume_path), str(latex_cover_letter_path), str(pdf_cover_letter_path), gr.update(visible=True)


def generate_resume(job_analysis : JobAnalysis, candidate_profile : CandidateProfile, mistral_client : MistralClient):
    """Generates a resume."""
    try:
        with open(RESUME_GENERATOR_PROMPT_PATH, 'r') as prompt_file : 
            system_prompt = prompt_file.read()

        resume_generator = ResumeGenerator(mistral_client)
        resume = resume_generator.generate(job_analysis, candidate_profile, system_prompt)
        return resume

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

def generate_cover_letter(job_analysis : JobAnalysis, candidate_profile : CandidateProfile, mistral_client : MistralClient):
    """Generates a cover letter."""
    try :   
        with open(COVER_LETTER_GENERATOR_PROMPT_PATH, 'r') as prompt_file : 
            system_prompt = prompt_file.read()

        cover_letter_generator = CoverLetterGenerator(mistral_client)
        cover_letter = cover_letter_generator.generate(job_analysis, candidate_profile, system_prompt)
        return cover_letter
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
    

def clean_temp_files(directory: Path):

    for file in directory.iterdir():
        if file.is_file() and any(
            file.name.endswith(ext) for ext in TEMP_EXTENSIONS
        ):
            file.unlink()

def compile_latex(tex_path: str) -> Path:

    # Vérifier la présence de pdflatex
    if not shutil.which("pdflatex"):
        raise RuntimeError("pdflatex is not installed.")

    tex_path = Path(tex_path).resolve()

    # Vérifier l'existence du fichier
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    if not tex_path.is_file():
        raise ValueError(f"{tex_path} is not a file.")

    clean_temp_files(tex_path.parent)

    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name
            ],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            encoding="latin-1",
            timeout=60,
            check=False
        )

    except subprocess.TimeoutExpired as error:
        raise RuntimeError("LaTeX compilation timed out.") from error

    except OSError as error:
        raise RuntimeError("Unable to execute pdflatex.") from error

    pdf_path = tex_path.with_suffix(".pdf")

    if result.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(
            f"LaTeX compilation failed:\n{result.stdout}\n{result.stderr}"
        )

    return pdf_path