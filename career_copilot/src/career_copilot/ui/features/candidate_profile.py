import gradio as gr
from fastapi import UploadFile
from pathlib import Path

from career_copilot.profile.pdf_reader import read_pdf
from career_copilot.profile.profile_builder import ProfileBuilder
from career_copilot.models.candidate_profile import CandidateProfile

from career_copilot.ui.utils.utils import build_llm_client

BASE_DIR = Path(__file__).resolve().parents[4]
CANDIDATE_PROFILE_PROMPT_PATH = BASE_DIR / "prompts" / "profile" / "prompt_v1.txt"

def build_candidate_profile(candidate_profile_state):

    gr.Markdown("## Candidate Profile")
    gr.Markdown("Load your resume and build your profile.")

    file = gr.File(
        label="Candidate Resume",
        file_count="single",
        file_types=[".pdf"],
        type="filepath"
    )

    profile_builder_button = gr.Button(
        "Build Candidate Profile",
        variant="primary"
    )

    with gr.Group(visible=False) as candidate_profile_group:
        gr.Markdown("## Personal Information")
        identity = gr.Markdown()

        gr.Markdown("## Professional Summary")
        summary = gr.Markdown()

        gr.Markdown("## Education")
        education = gr.Dataframe(
        headers=[
                "Company",
                "Position",
                "Start Date",
                "End Date",
                "Location"
            ],
            interactive=False,
            wrap=True
        )

        gr.Markdown("## Experiences")
        experiences = gr.Dataframe(
        headers=[
                "Institution",
                "Degree",
                "Field",
                "Start Date",
                "End Date",
                "Location",
                "Description",
                "Technologies",
                "Achievements"
            ],
            interactive=False,
            wrap=True
        )

        gr.Markdown("## Projects")
        projects = gr.Dataframe(
        headers=[
                "Name",
                "Description",
                "Technologies",
                "Start Date",
                "End Date"
            ],
            interactive=False,
            wrap=True
        )

        gr.Markdown("## Skills")
        skills = gr.Dataframe(
            headers=["Category", "Skills"],
            interactive=False,
        )

    profile_builder_button.click(
        fn=build_profile,
        inputs=[file],
        outputs=[candidate_profile_state, identity, summary, skills, experiences, education, projects, candidate_profile_group],
        show_progress="full"
    )

def display_candidate_profile(profile: CandidateProfile):

    skills_data = [
        [group.category, ", ".join(group.skills)]
        for group in profile.skills
    ]

    experiences_data = [
    [
        exp.company,
        exp.position,
        exp.start_date,
        exp.end_date or ("Present" if exp.current else ""),
        exp.location or "",
        exp.description,
        ", ".join(exp.technologies),
        ", ".join(exp.achievements)
    ]
    for exp in profile.experiences
    ]

    education_data = [
        [
            education.institution,
            education.degree,
            education.field,
            education.start_date,
            education.end_date,
            education.location or ""
        ]
        for education in profile.education
    ]

    project_data = [
        [
            project.name,
            project.description,
            ", ".join(project.technologies),
            project.start_date,
            project.end_date
        ]
        for project in profile.projects
    ]


    identity = (
        f"**{profile.first_name} {profile.last_name}**\n\n"
        f"**{profile.title}**\n\n"
        f"Email : {profile.email}  \n"
        f"Phone : {profile.phone}  \n"
        f"Location : {profile.location}"
    )

    return (
        identity,
        profile.summary,
        skills_data,
        experiences_data,
        education_data, 
        project_data,
        gr.update(visible=True)
    )

def build_profile(file):
    """
    Build the candidate profile.
    """
    if file is None:
        raise gr.Error("Please upload a CV.")
    try :
        candidate_profile = read_pdf(UploadFile(file))
        with open(CANDIDATE_PROFILE_PROMPT_PATH, "r") as prompt_file:
            system_prompt = prompt_file.read()
        mistral_client = build_llm_client()
        profile_builder = ProfileBuilder(mistral_client)
        profile = profile_builder.build_profile(candidate_profile, system_prompt)

        return (profile, *display_candidate_profile(profile))

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