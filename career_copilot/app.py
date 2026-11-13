import gradio as gr

from career_copilot.ui.features.job_analysis import render_job_analysis
from career_copilot.ui.features.candidate_profile import build_candidate_profile
from career_copilot.ui.features.scoring import render_compatibility_score
from career_copilot.ui.features.documents_generation import render_generated_documents


with gr.Blocks(title="CareerCopilot") as demo:

    # Shared states
    job_offer_state = gr.State(value=None)
    candidate_profile_state = gr.State(value=None)
    job_analysis_state = gr.State(value=None)
    scoring_state = gr.State(value=None)

    gr.Markdown("# CareerCopilot")

    # STEP 1 - Job Offer
    with gr.Column() as job_analysis_section:
        render_job_analysis(job_offer_state, job_analysis_state)

    gr.Markdown("---")

    # STEP 2 - Candidate Profile
    with gr.Column() as candidate_profile_section:
        build_candidate_profile(candidate_profile_state)

    gr.Markdown("---")

    # STEP 3 - Scoring
    with gr.Column() as scoring_section:
        render_compatibility_score(job_analysis_state, candidate_profile_state)

    gr.Markdown("---")

    # STEP 4 - Documents generation
    with gr.Column() as documents_generation_section:
        render_generated_documents(job_analysis_state, candidate_profile_state)

    gr.Markdown("---")


if __name__ == "__main__":
    demo.launch()