# CareerCopilot

**CareerCopilot** is an AI-powered job application assistant designed to help candidates analyze job opportunities, evaluate their compatibility with job requirements, and generate personalized application documents.

The project combines Large Language Models (LLMs), structured information extraction, candidate evaluation, and automated document generation to streamline the job application process.

## Overview

CareerCopilot provides an end-to-end workflow for preparing job applications.

Users can:
- Import a job offer in text format.
- Extract and analyze job requirements using an LLM.
- Upload their CV and extract a structured candidate profile.
- Evaluate candidate-job compatibility through an AI-based scoring system.
- Generate a personalized CV and a tailored cover letter.
- Download the generated documents in PDF and LATEX format.

The application relies on structured outputs validated with Pydantic and uses LaTeX to generate professional PDF documents.

## Features

### 1. Job Offer Analysis

- Import job descriptions from pasted text.
- Convert job descriptions into structured `JobOffer` objects.
- Analyze job requirements using the Mistral API.
- Extract key information:
  - Job summary
  - Required skills
  - Technologies
  - Experience requirements
  - Education requirements
  - Languages
  - Keywords
- Validate structured outputs using Pydantic.

### 2. Candidate Profile Extraction

- Upload a candidate CV in PDF format.
- Extract personal and professional information using an LLM.
- Build a structured `CandidateProfile` object.
- Organize extracted information into:
  - Personal information
  - Professional summary
  - Skills
  - Work experience
  - Education
  - Projects
  - Certifications
  - Languages
  - Extracurricular activities

### 3. Compatibility Scoring

Evaluate the compatibility between a candidate profile and a job offer.

The LLM generates:
- Overall compatibility score.
- Skills compatibility score.
- Experience compatibility score.
- Education compatibility score.
- Main strengths.
- Areas of improvement.
- Overall evaluation summary.

### 4. Application Document Generation

Generate two personalized application documents:

- **Personalized CV:** adapts the candidate's existing profile to the requirements of the selected position.
- **Cover Letter:** generates a tailored cover letter based on the candidate's profile and job analysis.

Document generation workflow:

1. Generate structured document content using an LLM.
2. Render the content into LaTeX templates.
3. Compile LaTeX files into PDF documents.
4. Provide PDF downloads through the Gradio interface.

LaTeX source files are also retained for further editing.

### 5. LLM Reliability, Tracing and Evaluation

- Structured output validation using Pydantic.
- LLM request tracing.
- Latency and token usage monitoring.
- Prompt experimentation and comparison.
- Evaluation datasets with reference outputs (ground truth).
- Deterministic evaluation using Precision, Recall and F1-score.
- LLM-as-a-Judge for semantic evaluation of generated summaries.

## User Interface

CareerCopilot provides an interactive interface built with **Gradio Blocks**.

The application follows a sequential workflow:

1. Job offer import and analysis.
2. Candidate CV upload and profile extraction.
3. Candidate-job compatibility scoring.
4. Personalized application document generation.

The interface displays the analysis results, candidate information, compatibility scores, and downloadable application documents.

## Project Structure

```text
career_copilot/
│
├── src/
│   └── career_copilot/
│       ├── collectors/
│       ├── database/
│       ├── generators/
|       ├── llm/
│       ├── models/
|       ├── profile/
│       ├── renderers/
│       └── ui/
│           └── features/
│
├── evaluation/
│   ├── eval_dataset.json
│   ├── ground_truth.json
│   ├── deterministic_evaluator.py
│   ├── llm_judge.py
│   ├── run_evaluation.py
│   ├── baseline.json
│   └── reports/
│
├── prompts/
├── templates/
├── data/
├── generated_documents/
├── tests/
├── traces/
│
├── app.py
├── .env
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| LLM | Mistral API |
| Data validation | Pydantic |
| User interface | Gradio Blocks |
| Database | SQLite |
| PDF processing | PDF extraction libraries |
| Document generation | LaTeX, pdfLaTeX |
| Web automation | Playwright |
| LLM evaluation | Precision, Recall, F1-score, LLM-as-a-Judge |
| Development | Git, GitHub |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MlleThalia/DataSciencePortfolio.git
cd DataSciencePortfolio/career_copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browsers

```bash
playwright install
```

### 5. Install LaTeX

CareerCopilot uses `pdflatex` to compile generated LaTeX documents into PDFs.

Install a LaTeX distribution according to your operating system:

- **macOS:** MacTeX or BasicTeX.
- **Linux:** TeX Live.
- **Windows:** MiKTeX.

On macOS, with Homebrew:

```bash
brew install --cask mactex-no-gui
```

Verify the installation:

```bash
pdflatex --version
```

Make sure `pdflatex` is accessible from your system PATH.

## Configuration

Create a `.env` file at the project root:

```env
MISTRALAI_API_URL=https://api.eu.mistral.ai/v1/chat/completions
MISTRALAI_API_KEY=your_api_key
MISTRAL_MODEL=your_model
MAX_RETRY=2
```

Never commit API credentials to the repository.

## Running the Application

Install the project in editable mode if required by your configuration:

```bash
pip install -e .
```

Launch the Gradio application from the project root:

```bash
python app.py
```

## Project Milestones

- [x] Job offer collection and SQLite storage.
- [x] LLM-based job offer analysis.
- [x] Structured output validation with Pydantic.
- [x] Candidate CV extraction and profile structuring.
- [x] Candidate-job compatibility scoring.
- [x] Personalized CV generation.
- [x] Cover letter generation.
- [x] LaTeX document rendering and PDF compilation.
- [x] LLM tracing and evaluation.
- [x] Gradio interface.
- [x] Integration of the application workflow.
- [x] Final end-to-end testing and documentation.

## Project Goal

CareerCopilot is a practical AI Engineering project focused on the development of an LLM-powered application.

It explores several important aspects of AI Engineering:

- Prompt engineering and LLM integration.
- Structured information extraction.
- Pydantic-based data validation.
- LLM evaluation and monitoring.
- Automated document generation.
- Application development with Gradio.