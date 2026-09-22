# CareerCopilot

CareerCopilot is an AI-powered job search assistant designed to support and accelerate the job application process.

The project aims to automate several stages of a job search workflow, from collecting job offers to analyzing them and generating tailored application documents.

> **Project status:** Work in progress  
> This repository currently focuses on job offer collection, LLM-based job analysis, prompt experimentation, and LLM evaluation. Additional application and API features are planned.

---

## Overview

Searching and applying for jobs involves several repetitive tasks:

- collecting relevant job offers;
- extracting useful information from job descriptions;
- identifying required skills and technologies;
- adapting a candidate profile to a specific position;
- generating tailored resumes and cover letters;
- reviewing generated content before applying;
- tracking applications.

CareerCopilot aims to bring these steps together into a single workflow while keeping the candidate in control of the final application.

---

## Current Features

### Job offer collection

CareerCopilot currently includes a LinkedIn collector based on Playwright.

The collector extracts information such as:

- job title;
- company;
- location;
- contract type;
- publication date;
- description;
- salary;
- URL;
- source.

Collected job offers are stored locally.

### Job analysis with LLMs

Job offers can be analyzed using an LLM through the Mistral API.

The analysis extracts structured information such as:

- summary;
- skills;
- technologies;
- experience requirements;
- education requirements;
- languages;
- keywords.

The generated output is validated using Pydantic models.

### Retry and tracing

The LLM workflow includes basic reliability mechanisms:

- retry handling;
- HTTP error tracking;
- JSON parsing error tracking;
- Pydantic validation error tracking;
- latency measurement;
- token usage tracking;
- individual LLM call tracing.

Each execution can therefore retain information about the different attempts made during the generation process.

### LLM evaluation

CareerCopilot includes an evaluation framework for measuring the quality of generated job analyses.

The current evaluation approach combines:

#### Deterministic evaluation

Structured list fields are evaluated using:

- Precision;
- Recall;
- F1-score.

This currently applies to:

- skills;
- technologies;
- languages;
- keywords.

The evaluation ignores the order of list elements.

#### LLM-as-a-judge

An LLM-based evaluator is being developed to evaluate fields where semantic similarity is more relevant than exact matching.

The first target is the generated `summary`.

The judge compares:

- the reference summary (`ground truth`);
- the generated summary (`prediction`).

This approach is intended to complement deterministic metrics rather than replace them.

### Prompt experimentation

Different versions of prompts can be evaluated on the same evaluation dataset.

The objective is to measure the impact of prompt modifications while keeping the evaluation conditions consistent.

---

## Project Structure

career_copilot/
├── evaluation/
│   ├── eval_dataset.json
│   ├── ground_truth.json
│   ├── evaluator.py
│   ├── run_evaluation.py
│   ├── reports/
│   └── baseline.json
│
├── src/
│   └── career_copilot/
│       ├── llm/
│       ├── collectors/
│       ├── database/
│       ├── models/
│       ├── renderers/
│       ├── generators/
│       └── profile/
│
├── data/
│
├── prompts/
│   ├── profile/
│   ├── job_analyzer/
│   ├── resume/
│   └── cover_letter/
│
├── templates/
│
├── pyproject.toml
├── requirements.txt
└── README.md

## Tech Stack

### Core

- Python
- Pydantic
- SQLite
- Requests

### AI & LLM

- Mistral API
- Large Language Models (LLMs)
- Prompt Engineering
- LLM Evaluation

### Data & Documents

- pdfplumber
- Jinja2
- LaTeX

### Automation

- Playwright

### Development

- Git
- GitHub

## Installation

### Clone the repository

```bash
git clone <https://github.com/MlleThalia/DataSciencePortfolio>
cd career_copilot
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment:

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install Playwright browsers

```bash
playwright install
```

## Configuration

CareerCopilot uses environment variables for API credentials and configuration.

Create a `.env` file at the root of the project:

```env
MISTRALAI_API_URL=https://api.eu.mistral.ai/v1/chat/completions
MISTRALAI_API_KEY=your_api_key
MISTRAL_MODEL=ministral-14b-2512
MAX_RETRY=2
```

The `.env` file must not be committed to the repository.

## Roadmap

### LLM Evaluation

- [x] Implement Precision, Recall and F1-score
- [x] Create an evaluation dataset
- [x] Implement LLM tracing
- [x] Compare multiple prompt versions
- [ ] Implement LLM-as-a-judge for summary evaluation
- [ ] Combine deterministic metrics and LLM judge results
- [ ] Implement automated regression testing
- [ ] Expand the evaluation dataset
- [ ] Track evaluation results across prompt and model versions

### Candidate Profile & Application Generation

- [ ] Improve candidate profile extraction
- [ ] Generate tailored resumes
- [ ] Generate tailored cover letters
- [ ] Render generated documents
- [ ] Add human review before applications
- [ ] Track applications

### LLM & Retrieval

- [ ] Add Retrieval-Augmented Generation (RAG)
- [ ] Explore Hugging Face tooling
- [ ] Explore LangChain
- [ ] Explore advanced LLM evaluation techniques
- [ ] Explore agent and workflow architectures

### API

- [ ] Add a FastAPI API
- [ ] Expose job collection and analysis services
- [ ] Add API validation and authentication
- [ ] Add automated API tests

### Engineering & Production

- [ ] Improve Docker support
- [ ] Add CI/CD
- [ ] Improve logging, tracing and observability
- [ ] Add monitoring
- [ ] Explore cloud deployment
- [ ] Explore AWS

### User Interface

- [ ] Add a user interface for job offers
- [ ] Add candidate profile management
- [ ] Add human-in-the-loop review
- [ ] Connect the UI to the API

## Project Goal

The long-term goal of CareerCopilot is to build an AI-assisted job application workflow that automates repetitive tasks while keeping the candidate in control of the final application.

The project also serves as a practical environment for exploring production-oriented AI engineering concepts, including:

- LLM evaluation
- Prompt Engineering
- Structured outputs
- LLM tracing
- API development
- Testing
- CI/CD
- Monitoring
- Deployment