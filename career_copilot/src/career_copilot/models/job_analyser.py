from pydantic import BaseModel

class JobAnalysis(BaseModel):
    summary: str
    skills: list[str]
    technologies: list[str]
    experience: str
    education: str
    languages: list[str]
    keywords: list[str]