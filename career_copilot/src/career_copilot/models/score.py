from pydantic import BaseModel, Field

class CompatibilityScore(BaseModel):

    overall_score: int

    skills_score: int
    experience_score: int
    education_score: int

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    summary: str