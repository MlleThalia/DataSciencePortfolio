from pydantic import BaseModel, Field

from career_copilot.models.candidate_profile import (
    Certification,
    Education,
    Experience,
    ExtracurricularActivity,
    Language,
    Project,
    SkillGroup,
)


class Resume(BaseModel):
    """Represents a tailored resume generated for a specific job offer."""

    # En-tête du template resume.tex
    first_name: str
    last_name: str
    email: str
    phone: str
    location: str
    linkedin: str | None = None
    github: str | None = None

    title: str
    summary: str

    # Sections du template resume.tex
    education: list[Education] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)
    extracurricular_activities: list[ExtracurricularActivity] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)

    # Sections optionnelles, à afficher seulement si le template les prend en charge.
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
