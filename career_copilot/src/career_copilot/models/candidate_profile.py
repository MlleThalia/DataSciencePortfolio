
from pydantic import BaseModel, Field


class Experience(BaseModel):
    """Une expérience professionnelle, y compris un stage."""

    company: str
    position: str
    start_date: str
    end_date: str | None = None
    location: str | None = None
    current: bool = False

    description: str

    technologies: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """Une formation académique."""

    institution: str
    degree: str
    field: str

    start_date: str
    end_date: str | None = None
    location: str | None = None


class Project(BaseModel):
    """Un projet académique ou personnel figurant sur le CV."""

    name: str
    description: str

    technologies: list[str] = Field(default_factory=list)
    start_date: str
    end_date: str | None = None

    url: str | None = None

class SkillGroup(BaseModel):
    category: str
    skills: list[str]

class Certification(BaseModel):
    name: str
    issuer: str

    date: str | None = None


class Language(BaseModel):
    name: str
    level: str


class ExtracurricularActivity(BaseModel):
    """Une activité associative ou extra-académique figurant sur le CV."""

    organization: str
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    achievements: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """Informations structurées extraites du CV d'un candidat."""

    # Identity
    first_name: str
    last_name: str

    email: str
    phone: str

    location: str

    linkedin: str | None = None
    github: str | None = None
    website: str | None = None

    # Presentation
    title: str
    summary: str

    # Skills
    skills: list[SkillGroup] = Field(default_factory=list)

    # Professional background
    experiences: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)

    # Others
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    extracurricular_activities: list[ExtracurricularActivity] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
