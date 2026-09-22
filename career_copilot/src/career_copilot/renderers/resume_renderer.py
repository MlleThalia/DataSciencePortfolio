
from pathlib import Path
from jinja2 import Template
import logging

from career_copilot.models.candidate_profile import *
from career_copilot.models.resume import Resume

logger = logging.getLogger(__name__)
class ResumeRenderer:

    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path
        self.template = Template(
            template_path.read_text(encoding="utf-8"),
            comment_start_string="/*",
            comment_end_string="*/",
        )

    def render(self, profile: CandidateProfile, resume: Resume) -> str:
        # Header information comes from the candidate profile.
        header = self._render_header(profile, resume)

        # Tailored content comes from the generated resume.
        education = self._render_education(resume.education)
        experiences = self._render_experiences(resume.experiences)
        projects = self._render_projects(resume.projects)
        skills = self._render_skills(resume.skills)
        languages = self._render_languages(resume.languages)
        extracurricular_activities = self._render_extracurricular_activities(
            resume.extracurricular_activities
        )

        logger.debug(f"Rendering resume")
        return self.template.render(
            header=header,
            education=education,
            experiences=experiences,
            projects=projects,
            skills=skills,
            languages=languages,
            extracurricular_activities=extracurricular_activities,
        )
    
    def save(self, latex_content: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            latex_content,
            encoding="utf-8",
        )
        print(f"✓ Saved {output_path}")
    
    def _render_header(
        self,
        profile: CandidateProfile,
        resume: Resume,
    ) -> str:
        return rf"""
    \begin{{center}}
        {{\Large \scshape {profile.first_name} {profile.last_name}}} \\ \vspace{{1pt}}
        {{\large {self._escape_latex(resume.title)}}} \\ \vspace{{1pt}}
        {{\large {self._escape_latex(resume.summary)}}} \\ \vspace{{1pt}}

        \faLinkedin \hspace{{2pt}}
        \href{{https://www.linkedin.com/in/{profile.linkedin}/}}{{LinkedIn : {profile.linkedin}}} ~
        \faGithub \hspace{{2pt}}
        \href{{https://{profile.github}}}{{ {profile.github}}} \\ \vspace{{1pt}}

        \raisebox{{-0.05\height}}\faHome\ {profile.location}~
        \small \raisebox{{-0.1\height}}\faPhone\ {profile.phone} ~
        \href{{mailto:{profile.email}}}{{\raisebox{{-0.2\height}}\faEnvelope\ \underline{{{profile.email}}}}} \\

        \vspace{{-8pt}}
    \end{{center}}
    """

    def _render_education(self, education: list[Education]) -> str:
        entries = []

        for item in education:
            period = f"{item.start_date} -- {item.end_date or 'Present'}"

            entries.append(
                rf"""
                    \resumeSubheading
                    {{{self._escape_latex(item.institution)}}}{{{period}}}
                    {{{self._escape_latex(item.degree)} -- {self._escape_latex(item.field)}}}{{{self._escape_latex(item.location) or ""}}}
                """
            )

        return "\n".join(entries)

    def _render_experiences(
        self,
        experiences: list[Experience],
    ) -> str:
        entries = []

        for experience in experiences:
            period = f"{experience.start_date} -- {experience.end_date or 'Present'}"
            bullet_points = "\n".join(
                rf"        \resumeItem{{{self._escape_latex(bullet)}}}"
                for bullet in experience.achievements
            )

            entry = rf"""
        \resumeSubheading
        {{{self._escape_latex(experience.company)}}}{{{period}}}
        {{{self._escape_latex(experience.position)}}}{{}}

        \resumeItemListStart
            {bullet_points}
        \resumeItemListEnd
    """
            entries.append(entry)

        return "\n".join(entries)

    def _render_projects(self, projects: list[Project]) -> str:
        entries = []

        for project in projects:
            technologies = ", ".join(self._escape_latex(tech) for tech in project.technologies)

            entry = rf"""
        \resumeProjectHeading
        {{\textbf{{{self._escape_latex(project.name)}}} $|$ \emph{{{technologies}}}}}{{{self._escape_latex(project.start_date)}}}
        \resumeItemListStart
            \resumeItem{{{self._escape_latex(project.description)}}}
        \resumeItemListEnd

        \vspace{{-13pt}}
    """
            entries.append(entry)

        return "\n".join(entries)

    def _render_skills(self, skill_groups: list[SkillGroup]) -> str:
        entries = []

        for group in skill_groups:
            skills = ", ".join(self._escape_latex(skill) for skill in group.skills)

            entries.append(
                rf"""
        \item{{\textbf{{{self._escape_latex(group.category)} : }}{{{skills}.}}}} \\
    """
            )

        return "\n".join(entries)

    def _render_languages(self, languages: list[Language]) -> str:
        entries = ", ".join(
            f"{self._escape_latex(language.name)} ({self._escape_latex(language.level)})"
            for language in languages
        )

        return rf"""
        \item{{\textbf{{Langues : }}{{{entries}.}}}} \\
    """

    def _render_extracurricular_activities(
        self,
        activities: list[ExtracurricularActivity],
    ) -> str:
        entries = []

        for activity in activities:
            period = ""
            if activity.start_date:
                period = activity.start_date

                if activity.end_date:
                    period += f" -- {activity.end_date}"

            bullet_points = "\n".join(
                rf"        \resumeItem{{{self._escape_latex(achievement)}}}"
                for achievement in activity.achievements
            )

            entry = rf"""
            \begin{{tabular*}}{{1.0\textwidth}}[t]{{l@{{\extracolsep{{\fill}}}}r}}
                \textbf{{{self._escape_latex(activity.organization)}}} & \textbf{{\small {period}}} \\
            \end{{tabular*}}

            \resumeItemListStart
                {bullet_points}
            \resumeItemListEnd
        """

            entries.append(entry)

        return "\n".join(entries)
    
    def _escape_latex(self, text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text