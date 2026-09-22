from pathlib import Path
from jinja2 import Template
import logging

from career_copilot.models.cover_letter import *

logger = logging.getLogger(__name__)

class CoverLetterRenderer:

    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path
        self.template = Template(
            template_path.read_text(encoding="utf-8"),
            comment_start_string="/*",
            comment_end_string="*/",
        )

    def render(self, cover_letter: CoverLetter) -> str:

        # Tailored content comes from the generated resume.
        sender_address = self._render_sender_address(cover_letter.sender_address)
        recipient_address = self._render_recipient(cover_letter.recipient_address)
        
        logger.debug(f"Rendering cover letter")
        return self.template.render(
            sender_address=sender_address,
            recipient_address=recipient_address,
            subject=cover_letter.subject,
            greeting=cover_letter.greeting,
            introduction=cover_letter.introduction,
            body=self._render_body(cover_letter.body),
            conclusion=cover_letter.conclusion,
            closing=cover_letter.closing,
            signature=cover_letter.signature,
        )
    
    def save(self, latex_content: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            latex_content,
            encoding="utf-8",
        )
        print(f"✓ Saved {output_path}")
    
    def _render_sender_address(self, sender_address: Address) -> str:
        lines = [
            self._escape_latex(sender_address.name),
        ]

        if sender_address.street:
            lines.append(self._escape_latex(sender_address.street))

        if sender_address.postal_code or sender_address.city:
            location = ", ".join(
                value
                for value in [
                    sender_address.postal_code,
                    sender_address.city,
                ]
                if value
            )
            lines.append(self._escape_latex(location))

        if sender_address.tel:
            lines.append(
                f"Téléphone : {self._escape_latex(sender_address.tel)}"
            )

        if sender_address.email:
            lines.append(
                f"Email : {self._escape_latex(sender_address.email)}"
            )

        return r" \\".join(lines) + r" \\"
    
    def _render_recipient(self, recipient: Address) -> str:
        return rf"""À l'attention de\\
    {self._escape_latex(recipient.name)}
    """

    def _render_body(self, paragraphs: list[str]) -> str:
        return "\n\n".join(
            self._escape_latex(paragraph)
            for paragraph in paragraphs
        )

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