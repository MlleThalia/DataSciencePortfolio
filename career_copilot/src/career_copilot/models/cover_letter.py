from pydantic import BaseModel

class Address(BaseModel):
    """Une expérience professionnelle, y compris un stage."""

    name: str
    street: str | None = None
    postal_code: str | None = None
    city: str | None = None
    tel: str | None = None
    email: str | None = None

class CoverLetter(BaseModel):
    """Represents a cover letter tailored to a specific job offer."""

    sender_address: Address

    recipient_address: Address

    subject: str

    greeting: str

    introduction: str

    body: list[str]

    conclusion: str

    closing: str

    signature: str