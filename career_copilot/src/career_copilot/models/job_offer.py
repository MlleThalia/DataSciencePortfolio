from dataclasses import dataclass
from typing import Optional
from datetime import date


@dataclass
class JobOffer:
    id: str
    title: str
    company: str
    city: str
    country : str

    contract: Optional[str]
    publication_date: Optional[date]

    description: Optional[str]
    salary: Optional[str]

    url: str
    source: str