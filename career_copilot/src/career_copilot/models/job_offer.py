
from typing import Optional
from datetime import date
from pydantic import BaseModel


class JobOffer(BaseModel):
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