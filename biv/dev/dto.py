import datetime
from dataclasses import dataclass

from pydantic import BaseModel


class Payment(BaseModel):
    id: int
    date: str
    sum: str
    description: str
    category: str | None
