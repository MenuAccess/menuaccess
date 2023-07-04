from pydantic import BaseModel, Field


class ParsedOutput(BaseModel):
    question: str = Field(...)
    answer: str = Field(...)
