from typing import List, Optional, Any
from pydantic import BaseModel, Field, validator


class Output(BaseModel):
    response: str = Field(...)
    extra_info: Optional[str] = Field(None)


class FormattedOutput(BaseModel):
    user_id: str = Field(...)
    question: str = Field(...)
    answer: str = Field(...)

    @validator("question")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field
