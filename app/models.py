from pydantic import BaseModel, HttpUrl
from typing import List

class HackRxRequest(BaseModel):
    # incoming payload
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    # api response type
    answers: List[str]