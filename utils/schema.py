
from pydantic import BaseModel


class UniTextItem(BaseModel):
    text: str | None
    
class MultiTextItem(BaseModel):
    id: str | int 
    text: str | None

class GenerateStructuredRequest(BaseModel):
    text: UniTextItem
    json_schema: dict

class GenerateBatchStructuredRequest(BaseModel):
    texts: list[MultiTextItem]
    json_schema: dict