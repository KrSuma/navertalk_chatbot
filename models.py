from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# NaverTalk Webhook Event Models
class TextContent(BaseModel):
    text: str


class NaverTalkEvent(BaseModel):
    event: str
    user: str
    textContent: Optional[TextContent] = None
    options: Optional[Dict[str, Any]] = None


# NaverTalk Response Models
class TextResponseContent(BaseModel):
    text: str


class NaverTalkResponse(BaseModel):
    event: str = "send"
    user: str
    textContent: TextResponseContent