from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class AssistantChatRequest(BaseModel):
    message: str
    language: Optional[str] = "auto"  # "auto" | "en" | "hi"
    mine_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = []


class QueryQuickAction(BaseModel):
    label: str
    action_type: str  # "navigate" | "filter" | "query"
    payload: str


class LLMResponse(BaseModel):
    reply: str
    language: str
    data_context: Optional[Dict[str, Any]] = None
    quick_actions: Optional[List[QueryQuickAction]] = None
    used_llm: bool = False
