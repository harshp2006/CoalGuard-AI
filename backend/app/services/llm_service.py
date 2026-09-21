import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service for CoalGuard AI Conversational Assistant.
    Supports OpenAI-compatible, OpenAI native, Anthropic, and Ollama endpoints.
    Used for Retrieval-Augmented Generation (RAG) over mine compliance data.
    """

    def __init__(self):
        self.provider = settings.llm_provider or "none"
        self.api_base = settings.llm_api_base
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.provider != "ollama":
                headers["Authorization"] = f"Bearer {self.api_key}"
            base_url = self.api_base or "https://api.openai.com/v1"
            self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)
        return self._client

    def is_available(self) -> bool:
        return self.provider != "none" and bool(self.api_key and self.model)

    def _build_system_prompt(self, context_data: Optional[str] = None) -> str:
        system_prompt = (
            "You are CoalGuard AI, a professional smart governance and compliance monitoring assistant "
            "for Indian coal mining operations. Answer questions concisely, accurately, and authoritatively "
            "based ONLY on the provided context data. If a user asks something not covered in the context data, "
            "respond with: 'I don't have that information in the current dataset. Please check the official CoalGuard dashboard or register.' "
            "Always present compliance, safety, inspection, contractor, and governance data clearly. "
            "If the user speaks in Hindi (Devanagari), reply in Hindi. Otherwise reply in English."
        )
        if context_data:
            system_prompt += f"\n\n--- Context Data from CoalGuard Database ---\n{context_data}"
        return system_prompt

    async def chat_completion(
        self,
        user_message: str,
        context_data: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if not self.is_available():
            logger.warning("LLM not configured. Falling back to rule-based responses.")
            return ""

        messages: List[Dict[str, str]] = []

        # System Context
        system_msg = self._build_system_prompt(context_data)
        messages.append({"role": "system", "content": system_msg})

        # Previous conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 turns to manage token limits
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # Current User Query
        messages.append({"role": "user", "content": user_message})

        try:
            if self.provider == "anthropic":
                response_text = await self._call_anthropic(messages)
            else:
                response_text = await self._call_openai_compatible(messages)
            logger.info("Successfully received LLM response.")
            return response_text
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return ""

    async def _call_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Handle Anthropic Claude API format."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
            ],
            "system": self._build_system_prompt(),
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    def build_retrieval_context(
        self,
        mines_summary: Optional[str] = None,
        compliance_summary: Optional[str] = None,
        observations_summary: Optional[str] = None,
        inspections_summary: Optional[str] = None,
        contractors_summary: Optional[str] = None,
    ) -> str:
        """
        Assembles a structured context string from CoalGuard DB query results for RAG.
        """
        parts = []
        if mines_summary:
            parts.append(f"Mines Overview:\n{mines_summary}")
        if compliance_summary:
            parts.append(f"Compliance Status:\n{compliance_summary}")
        if observations_summary:
            parts.append(f"Safety Observations:\n{observations_summary}")
        if inspections_summary:
            parts.append(f"Recent Inspections:\n{inspections_summary}")
        if contractors_summary:
            parts.append(f"Contractor Details:\n{contractors_summary}")

        if not parts:
            return ""
        return "\n\n".join(parts)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
