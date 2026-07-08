from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_THINKING_LEVEL = "minimal"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _message_to_text(message: BaseMessage | tuple[str, Any] | Any) -> str:
    if isinstance(message, BaseMessage):
        return _content_to_text(message.content)
    if isinstance(message, tuple) and len(message) == 2:
        return _content_to_text(message[1])
    return _content_to_text(message)


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, StringPromptValue):
        return prompt.text
    if isinstance(prompt, ChatPromptValue):
        return "\n\n".join(_message_to_text(message) for message in prompt.messages)
    if isinstance(prompt, PromptValue):
        return prompt.to_string()
    if isinstance(prompt, list):
        return "\n\n".join(_message_to_text(message) for message in prompt)
    return str(prompt)


@dataclass
class GeminiGenAIRunnable(Runnable[Any, str]):
    model: str = GEMINI_MODEL
    thinking_level: str = GEMINI_THINKING_LEVEL
    _client: genai.Client | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            load_dotenv()
            self._client = genai.Client()
        return self._client

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=_prompt_to_text(input),
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.thinking_level
                ),
            ),
        )
        return response.text or ""


@cache
def default_llm() -> GeminiGenAIRunnable:
    return GeminiGenAIRunnable()


@cache
def aux_llm() -> GeminiGenAIRunnable:
    return default_llm()
