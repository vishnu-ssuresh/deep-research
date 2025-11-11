import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from ..exceptions import APIKeyException, LLMServiceException

T = TypeVar("T", bound=BaseModel)

class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIKeyException(
                "OpenAI API key must be set in OPENAI_API_KEY environment variable"
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        response_format: type[T] | None = None,
        model: str | None = None,
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if response_format:
                kwargs["response_format"] = {"type": "json_object"}
                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
                parsed_data = json.loads(content)
                return response_format(**parsed_data)

            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

        except Exception as e:
            raise LLMServiceException(f"OpenAI API call failed: {str(e)}") from e
