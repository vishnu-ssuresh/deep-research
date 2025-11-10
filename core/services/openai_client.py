"""OpenAI client for LLM interactions."""

import json
import os
from typing import Optional, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from ..errors import APIKeyError, LLMServiceError

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            model: Default model to use for completions
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIKeyError(
                "OpenAI API key must be set in OPENAI_API_KEY environment variable"
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        response_format: Optional[type[T]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Make a call to OpenAI API.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message/query
            temperature: Sampling temperature (0-1)
            response_format: Optional Pydantic model for structured output
            model: Model to use (overrides default)

        Returns:
            String response or Pydantic model instance if response_format provided

        Raises:
            LLMServiceError: If the API call fails
        """
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

            # Handle structured output
            if response_format:
                kwargs["response_format"] = {"type": "json_object"}
                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
                parsed_data = json.loads(content)
                return response_format(**parsed_data)

            # Regular string output
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

        except Exception as e:
            raise LLMServiceError(f"OpenAI API call failed: {str(e)}") from e
