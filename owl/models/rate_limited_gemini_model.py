# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import time
import logging
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from camel.models.gemini_model import GeminiModel
from camel.messages import OpenAIMessage
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
)


class RateLimitedGeminiModel(GeminiModel):
    """A rate-limited version of GeminiModel to prevent exceeding API rate limits.
    
    This implementation adds a delay between requests to ensure we don't exceed
    the 30 requests per minute limit imposed by Gemini API.
    
    Args:
        model_type (Union[ModelType, str]): Model for which a backend is created.
        model_config_dict (Optional[Dict[str, Any]], optional): Configuration.
        api_key (Optional[str], optional): The API key for Gemini service.
        url (Optional[str], optional): The API URL.
        token_counter (Optional[BaseTokenCounter], optional): Token counter.
        min_request_interval (float, optional): Minimum time in seconds between
            requests (default: 2.0 seconds, allowing max 30 RPM).
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        min_request_interval: float = 2.0,
    ) -> None:
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Rate limited Gemini model initialized with {min_request_interval:.2f}s "
            f"interval (max {60/min_request_interval:.1f} RPM)"
        )

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by waiting if needed."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < self._min_request_interval and self._last_request_time > 0:
            wait_time = self._min_request_interval - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
        
        self._last_request_time = time.time()

    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        """Override to add rate limiting before making the request."""
        self._enforce_rate_limit()
        self.logger.debug("Making rate-limited Gemini API request")
        return super()._request_chat_completion(messages, tools)

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        """Override to add rate limiting before making the async request."""
        self._enforce_rate_limit()
        self.logger.debug("Making rate-limited async Gemini API request")
        return await super()._arequest_chat_completion(messages, tools)

    def _request_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
    ) -> ChatCompletion:
        """Override to add rate limiting before making the parse request."""
        self._enforce_rate_limit()
        self.logger.debug("Making rate-limited Gemini API parse request")
        return super()._request_parse(messages, response_format)

    async def _arequest_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
    ) -> ChatCompletion:
        """Override to add rate limiting before making the async parse request."""
        self._enforce_rate_limit()
        self.logger.debug("Making rate-limited async Gemini API parse request")
        return await super()._arequest_parse(messages, response_format) 