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
from typing import Any, Dict, Optional, Union

from camel.models.model_factory import ModelFactory as OriginalModelFactory
from camel.models.base_model import BaseModelBackend
from camel.types import ModelPlatformType, ModelType
from owl.models.rate_limited_gemini_model import RateLimitedGeminiModel


class RateLimitedModelFactory:
    """A factory class for creating model backends with rate limiting support.
    
    This factory overrides the original ModelFactory to use RateLimitedGeminiModel
    for any Gemini model types, ensuring they respect rate limits.
    """

    @staticmethod
    def create(
        model_platform: Union[ModelPlatformType, str],
        model_type: Optional[Union[ModelType, str]] = None,
        model_config_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseModelBackend:
        """Create a model backend based on the specified model platform and type.
        
        Args:
            model_platform: The platform of the model.
            model_type: The specific type of the model.
            model_config_dict: Configuration parameters for the model.
            **kwargs: Additional keyword arguments to pass to the model constructor.
            
        Returns:
            A model backend instance with rate limiting if applicable.
        """
        # Convert string to enum if needed
        if isinstance(model_platform, str):
            model_platform = ModelPlatformType(model_platform)
            
        # If it's a Gemini model, use our rate-limited version
        if model_platform == ModelPlatformType.GEMINI:
            # Set default rate limit interval if not specified
            if "min_request_interval" not in kwargs:
                kwargs["min_request_interval"] = 2.0  # 2 seconds between requests
                
            return RateLimitedGeminiModel(
                model_type=model_type,
                model_config_dict=model_config_dict,
                **kwargs,
            )
        
        # For other model platforms, use the original factory
        return OriginalModelFactory.create(
            model_platform=model_platform,
            model_type=model_type,
            model_config_dict=model_config_dict,
            **kwargs,
        ) 