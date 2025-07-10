import os
import sys
import copy
import time
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
import requests

from ._mabc import APIModel, APITrainingJob, APIReinforceJob

# Handle imports for both direct execution and module imports
try:
    from ...utils import AppLogger
except ImportError:
    # If relative import fails, try absolute import for direct execution
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from ...utils import AppLogger


class OpenAIClient(APIModel):
    """OpenAI API implementation of the universal model interface with full DSPy support."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: int = 30,
        model: str = "gpt-4",
        model_type: str = "chat",
        **kwargs,
    ):
        """
        Initialize OpenAI client with DSPy compatibility.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            base_url: Custom OpenAI-compatible endpoint URL
            timeout: Request timeout in seconds
            model: Default model to use
            model_type: Type of model ("chat" or "text")
            **kwargs: Additional OpenAI parameters
        """
        self.logger = AppLogger(__name__)

        # Set DSPy required properties
        self.model = model
        self.model_type = model_type
        self.provider = "openai"
        self.cache = kwargs.get("cache", True)
        self.num_retries = kwargs.get("num_retries", 3)
        self.kwargs = kwargs
        self.history = []

        # Set API-specific properties
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

        if not self.api_key:
            self.logger.error(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )
            raise ValueError("OpenAI API key is required")

        try:
            # Initialize sync and async clients
            client_kwargs = {"api_key": self.api_key, "timeout": timeout}
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)

            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    # === CORE DSPy INTERFACE ===

    def __call__(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Any:
        """Main callable interface for DSPy modules."""
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_response(response, prompt, messages, **kwargs)

    def forward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Synchronous forward pass through OpenAI model."""
        # Use inherited message formatting
        messages = self.format_messages_for_api(prompt, messages)

        # Merge kwargs with defaults and filter out client-specific parameters
        request_kwargs = {**self.kwargs, **kwargs}
        # Remove client-specific parameters that shouldn't be sent to API
        client_params = {"num_retries", "cache", "provider", "model_type", "timeout"}
        request_kwargs = {
            k: v for k, v in request_kwargs.items() if k not in client_params
        }

        # Retry logic with inherited rate limiting
        for attempt in range(self.num_retries + 1):
            try:
                if self.model_type == "chat":
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, **request_kwargs
                    )
                else:
                    # For text completion models
                    prompt_text = messages[0]["content"] if messages else prompt
                    response = self.client.completions.create(
                        model=self.model, prompt=prompt_text, **request_kwargs
                    )

                # Update history
                history_entry = {
                    "prompt": prompt,
                    "messages": messages,
                    "response": (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else dict(response)
                    ),
                    "kwargs": request_kwargs,
                }
                self.history.append(history_entry)
                self.update_global_history(
                    history_entry
                )  # Uses inherited implementation

                return (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else dict(response)
                )

            except Exception as e:
                if attempt < self.num_retries:
                    self._handle_rate_limit(attempt)  # Uses inherited implementation
                else:
                    self.logger.error(
                        f"OpenAI request failed after {self.num_retries} retries: {e}"
                    )
                    raise

    async def aforward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Asynchronous forward pass through OpenAI model."""
        # Use inherited message formatting
        messages = self.format_messages_for_api(prompt, messages)

        # Merge kwargs with defaults and filter out client-specific parameters
        request_kwargs = {**self.kwargs, **kwargs}
        # Remove client-specific parameters that shouldn't be sent to API
        client_params = {"num_retries", "cache", "provider", "model_type", "timeout"}
        request_kwargs = {
            k: v for k, v in request_kwargs.items() if k not in client_params
        }

        # Retry logic
        for attempt in range(self.num_retries + 1):
            try:
                if self.model_type == "chat":
                    response = await self.async_client.chat.completions.create(
                        model=self.model, messages=messages, **request_kwargs
                    )
                else:
                    # For text completion models
                    prompt_text = messages[0]["content"] if messages else prompt
                    response = await self.async_client.completions.create(
                        model=self.model, prompt=prompt_text, **request_kwargs
                    )

                # Update history
                history_entry = {
                    "prompt": prompt,
                    "messages": messages,
                    "response": (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else dict(response)
                    ),
                    "kwargs": request_kwargs,
                }
                self.history.append(history_entry)
                self.update_global_history(
                    history_entry
                )  # Uses inherited implementation

                return (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else dict(response)
                )

            except Exception as e:
                if attempt < self.num_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(
                        f"OpenAI async request failed after {self.num_retries} retries: {e}"
                    )
                    raise

    async def acall(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Any:
        """Asynchronous callable interface for DSPy modules."""
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_response(response, prompt, messages, **kwargs)

    async def stream_forward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ):
        """Stream model response chunks."""
        messages = self.format_messages_for_api(prompt, messages)
        kwargs["stream"] = True

        # Remove client-specific parameters that shouldn't be sent to API
        client_params = {"num_retries", "cache", "provider", "model_type", "timeout"}
        request_kwargs = {k: v for k, v in kwargs.items() if k not in client_params}

        response = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, **request_kwargs
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # === DSPy FRAMEWORK METHODS ===

    def copy(self, **kwargs) -> "OpenAIClient":
        """Create a copy with updated parameters."""
        new_instance = copy.deepcopy(self)
        new_instance.history = []  # Fresh history for the copy

        # Update any provided parameters
        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in new_instance.kwargs or not hasattr(new_instance, key):
                new_instance.kwargs[key] = value

        return new_instance

    def finetune(
        self,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[str] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "OpenAITrainingJob":
        """Start OpenAI fine-tuning job."""
        return OpenAITrainingJob(
            api_client=self,
            model=self.model,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs or {},
        )

    def reinforce(self, train_kwargs: Dict[str, Any]) -> "OpenAIReinforceJob":
        """Start OpenAI reinforcement learning job."""
        return OpenAIReinforceJob(api_client=self, train_kwargs=train_kwargs)

    # === EMBEDDING INTERFACE ===

    def generate_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Generate embedding vector for given text."""
        try:
            response = self.client.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            self.logger.debug(f"Generated embedding for text length: {len(text)}")
            return embedding

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    # === VALIDATION ===

    def validate_connection(self) -> bool:
        """Test if OpenAI connection is working."""
        try:
            test_response = self.client.embeddings.create(
                input=["test"], model="text-embedding-3-small"
            )

            if test_response and test_response.data:
                self.logger.info("OpenAI connection validated successfully")
                return True
            else:
                self.logger.error(
                    "OpenAI connection validation failed - no data returned"
                )
                return False

        except Exception as e:
            self.logger.error(f"OpenAI connection validation failed: {e}")
            return False

    # === API-SPECIFIC METHODS ===

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenAI API endpoint."""
        url = f"{self.base_url or 'https://api.openai.com'}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url, json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    # REMOVED: _handle_rate_limit() - now uses inherited implementation

    def _validate_api_key(self) -> bool:
        """Validate OpenAI API key with actual API call."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            # Return known models from registry as fallback
            from ..registry import get_available_models

            return get_available_models(provider="openai")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about an OpenAI model."""
        try:
            model = self.client.models.retrieve(model_name)
            return model.model_dump() if hasattr(model, "model_dump") else dict(model)
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            # Fallback to registry information
            from ..registry import get_model_info

            model_info = get_model_info(model_name)
            if model_info:
                return {
                    "id": model_info.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "openai",
                    "max_tokens": model_info.max_tokens,
                    "context_window": model_info.context_window,
                    "description": model_info.description,
                    "display_name": model_info.display_name,
                    "provider": model_info.provider.value,
                    "model_type": model_info.model_type.value,
                    "supports_streaming": model_info.supports_streaming,
                    "supports_functions": model_info.supports_functions,
                    "supports_vision": model_info.supports_vision,
                    "input_price_per_million": model_info.input_price_per_million,
                    "output_price_per_million": model_info.output_price_per_million,
                }
            return {}

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: str
    ) -> float:
        """Estimate cost for OpenAI model usage."""
        # Import here to avoid circular imports
        from ..registry import estimate_cost

        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        if cost > 0:
            return cost

        # Fallback for unknown models
        self.logger.warning(f"Unknown model for cost estimation: {model}")
        return 0.0

    # === OVERRIDE INHERITED METHODS FOR OPENAI-SPECIFIC BEHAVIOR ===

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get OpenAI-specific rate limit information."""
        return {
            "provider": self.provider,
            "rate_limit_available": True,
            "limits": {
                "requests_per_minute": "Varies by model and tier",
                "tokens_per_minute": "Varies by model and tier",
                "note": "Check OpenAI dashboard for current limits",
            },
        }

    def calculate_tokens_estimate(self, text: str) -> int:
        """
        More accurate token estimation for OpenAI models.
        Override the basic estimation from parent class.
        """
        # More accurate estimation for OpenAI: ~4 chars per token for English
        # Add some padding for special tokens
        base_estimate = len(text) // 4
        return base_estimate + max(1, base_estimate // 10)  # Add ~10% padding

    # === HELPER METHODS ===

    def _process_response(
        self,
        response: Dict[str, Any],
        prompt: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """Process raw API response into DSPy-compatible format."""
        # This would return a DSPy Prediction object in a full implementation
        # For now, return the processed response
        if "choices" in response and response["choices"]:
            if self.model_type == "chat":
                return response["choices"][0]["message"]["content"]
            else:
                return response["choices"][0]["text"]
        return ""


class OpenAITrainingJob(APITrainingJob):
    """OpenAI fine-tuning job implementation using inherited utilities."""

    # REMOVED: __init__() - now uses inherited implementation
    # REMOVED: status() - now uses inherited implementation

    def start(self) -> None:
        """Start OpenAI fine-tuning job."""
        # Set start time for inherited duration tracking
        self._start_time = time.time()
        # Implementation would upload data and start fine-tuning

    def result(self) -> Optional[str]:
        """Get fine-tuned model ID when complete."""
        # Would return the fine-tuned model ID
        return None

    def _upload_training_data(self) -> str:
        """Upload training data to OpenAI."""
        # Implementation would format and upload data
        return "file_id"

    def _start_training(self) -> str:
        """Start training via OpenAI API."""
        # Implementation would create fine-tuning job
        return "job_id"

    def _check_status(self) -> str:
        """Check training status via OpenAI API."""
        # Implementation would check job status
        return "running"


class OpenAIReinforceJob(APIReinforceJob):
    """OpenAI reinforcement learning job implementation using inherited utilities."""

    # REMOVED: __init__() - now uses inherited implementation

    def initialize(self) -> None:
        """Initialize RL training setup."""
        self._start_time = time.time()  # Use inherited tracking

    def step(self) -> Dict[str, Any]:
        """Perform one RL training step."""
        self._step_count += 1  # Use inherited step tracking
        reward = 0.0  # Placeholder
        self.add_reward(reward)  # Use inherited reward tracking
        return {"step": self._step_count, "reward": reward}

    def finalize(self) -> OpenAIClient:
        """Finalize training and return updated model."""
        return self.api_client
