import os
import sys
import copy
import time
import asyncio
from typing import List, Dict, Any, Optional
import requests

from ._mabc import APIModel, APITrainingJob, APIReinforceJob

# Handle imports for both direct execution and module imports
try:
    from ...utils import AppLogger
except ImportError:
    # If relative import fails, try absolute import for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from ...utils import AppLogger

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None
    Anthropic = None
    AsyncAnthropic = None


class AnthropicClient(APIModel):
    """Anthropic API implementation of the universal model interface with full DSPy support."""
    
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30, 
                 model: str = "claude-3-haiku-20240307", model_type: str = "chat", **kwargs):
        """
        Initialize Anthropic client with DSPy compatibility.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
            base_url: Custom Anthropic-compatible endpoint URL
            timeout: Request timeout in seconds
            model: Default model to use
            model_type: Type of model ("chat" or "text")
            **kwargs: Additional Anthropic parameters
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not available. Install with: pip install anthropic"
            )
        
        self.logger = AppLogger(__name__)
        
        # Set DSPy required properties
        self.model = model
        self.model_type = model_type
        self.provider = "anthropic"
        self.cache = kwargs.get("cache", True)
        self.num_retries = kwargs.get("num_retries", 3)
        self.kwargs = kwargs
        self.history = []
        
        # Set API-specific properties
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        
        if not self.api_key:
            self.logger.error("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")
            raise ValueError("Anthropic API key is required")
        
        try:
            # Initialize sync and async clients
            client_kwargs = {"api_key": self.api_key, "timeout": timeout}
            if base_url:
                client_kwargs["base_url"] = base_url
                
            self.client = Anthropic(**client_kwargs)
            self.async_client = AsyncAnthropic(**client_kwargs)
            
            self.logger.info("Anthropic client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    # === CORE DSPy INTERFACE ===
    
    def __call__(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """Main callable interface for DSPy modules."""
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_response(response, prompt, messages, **kwargs)
    
    def forward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """Forward pass through the model (synchronous)."""
        formatted_messages = self.format_messages_for_api(prompt, messages)
        
        # Prepare request parameters
        request_params = self._prepare_request_params(formatted_messages, **kwargs)
        
        # Make request with retries
        for attempt in range(self.num_retries):
            try:
                response = self.client.messages.create(**request_params)
                
                # Convert to standard format
                response_dict = self._convert_response_to_dict(response)
                
                # Store in history
                self._update_history(formatted_messages, response_dict)
                
                return response_dict
                
            except Exception as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.num_retries - 1:
                    self._handle_rate_limit(attempt)
                else:
                    self.logger.error(f"All {self.num_retries} attempts failed")
                    raise
    
    async def aforward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """Asynchronous forward pass through the model."""
        formatted_messages = self.format_messages_for_api(prompt, messages)
        
        # Prepare request parameters
        request_params = self._prepare_request_params(formatted_messages, **kwargs)
        
        # Make async request with retries
        for attempt in range(self.num_retries):
            try:
                response = await self.async_client.messages.create(**request_params)
                
                # Convert to standard format
                response_dict = self._convert_response_to_dict(response)
                
                # Store in history
                self._update_history(formatted_messages, response_dict)
                
                return response_dict
                
            except Exception as e:
                self.logger.warning(f"Async request attempt {attempt + 1} failed: {e}")
                if attempt < self.num_retries - 1:
                    await asyncio.sleep((2 ** attempt) + 1)
                else:
                    self.logger.error(f"All {self.num_retries} async attempts failed")
                    raise
    
    async def acall(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """Asynchronous callable interface."""
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_response(response, prompt, messages, **kwargs)
    
    # === DSPy FRAMEWORK METHODS ===
    
    def copy(self, **kwargs) -> 'AnthropicClient':
        """Create a copy of the model with updated parameters."""
        new_kwargs = self.kwargs.copy()
        new_kwargs.update(kwargs)
        
        return AnthropicClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            model=kwargs.get('model', self.model),
            model_type=kwargs.get('model_type', self.model_type),
            **new_kwargs
        )
    
    def validate_connection(self) -> bool:
        """Test if the Anthropic API connection is working."""
        try:
            # Test with a simple message
            test_response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    # === TRAINING & OPTIMIZATION (Not supported by Anthropic API) ===
    
    def finetune(self, train_data: List[Dict[str, Any]], 
                train_data_format: Optional[str] = None,
                train_kwargs: Optional[Dict[str, Any]] = None) -> 'AnthropicTrainingJob':
        """Anthropic API does not support fine-tuning."""
        raise NotImplementedError("Anthropic API does not support fine-tuning")
    
    def reinforce(self, train_kwargs: Dict[str, Any]) -> 'AnthropicReinforceJob':
        """Anthropic API does not support reinforcement learning."""
        raise NotImplementedError("Anthropic API does not support reinforcement learning")
    
    def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Anthropic API does not provide embedding endpoints."""
        raise NotImplementedError("Anthropic API does not provide embedding endpoints")
    
    # === API-SPECIFIC REQUIREMENTS ===
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Anthropic API endpoint."""
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        url = f"{self.base_url or 'https://api.anthropic.com'}{endpoint}"
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        # Import here to avoid circular imports
        from ..registry import get_available_models
        return get_available_models(provider="anthropic")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific Anthropic model."""
        # Import here to avoid circular imports
        from ..registry import get_model_info
        
        model_info = get_model_info(model_name)
        if model_info:
            return {
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
        
        return {
            "max_tokens": 4096,
            "context_window": 200000,
            "description": "Unknown model"
        }
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate the cost for Anthropic model usage."""
        # Import here to avoid circular imports
        from ..registry import estimate_cost
        
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        if cost > 0:
            return cost
        
        # Fallback for unknown models
        self.logger.warning(f"Unknown model for cost estimation: {model}")
        return 0.0
    
    # === HELPER METHODS ===
    
    def _prepare_request_params(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Prepare request parameters for Anthropic API."""
        # Merge instance kwargs with call-time kwargs
        params = self.kwargs.copy()
        params.update(kwargs)
        
        # Set required parameters
        request_params = {
            "model": params.get("model", self.model),
            "messages": messages,
            "max_tokens": params.get("max_tokens", 1000)
        }
        
        # Add optional parameters
        if "temperature" in params:
            request_params["temperature"] = params["temperature"]
        if "top_p" in params:
            request_params["top_p"] = params["top_p"]
        if "top_k" in params:
            request_params["top_k"] = params["top_k"]
        if "stop_sequences" in params:
            request_params["stop_sequences"] = params["stop_sequences"]
        if "system" in params:
            request_params["system"] = params["system"]
        
        return request_params
    
    def _convert_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert Anthropic response to standard dictionary format."""
        # Extract content from response
        content = ""
        if hasattr(response, 'content') and response.content:
            # Handle list of content blocks
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
        
        return {
            "id": getattr(response, 'id', 'unknown'),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(response, 'model', self.model),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": getattr(response, 'stop_reason', 'stop')
            }],
            "usage": {
                "prompt_tokens": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                "completion_tokens": getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0,
                "total_tokens": (getattr(response.usage, 'input_tokens', 0) + 
                               getattr(response.usage, 'output_tokens', 0)) if hasattr(response, 'usage') else 0
            }
        }
    
    def _process_response(self, response: Dict[str, Any], prompt: str = None, 
                         messages: List[Dict[str, str]] = None, **kwargs) -> str:
        """Process response and extract text content."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        self.logger.warning("Unexpected response format from Anthropic API")
        return str(response)
    
    def _update_history(self, messages: List[Dict[str, str]], response: Dict[str, Any]) -> None:
        """Update interaction history."""
        history_entry = {
            "timestamp": time.time(),
            "messages": messages,
            "response": response,
            "model": self.model,
            "provider": self.provider
        }
        
        self.history.append(history_entry)
        self.update_global_history(history_entry)
    
    # === PROVIDER MANAGEMENT ===
    
    def infer_provider(self) -> str:
        """Infer provider."""
        return "anthropic"
    
    def launch(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Anthropic models don't require server launch."""
        self.logger.info("Anthropic models are API-based - no launch required")
    
    def kill(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Anthropic models don't require termination."""
        self.logger.info("Anthropic models are API-based - no termination required")


# === TRAINING JOB IMPLEMENTATIONS (Not supported) ===

class AnthropicTrainingJob(APITrainingJob):
    """Training job for Anthropic models (not supported)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Anthropic API does not support fine-tuning")
    
    def start(self) -> None:
        raise NotImplementedError("Anthropic API does not support fine-tuning")
    
    def result(self) -> Optional[str]:
        raise NotImplementedError("Anthropic API does not support fine-tuning")


class AnthropicReinforceJob(APIReinforceJob):
    """Reinforcement learning job for Anthropic models (not supported)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Anthropic API does not support reinforcement learning")
    
    def initialize(self) -> None:
        raise NotImplementedError("Anthropic API does not support reinforcement learning")
    
    def step(self) -> Dict[str, Any]:
        raise NotImplementedError("Anthropic API does not support reinforcement learning")
    
    def finalize(self) -> 'AnthropicClient':
        raise NotImplementedError("Anthropic API does not support reinforcement learning")
