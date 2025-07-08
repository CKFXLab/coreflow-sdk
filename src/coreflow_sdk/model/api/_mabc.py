from abc import abstractmethod
from typing import Dict, Any, Optional, List
import time
import requests

from .._mabc import Model, TrainingJob, ReinforceJob


class APIModel(Model):
    """
    Abstract base class for API-based language models.
    
    This class inherits all DSPy requirements from Model and adds
    API-specific mandatory methods that all API providers must implement.
    """
    
    # API-specific properties that must be set
    api_key: str
    base_url: Optional[str]
    timeout: int
    
    # === API-SPECIFIC REQUIREMENTS ===
    
    @abstractmethod
    def __init__(self, api_key: str = None, base_url: str = None, timeout: int = 30, **kwargs):
        """
        Initialize API-based model client.
        
        Args:
            api_key: API authentication key
            base_url: Custom API endpoint URL
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        """
        pass
    
    @abstractmethod
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to API endpoint.
        All API providers must implement this for HTTP communication.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            
        Returns:
            Response data dictionary
        """
        pass
    
    def _handle_rate_limit(self, retry_count: int) -> None:
        """
        Handle rate limiting from API.
        Concrete implementation with exponential backoff.
        
        Args:
            retry_count: Current retry attempt number
        """
        # Exponential backoff with jitter
        import random
        base_delay = 2 ** retry_count
        jitter = random.uniform(0.1, 0.5)
        wait_time = min(base_delay + jitter, 60)  # Max 60 seconds
        
        time.sleep(wait_time)
    
    def _validate_api_key(self) -> bool:
        """
        Validate the API key format.
        Basic validation - override for provider-specific validation.
        
        Returns:
            True if API key appears valid, False otherwise
        """
        if not self.api_key:
            return False
        
        # Basic validation - not empty and reasonable length
        api_key = self.api_key.strip()
        return len(api_key) > 10 and not api_key.isspace()
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the API provider.
        DSPy can use this for dynamic model selection.
        
        Returns:
            List of available model names
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to query
            
        Returns:
            Dictionary with model capabilities, limits, etc.
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """
        Estimate the cost for a given number of tokens.
        DSPy optimizers can use this for cost-aware optimization.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens  
            model: Model name for pricing
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    # === UTILITY METHODS (Concrete - reusable) ===
    
    def test_connection(self) -> bool:
        """
        Test API connection with a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get available models as a connection test
            models = self.get_available_models()
            return len(models) > 0
        except Exception:
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get rate limit information if available.
        Override in provider implementations for specific rate limit details.
        
        Returns:
            Dictionary with rate limit info
        """
        return {
            'provider': self.provider,
            'rate_limit_available': False,
            'message': 'Rate limit info not available for this provider'
        }
    
    def format_messages_for_api(self, prompt: str = None, messages: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Format prompt/messages for API consumption.
        Concrete implementation handling common message formatting.
        
        Args:
            prompt: Text prompt for text completion models
            messages: Message list for chat completion models
            
        Returns:
            Formatted messages list
        """
        if messages is not None:
            return messages
        elif prompt is not None:
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")
    
    def calculate_tokens_estimate(self, text: str) -> int:
        """
        Rough estimate of token count for text.
        Override with provider-specific tokenization for accuracy.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4 + 1
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get API status information.
        
        Returns:
            Dictionary with API status details
        """
        return {
            'provider': self.provider,
            'base_url': self.base_url,
            'timeout': self.timeout,
            'api_key_configured': self.api_key is not None,
            'api_key_valid': self._validate_api_key(),
            'connection_test': self.test_connection()
        }


class APITrainingJob(TrainingJob):
    """Training job implementation for API-based models with concrete utilities."""
    
    def __init__(self, api_client: APIModel, model: str, train_data: List[Dict[str, Any]], 
                 train_data_format: Optional[str] = None, train_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize API training job.
        
        Args:
            api_client: API model client instance
            model: Model identifier
            train_data: Training examples
            train_data_format: Format specification for training data
            train_kwargs: Training-specific parameters
        """
        super().__init__(model, train_data, train_data_format, train_kwargs)
        self.api_client = api_client
        self.job_id = None
    
    @abstractmethod
    def _upload_training_data(self) -> str:
        """
        Upload training data to API provider.
        
        Returns:
            Data file ID or identifier
        """
        pass
    
    @abstractmethod
    def _start_training(self) -> str:
        """
        Start the training job via API.
        
        Returns:
            Training job ID
        """
        pass
    
    @abstractmethod
    def _check_status(self) -> str:
        """
        Check training job status via API.
        
        Returns:
            Current status string
        """
        pass
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get comprehensive training information.
        
        Returns:
            Dictionary with training details
        """
        info = self.get_progress_info()
        info.update({
            'job_id': self.job_id,
            'api_provider': self.api_client.provider,
            'api_model': self.api_client.model,
            'data_format': self.train_data_format,
            'training_kwargs': self.train_kwargs
        })
        return info


class APIReinforceJob(ReinforceJob):
    """Reinforcement learning job implementation for API-based models with concrete utilities."""
    
    def __init__(self, api_client: APIModel, train_kwargs: Dict[str, Any]):
        """
        Initialize API reinforcement learning job.
        
        Args:
            api_client: API model client instance
            train_kwargs: RL training parameters
        """
        super().__init__(api_client, train_kwargs)
        self.api_client = api_client
    
    def get_api_rl_info(self) -> Dict[str, Any]:
        """
        Get API-specific RL information.
        
        Returns:
            Dictionary with API RL details
        """
        stats = self.get_training_stats()
        stats.update({
            'api_provider': self.api_client.provider,
            'api_model': self.api_client.model,
            'base_url': self.api_client.base_url,
            'training_kwargs': self.train_kwargs
        })
        return stats