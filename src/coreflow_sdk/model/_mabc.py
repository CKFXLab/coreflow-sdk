from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal, Callable
import threading
import time
import copy

class Model(ABC):
    """
    Universal model abstraction that all model providers must implement.
    
    This abstract base class defines the complete interface required for DSPy framework integration.
    All model providers (API, Bedrock, LlamaServer, etc.) must implement these methods to ensure
    compatibility with DSPy modules, signatures, and optimizers.
    """
    
    # Core Model Properties (must be set by implementations)
    model: str
    model_type: Literal["chat", "text"]
    provider: str
    cache: bool
    num_retries: int
    kwargs: Dict[str, Any]
    history: List[Dict[str, Any]]
    
    # Global history tracking (class variable)
    _global_history: List[Dict[str, Any]] = []
    
    # === CORE LM INTERFACE (Required by DSPy) ===
    
    @abstractmethod
    def __call__(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """
        Main callable interface for the model.
        DSPy modules use this method to invoke the language model.
        
        Args:
            prompt: Text prompt for text completion models
            messages: Message list for chat completion models  
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model response object with standardized interface
        """
        pass
    
    @abstractmethod
    def forward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Forward pass through the model (synchronous).
        Core method that DSPy uses for inference.
        
        Args:
            prompt: Text prompt for text completion models
            messages: Message list for chat completion models
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Raw model response dictionary with choices, usage, etc.
        """
        pass
        
    @abstractmethod 
    async def aforward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Asynchronous forward pass through the model.
        Required for async DSPy operations.
        
        Args:
            prompt: Text prompt for text completion models
            messages: Message list for chat completion models
            **kwargs: Additional parameters
            
        Returns:
            Raw model response dictionary
        """
        pass
    
    @abstractmethod
    async def acall(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """
        Asynchronous callable interface.
        
        Args:
            prompt: Text prompt for text completion models
            messages: Message list for chat completion models
            **kwargs: Additional parameters
            
        Returns:
            Model response object
        """
        pass
    
    # === DSPy FRAMEWORK METHODS (Some concrete, some abstract) ===
    
    @abstractmethod
    def copy(self, **kwargs) -> 'Model':
        """
        Create a copy of the model with updated parameters.
        DSPy optimizers use this to create model variants.
        
        Args:
            **kwargs: Parameters to update in the copy
            
        Returns:
            New model instance with updated parameters
        """
        pass
    
    def inspect_history(self, n: int = 1) -> str:
        """
        Inspect recent model interactions.
        Used by DSPy for debugging and analysis.
        
        Args:
            n: Number of recent interactions to show
            
        Returns:
            Formatted string showing interaction history
        """
        if not self.history:
            return "No history available"
        
        recent_entries = self.history[-n:]
        formatted = []
        for i, entry in enumerate(recent_entries, 1):
            formatted.append(f"Entry {i}:")
            formatted.append(f"  Prompt: {entry.get('prompt', 'N/A')}")
            
            # Format response preview
            response = entry.get('response', 'N/A')
            if isinstance(response, dict):
                response_preview = str(response).replace('\n', ' ')[:100] + "..."
            else:
                response_preview = str(response)[:100] + "..."
            formatted.append(f"  Response: {response_preview}")
            
            formatted.append(f"  Timestamp: {entry.get('timestamp', 'N/A')}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def update_global_history(self, entry: Dict[str, Any]) -> None:
        """
        Update the global history with a new entry.
        DSPy uses this to track all model interactions.
        
        Args:
            entry: History entry dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in entry:
            entry['timestamp'] = time.time()
        
        # Add provider info
        entry['provider'] = getattr(self, 'provider', 'unknown')
        entry['model'] = getattr(self, 'model', 'unknown')
        
        # Add to global history
        if not hasattr(self.__class__, '_global_history'):
            self.__class__._global_history = []
        self.__class__._global_history.append(entry)
    
    def dump_state(self) -> Dict[str, Any]:
        """
        Serialize model state for saving/loading.
        DSPy optimizers use this for checkpointing.
        
        Returns:
            Dictionary containing all model state
        """
        return {
            "model": getattr(self, 'model', None),
            "model_type": getattr(self, 'model_type', None),
            "provider": getattr(self, 'provider', None),
            "cache": getattr(self, 'cache', True),
            "num_retries": getattr(self, 'num_retries', 3),
            "kwargs": getattr(self, 'kwargs', {}),
            "history_count": len(getattr(self, 'history', [])),
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "timestamp": time.time()
        }
    
    # === PROVIDER MANAGEMENT (Some concrete, some abstract) ===
    
    def infer_provider(self) -> str:
        """
        Infer the provider from model name/configuration.
        DSPy uses this for provider-specific optimizations.
        
        Returns:
            Provider name (e.g., 'openai', 'anthropic', 'bedrock')
        """
        return getattr(self, 'provider', 'unknown')
    
    def launch(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Launch model server/service if needed.
        Used for models that require server startup.
        
        Default implementation is no-op for API-based models.
        Override in LlamaServer implementations.
        
        Args:
            launch_kwargs: Provider-specific launch parameters
        """
        pass  # Default: no-op for most models
    
    def kill(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Terminate model server/service if running.
        
        Default implementation is no-op for API-based models.
        Override in LlamaServer implementations.
        
        Args:
            launch_kwargs: Provider-specific termination parameters
        """
        pass  # Default: no-op for most models
    
    # === TRAINING & OPTIMIZATION INTERFACES (Abstract - provider-specific) ===
    
    @abstractmethod
    def finetune(self, 
                train_data: List[Dict[str, Any]], 
                train_data_format: Optional[str] = None,
                train_kwargs: Optional[Dict[str, Any]] = None) -> 'TrainingJob':
        """
        Fine-tune the model on provided data.
        DSPy optimizers use this for model improvement.
        
        Args:
            train_data: Training examples
            train_data_format: Format specification for training data
            train_kwargs: Training-specific parameters
            
        Returns:
            TrainingJob object for monitoring progress
        """
        pass
    
    @abstractmethod
    def reinforce(self, train_kwargs: Dict[str, Any]) -> 'ReinforceJob':
        """
        Reinforcement learning interface for model improvement.
        Advanced DSPy optimizers use this for RL-based optimization.
        
        Args:
            train_kwargs: RL training parameters
            
        Returns:
            ReinforceJob object for managing RL training
        """
        pass
    
    # === EMBEDDING INTERFACE (Abstract - provider-specific) ===
    
    @abstractmethod
    def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generate embedding vector for given text.
        Used by DSPy for semantic similarity and retrieval.
        
        Args:
            text: Text to embed
            model: Specific embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    # === VALIDATION & HEALTH CHECKS (Abstract - provider-specific) ===
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Test if the model connection is working.
        DSPy uses this for health checks and error handling.
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    # === UTILITY METHODS (Concrete - reusable) ===
    
    def get_global_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get global history across all model instances.
        
        Args:
            n: Number of recent entries to return (None for all)
            
        Returns:
            List of history entries
        """
        history = getattr(self.__class__, '_global_history', [])
        if n is None:
            return history
        return history[-n:]
    
    def clear_history(self, global_history: bool = False) -> None:
        """
        Clear model history.
        
        Args:
            global_history: Whether to also clear global history
        """
        self.history = []
        if global_history:
            self.__class__._global_history = []
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for this model instance.
        
        Returns:
            Dictionary with usage statistics
        """
        total_requests = len(self.history)
        
        # Calculate token usage if available
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for entry in self.history:
            response = entry.get('response', {})
            if isinstance(response, dict) and 'usage' in response:
                usage = response['usage']
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
        
        return {
            'total_requests': total_requests,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens,
            'provider': self.provider,
            'model': self.model
        }


# === HELPER CLASSES FOR TRAINING/RL ===

class TrainingJob(ABC):
    """Abstract base class for training jobs with concrete utilities."""
    
    def __init__(self, model: str, train_data: List[Dict[str, Any]], 
                 train_data_format: Optional[str] = None, train_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize training job with standard parameters.
        
        Args:
            model: Model identifier
            train_data: Training examples
            train_data_format: Format specification for training data
            train_kwargs: Training-specific parameters
        """
        self.model = model
        self.train_data = train_data
        self.train_data_format = train_data_format
        self.train_kwargs = train_kwargs or {}
        self.thread = None
        self._status = "initialized"
        self._start_time = None
        self._end_time = None
    
    @abstractmethod
    def start(self) -> None:
        """Start the training job."""
        pass
    
    def status(self) -> str:
        """
        Get current training status.
        
        Returns:
            Status string: "not_started", "running", "completed", "failed"
        """
        if self.thread is None:
            return "not_started"
        elif self.thread.is_alive():
            return "running"
        else:
            return getattr(self, '_status', 'completed')
    
    @abstractmethod
    def result(self) -> Optional[str]:
        """Get training result/model ID when complete."""
        pass
    
    def get_duration(self) -> Optional[float]:
        """
        Get training duration in seconds.
        
        Returns:
            Duration in seconds, None if not started or still running
        """
        if self._start_time is None:
            return None
        
        end_time = self._end_time or time.time()
        return end_time - self._start_time
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get training progress information.
        
        Returns:
            Dictionary with progress details
        """
        return {
            'status': self.status(),
            'model': self.model,
            'train_data_size': len(self.train_data),
            'train_data_format': self.train_data_format,
            'duration': self.get_duration(),
            'start_time': self._start_time,
            'end_time': self._end_time
        }


class ReinforceJob(ABC):
    """Abstract base class for reinforcement learning jobs with concrete utilities."""
    
    def __init__(self, lm: Model, train_kwargs: Dict[str, Any]):
        """
        Initialize reinforcement learning job.
        
        Args:
            lm: Language model instance
            train_kwargs: RL training parameters
        """
        self.lm = lm
        self.train_kwargs = train_kwargs
        self._step_count = 0
        self._rewards = []
        self._start_time = None
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the RL training setup."""
        pass
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """Perform one RL training step."""
        pass
    
    @abstractmethod
    def finalize(self) -> Model:
        """Finalize training and return updated model."""
        pass
    
    def get_step_count(self) -> int:
        """Get current step count."""
        return self._step_count
    
    def add_reward(self, reward: float) -> None:
        """Add a reward to the tracking."""
        self._rewards.append(reward)
    
    def get_average_reward(self) -> float:
        """Get average reward across all steps."""
        if not self._rewards:
            return 0.0
        return sum(self._rewards) / len(self._rewards)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get RL training statistics.
        
        Returns:
            Dictionary with training stats
        """
        duration = None
        if self._start_time:
            duration = time.time() - self._start_time
        
        return {
            'step_count': self._step_count,
            'total_rewards': len(self._rewards),
            'average_reward': self.get_average_reward(),
            'max_reward': max(self._rewards) if self._rewards else 0.0,
            'min_reward': min(self._rewards) if self._rewards else 0.0,
            'duration': duration,
            'model': self.lm.model if self.lm else None,
            'provider': self.lm.provider if self.lm else None
        }
