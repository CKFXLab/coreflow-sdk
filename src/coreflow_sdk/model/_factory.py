"""
ModelFactory for creating model instances from configuration dictionaries.

This factory supports creating models from:
- Configuration dictionaries: {"provider": "openai", "model": "gpt-4", "api_key": "..."}
- Already instantiated Model objects (pass-through)

For type safety and better IDE support, use configuration helper functions:
- openai_config("gpt-4", temperature=0.7)
- bedrock_config("claude-3-sonnet", region="us-west-2")
- llamaserver_config("llama-2-7b")
"""

import importlib
import os
from typing import Union, Dict, Any, Optional
from ._mabc import Model
from ._default import (
    DEFAULT_MODELS, DEFAULT_PROVIDER_PARAMS, MODEL_STRING_PATTERNS,
    PROVIDER_CLASS_MAP, PROVIDER_ENV_VARS, PROVIDER_IMPORT_PATHS,
    DEFAULT_FALLBACK_CONFIG, Provider, ModelType
)

try:
    from ..utils import AppLogger
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ..utils import AppLogger


class ModelFactory:
    """
    Factory for creating model instances from configuration dictionaries.
    
    Supports these input formats:
    1. Dictionary format: {"provider": "openai", "model": "gpt-4", ...}
    2. Model instance: Returns as-is (pass-through)
    
    For better type safety and IDE support, use configuration helper functions
    from sdk.model.utils.config instead of raw dictionaries.
    """
    
    def __init__(self):
        """Initialize the ModelFactory."""
        self.logger = AppLogger(__name__)
        self._provider_cache = {}  # Cache for imported provider classes
    
    def create_model(self, 
                    model_config: Union[Dict[str, Any], Model],
                    **override_kwargs) -> Model:
        """
        Create a model instance from configuration dictionary or pre-instantiated model.
        
        Args:
            model_config: Model configuration in one of these formats:
                - Dict: {"provider": "openai", "model": "gpt-4", "api_key": "..."}
                - Model: Already instantiated model (returned as-is)
            **override_kwargs: Additional parameters to override defaults
            
        Returns:
            Instantiated Model object
            
        Raises:
            ValueError: If configuration is invalid or provider not supported
            ImportError: If provider dependencies are not available
        """
        try:
            # Handle already instantiated models
            if isinstance(model_config, Model):
                self.logger.info(f"Using pre-instantiated model: {model_config.__class__.__name__}")
                return model_config
            
            # Validate that we have a dictionary
            if not isinstance(model_config, dict):
                raise ValueError(f"model_config must be a dictionary, got {type(model_config)}")
            
            # Copy config and merge with override kwargs
            config = model_config.copy()
            config.update(override_kwargs)
            
            # Validate configuration
            self._validate_config(config)
            
            # Create and return model instance
            return self._instantiate_model(config)
            
        except Exception as e:
            self.logger.error(f"Failed to create model from config {model_config}: {e}")
            raise
    

    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate and normalize the configuration dictionary.
        
        Args:
            config: Configuration to validate and normalize
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Ensure required fields
        provider = config.get("provider")
        if not provider:
            raise ValueError("Provider must be specified in configuration")
        
        if provider not in PROVIDER_CLASS_MAP:
            available = list(PROVIDER_CLASS_MAP.keys())
            raise ValueError(f"Unsupported provider: {provider}. Available: {available}")
        
        # Set default model if not specified
        if "model" not in config:
            if provider in DEFAULT_MODELS:
                config["model"] = DEFAULT_MODELS[provider]
            else:
                raise ValueError(f"Model must be specified for provider: {provider}")
        
        # Merge with provider defaults (user config takes precedence)
        if provider in DEFAULT_PROVIDER_PARAMS:
            defaults = DEFAULT_PROVIDER_PARAMS[provider].copy()
            defaults.update(config)
            config.clear()
            config.update(defaults)
        
        # Check required environment variables
        if provider in PROVIDER_ENV_VARS:
            missing_vars = []
            for var in PROVIDER_ENV_VARS[provider]:
                if not os.getenv(var) and var.lower().replace("_", "") not in config:
                    missing_vars.append(var)
            
            if missing_vars:
                self.logger.warning(
                    f"Missing environment variables for {provider}: {missing_vars}. "
                    "Make sure they're provided in config or set as environment variables."
                )
    
    def _instantiate_model(self, config: Dict[str, Any]) -> Model:
        """
        Instantiate the model from validated configuration.
        
        Args:
            config: Validated configuration dictionary
            
        Returns:
            Model instance
        """
        provider = config["provider"]
        
        # Get the provider class
        provider_class = self._get_provider_class(provider)
        
        # Prepare initialization arguments
        init_kwargs = self._prepare_init_kwargs(config, provider)
        
        # Instantiate the model
        self.logger.info(f"Creating {provider} model: {config['model']}")
        model_instance = provider_class(**init_kwargs)
        
        self.logger.info(f"Successfully created {provider} model instance")
        return model_instance
    
    def _get_provider_class(self, provider: str):
        """
        Get the provider class, using cache for performance.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider class
        """
        if provider in self._provider_cache:
            return self._provider_cache[provider]
        
        # Import the provider module
        if provider not in PROVIDER_IMPORT_PATHS:
            raise ValueError(f"No import path configured for provider: {provider}")
        
        module_path = PROVIDER_IMPORT_PATHS[provider]
        class_name = PROVIDER_CLASS_MAP[provider]
        
        try:
            # Dynamic import
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            
            # Cache for future use
            self._provider_cache[provider] = provider_class
            
            return provider_class
            
        except ImportError as e:
            raise ImportError(f"Failed to import {provider} provider. "
                            f"Make sure dependencies are installed: {e}")
        except AttributeError as e:
            raise ImportError(f"Provider class {class_name} not found in {module_path}: {e}")
    
    def _prepare_init_kwargs(self, config: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Prepare initialization kwargs for the specific provider.
        
        Args:
            config: Full configuration dictionary
            provider: Provider name
            
        Returns:
            Kwargs suitable for provider class initialization
        """
        # Start with all config
        kwargs = config.copy()
        
        # Remove factory-specific keys that shouldn't be passed to model
        factory_keys = {"provider"}
        for key in factory_keys:
            kwargs.pop(key, None)
        
        # Provider-specific argument mapping
        if provider == "openai":
            # Map generic keys to OpenAI-specific parameter names
            if "api_key" not in kwargs:
                kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        
        elif provider == "bedrock":
            # Map generic keys to Bedrock-specific parameter names
            if "model" in kwargs:
                kwargs["model_id"] = kwargs.pop("model")  # Bedrock uses model_id parameter
            if "aws_access_key_id" not in kwargs:
                kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            if "aws_secret_access_key" not in kwargs:
                kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
            if "aws_session_token" not in kwargs:
                kwargs["aws_session_token"] = os.getenv("AWS_SESSION_TOKEN")
        
        elif provider == "llamaserver":
            # LlamaServer-specific parameters
            pass  # No special mapping needed currently
        
        return kwargs
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available providers.
        
        Returns:
            Dictionary with provider information
        """
        providers = {}
        
        for provider in PROVIDER_CLASS_MAP.keys():
            providers[provider] = {
                "class_name": PROVIDER_CLASS_MAP[provider],
                "default_model": DEFAULT_MODELS.get(provider),
                "required_env_vars": PROVIDER_ENV_VARS.get(provider, []),
                "default_params": DEFAULT_PROVIDER_PARAMS.get(provider, {}),
                "available": self._check_provider_availability(provider)
            }
        
        return providers
    
    def _check_provider_availability(self, provider: str) -> bool:
        """
        Check if a provider is available (dependencies installed).
        
        Args:
            provider: Provider name
            
        Returns:
            True if provider is available, False otherwise
        """
        try:
            self._get_provider_class(provider)
            return True
        except (ImportError, AttributeError):
            return False
    
    def create_default_model(self, provider: Optional[str] = None) -> Model:
        """
        Create a model with default configuration.
        
        Args:
            provider: Provider to use (defaults to openai)
            
        Returns:
            Model instance with default configuration
        """
        if provider is None:
            provider = Provider.OPENAI
        
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"No default configuration for provider: {provider}")
        
        config = {
            "provider": provider,
            "model": DEFAULT_MODELS[provider]
        }
        
        return self.create_model(config)


# Global factory instance for convenience
_factory = ModelFactory()

# Convenience functions
def create_model(model_config: Union[Dict[str, Any], Model], **kwargs) -> Model:
    """
    Convenience function to create a model using the global factory.
    
    Args:
        model_config: Model configuration dictionary or Model instance
        **kwargs: Additional parameters
        
    Returns:
        Model instance
    """
    return _factory.create_model(model_config, **kwargs)

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get available providers.
    
    Returns:
        Dictionary with provider information
    """
    return _factory.get_available_providers()

def create_default_model(provider: Optional[str] = None) -> Model:
    """
    Convenience function to create a default model.
    
    Args:
        provider: Provider to use
        
    Returns:
        Model instance
    """
    return _factory.create_default_model(provider) 