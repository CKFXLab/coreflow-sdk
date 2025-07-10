"""
Centralized Model Registry - Single Source of Truth for All Model Information

This registry contains comprehensive model definitions, capabilities, and metadata
for all supported providers. It serves as the single source of truth to eliminate
duplication across _default.py, config.py, and provider implementations.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import copy


class Provider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


class ModelType(Enum):
    """Supported model types."""

    CHAT = "chat"
    TEXT = "text"


@dataclass
class ModelInfo:
    """Complete model information and capabilities."""

    # Core identification
    model_id: str
    provider: Provider
    model_type: ModelType
    display_name: str
    description: str

    # Technical capabilities
    max_tokens: int
    context_window: int
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False

    # Pricing (USD per million tokens)
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    # Provider-specific settings
    provider_settings: Dict[str, Any] = field(default_factory=dict)

    # Availability and status
    is_available: bool = True
    is_deprecated: bool = False
    deprecation_date: Optional[str] = None
    replacement_model: Optional[str] = None

    # API-specific information
    api_version: Optional[str] = None
    release_date: Optional[str] = None

    # Aliases and shortcuts
    aliases: List[str] = field(default_factory=list)


class ModelRegistry:
    """Centralized registry for all model information."""

    def __init__(self):
        """Initialize the model registry with all supported models."""
        self._models: Dict[str, ModelInfo] = {}
        self._provider_defaults: Dict[Provider, Dict[str, Any]] = {}
        self._initialize_models()
        self._initialize_provider_defaults()

    def _initialize_models(self):
        """Initialize all model definitions."""

        # OpenAI Models
        self._add_model(
            ModelInfo(
                model_id="gpt-4o",
                provider=Provider.OPENAI,
                model_type=ModelType.CHAT,
                display_name="GPT-4o",
                description="Most advanced multimodal flagship model",
                max_tokens=4096,
                context_window=128000,
                supports_vision=True,
                supports_functions=True,
                supports_json_mode=True,
                input_price_per_million=5.00,
                output_price_per_million=15.00,
                aliases=["gpt-4o", "gpt4o"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="gpt-4o-mini",
                provider=Provider.OPENAI,
                model_type=ModelType.CHAT,
                display_name="GPT-4o Mini",
                description="Affordable and intelligent small model for fast, lightweight tasks",
                max_tokens=16384,
                context_window=128000,
                supports_vision=True,
                supports_functions=True,
                supports_json_mode=True,
                input_price_per_million=0.15,
                output_price_per_million=0.60,
                aliases=["gpt-4o-mini", "gpt4o-mini"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="gpt-4",
                provider=Provider.OPENAI,
                model_type=ModelType.CHAT,
                display_name="GPT-4",
                description="Previous generation flagship model",
                max_tokens=4096,
                context_window=8192,
                supports_functions=True,
                supports_json_mode=True,
                input_price_per_million=30.00,
                output_price_per_million=60.00,
                aliases=["gpt-4", "gpt4"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="gpt-3.5-turbo",
                provider=Provider.OPENAI,
                model_type=ModelType.CHAT,
                display_name="GPT-3.5 Turbo",
                description="Fast, inexpensive model for simple tasks",
                max_tokens=4096,
                context_window=16385,
                supports_functions=True,
                supports_json_mode=True,
                input_price_per_million=0.50,
                output_price_per_million=1.50,
                aliases=["gpt-3.5-turbo", "gpt35-turbo"],
            )
        )

        # Anthropic API Models
        self._add_model(
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude 3.5 Sonnet",
                description="Latest and most capable model with enhanced reasoning",
                max_tokens=8192,
                context_window=200000,
                input_price_per_million=3.00,
                output_price_per_million=15.00,
                release_date="2024-10-22",
                aliases=["claude-3.5-sonnet", "claude35-sonnet"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="claude-3-5-haiku-20241022",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude 3.5 Haiku",
                description="Fast and efficient model with improved capabilities",
                max_tokens=8192,
                context_window=200000,
                input_price_per_million=1.00,
                output_price_per_million=5.00,
                release_date="2024-10-22",
                aliases=["claude-3.5-haiku", "claude35-haiku"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="claude-3-opus-20240229",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude 3 Opus",
                description="Most powerful model for highly complex tasks",
                max_tokens=4096,
                context_window=200000,
                input_price_per_million=15.00,
                output_price_per_million=75.00,
                release_date="2024-02-29",
                aliases=["claude-3-opus", "claude3-opus"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="claude-3-haiku-20240307",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude 3 Haiku",
                description="Fastest and most compact model",
                max_tokens=4096,
                context_window=200000,
                input_price_per_million=0.25,
                output_price_per_million=1.25,
                release_date="2024-03-07",
                aliases=["claude-3-haiku", "claude3-haiku"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="claude-2.1",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude 2.1",
                description="Previous generation model",
                max_tokens=4096,
                context_window=200000,
                input_price_per_million=8.00,
                output_price_per_million=24.00,
                aliases=["claude-2.1", "claude2.1"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="claude-instant-1.2",
                provider=Provider.ANTHROPIC,
                model_type=ModelType.CHAT,
                display_name="Claude Instant 1.2",
                description="Fast and cost-effective model",
                max_tokens=4096,
                context_window=100000,
                input_price_per_million=0.80,
                output_price_per_million=2.40,
                aliases=["claude-instant", "claude-instant-1.2"],
            )
        )

        # AWS Bedrock Models (System Inference Profiles - Working Models Only)
        self._add_model(
            ModelInfo(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                provider=Provider.BEDROCK,
                model_type=ModelType.CHAT,
                display_name="Claude 4 Sonnet",
                description="Latest Claude 4 model via AWS Bedrock system inference profile",
                max_tokens=8192,
                context_window=200000,
                provider_settings={"region_name": "us-east-1"},
                input_price_per_million=3.00,  # Estimated
                output_price_per_million=15.00,  # Estimated
                aliases=["bedrock-claude-4-sonnet", "claude-4-sonnet"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                provider=Provider.BEDROCK,
                model_type=ModelType.CHAT,
                display_name="Claude 3.7 Sonnet",
                description="Claude 3.7 model via AWS Bedrock system inference profile",
                max_tokens=8192,
                context_window=200000,
                provider_settings={"region_name": "us-east-1"},
                input_price_per_million=3.00,  # Estimated
                output_price_per_million=15.00,  # Estimated
                aliases=["bedrock-claude-3.7-sonnet", "claude-3.7-sonnet"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                provider=Provider.BEDROCK,
                model_type=ModelType.CHAT,
                display_name="Claude 3.5 Sonnet v2",
                description="Claude 3.5 Sonnet v2 via AWS Bedrock system inference profile",
                max_tokens=8192,
                context_window=200000,
                provider_settings={"region_name": "us-east-1"},
                input_price_per_million=3.00,
                output_price_per_million=15.00,
                aliases=["bedrock-claude-3.5-sonnet-v2", "claude-3.5-sonnet-v2"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                provider=Provider.BEDROCK,
                model_type=ModelType.CHAT,
                display_name="Claude 3.5 Sonnet",
                description="Claude 3.5 Sonnet via AWS Bedrock system inference profile",
                max_tokens=8192,
                context_window=200000,
                provider_settings={"region_name": "us-east-1"},
                input_price_per_million=3.00,
                output_price_per_million=15.00,
                aliases=["bedrock-claude-3.5-sonnet", "claude-3.5-sonnet-bedrock"],
            )
        )

        self._add_model(
            ModelInfo(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider=Provider.BEDROCK,
                model_type=ModelType.CHAT,
                display_name="Claude 3 Haiku",
                description="Claude 3 Haiku via AWS Bedrock",
                max_tokens=4096,
                context_window=200000,
                provider_settings={"region_name": "us-east-1"},
                input_price_per_million=0.25,
                output_price_per_million=1.25,
                aliases=["bedrock-claude-3-haiku", "claude-3-haiku-bedrock"],
            )
        )

    def _initialize_provider_defaults(self):
        """Initialize provider-specific default parameters."""
        self._provider_defaults = {
            Provider.OPENAI: {
                "model_type": "chat",
                "timeout": 30,
                "num_retries": 3,
                "cache": True,
            },
            Provider.ANTHROPIC: {
                "model_type": "chat",
                "timeout": 30,
                "num_retries": 3,
                "cache": True,
            },
            Provider.BEDROCK: {
                "model_type": "chat",
                "region_name": "us-east-1",
                "num_retries": 3,
                "cache": True,
            },
        }

    def _add_model(self, model_info: ModelInfo):
        """Add a model to the registry."""
        self._models[model_info.model_id] = model_info

        # Also register aliases
        for alias in model_info.aliases:
            self._models[alias] = model_info

    # === QUERY METHODS ===

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID or alias."""
        return self._models.get(model_id)

    def get_models_by_provider(self, provider: Union[Provider, str]) -> List[ModelInfo]:
        """Get all models for a specific provider."""
        if isinstance(provider, str):
            provider = Provider(provider)

        # Use a set to avoid duplicates from aliases
        unique_models = set()
        for model in self._models.values():
            if model.provider == provider:
                unique_models.add(model.model_id)

        return [self._models[model_id] for model_id in unique_models]

    def get_available_models(self, provider: Union[Provider, str] = None) -> List[str]:
        """Get list of available model IDs, optionally filtered by provider."""
        if provider is None:
            # Return all unique model IDs (no aliases)
            unique_models = set()
            for model in self._models.values():
                if model.is_available and not model.is_deprecated:
                    unique_models.add(model.model_id)
            return sorted(list(unique_models))

        models = self.get_models_by_provider(provider)
        return [m.model_id for m in models if m.is_available and not m.is_deprecated]

    def get_default_model(self, provider: Union[Provider, str]) -> Optional[str]:
        """Get the default model for a provider."""
        if isinstance(provider, str):
            provider = Provider(provider)

        # Define default models per provider
        defaults = {
            Provider.OPENAI: "gpt-4o-mini",
            Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            Provider.BEDROCK: "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        }

        return defaults.get(provider)

    def get_provider_defaults(self, provider: Union[Provider, str]) -> Dict[str, Any]:
        """Get default parameters for a provider."""
        if isinstance(provider, str):
            provider = Provider(provider)

        return copy.deepcopy(self._provider_defaults.get(provider, {}))

    def estimate_cost(
        self, model_id: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost for a model usage."""
        model = self.get_model(model_id)
        if not model:
            return 0.0

        input_cost = prompt_tokens * (model.input_price_per_million / 1_000_000)
        output_cost = completion_tokens * (model.output_price_per_million / 1_000_000)
        return input_cost + output_cost

    def search_models(
        self, query: str, provider: Union[Provider, str] = None
    ) -> List[ModelInfo]:
        """Search models by name, description, or alias."""
        query = query.lower()

        models = (
            self.get_models_by_provider(provider)
            if provider
            else list(self._models.values())
        )

        # Use set to avoid duplicates from aliases
        unique_results = set()
        for model in models:
            if (
                query in model.model_id.lower()
                or query in model.display_name.lower()
                or query in model.description.lower()
                or any(query in alias.lower() for alias in model.aliases)
            ):
                unique_results.add(model.model_id)

        return [self._models[model_id] for model_id in unique_results]

    def get_model_string_patterns(self) -> Dict[str, Dict[str, str]]:
        """Generate model string patterns for parsing (provider:model format)."""
        patterns = {}

        for model in self._models.values():
            # Skip aliases to avoid duplicates
            if model.model_id in [m.model_id for m in self._models.values()]:
                # Create short name patterns
                if model.provider == Provider.OPENAI:
                    if "gpt-4o" in model.model_id:
                        short_name = model.model_id  # Keep full name for GPT models
                    elif "gpt-4" in model.model_id:
                        short_name = "gpt-4"
                    elif "gpt-3.5" in model.model_id:
                        short_name = "gpt-3.5-turbo"
                    else:
                        short_name = model.model_id
                elif model.provider == Provider.ANTHROPIC:
                    if "claude-3-5-sonnet" in model.model_id:
                        short_name = "claude-3.5-sonnet"
                    elif "claude-3-5-haiku" in model.model_id:
                        short_name = "claude-3.5-haiku"
                    elif "claude-3-opus" in model.model_id:
                        short_name = "claude-3-opus"
                    elif "claude-3-haiku" in model.model_id:
                        short_name = "claude-3-haiku"
                    else:
                        short_name = model.model_id
                elif model.provider == Provider.BEDROCK:
                    if "claude-sonnet-4" in model.model_id:
                        short_name = "claude-4-sonnet"
                    elif "claude-3-7-sonnet" in model.model_id:
                        short_name = "claude-3.7-sonnet"
                    elif "claude-3-5-sonnet-20241022-v2" in model.model_id:
                        short_name = "claude-3.5-sonnet-v2"
                    elif "claude-3-5-sonnet-20240620" in model.model_id:
                        short_name = "claude-3.5-sonnet"
                    elif "claude-3-haiku" in model.model_id:
                        short_name = "claude-3-haiku"
                    else:
                        short_name = model.model_id
                else:
                    short_name = model.model_id

                pattern_key = f"{model.provider.value}:{short_name}"
                patterns[pattern_key] = {
                    "provider": model.provider.value,
                    "model": model.model_id,
                }

        return patterns

    def validate_model_config(self, provider: str, model_id: str) -> bool:
        """Validate that a model configuration is valid."""
        model = self.get_model(model_id)
        if not model:
            return False

        return model.provider.value == provider and model.is_available

    def get_all_providers(self) -> List[str]:
        """Get list of all supported providers."""
        return [provider.value for provider in Provider]

    def get_available_models_by_credentials(self, env=None) -> Dict[str, Any]:
        """
        Get available models filtered by actual credentials.

        Args:
            env: ENV instance to check credentials (creates new one if None)

        Returns:
            Dictionary with provider availability and filtered models
        """
        if env is None:
            from ..utils.env import ENV

            env = ENV()

        # Check credential availability
        available_credentials = env.get_available_credentials()

        providers = {}
        available_models = []

        for provider in Provider:
            provider_available = False
            provider_models = []

            # Check if provider has required credentials
            if provider == Provider.OPENAI:
                provider_available = available_credentials.get("openai_api_key", False)
            elif provider == Provider.ANTHROPIC:
                provider_available = available_credentials.get(
                    "anthropic_api_key", False
                )
            elif provider == Provider.BEDROCK:
                # AWS can work with multiple credential methods
                provider_available = (
                    available_credentials.get("aws_access_key_id", False)
                    or available_credentials.get("aws_profile_available", False)
                    or available_credentials.get("aws_instance_profile", False)
                )

            if provider_available:
                # Get models for this provider
                provider_models = self.get_available_models(provider)
                for model_id in provider_models:
                    model_info = self.get_model(model_id)
                    if model_info:
                        available_models.append(
                            {
                                "model_id": model_id,
                                "provider": provider.value,
                                "display_name": model_info.display_name,
                                "description": model_info.description,
                                "max_tokens": model_info.max_tokens,
                                "context_window": model_info.context_window,
                                "supports_streaming": model_info.supports_streaming,
                                "supports_functions": model_info.supports_functions,
                                "supports_vision": model_info.supports_vision,
                                "supports_json_mode": model_info.supports_json_mode,
                                "input_price_per_million": model_info.input_price_per_million,
                                "output_price_per_million": model_info.output_price_per_million,
                                "is_available": True,  # Only available models are included
                            }
                        )

            providers[provider.value] = {
                "available": provider_available,
                "models": provider_models,
                "credential_status": self._get_provider_credential_status(
                    provider, available_credentials
                ),
            }

        return {
            "providers": providers,
            "models": available_models,
            "total_models": len(available_models),
            "credential_summary": available_credentials,
        }

    def _get_provider_credential_status(
        self, provider: Provider, credentials: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Get detailed credential status for a provider."""
        if provider == Provider.OPENAI:
            return {
                "required": ["OPENAI_API_KEY"],
                "available": credentials.get("openai_api_key", False),
                "missing": (
                    []
                    if credentials.get("openai_api_key", False)
                    else ["OPENAI_API_KEY"]
                ),
            }
        elif provider == Provider.ANTHROPIC:
            return {
                "required": ["ANTHROPIC_API_KEY"],
                "available": credentials.get("anthropic_api_key", False),
                "missing": (
                    []
                    if credentials.get("anthropic_api_key", False)
                    else ["ANTHROPIC_API_KEY"]
                ),
            }
        elif provider == Provider.BEDROCK:
            aws_available = (
                credentials.get("aws_access_key_id", False)
                or credentials.get("aws_profile_available", False)
                or credentials.get("aws_instance_profile", False)
            )
            missing = []
            if not aws_available:
                missing = [
                    "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY",
                    "AWS_PROFILE",
                    "IAM_ROLE",
                ]

            return {
                "required": ["AWS credentials (multiple methods supported)"],
                "available": aws_available,
                "missing": missing,
                "methods": {
                    "environment_vars": credentials.get("aws_access_key_id", False),
                    "aws_profile": credentials.get("aws_profile_available", False),
                    "instance_profile": credentials.get("aws_instance_profile", False),
                },
            }

        return {"required": [], "available": False, "missing": ["Unknown provider"]}

    def get_best_available_model(self, env=None) -> Optional[str]:
        """
        Get the best available model ID based on credentials.

        Args:
            env: ENV instance to check credentials

        Returns:
            Model ID of the best available model, or None if no models available
        """
        if env is None:
            from ..utils.env import ENV

            env = ENV()

        model_config = env.get_best_available_model_config()
        if model_config:
            return model_config.get("model")
        return None

    def get_available_providers_list(self, env=None) -> List[str]:
        """
        Get list of available providers based on credentials.

        Args:
            env: ENV instance to check credentials

        Returns:
            List of available provider names
        """
        if env is None:
            from ..utils.env import ENV

            env = ENV()

        credentials = env.get_available_credentials()
        available_providers = []

        if credentials.get("openai_api_key", False):
            available_providers.append(Provider.OPENAI.value)
        if credentials.get("anthropic_api_key", False):
            available_providers.append(Provider.ANTHROPIC.value)
        if (
            credentials.get("aws_access_key_id", False)
            or credentials.get("aws_profile_available", False)
            or credentials.get("aws_instance_profile", False)
        ):
            available_providers.append(Provider.BEDROCK.value)

        return available_providers


# Global registry instance
_registry = ModelRegistry()


# Convenience functions
def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _registry


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model information by ID or alias."""
    return _registry.get_model(model_id)


def get_available_models(provider: str = None) -> List[str]:
    """Get list of available model IDs."""
    return _registry.get_available_models(provider)


def get_default_model(provider: str) -> Optional[str]:
    """Get the default model for a provider."""
    return _registry.get_default_model(provider)


def get_provider_defaults(provider: str) -> Dict[str, Any]:
    """Get default parameters for a provider."""
    return _registry.get_provider_defaults(provider)


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost for model usage."""
    return _registry.estimate_cost(model_id, prompt_tokens, completion_tokens)
