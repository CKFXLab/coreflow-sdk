# Default model configurations and provider mappings for the SDK
# Now powered by centralized model registry

from .registry import (
    get_model_registry, 
    get_default_model, 
    get_provider_defaults,
    Provider,
    ModelType
)

# Get registry instance
_registry = get_model_registry()

# Default model configurations by provider (from registry)
DEFAULT_MODELS = {
    Provider.OPENAI.value: get_default_model(Provider.OPENAI),
    Provider.ANTHROPIC.value: get_default_model(Provider.ANTHROPIC),
    Provider.BEDROCK.value: get_default_model(Provider.BEDROCK),
}

# Provider-specific default parameters (from registry)
DEFAULT_PROVIDER_PARAMS = {
    Provider.OPENAI.value: get_provider_defaults(Provider.OPENAI),
    Provider.ANTHROPIC.value: get_provider_defaults(Provider.ANTHROPIC),
    Provider.BEDROCK.value: get_provider_defaults(Provider.BEDROCK),
}

# Model string format patterns for parsing (from registry)
MODEL_STRING_PATTERNS = _registry.get_model_string_patterns()

# Provider class mappings - maps provider names to their implementation classes
PROVIDER_CLASS_MAP = {
    Provider.OPENAI.value: "OpenAIClient",
    Provider.ANTHROPIC.value: "AnthropicClient",
    Provider.BEDROCK.value: "BedrockAnthropicClient",
}

# Required environment variables by provider
PROVIDER_ENV_VARS = {
    Provider.OPENAI.value: ["OPENAI_API_KEY"],
    Provider.ANTHROPIC.value: ["ANTHROPIC_API_KEY"],
    Provider.BEDROCK.value: [],  # Uses AWS credentials (optional env vars)
}

# Provider import paths for dynamic loading
PROVIDER_IMPORT_PATHS = {
    Provider.OPENAI.value: "sdk.model.api.openai",
    Provider.ANTHROPIC.value: "sdk.model.api.anthropic",
    Provider.BEDROCK.value: "sdk.model.bedrock.anthropic",
}

# Export ModelType and Provider from registry for backward compatibility
# (These are now defined in registry.py)

# Default fallback configuration (using registry)
DEFAULT_FALLBACK_CONFIG = {
    "provider": Provider.OPENAI.value,
    "model": DEFAULT_MODELS[Provider.OPENAI.value],
    "model_type": ModelType.CHAT.value,
    "timeout": 30,
    "num_retries": 3,
    "cache": True,
} 