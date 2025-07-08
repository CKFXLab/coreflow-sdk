"""Configuration helper functions for model creation."""

from typing import Dict, Any, Optional
from ..registry import get_model_registry, get_provider_defaults, Provider

# Get registry instance
_registry = get_model_registry()


def openai_config(model: str, 
                  api_key: Optional[str] = None,
                  temperature: float = 0.7,
                  max_tokens: int = 1000,
                  timeout: int = 30,
                  **kwargs) -> Dict[str, Any]:
    """
    Create OpenAI model configuration with sensible defaults.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo")
        api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        temperature: Sampling temperature between 0.0 and 1.0
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        **kwargs: Additional OpenAI-specific parameters
        
    Returns:
        Dictionary configuration for ModelFactory
        
    Example:
        >>> config = openai_config("gpt-4", temperature=0.3)
        >>> model = create_model(config)
    """
    config = get_provider_defaults(Provider.OPENAI)
    config.update({
        "provider": Provider.OPENAI.value, 
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout
    })
    if api_key:
        config["api_key"] = api_key
    config.update(kwargs)
    return config


def anthropic_config(model: str,
                     api_key: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: int = 1000,
                     timeout: int = 30,
                     **kwargs) -> Dict[str, Any]:
    """
    Create Anthropic API model configuration with sensible defaults.
    
    Args:
        model: Anthropic model name (e.g., "claude-3-opus-20240229", "claude-3-haiku-20240307")
        api_key: Anthropic API key (if not provided, will use ANTHROPIC_API_KEY env var)
        temperature: Sampling temperature between 0.0 and 1.0
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        **kwargs: Additional Anthropic-specific parameters
        
    Returns:
        Dictionary configuration for ModelFactory
        
    Example:
        >>> config = anthropic_config("claude-3-haiku-20240307", temperature=0.3)
        >>> model = create_model(config)
    """
    config = get_provider_defaults(Provider.ANTHROPIC)
    config.update({
        "provider": Provider.ANTHROPIC.value,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout
    })
    if api_key:
        config["api_key"] = api_key
    config.update(kwargs)
    return config


def bedrock_config(model: str,
                   region_name: str = "us-east-1",
                   aws_access_key_id: Optional[str] = None,
                   aws_secret_access_key: Optional[str] = None,
                   aws_session_token: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Create Bedrock model configuration with sensible defaults.
    
    Args:
        model: Bedrock model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
        region_name: AWS region name
        aws_access_key_id: AWS access key (if not provided, will use env var or IAM)
        aws_secret_access_key: AWS secret key (if not provided, will use env var or IAM)
        aws_session_token: AWS session token for temporary credentials
        **kwargs: Additional Bedrock-specific parameters
        
    Returns:
        Dictionary configuration for ModelFactory
        
    Example:
        >>> config = bedrock_config("anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")
        >>> model = create_model(config)
    """
    config = get_provider_defaults(Provider.BEDROCK)
    config.update({
        "provider": Provider.BEDROCK.value,
        "model": model, 
        "region_name": region_name
    })
    
    # Add AWS credentials if provided
    if aws_access_key_id:
        config["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        config["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        config["aws_session_token"] = aws_session_token
        
    config.update(kwargs)
    return config





# === CONVENIENCE SHORTCUTS (using registry data) ===

def gpt4_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for GPT-4."""
    return openai_config("gpt-4", temperature=temperature, **kwargs)


def gpt4o_mini_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for GPT-4o Mini."""
    return openai_config("gpt-4o-mini", temperature=temperature, **kwargs)


# === ANTHROPIC API CONVENIENCE SHORTCUTS ===

def claude35_sonnet_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.5 Sonnet (Anthropic API)."""
    model_info = _registry.get_model("claude-3.5-sonnet")
    if model_info:
        return anthropic_config(model_info.model_id, temperature=temperature, **kwargs)
    return anthropic_config("claude-3-5-sonnet-20241022", temperature=temperature, **kwargs)


def claude35_haiku_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.5 Haiku (Anthropic API)."""
    model_info = _registry.get_model("claude-3.5-haiku")
    if model_info:
        return anthropic_config(model_info.model_id, temperature=temperature, **kwargs)
    return anthropic_config("claude-3-5-haiku-20241022", temperature=temperature, **kwargs)


def claude3_opus_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3 Opus (Anthropic API)."""
    model_info = _registry.get_model("claude-3-opus")
    if model_info:
        return anthropic_config(model_info.model_id, temperature=temperature, **kwargs)
    return anthropic_config("claude-3-opus-20240229", temperature=temperature, **kwargs)


def claude3_sonnet_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.5 Sonnet (Anthropic API) - alias for backward compatibility."""
    return claude35_sonnet_config(temperature=temperature, **kwargs)


def claude3_haiku_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3 Haiku (Anthropic API)."""
    model_info = _registry.get_model("claude-3-haiku")
    if model_info:
        return anthropic_config(model_info.model_id, temperature=temperature, **kwargs)
    return anthropic_config("claude-3-haiku-20240307", temperature=temperature, **kwargs)


def claude2_config(temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 2.1 (Anthropic API)."""
    model_info = _registry.get_model("claude-2.1")
    if model_info:
        return anthropic_config(model_info.model_id, temperature=temperature, **kwargs)
    return anthropic_config("claude-2.1", temperature=temperature, **kwargs)


# === BEDROCK CONVENIENCE SHORTCUTS (using registry data) ===

def bedrock_claude4_sonnet_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 4 Sonnet (AWS Bedrock)."""
    model_info = _registry.get_model("bedrock-claude-4-sonnet")
    if model_info:
        return bedrock_config(model_info.model_id, region_name=region_name, **kwargs)
    return bedrock_config("us.anthropic.claude-sonnet-4-20250514-v1:0", region_name=region_name, **kwargs)


def bedrock_claude37_sonnet_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.7 Sonnet (AWS Bedrock)."""
    model_info = _registry.get_model("bedrock-claude-3.7-sonnet")
    if model_info:
        return bedrock_config(model_info.model_id, region_name=region_name, **kwargs)
    return bedrock_config("us.anthropic.claude-3-7-sonnet-20250219-v1:0", region_name=region_name, **kwargs)


def bedrock_claude35_sonnet_v2_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.5 Sonnet v2 (AWS Bedrock)."""
    model_info = _registry.get_model("bedrock-claude-3.5-sonnet-v2")
    if model_info:
        return bedrock_config(model_info.model_id, region_name=region_name, **kwargs)
    return bedrock_config("us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name=region_name, **kwargs)


def bedrock_claude35_sonnet_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3.5 Sonnet (AWS Bedrock)."""
    model_info = _registry.get_model("bedrock-claude-3.5-sonnet")
    if model_info:
        return bedrock_config(model_info.model_id, region_name=region_name, **kwargs)
    return bedrock_config("us.anthropic.claude-3-5-sonnet-20240620-v1:0", region_name=region_name, **kwargs)


def bedrock_claude3_sonnet_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3 Sonnet (AWS Bedrock) - legacy model."""
    # This model is not in the registry (removed for access issues), so use hardcoded fallback
    return bedrock_config("anthropic.claude-3-sonnet-20240229-v1:0", region_name=region_name, **kwargs)


def bedrock_claude3_haiku_config(region_name: str = "us-east-1", **kwargs) -> Dict[str, Any]:
    """Quick configuration for Claude 3 Haiku (AWS Bedrock)."""
    model_info = _registry.get_model("bedrock-claude-3-haiku")
    if model_info:
        return bedrock_config(model_info.model_id, region_name=region_name, **kwargs)
    return bedrock_config("anthropic.claude-3-haiku-20240307-v1:0", region_name=region_name, **kwargs)


 