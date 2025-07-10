"""
Models endpoint for listing available models based on credentials.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from ....model.registry import get_model_registry
from ....utils.env import ENV

router = APIRouter()


@router.get("/")
async def list_models(provider: Optional[str] = None):
    """
    List all available models and providers based on credentials.

    Args:
        provider: Optional provider filter (openai, anthropic, bedrock)

    Returns:
        Dictionary with providers, models, and total count
    """
    try:
        registry = get_model_registry()
        available = registry.get_available_models_by_credentials()

        # Filter by provider if specified
        if provider:
            if provider in available["providers"]:
                filtered_models = [
                    model
                    for model in available["models"]
                    if model.get("provider") == provider
                ]
                return {
                    "providers": {provider: available["providers"][provider]},
                    "models": filtered_models,
                    "total_models": len(filtered_models),
                }
            else:
                return {"providers": {}, "models": [], "total_models": 0}

        return {
            "providers": available["providers"],
            "models": available["models"],
            "total_models": available["total_models"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/{model_name}")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.

    Args:
        model_name: Model identifier

    Returns:
        Detailed model information
    """
    try:
        registry = get_model_registry()
        model_info = registry.get_model(model_name)

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        return {
            "model_id": model_info.model_id,
            "provider": model_info.provider.value,
            "model_type": model_info.model_type.value,
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
            "is_available": model_info.is_available,
            "is_deprecated": model_info.is_deprecated,
            "aliases": model_info.aliases,
            "release_date": model_info.release_date,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/{model_name}/health")
async def check_model_health(model_name: str):
    """
    Perform a health check on a specific model.

    Args:
        model_name: Model identifier

    Returns:
        Health status information
    """
    try:
        registry = get_model_registry()
        model_info = registry.get_model(model_name)

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Check if model is available based on credentials
        env = ENV()
        credentials = env.get_available_credentials()

        # Determine if model is healthy based on provider credentials
        healthy = False
        status = "unavailable"
        error = None

        if model_info.provider.value == "openai" and credentials.get("openai_api_key"):
            healthy = True
            status = "operational"
        elif model_info.provider.value == "anthropic" and credentials.get(
            "anthropic_api_key"
        ):
            healthy = True
            status = "operational"
        elif model_info.provider.value == "bedrock" and (
            credentials.get("aws_access_key_id")
            or credentials.get("aws_profile_available")
            or credentials.get("aws_instance_profile")
        ):
            healthy = True
            status = "operational"
        else:
            error = f"Missing credentials for {model_info.provider.value} provider"

        return {
            "model_name": model_name,
            "provider": model_info.provider.value,
            "healthy": healthy,
            "status": status,
            "error": error,
            "timestamp": __import__("time").time(),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "model_name": model_name,
            "healthy": False,
            "status": "error",
            "error": str(e),
            "timestamp": __import__("time").time(),
        }
