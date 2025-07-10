"""
Credentials endpoint for checking credential availability and feature status.
"""

from fastapi import APIRouter, HTTPException
import time

from ....utils.env import ENV

router = APIRouter()


@router.get("/")
async def get_credentials_status():
    """
    Get detailed credential availability and feature status.

    Returns:
        Comprehensive credential and feature status information
    """
    try:
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()
        provider_availability = env.get_provider_availability()
        model_config = env.get_best_available_model_config()

        return {
            "timestamp": time.time(),
            "credentials": {
                "available": credentials,
                "disabled_features": disabled_features,
                "selected_model": (
                    {
                        "provider": (
                            model_config.get("provider") if model_config else None
                        ),
                        "model": model_config.get("model") if model_config else None,
                    }
                    if model_config
                    else None
                ),
            },
            "providers": provider_availability,
            "recommendations": {
                "critical": "Ensure at least one model provider (OpenAI, Anthropic, or AWS) is configured",
                "optional": "Add SERPER_API_KEY for web search, MEM0_API_KEY for cloud memory",
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get credential status: {str(e)}"
        )


@router.get("/check")
async def check_specific_credentials(provider: str = None):
    """
    Check availability of specific provider credentials.

    Args:
        provider: Provider to check (openai, anthropic, bedrock, websearch, memory, huggingface)

    Returns:
        Specific provider credential status
    """
    try:
        env = ENV()
        provider_availability = env.get_provider_availability()

        if provider:
            if provider in provider_availability:
                return {
                    "provider": provider,
                    "status": provider_availability[provider],
                    "timestamp": time.time(),
                }
            else:
                raise HTTPException(
                    status_code=404, detail=f"Provider '{provider}' not found"
                )

        return {"providers": provider_availability, "timestamp": time.time()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to check credentials: {str(e)}"
        )


@router.get("/features")
async def get_feature_status():
    """
    Get status of all features based on available credentials.

    Returns:
        Feature availability status
    """
    try:
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()

        # Map features to their status
        features = {
            "model_openai": credentials.get("openai_api_key", False),
            "model_anthropic": credentials.get("anthropic_api_key", False),
            "model_bedrock": (
                credentials.get("aws_access_key_id", False)
                or credentials.get("aws_profile_available", False)
                or credentials.get("aws_instance_profile", False)
            ),
            "websearch": credentials.get("serper_api_key", False),
            "memory_cloud": credentials.get("mem0_api_key", False),
            "memory_local": True,  # Always available
            "huggingface": credentials.get("hf_token", False),
            "vector_storage": True,  # Always available with local Qdrant
        }

        return {
            "features": features,
            "disabled_features": disabled_features,
            "enabled_count": sum(1 for enabled in features.values() if enabled),
            "total_features": len(features),
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get feature status: {str(e)}"
        )
