"""
Health endpoint for system health checks.
"""

from fastapi import APIRouter
import time

from ....utils.env import ENV

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        Basic health status
    """
    return {
        "status": "healthy",
        "service": "coreflow-sdk-api",
        "timestamp": time.time(),
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with component status.

    Returns:
        Comprehensive health information
    """
    try:
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()

        # Count available features
        available_features = sum(1 for available in credentials.values() if available)
        total_features = len(credentials)

        # Determine overall health
        health_status = "healthy" if available_features > 0 else "degraded"

        return {
            "status": health_status,
            "service": "coreflow-sdk-api",
            "credentials": {
                "available_count": available_features,
                "total_count": total_features,
                "details": credentials,
            },
            "features": {
                "disabled_count": len(disabled_features),
                "disabled_features": disabled_features,
            },
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "coreflow-sdk-api",
            "error": str(e),
            "timestamp": time.time(),
        }


@router.get("/components")
async def component_health_check():
    """
    Check health of individual components.

    Returns:
        Component-specific health status
    """
    try:
        from ....workflow import BaseWorkflow

        workflow = BaseWorkflow()
        component_status = workflow.get_component_status()

        return {
            "status": "healthy" if all(component_status.values()) else "degraded",
            "components": component_status,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": time.time()}
