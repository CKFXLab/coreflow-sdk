"""
Main FastAPI application with credential-aware endpoints.

This module provides a complete FastAPI server that users can import and run.
It includes all the endpoints documented in the API reference with proper
credential detection and graceful degradation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from ...utils.env import ENV
from ...model.registry import get_model_registry
from ...workflow import BaseWorkflow
from .endpoints import chat, models, credentials, workflow, health


def create_app(
    title: str = "CoreFlow SDK API",
    version: str = "1.0.0",
    description: str = "AI-powered API with credential awareness and graceful degradation",
    enable_cors: bool = True,
    cors_origins: list = None,
) -> FastAPI:
    """
    Create a FastAPI application with CoreFlow SDK endpoints.

    Args:
        title: API title
        version: API version
        description: API description
        enable_cors: Whether to enable CORS middleware
        cors_origins: List of allowed CORS origins

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware if enabled
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include endpoint routers
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(credentials.router, prefix="/credentials", tags=["credentials"])
    app.include_router(workflow.router, prefix="/workflow", tags=["workflow"])
    app.include_router(health.router, prefix="/health", tags=["health"])

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": title,
            "version": version,
            "description": description,
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json",
            "timestamp": time.time(),
        }

    return app


# Default app instance that users can import directly
app = create_app()


# For backward compatibility with tests
@app.get("/models")
async def list_models_compat():
    """List available models based on credentials (compatibility endpoint)."""
    try:
        registry = get_model_registry()
        available = registry.get_available_models_by_credentials()

        return {
            "providers": available["providers"],
            "models": available["models"],
            "total_models": available["total_models"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/credentials")
async def get_credentials_status_compat():
    """Get credential availability status (compatibility endpoint)."""
    try:
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()

        return {"credentials": credentials, "disabled_features": disabled_features}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get credentials: {str(e)}"
        )


@app.get("/workflow/status")
async def get_workflow_status_compat():
    """Get workflow component status (compatibility endpoint)."""
    try:
        workflow_instance = BaseWorkflow()
        status = workflow_instance.get_component_status()
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()

        return {
            "credentials": credentials,
            "disabled_features": disabled_features,
            "component_status": status,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
