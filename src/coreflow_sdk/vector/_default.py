import os
from typing import Dict, Any, Optional, Union
from ..utils.env import ENV
from .utils.config import (
    create_vector_config_from_env,
    create_qdrant_config_from_env,
    create_embedding_client_from_env,
    get_config as get_vector_config,
    DEFAULT_CONFIG
)

USE_DOCKER_QDRANT = True  # Set to False to use in-memory mode
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection naming patterns for multi-tenant support using __ namespace separator
SYSTEM_COLLECTION_NAME = "system__default"
USER_COLLECTION_PREFIX = "user__"

# Default collection name for backwards compatibility
COLLECTION_NAME = "docs"

# Collection type enumeration
class CollectionType:
    USER = "user"
    SYSTEM = "system"


def create_embedding_client(env: Optional[ENV] = None):
    """
    Create an embedding client based on the configured provider.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Embedding client instance (OpenAI or Bedrock)
    """
    return create_embedding_client_from_env(env)


def create_vector_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Create dynamic vector store configuration based on environment variables.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Configuration dictionary for vector store, LLM, and embedder
    """
    return create_vector_config_from_env(env)


def create_qdrant_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Create Qdrant-specific configuration for RAG system.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with Qdrant configuration
    """
    return create_qdrant_config_from_env(env)


def get_embedding_client_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Get embedding client configuration based on provider.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with embedding client configuration
    """
    if env is None:
        env = ENV()
    
    provider = env.get_embedding_provider()
    config = {
        "provider": provider,
        "model": env.get_embedding_model(),
        "dimensions": env.get_embedding_dimensions(),
    }
    
    if provider == "bedrock":
        config.update({
            "region_name": env.get_aws_region(),
            "aws_access_key_id": env.get_aws_access_key_id(),
            "aws_secret_access_key": env.get_aws_secret_access_key(),
            "aws_session_token": env.get_aws_session_token(),
        })
    
    return config


# Static configuration for backward compatibility
# This will be used if no environment variables are set
CONFIG = DEFAULT_CONFIG


def get_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Get configuration with environment variable support and fallback to static config.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Configuration dictionary
    """
    return get_vector_config(env)