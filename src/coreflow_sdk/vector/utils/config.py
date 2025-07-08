"""
Configuration utilities for vector operations, embeddings, and RAG systems.

This module provides convenient configuration functions for:
- OpenAI and Bedrock embedding providers
- Qdrant vector store (local Docker and AWS deployments)
- Mem0 memory system configurations
- RAG system configurations
"""

from typing import Dict, Any, Optional
from ...utils.env import ENV


# === EMBEDDING PROVIDER CONFIGURATIONS ===

def openai_embedding_config(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create OpenAI embedding configuration.
    
    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        **kwargs: Additional OpenAI-specific parameters
        
    Returns:
        Dictionary configuration for OpenAI embeddings
        
    Example:
        >>> config = openai_embedding_config("text-embedding-3-large")
        >>> embedding_client = create_embedding_client_from_config(config)
    """
    config = {
        "provider": "openai",
        "model": model,
        "dimensions": _get_openai_embedding_dimensions(model),
    }
    
    if api_key:
        config["api_key"] = api_key
    
    config.update(kwargs)
    return config


def bedrock_embedding_config(
    model: str = "amazon.titan-embed-text-v1",
    region_name: str = "us-east-1",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create AWS Bedrock embedding configuration.
    
    Args:
        model: Bedrock embedding model ID
        region_name: AWS region name
        aws_access_key_id: AWS access key (if not provided, uses env var or IAM)
        aws_secret_access_key: AWS secret key (if not provided, uses env var or IAM)
        aws_session_token: AWS session token for temporary credentials
        **kwargs: Additional Bedrock-specific parameters
        
    Returns:
        Dictionary configuration for Bedrock embeddings
        
    Example:
        >>> config = bedrock_embedding_config("amazon.titan-embed-text-v2:0", region_name="us-west-2")
        >>> embedding_client = create_embedding_client_from_config(config)
    """
    config = {
        "provider": "bedrock",
        "model": model,
        "region_name": region_name,
        "dimensions": _get_bedrock_embedding_dimensions(model),
    }
    
    # Add AWS credentials if provided
    if aws_access_key_id:
        config["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        config["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        config["aws_session_token"] = aws_session_token
    
    config.update(kwargs)
    return config


# === QDRANT VECTOR STORE CONFIGURATIONS ===

def qdrant_local_config(
    collection_name: str = "docs",
    host: str = "localhost",
    port: int = 6333,
    use_docker: bool = True,
    embedding_dimensions: int = 1536,
    **kwargs
) -> Dict[str, Any]:
    """
    Create Qdrant local deployment configuration.
    
    Args:
        collection_name: Name of the Qdrant collection
        host: Qdrant host address
        port: Qdrant port number
        use_docker: Whether to use Docker deployment
        embedding_dimensions: Dimension size for embeddings
        **kwargs: Additional Qdrant-specific parameters
        
    Returns:
        Dictionary configuration for local Qdrant
        
    Example:
        >>> config = qdrant_local_config("my_documents", embedding_dimensions=1536)
        >>> rag_system = create_rag_system_from_config(config)
    """
    config = {
        "provider": "qdrant",
        "deployment_mode": "local",
        "host": host,
        "port": port,
        "collection_name": collection_name,
        "use_docker": use_docker,
        "embedding_dimensions": embedding_dimensions,
    }
    
    config.update(kwargs)
    return config


def qdrant_cloud_config(
    collection_name: str = "docs",
    url: str = None,
    api_key: Optional[str] = None,
    embedding_dimensions: int = 1536,
    **kwargs
) -> Dict[str, Any]:
    """
    Create Qdrant cloud deployment configuration.
    
    Args:
        collection_name: Name of the Qdrant collection
        url: Qdrant cloud URL
        api_key: Qdrant API key for authentication
        embedding_dimensions: Dimension size for embeddings
        **kwargs: Additional Qdrant-specific parameters
        
    Returns:
        Dictionary configuration for cloud Qdrant
        
    Example:
        >>> config = qdrant_cloud_config("my_documents", url="https://xyz.qdrant.tech", api_key="your-key")
        >>> rag_system = create_rag_system_from_config(config)
    """
    if not url:
        raise ValueError("URL is required for cloud Qdrant deployment")
    
    config = {
        "provider": "qdrant",
        "deployment_mode": "cloud",
        "url": url,
        "collection_name": collection_name,
        "embedding_dimensions": embedding_dimensions,
    }
    
    if api_key:
        config["api_key"] = api_key
    
    config.update(kwargs)
    return config


def qdrant_fargate_config(
    collection_name: str = "docs",
    service_name: str = "qdrant-service",
    cluster_name: str = "qdrant-cluster",
    region_name: str = "us-east-1",
    port: int = 6333,
    embedding_dimensions: int = 1536,
    **kwargs
) -> Dict[str, Any]:
    """
    Create Qdrant AWS Fargate deployment configuration.
    
    Args:
        collection_name: Name of the Qdrant collection
        service_name: ECS service name for Qdrant
        cluster_name: ECS cluster name
        region_name: AWS region name
        port: Qdrant port number
        embedding_dimensions: Dimension size for embeddings
        **kwargs: Additional Qdrant-specific parameters
        
    Returns:
        Dictionary configuration for Fargate Qdrant
        
    Example:
        >>> config = qdrant_fargate_config("my_documents", service_name="my-qdrant", cluster_name="my-cluster")
        >>> rag_system = create_rag_system_from_config(config)
    """
    config = {
        "provider": "qdrant",
        "deployment_mode": "fargate",
        "service_name": service_name,
        "cluster_name": cluster_name,
        "region_name": region_name,
        "port": port,
        "collection_name": collection_name,
        "embedding_dimensions": embedding_dimensions,
    }
    
    config.update(kwargs)
    return config


# === COMPLETE SYSTEM CONFIGURATIONS ===

def openai_qdrant_local_config(
    collection_name: str = "docs",
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete configuration for OpenAI embeddings with local Qdrant.
    
    Args:
        collection_name: Name of the Qdrant collection
        embedding_model: OpenAI embedding model
        llm_model: OpenAI LLM model for responses
        qdrant_host: Qdrant host address
        qdrant_port: Qdrant port number
        **kwargs: Additional parameters
        
    Returns:
        Complete system configuration dictionary
        
    Example:
        >>> config = openai_qdrant_local_config("my_docs", embedding_model="text-embedding-3-large")
        >>> rag_system = create_rag_system_from_config(config)
    """
    embedding_dims = _get_openai_embedding_dimensions(embedding_model)
    
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "host": qdrant_host,
                "port": qdrant_port,
                "embedding_model_dims": embedding_dims,
                "deployment_mode": "local",
                "use_docker": True,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0,
                "max_tokens": 2000,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
            }
        },
        **kwargs
    }


def bedrock_qdrant_cloud_config(
    collection_name: str = "docs",
    embedding_model: str = "amazon.titan-embed-text-v1",
    llm_model: str = "anthropic.claude-3-haiku-20240307-v1:0",
    qdrant_url: str = None,
    qdrant_api_key: Optional[str] = None,
    region_name: str = "us-east-1",
    **kwargs
) -> Dict[str, Any]:
    """
    Complete configuration for Bedrock embeddings with cloud Qdrant.
    
    Args:
        collection_name: Name of the Qdrant collection
        embedding_model: Bedrock embedding model
        llm_model: Bedrock LLM model for responses
        qdrant_url: Qdrant cloud URL
        qdrant_api_key: Qdrant API key
        region_name: AWS region name
        **kwargs: Additional parameters
        
    Returns:
        Complete system configuration dictionary
        
    Example:
        >>> config = bedrock_qdrant_cloud_config("my_docs", qdrant_url="https://xyz.qdrant.tech", qdrant_api_key="key")
        >>> rag_system = create_rag_system_from_config(config)
    """
    if not qdrant_url:
        raise ValueError("qdrant_url is required for cloud deployment")
    
    embedding_dims = _get_bedrock_embedding_dimensions(embedding_model)
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "url": qdrant_url,
                "embedding_model_dims": embedding_dims,
                "deployment_mode": "cloud",
            }
        },
        "llm": {
            "provider": "bedrock",
            "config": {
                "model": llm_model,
                "region_name": region_name,
                "temperature": 0,
                "max_tokens": 2000,
            }
        },
        "embedder": {
            "provider": "bedrock",
            "config": {
                "model": embedding_model,
                "region_name": region_name,
            }
        }
    }
    
    # Add Qdrant API key if provided
    if qdrant_api_key:
        config["vector_store"]["config"]["api_key"] = qdrant_api_key
    
    # Add AWS credentials if available from environment
    env = ENV()
    if env.get_aws_access_key_id():
        config["llm"]["config"]["aws_access_key_id"] = env.get_aws_access_key_id()
        config["embedder"]["config"]["aws_access_key_id"] = env.get_aws_access_key_id()
    if env.get_aws_secret_access_key():
        config["llm"]["config"]["aws_secret_access_key"] = env.get_aws_secret_access_key()
        config["embedder"]["config"]["aws_secret_access_key"] = env.get_aws_secret_access_key()
    if env.get_aws_session_token():
        config["llm"]["config"]["aws_session_token"] = env.get_aws_session_token()
        config["embedder"]["config"]["aws_session_token"] = env.get_aws_session_token()
    
    config.update(kwargs)
    return config


def bedrock_qdrant_fargate_config(
    collection_name: str = "docs",
    embedding_model: str = "amazon.titan-embed-text-v1",
    llm_model: str = "anthropic.claude-3-haiku-20240307-v1:0",
    service_name: str = "qdrant-service",
    cluster_name: str = "qdrant-cluster",
    region_name: str = "us-east-1",
    **kwargs
) -> Dict[str, Any]:
    """
    Complete configuration for Bedrock embeddings with Fargate Qdrant.
    
    Args:
        collection_name: Name of the Qdrant collection
        embedding_model: Bedrock embedding model
        llm_model: Bedrock LLM model for responses
        service_name: ECS service name for Qdrant
        cluster_name: ECS cluster name
        region_name: AWS region name
        **kwargs: Additional parameters
        
    Returns:
        Complete system configuration dictionary
        
    Example:
        >>> config = bedrock_qdrant_fargate_config("my_docs", service_name="my-qdrant", cluster_name="my-cluster")
        >>> rag_system = create_rag_system_from_config(config)
    """
    embedding_dims = _get_bedrock_embedding_dimensions(embedding_model)
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "service_name": service_name,
                "cluster_name": cluster_name,
                "region_name": region_name,
                "port": 6333,
                "embedding_model_dims": embedding_dims,
                "deployment_mode": "fargate",
            }
        },
        "llm": {
            "provider": "bedrock",
            "config": {
                "model": llm_model,
                "region_name": region_name,
                "temperature": 0,
                "max_tokens": 2000,
            }
        },
        "embedder": {
            "provider": "bedrock",
            "config": {
                "model": embedding_model,
                "region_name": region_name,
            }
        }
    }
    
    # Add AWS credentials if available from environment
    env = ENV()
    if env.get_aws_access_key_id():
        config["llm"]["config"]["aws_access_key_id"] = env.get_aws_access_key_id()
        config["embedder"]["config"]["aws_access_key_id"] = env.get_aws_access_key_id()
    if env.get_aws_secret_access_key():
        config["llm"]["config"]["aws_secret_access_key"] = env.get_aws_secret_access_key()
        config["embedder"]["config"]["aws_secret_access_key"] = env.get_aws_secret_access_key()
    if env.get_aws_session_token():
        config["llm"]["config"]["aws_session_token"] = env.get_aws_session_token()
        config["embedder"]["config"]["aws_session_token"] = env.get_aws_session_token()
    
    config.update(kwargs)
    return config


# === ENVIRONMENT-BASED CONFIGURATION ===

def create_vector_config_from_env(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Create dynamic vector store configuration based on environment variables.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Configuration dictionary for vector store, LLM, and embedder
    """
    if env is None:
        env = ENV()
    
    # Get configuration from environment
    vector_provider = env.get_vector_store_provider()
    embedding_provider = env.get_embedding_provider()
    embedding_model = env.get_embedding_model()
    embedding_dims = env.get_embedding_dimensions()
    
    # Build vector store config
    vector_config = {
        "provider": vector_provider,
        "config": {}
    }
    
    if vector_provider == "qdrant":
        # Start with basic required fields for Mem0 compatibility
        vector_config["config"] = {
            "collection_name": env.get_qdrant_collection_name(),
            "embedding_model_dims": embedding_dims,
        }
        
        # Handle different deployment modes
        deployment_mode = env.get_qdrant_deployment_mode()
        qdrant_url = env.get_qdrant_url()
        
        if qdrant_url:
            # URL-based configuration (for Fargate, cloud deployments, etc.)
            vector_config["config"]["url"] = qdrant_url
            
            # Add API key if provided
            api_key = env.get_qdrant_api_key()
            if api_key:
                vector_config["config"]["api_key"] = api_key
        else:
            # Host/port-based configuration (for local Docker or direct connection)
            vector_config["config"]["host"] = env.get_qdrant_host()
            vector_config["config"]["port"] = env.get_qdrant_port()
            
            # For local deployments, we may need to set path for in-memory fallback
            vector_config["config"]["path"] = "/tmp/qdrant"
    
    # Build LLM config (for mem0)
    llm_config = {
        "provider": "openai",  # Primary LLM provider for mem0
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 2000,
        },
    }
    
    # Build embedder config based on provider
    embedder_config = {
        "provider": embedding_provider,
        "config": {}
    }
    
    if embedding_provider == "openai":
        embedder_config["config"] = {
            "model": embedding_model,
        }
    elif embedding_provider == "bedrock":
        embedder_config["config"] = {
            "model": embedding_model,
            "region_name": env.get_aws_region(),
        }
        # Add AWS credentials if available
        if env.get_aws_access_key_id():
            embedder_config["config"]["aws_access_key_id"] = env.get_aws_access_key_id()
        if env.get_aws_secret_access_key():
            embedder_config["config"]["aws_secret_access_key"] = env.get_aws_secret_access_key()
        if env.get_aws_session_token():
            embedder_config["config"]["aws_session_token"] = env.get_aws_session_token()
    
    return {
        "vector_store": vector_config,
        "llm": llm_config,
        "embedder": embedder_config,
    }


def create_qdrant_config_from_env(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Create Qdrant-specific configuration for RAG system from environment.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with Qdrant configuration
    """
    if env is None:
        env = ENV()
    
    config = {
        "use_docker": env.get_qdrant_use_docker(),
        "embedding_dimensions": env.get_embedding_dimensions(),
        "collection_name": env.get_qdrant_collection_name(),
    }
    
    # Handle different deployment modes
    deployment_mode = env.get_qdrant_deployment_mode()
    qdrant_url = env.get_qdrant_url()
    
    if qdrant_url:
        # URL-based configuration (for Fargate, cloud deployments, etc.)
        config["url"] = qdrant_url
        config["deployment_mode"] = deployment_mode or "cloud"
        
        # Add API key if provided
        api_key = env.get_qdrant_api_key()
        if api_key:
            config["api_key"] = api_key
    else:
        # Host/port-based configuration (for local Docker or direct connection)
        config["host"] = env.get_qdrant_host()
        config["port"] = env.get_qdrant_port()
        config["deployment_mode"] = deployment_mode or "local"
    
    return config


def create_embedding_client_from_env(env: Optional[ENV] = None):
    """
    Create an embedding client based on environment configuration.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Embedding client instance (OpenAI or Bedrock)
    """
    if env is None:
        env = ENV()
    
    provider = env.get_embedding_provider()
    
    if provider == "openai":
        from ...model.api.openai import OpenAIClient
        return OpenAIClient()
    elif provider == "bedrock":
        from ...model.bedrock.anthropic import BedrockAnthropicClient
        return BedrockAnthropicClient(
            region_name=env.get_aws_region(),
            aws_access_key_id=env.get_aws_access_key_id(),
            aws_secret_access_key=env.get_aws_secret_access_key(),
            aws_session_token=env.get_aws_session_token()
        )
    else:
        # Default to OpenAI
        from ...model.api.openai import OpenAIClient
        return OpenAIClient()


# === UTILITY FUNCTIONS ===

def _get_openai_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for OpenAI models."""
    dimensions_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return dimensions_map.get(model, 1536)


def _get_bedrock_embedding_dimensions(model: str) -> int:
    """Get embedding dimensions for Bedrock models."""
    dimensions_map = {
        "amazon.titan-embed-text-v1": 1536,
        "amazon.titan-embed-text-v2:0": 1024,
        "cohere.embed-english-v3": 1024,
        "cohere.embed-multilingual-v3": 1024,
    }
    return dimensions_map.get(model, 1536)


# === BACKWARD COMPATIBILITY ===

# Static configuration for backward compatibility
DEFAULT_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "system__mem0",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536,
            "path": "/tmp/qdrant",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        },
    },
}


def get_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Get configuration with environment variable support and fallback to static config.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Configuration dictionary
    """
    try:
        return create_vector_config_from_env(env)
    except Exception:
        # Fallback to static config if environment loading fails
        return DEFAULT_CONFIG


# === CONVENIENCE SHORTCUTS ===

def openai_local_config(**kwargs) -> Dict[str, Any]:
    """Quick configuration for OpenAI with local Qdrant."""
    return openai_qdrant_local_config(**kwargs)


def bedrock_cloud_config(**kwargs) -> Dict[str, Any]:
    """Quick configuration for Bedrock with cloud Qdrant."""
    return bedrock_qdrant_cloud_config(**kwargs)


def bedrock_fargate_config(**kwargs) -> Dict[str, Any]:
    """Quick configuration for Bedrock with Fargate Qdrant."""
    return bedrock_qdrant_fargate_config(**kwargs) 