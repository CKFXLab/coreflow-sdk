"""
Vector utilities package.

This package provides utilities for vector operations including:
- Configuration management
- Collection naming and management
- File operations for RAG systems
"""

from .config import (
    # Embedding provider configurations
    openai_embedding_config,
    bedrock_embedding_config,
    # Qdrant vector store configurations
    qdrant_local_config,
    qdrant_cloud_config,
    qdrant_fargate_config,
    # Complete system configurations
    openai_qdrant_local_config,
    bedrock_qdrant_cloud_config,
    bedrock_qdrant_fargate_config,
    # Environment-based configurations
    create_vector_config_from_env,
    create_qdrant_config_from_env,
    create_embedding_client_from_env,
    # Convenience shortcuts
    openai_local_config,
    bedrock_cloud_config,
    bedrock_fargate_config,
    # Legacy support
    get_config,
    DEFAULT_CONFIG,
)

from .collections import (
    sanitize_username,
    create_user_collection_name,
    create_system_collection_name,
    parse_collection_name,
    is_user_collection,
    is_system_collection,
    validate_collection_name,
)

from .file_operations import FileInfo, TextChunk, FileOperations

__all__ = [
    # Configuration functions
    "openai_embedding_config",
    "bedrock_embedding_config",
    "qdrant_local_config",
    "qdrant_cloud_config",
    "qdrant_fargate_config",
    "openai_qdrant_local_config",
    "bedrock_qdrant_cloud_config",
    "bedrock_qdrant_fargate_config",
    "create_vector_config_from_env",
    "create_qdrant_config_from_env",
    "create_embedding_client_from_env",
    "openai_local_config",
    "bedrock_cloud_config",
    "bedrock_fargate_config",
    "get_config",
    "DEFAULT_CONFIG",
    # Collection utilities
    "sanitize_username",
    "create_user_collection_name",
    "create_system_collection_name",
    "parse_collection_name",
    "is_user_collection",
    "is_system_collection",
    "validate_collection_name",
    # File operations
    "FileInfo",
    "TextChunk",
    "FileOperations",
]
