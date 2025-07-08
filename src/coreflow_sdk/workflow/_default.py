"""
Default configurations and constants for workflow components.

This module provides default values, prompt templates, and configuration helpers
for the workflow system. It includes credential-aware configuration functions
that automatically select the best available model based on available credentials.
"""

import os
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from ..utils.env import ENV

# Enum for workflow types
class WorkflowType(Enum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    API_ENHANCED = "api_enhanced"
    CUSTOM = "custom"

# Default agent role and instructions
AGENT_ROLE = """You are a helpful AI assistant with access to real-time information and knowledge bases."""

DEFAULT_INSTRUCTION = """Please provide a comprehensive and accurate response based on the context provided. 
If you don't have enough information to answer fully, please say so."""

WEB_DATA_INSTRUCTION = """Please provide a comprehensive response using both the knowledge base context and the current web information provided. 
Prioritize recent information when there are conflicts between sources."""

# Standardized prompt structure for consistent formatting
PROMPT_STRUCTURE = """Role: {agent_role}

Query: {query}

Context:
{context}"""

# Default workflow configuration
DEFAULT_WORKFLOW_CONFIG = {
    "use_docker_qdrant": True,
    "enable_websearch": True,
    "enable_rag": True,
    "enable_memory": True,
    "log_format": "ascii",
    "user_collection_prefix": "user_",
    "system_collection": "system"
}

# Default configurations for different workflow types
DEFAULT_WORKFLOW_CONFIGS = {
    "single_agent": {
        "workflow_type": "single_agent",
        "enable_memory": True,
        "enable_rag": True,
        "enable_websearch": True,
        "use_docker_qdrant": True,
        "log_format": "ascii",
        "user_collection_prefix": "user_",
        "system_collection": "system"
    },
    "multi_agent": {
        "workflow_type": "multi_agent",
        "coordination_strategy": "sequential",
        "enable_memory": True,
        "enable_rag": True,
        "enable_websearch": True,
        "use_docker_qdrant": True,
        "log_format": "ascii",
        "user_collection_prefix": "user_",
        "system_collection": "system"
    },
    "api_enhanced": {
        "workflow_type": "api_enhanced",
        "enable_function_calling": True,
        "enable_memory": True,
        "enable_rag": True,
        "enable_websearch": True,
        "use_docker_qdrant": True,
        "log_format": "ascii",
        "user_collection_prefix": "user_",
        "system_collection": "system"
    },
    "custom": {
        "workflow_type": "custom",
        "enable_memory": True,
        "enable_rag": True,
        "enable_websearch": True,
        "use_docker_qdrant": True,
        "log_format": "ascii",
        "user_collection_prefix": "user_",
        "system_collection": "system"
    }
}

# Mapping of workflow types to their corresponding classes
WORKFLOW_CLASS_MAP = {
    "single_agent": "BaseWorkflow",
    "multi_agent": "MultiAgentWorkflow",
    "api_enhanced": "APIWorkflow",
    "custom": "CustomWorkflow"
}

# Import paths for workflow classes
WORKFLOW_IMPORT_PATHS = {
    "single_agent": "sdk.workflow._wabc",
    "multi_agent": "sdk.workflow.multi_agent",
    "api_enhanced": "sdk.workflow.api_enhanced",
    "custom": "sdk.workflow.custom"
}

# Dependencies required for each workflow type
WORKFLOW_DEPENDENCIES = {
    "single_agent": ["model", "memory", "vector", "websearch"],
    "multi_agent": ["model", "memory", "vector", "websearch", "coordination"],
    "api_enhanced": ["model", "memory", "vector", "websearch", "function_calling"],
    "custom": ["model"]
}

# Fallback configuration when no specific config is provided
DEFAULT_FALLBACK_CONFIG = {
    "workflow_type": "single_agent",
    "enable_memory": True,
    "enable_rag": True,
    "enable_websearch": True,
    "use_docker_qdrant": True,
    "log_format": "ascii",
    "user_collection_prefix": "user_",
    "system_collection": "system"
}

# Coordination strategies for multi-agent workflows
COORDINATION_STRATEGIES = {
    "sequential": "Execute agents in sequence",
    "parallel": "Execute agents in parallel",
    "hierarchical": "Execute agents in hierarchical order",
    "consensus": "Execute agents and reach consensus"
}

# Default API configurations
DEFAULT_API_CONFIGS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "bedrock": {
        "provider": "bedrock",
        "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

def get_credential_aware_workflow_config(env: Optional[ENV] = None) -> Dict[str, Any]:
    """
    Get workflow configuration adjusted for available credentials.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with workflow configuration adjusted for credentials
    """
    if env is None:
        env = ENV()
    
    # Start with default config
    config = DEFAULT_WORKFLOW_CONFIG.copy()
    
    # Get available credentials
    credentials = env.get_available_credentials()
    
    # Adjust feature flags based on credentials
    config["enable_websearch"] = config["enable_websearch"] and credentials.get('serper_api_key', False)
    
    # Memory and RAG can always be enabled (they have fallbacks)
    # Model selection will be handled by get_best_available_model_config()
    
    return config

def get_best_available_model_config(env: Optional[ENV] = None) -> Optional[Dict[str, Any]]:
    """
    Get the best available model configuration based on credentials.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with model configuration for the best available provider, or None if no providers available
    """
    if env is None:
        env = ENV()
    
    return env.get_best_available_model_config()

def get_available_features(env: Optional[ENV] = None) -> Dict[str, bool]:
    """
    Get available features based on credentials.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary mapping feature names to availability
    """
    if env is None:
        env = ENV()
    
    credentials = env.get_available_credentials()
    
    return {
        "model_openai": credentials.get('openai_api_key', False),
        "model_anthropic": credentials.get('anthropic_api_key', False),
        "model_bedrock": (
            credentials.get('aws_access_key_id', False) or
            credentials.get('aws_profile_available', False) or
            credentials.get('aws_instance_profile', False)
        ),
        "websearch": credentials.get('serper_api_key', False),
        "memory_cloud": credentials.get('mem0_api_key', False),
        "memory_local": True,  # Always available
        "rag": True,  # Always available (with fallbacks)
        "huggingface": credentials.get('hf_token', False)
    }

def get_disabled_features_list(env: Optional[ENV] = None) -> List[str]:
    """
    Get list of disabled features due to missing credentials.
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        List of disabled feature names
    """
    if env is None:
        env = ENV()
    
    return env.get_disabled_features()

def create_workflow_with_best_model(
    use_docker_qdrant: bool = True,
    enable_websearch: bool = True,
    enable_rag: bool = True,
    enable_memory: bool = True,
    log_format: str = "ascii",
    user_collection_prefix: str = "user_",
    system_collection: str = "system",
    env: Optional[ENV] = None
) -> Dict[str, Any]:
    """
    Create workflow configuration with the best available model.
    
    Args:
        use_docker_qdrant: Whether to use Docker Qdrant or in-memory
        enable_websearch: Whether to enable web search (will be disabled if no SERPER_API_KEY)
        enable_rag: Whether to enable RAG/vector storage
        enable_memory: Whether to enable memory
        log_format: Logging format ("json" or "ascii")
        user_collection_prefix: Prefix for user-specific collections
        system_collection: Name of system-wide collection
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with complete workflow configuration
    """
    if env is None:
        env = ENV()
    
    # Get credential-aware config
    config = get_credential_aware_workflow_config(env)
    
    # Override with provided parameters
    config.update({
        "use_docker_qdrant": use_docker_qdrant,
        "enable_websearch": enable_websearch and config["enable_websearch"],  # Respect credential limitations
        "enable_rag": enable_rag,
        "enable_memory": enable_memory,
        "log_format": log_format,
        "user_collection_prefix": user_collection_prefix,
        "system_collection": system_collection
    })
    
    # Add model configuration
    model_config = get_best_available_model_config(env)
    if model_config:
        config["model_config"] = model_config
    
    return config

# Backward compatibility functions
def get_default_workflow_config() -> Dict[str, Any]:
    """
    Get default workflow configuration.
    
    Returns:
        Dictionary with default workflow configuration
    """
    return DEFAULT_WORKFLOW_CONFIG.copy()

# Legacy model configuration functions for backward compatibility
def get_default_model_config(env: Optional[ENV] = None) -> Optional[Dict[str, Any]]:
    """
    Get default model configuration (credential-aware).
    
    Args:
        env: ENV instance (creates new one if None)
        
    Returns:
        Dictionary with default model configuration
    """
    return get_best_available_model_config(env)