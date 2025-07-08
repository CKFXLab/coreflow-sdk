# Vector Configuration Guide

This guide covers the consolidated configuration utilities for vector operations, embeddings, and RAG systems in CoreFlow.

## Overview

The vector configuration system has been consolidated into `sdk/vector/utils/config.py` to provide convenient, reusable configuration functions for different deployment scenarios.

## Quick Start

### Development Setup (OpenAI + Local Qdrant)
```python
from sdk.vector.utils.config import openai_qdrant_local_config

config = openai_qdrant_local_config(
    collection_name="dev_documents",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)
```

### Production Setup (Bedrock + Cloud Qdrant)
```python
from sdk.vector.utils.config import bedrock_qdrant_cloud_config

config = bedrock_qdrant_cloud_config(
    collection_name="prod_documents",
    embedding_model="amazon.titan-embed-text-v1",
    llm_model="anthropic.claude-3-haiku-20240307-v1:0",
    qdrant_url="https://xyz.qdrant.tech",
    qdrant_api_key="your-api-key",
    region_name="us-east-1"
)
```

### Scalable Setup (Bedrock + Fargate Qdrant)
```python
from sdk.vector.utils.config import bedrock_qdrant_fargate_config

config = bedrock_qdrant_fargate_config(
    collection_name="scalable_documents",
    service_name="my-qdrant-service",
    cluster_name="my-qdrant-cluster",
    region_name="us-west-2"
)
```

## Configuration Functions

### Embedding Provider Configurations

#### OpenAI Embeddings
```python
from sdk.vector.utils.config import openai_embedding_config

config = openai_embedding_config(
    model="text-embedding-3-large",  # or "text-embedding-3-small"
    api_key="your-api-key"  # optional, uses env var if not provided
)
```

#### Bedrock Embeddings
```python
from sdk.vector.utils.config import bedrock_embedding_config

config = bedrock_embedding_config(
    model="amazon.titan-embed-text-v1",  # or "amazon.titan-embed-text-v2:0"
    region_name="us-east-1",
    aws_access_key_id="your-key",  # optional, uses env var or IAM
    aws_secret_access_key="your-secret"  # optional, uses env var or IAM
)
```

### Vector Store Configurations

#### Local Qdrant (Docker)
```python
from sdk.vector.utils.config import qdrant_local_config

config = qdrant_local_config(
    collection_name="my_documents",
    host="localhost",
    port=6333,
    use_docker=True,
    embedding_dimensions=1536
)
```

#### Cloud Qdrant
```python
from sdk.vector.utils.config import qdrant_cloud_config

config = qdrant_cloud_config(
    collection_name="my_documents",
    url="https://xyz.qdrant.tech",
    api_key="your-qdrant-api-key",
    embedding_dimensions=1536
)
```

#### Fargate Qdrant
```python
from sdk.vector.utils.config import qdrant_fargate_config

config = qdrant_fargate_config(
    collection_name="my_documents",
    service_name="qdrant-service",
    cluster_name="qdrant-cluster",
    region_name="us-east-1",
    embedding_dimensions=1536
)
```

### Complete System Configurations

#### OpenAI + Local Qdrant
```python
from sdk.vector.utils.config import openai_qdrant_local_config

config = openai_qdrant_local_config(
    collection_name="docs",
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    qdrant_host="localhost",
    qdrant_port=6333
)
```

#### Bedrock + Cloud Qdrant
```python
from sdk.vector.utils.config import bedrock_qdrant_cloud_config

config = bedrock_qdrant_cloud_config(
    collection_name="docs",
    embedding_model="amazon.titan-embed-text-v1",
    llm_model="anthropic.claude-3-haiku-20240307-v1:0",
    qdrant_url="https://xyz.qdrant.tech",
    qdrant_api_key="your-key",
    region_name="us-east-1"
)
```

#### Bedrock + Fargate Qdrant
```python
from sdk.vector.utils.config import bedrock_qdrant_fargate_config

config = bedrock_qdrant_fargate_config(
    collection_name="docs",
    embedding_model="amazon.titan-embed-text-v1",
    llm_model="anthropic.claude-3-haiku-20240307-v1:0",
    service_name="qdrant-service",
    cluster_name="qdrant-cluster",
    region_name="us-east-1"
)
```

## Environment-Based Configuration

### Using Environment Variables
```python
from sdk.vector.utils.config import create_vector_config_from_env

# Set environment variables:
# VECTOR_STORE_PROVIDER=qdrant
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# QDRANT_COLLECTION_NAME=my_collection

config = create_vector_config_from_env()
```

### Creating Embedding Client from Environment
```python
from sdk.vector.utils.config import create_embedding_client_from_env

# Uses EMBEDDING_PROVIDER and related env vars
client = create_embedding_client_from_env()
```

### Qdrant Configuration from Environment
```python
from sdk.vector.utils.config import create_qdrant_config_from_env

# Uses QDRANT_* environment variables
qdrant_config = create_qdrant_config_from_env()
```

## Convenience Shortcuts

### Quick Development Setup
```python
from sdk.vector.utils.config import openai_local_config

config = openai_local_config(collection_name="dev_test")
```

### Quick Production Setup
```python
from sdk.vector.utils.config import bedrock_cloud_config

config = bedrock_cloud_config(
    qdrant_url="https://xyz.qdrant.tech",
    qdrant_api_key="your-key"
)
```

### Quick Scalable Setup
```python
from sdk.vector.utils.config import bedrock_fargate_config

config = bedrock_fargate_config(
    service_name="my-qdrant",
    cluster_name="my-cluster"
)
```

## Supported Models

### OpenAI Embedding Models
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions)

### Bedrock Embedding Models
- `amazon.titan-embed-text-v1` (1536 dimensions)
- `amazon.titan-embed-text-v2:0` (1024 dimensions)
- `cohere.embed-english-v3` (1024 dimensions)
- `cohere.embed-multilingual-v3` (1024 dimensions)

## Deployment Modes

### Local Development
- **Qdrant**: Docker container on localhost
- **Embeddings**: OpenAI API
- **LLM**: OpenAI API (gpt-4o-mini)

### Cloud Production
- **Qdrant**: Managed Qdrant Cloud
- **Embeddings**: AWS Bedrock Titan
- **LLM**: AWS Bedrock Claude

### Scalable Fargate
- **Qdrant**: ECS Fargate service
- **Embeddings**: AWS Bedrock Titan
- **LLM**: AWS Bedrock Claude

## Migration from Legacy Configuration

### Before (in _default.py)
```python
from sdk.vector._default import CONFIG, create_vector_config

config = create_vector_config()
```

### After (using new utilities)
```python
from sdk.vector.utils.config import openai_qdrant_local_config

config = openai_qdrant_local_config()
```

## Examples

See `examples/vector_config_examples.py` for complete working examples of all configuration scenarios.

## Environment Variables

All configuration functions support environment variables for sensitive data:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `AWS_REGION`
- `QDRANT_API_KEY`
- `QDRANT_URL`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `QDRANT_COLLECTION_NAME`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `VECTOR_STORE_PROVIDER`

## Backward Compatibility

The legacy `CONFIG` dictionary and `get_config()` function are still available for backward compatibility, but new code should use the specific configuration functions for better type safety and clarity.

## Integration with Workflow System

The vector configuration system integrates seamlessly with the new workflow factory:

```python
from sdk.workflow import create_workflow, single_agent_config
from sdk.vector.utils.config import openai_qdrant_local_config

# Create vector config
vector_config = openai_qdrant_local_config()

# Use in workflow
workflow = create_workflow(single_agent_config(
    enable_rag=True,
    # Vector config is automatically loaded from environment
))
```

## Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Environment variable setup
- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Creating and configuring workflows
- [FastAPI Integration](FASTAPI_INTEGRATION.md) - Building APIs with vector support
- [API Reference](API_REFERENCE.md) - Complete API documentation 