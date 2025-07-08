# Embedding Configuration Guide

This guide explains how to configure the embedding and vector store system using environment variables.

## Overview

The system now supports dynamic configuration through environment variables, allowing you to:
- Switch between OpenAI and AWS Bedrock embedding providers
- Configure Qdrant for local Docker or cloud/Fargate deployments
- Customize embedding models and dimensions
- Set up different configurations for development, staging, and production

## Environment Variables

### Required API Keys

```bash
# Required for basic functionality
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HF_TOKEN=your_huggingface_token_here
MEM0_API_KEY=your_mem0_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

### Embedding Configuration

```bash
# Embedding provider (default: openai)
EMBEDDING_PROVIDER=openai
# Options: openai, bedrock

# Embedding model (auto-detected based on provider)
EMBEDDING_MODEL=text-embedding-3-small
# OpenAI options: text-embedding-3-small, text-embedding-3-large
# Bedrock options: amazon.titan-embed-text-v1, cohere.embed-english-v3

# Embedding dimensions (auto-detected based on model)
EMBEDDING_DIMENSIONS=1536
# 1536 for text-embedding-3-small and titan-embed
# 3072 for text-embedding-3-large
```

### Vector Store Configuration

```bash
# Vector store provider (default: qdrant)
VECTOR_STORE_PROVIDER=qdrant

# Collection name for mem0 (default: system__mem0)
QDRANT_COLLECTION_NAME=system__mem0
```

### Qdrant Deployment Modes

#### Local Docker (Default)
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_USE_DOCKER=true
QDRANT_DEPLOYMENT_MODE=local
```

#### Cloud/Fargate Deployment
```bash
QDRANT_URL=https://your-qdrant-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_DEPLOYMENT_MODE=cloud
```

#### AWS Fargate Service
```bash
QDRANT_URL=http://qdrant-service.your-namespace.svc.cluster.local:6333
QDRANT_DEPLOYMENT_MODE=fargate
```

### AWS Configuration (for Bedrock)

```bash
# AWS region (default: us-east-1)
AWS_REGION=us-east-1

# AWS credentials (optional - uses credential chain if not provided)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_SESSION_TOKEN=your_aws_session_token
```

## Configuration Examples

### Development Environment (.env.development)

```bash
# Local development with OpenAI embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_USE_DOCKER=true
```

### Production Environment (.env.production)

```bash
# Production with Bedrock embeddings and cloud Qdrant
EMBEDDING_PROVIDER=bedrock
EMBEDDING_MODEL=amazon.titan-embed-text-v1
AWS_REGION=us-east-1
QDRANT_URL=https://your-qdrant-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_DEPLOYMENT_MODE=cloud
```

### Fargate Deployment (.env.production)

```bash
# Fargate with OpenAI embeddings and service discovery
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=http://qdrant-service.your-namespace.svc.cluster.local:6333
QDRANT_DEPLOYMENT_MODE=fargate
```

## Usage in Code

### Automatic Configuration

The system automatically detects and applies environment configuration:

```python
from sdk.vector import RAG, Mem0

# Automatically uses environment configuration
rag = RAG()
memory = Mem0()
```

### Manual Configuration

You can also pass configuration explicitly:

```python
from sdk.utils.env import ENV
from sdk.vector import RAG, Mem0

# Create environment instance
env = ENV()

# Pass to components
rag = RAG(env=env)
memory = Mem0(env=env)
```

### Configuration Inspection

```python
from sdk.utils.env import ENV

env = ENV()
config_info = env.get_env_info()
print(config_info['vector_config'])
```

## Supported Embedding Models

### OpenAI Models
- `text-embedding-3-small` (1536 dimensions) - Default
- `text-embedding-3-large` (3072 dimensions) - Higher quality

### AWS Bedrock Models
- `amazon.titan-embed-text-v1` (1536 dimensions) - General purpose
- `cohere.embed-english-v3` (1024 dimensions) - English text
- `cohere.embed-multilingual-v3` (1024 dimensions) - Multilingual

## Deployment Scenarios

### Local Development
- Use Docker Qdrant on localhost
- OpenAI embeddings for quick setup
- Default configuration works out of the box

### Production Cloud
- Use managed Qdrant service
- Bedrock embeddings for cost optimization
- URL-based Qdrant connection with API key

### AWS Fargate
- Use ECS service discovery for Qdrant
- Bedrock embeddings in same region
- No API keys needed with IAM roles

## Migration from Static Configuration

The system maintains backward compatibility. Existing code continues to work without changes, but you can gradually adopt environment-based configuration:

1. **Phase 1**: Set environment variables alongside existing code
2. **Phase 2**: Remove hardcoded parameters from initialization
3. **Phase 3**: Leverage new deployment modes and providers

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set
2. **Qdrant Connection**: Check host/port or URL configuration
3. **Embedding Dimensions**: Verify dimensions match your model
4. **AWS Credentials**: Ensure proper AWS credential chain setup

### Debug Configuration

```python
from sdk.utils.env import ENV

env = ENV()
print(env.get_env_info())
```

This will show all loaded configuration values and help identify issues.

## Integration with New Features

### Workflow Factory Integration

The embedding configuration works seamlessly with the new workflow factory system:

```python
from sdk.workflow import create_workflow, single_agent_config
from sdk.utils.env import ENV

# Environment automatically configures embeddings
workflow = create_workflow(single_agent_config(
    enable_rag=True,
    enable_memory=True
))

# Check current embedding configuration
env = ENV()
print("Embedding provider:", env.get_embedding_provider())
print("Embedding model:", env.get_embedding_model())
print("Embedding dimensions:", env.get_embedding_dimensions())
```

### FastAPI Integration

Environment-based configuration works automatically with FastAPI endpoints:

```python
from fastapi import FastAPI
from sdk.workflow import BaseWorkflow

app = FastAPI()

# Workflow automatically uses environment configuration
workflow = BaseWorkflow(
    enable_memory=True,
    enable_rag=True
)

@app.post("/chat")
async def chat(request: dict):
    return workflow.generate_response(request["query"])
```

### Enhanced Environment Management

The new environment system provides better validation and debugging:

```python
from sdk.utils.env import ENV, validate_required_env

# Validate specific keys
validate_required_env(["OPENAI_API_KEY", "EMBEDDING_PROVIDER"])

# Get comprehensive environment info
env = ENV()
env_info = env.get_env_info()
print("Environment details:", env_info)
```

## Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Complete environment setup guide
- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Creating and configuring workflows
- [Vector Configuration](VECTOR_CONFIGURATION.md) - RAG and vector store setup
- [FastAPI Integration](FASTAPI_INTEGRATION.md) - Building APIs with proper configuration
- [API Reference](API_REFERENCE.md) - Complete API documentation 