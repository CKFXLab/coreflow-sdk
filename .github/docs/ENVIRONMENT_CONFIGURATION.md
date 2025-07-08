# Environment Configuration Guide

This guide explains how to configure the CoreFlow SDK using environment variables with enhanced dotenv support and proper environment precedence.

## 🚀 Overview

The CoreFlow SDK supports dynamic configuration through environment variables with Node.js-style dotenv file loading and **credential awareness**:

- **Environment Detection**: Automatic detection of development, production, test, and local environments
- **Dotenv File Precedence**: Proper loading order with local overrides
- **Project Root Detection**: Automatic detection of project root directory
- **Credential Awareness**: Automatic detection of available API keys and graceful degradation
- **Auto Model Selection**: Intelligent selection of best available model provider
- **Feature Disabling**: Graceful disabling of features when credentials are missing
- **Multi-Provider Support**: Switch between OpenAI, Anthropic, and AWS Bedrock providers

## 📁 Dotenv File Structure

The system follows Node.js conventions for environment file loading:

### Loading Order (Highest to Lowest Precedence)

1. **System Environment Variables** (highest precedence)
2. **`.env.local`** - Local overrides (gitignored)
3. **`.env.[environment].local`** - Environment-specific local overrides
4. **`.env.[environment]`** - Environment-specific settings

**Note**: `.env` and `.env.example` are NOT loaded automatically (following Node.js conventions)

### Example File Structure

```
project-root/
├── .env.development       # Development settings
├── .env.production       # Production settings
├── .env.test            # Test settings
├── .env.local           # Local overrides (gitignored)
├── .env.development.local  # Dev local overrides (gitignored)
├── .env.production.local   # Prod local overrides (gitignored)
└── .env.example         # Example file (not loaded)
```

## 🔧 Environment Variables

### Required Credentials (At least one)

**⚠️ Critical**: You MUST have at least one model provider credential for the system to work:

```bash
# Model Providers (choose at least one)
OPENAI_API_KEY=your_openai_api_key_here           # Priority 1 - Recommended
ANTHROPIC_API_KEY=your_anthropic_api_key_here     # Priority 2 - Alternative
AWS_ACCESS_KEY_ID=your_aws_access_key             # Priority 3 - For Bedrock
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_SESSION_TOKEN=your_aws_session_token          # Optional for temporary credentials
```

### Optional Credentials (Enable additional features)

```bash
# Optional features - system works without these but with reduced functionality
SERPER_API_KEY=your_serper_api_key_here           # Enables web search
MEM0_API_KEY=your_mem0_api_key_here               # Enables cloud memory (falls back to local)
HF_TOKEN=your_huggingface_token_here              # Enables HuggingFace models
```

### Environment Detection

```bash
# Environment setting (auto-detected)
NODE_ENV=development
# Alternatives: ENV, ENVIRONMENT
# Values: development, production, test, local
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

## 🔄 Environment-Specific Configuration

### Development Environment (`.env.development`)

```bash
# Development with OpenAI embeddings and local Qdrant
NODE_ENV=development
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_USE_DOCKER=true
QDRANT_DEPLOYMENT_MODE=local

# Development debugging
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Environment (`.env.production`)

```bash
# Production with Bedrock embeddings and cloud Qdrant
NODE_ENV=production
EMBEDDING_PROVIDER=bedrock
EMBEDDING_MODEL=amazon.titan-embed-text-v1
AWS_REGION=us-east-1
QDRANT_URL=https://your-qdrant-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_DEPLOYMENT_MODE=cloud

# Production settings
DEBUG=false
LOG_LEVEL=INFO
```

### Test Environment (`.env.test`)

```bash
# Test environment with minimal setup
NODE_ENV=test
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_USE_DOCKER=true

# Test-specific settings
LOG_LEVEL=WARNING
```

### Local Overrides (`.env.local`)

```bash
# Personal local overrides (gitignored)
# Override any setting for local development
OPENAI_API_KEY=your_personal_openai_key
DEBUG=true
LOG_LEVEL=DEBUG

# Local development customizations
QDRANT_PORT=6334  # Use different port
```

## 💻 Usage in Code

### Automatic Configuration

The system automatically detects and applies environment configuration:

```python
from sdk.utils.env import ENV
from sdk.vector import RAG, Mem0
from sdk.workflow import BaseWorkflow

# Environment is automatically loaded
env = ENV()

# Components automatically use environment configuration
rag = RAG()
memory = Mem0()
workflow = BaseWorkflow()
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

print("Environment:", config_info['environment'])
print("Project Root:", config_info['project_root'])
print("Dotenv Files:", config_info['dotenv_files'])
print("Vector Config:", config_info['vector_config'])

# Check credential availability
credentials = env.get_available_credentials()
print("Available credentials:")
for service, available in credentials.items():
    print(f"  {service}: {'✅' if available else '❌'}")

# Check disabled features
disabled = env.get_disabled_features()
if disabled:
    print(f"Disabled features: {disabled}")
else:
    print("All features available!")

# Get best available model
model_config = env.get_best_available_model_config()
if model_config:
    print(f"Selected model: {model_config['provider']}:{model_config['model']}")
else:
    print("❌ No model provider credentials available!")
```

### Environment Validation

```python
from sdk.utils.env import validate_required_env

# Validate specific environment variables
required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
try:
    validate_required_env(required_vars)
    print("All required environment variables are set")
except ValueError as e:
    print(f"Missing environment variables: {e}")
```

### Manual Dotenv Loading

```python
from sdk.utils.env import load_dotenv_files, find_dotenv_files

# Load dotenv files manually
loaded_vars = load_dotenv_files(verbose=True)
print("Loaded variables:", list(loaded_vars.keys()))

# Find available dotenv files
dotenv_files = find_dotenv_files()
print("Available dotenv files:", [str(f) for f in dotenv_files])
```

## 🏗️ Project Structure Detection

The system automatically detects your project root by looking for common markers:

```python
from sdk.utils.env import find_project_root

# Automatically find project root
project_root = find_project_root()
print("Project root:", project_root)
```

**Project markers searched (in order):**
- `requirements.txt`
- `pyproject.toml`
- `setup.py`
- `Pipfile`
- `poetry.lock`
- `.git`
- `package.json`
- `Dockerfile`
- `docker-compose.yml`

## 🔧 Supported Embedding Models

### OpenAI Models
- `text-embedding-3-small` (1536 dimensions) - Default, cost-effective
- `text-embedding-3-large` (3072 dimensions) - Higher quality
- `text-embedding-ada-002` (1536 dimensions) - Legacy model

### AWS Bedrock Models
- `amazon.titan-embed-text-v1` (1536 dimensions) - General purpose
- `amazon.titan-embed-text-v2:0` (1024 dimensions) - Newer version
- `cohere.embed-english-v3` (1024 dimensions) - English text
- `cohere.embed-multilingual-v3` (1024 dimensions) - Multilingual

## 🚀 Deployment Scenarios

### Local Development
```bash
# .env.development
NODE_ENV=development
EMBEDDING_PROVIDER=openai
QDRANT_HOST=localhost
QDRANT_USE_DOCKER=true
```

### Production Cloud
```bash
# .env.production
NODE_ENV=production
EMBEDDING_PROVIDER=bedrock
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_api_key
AWS_REGION=us-east-1
```

### AWS Fargate
```bash
# .env.production
NODE_ENV=production
EMBEDDING_PROVIDER=bedrock
QDRANT_URL=http://qdrant-service.namespace.svc.cluster.local:6333
QDRANT_DEPLOYMENT_MODE=fargate
AWS_REGION=us-west-2
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   # Check which keys are missing
   python -c "from sdk.utils.env import ENV; env = ENV(); print(env.missing)"
   ```

2. **Dotenv Files Not Loading**
   ```bash
   # Check which files are being loaded
   python -c "from sdk.utils.env import find_dotenv_files; print([str(f) for f in find_dotenv_files()])"
   ```

3. **Environment Detection Issues**
   ```bash
   # Check detected environment
   python -c "from sdk.utils.env import get_environment; print(get_environment())"
   ```

4. **Project Root Detection**
   ```bash
   # Check detected project root
   python -c "from sdk.utils.env import find_project_root; print(find_project_root())"
   ```

### Debug Configuration

```python
from sdk.utils.env import ENV

# Create ENV instance with debugging
env = ENV()

# Get comprehensive environment info
info = env.get_env_info()
print("=" * 50)
print("ENVIRONMENT CONFIGURATION DEBUG")
print("=" * 50)
print(f"Environment: {info['environment']}")
print(f"Project Root: {info['project_root']}")
print(f"Dotenv Files: {info['dotenv_files']}")
print(f"All Keys Present: {info['all_keys_present']}")
print(f"Missing Keys: {info['missing_keys']}")
print(f"Vector Config: {info['vector_config']}")
```

### Validation Errors

```python
from sdk.utils.env import ENV

try:
    env = ENV()
    print("Environment validation successful")
except ValueError as e:
    print(f"Environment validation failed: {e}")
    # The error message includes:
    # - Missing variables
    # - Current environment
    # - Expected dotenv files
    # - Available dotenv files
```

## 🔄 Migration from Static Configuration

### Phase 1: Add Environment Variables
```bash
# Add to .env.development
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### Phase 2: Update Code
```python
# Before
from sdk.vector import RAG

rag = RAG(api_key="hardcoded_key")

# After
from sdk.vector import RAG

rag = RAG()  # Automatically uses environment
```

### Phase 3: Environment-Specific Configuration
```bash
# .env.development
EMBEDDING_PROVIDER=openai
QDRANT_HOST=localhost

# .env.production
EMBEDDING_PROVIDER=bedrock
QDRANT_URL=https://prod-cluster.qdrant.tech
```

## 📖 Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Credential Awareness](CREDENTIAL_AWARENESS.md) - Complete guide to credential detection and graceful degradation
- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Creating and configuring workflows
- [Vector Configuration](VECTOR_CONFIGURATION.md) - RAG and vector store setup
- [FastAPI Integration](FASTAPI_INTEGRATION.md) - Building APIs with proper environment setup
- [Model Configuration](MODEL_CONFIGURATION.md) - Configuring LLM providers

## 🎯 Best Practices

1. **Never commit `.env.local` files** - Add to `.gitignore`
2. **Use environment-specific files** - `.env.development`, `.env.production`
3. **Validate required variables** - Use `validate_required_env()`
4. **Use system environment variables in production** - Don't rely on files
5. **Document your environment variables** - Keep `.env.example` updated
6. **Use proper precedence** - Local overrides should be in `.env.local`

The enhanced environment system provides robust, flexible configuration management that scales from development to production! 