# Credential Awareness in CoreFlow SDK

CoreFlow SDK implements a sophisticated credential-aware system that automatically detects available API keys and credentials, then gracefully disables features when credentials are missing. This ensures the system continues to function even with limited credentials while providing clear feedback about what's available.

## 🚨 Critical Requirements

**At least one of the following model provider credentials MUST be present for any functionality to work:**

- **OpenAI API Key** (`OPENAI_API_KEY`) - **RECOMMENDED**
- **AWS Credentials** (for Bedrock access) - **ALTERNATIVE**
- **Anthropic API Key** (`ANTHROPIC_API_KEY`) - **ALTERNATIVE**

**Without at least one model provider credential, the system will fail to initialize.**

## 📋 Complete Credential Reference

### Required Credentials (At least one)

| Credential | Environment Variable | Purpose | Priority |
|------------|---------------------|---------|----------|
| **OpenAI API Key** | `OPENAI_API_KEY` | Access to GPT models | 1st (Highest) |
| **AWS Credentials** | `AWS_ACCESS_KEY_ID`<br>`AWS_SECRET_ACCESS_KEY`<br>`AWS_SESSION_TOKEN` (optional) | Access to Bedrock models | 3rd |
| **Anthropic API Key** | `ANTHROPIC_API_KEY` | Access to Claude models | 2nd |

### Optional Credentials

| Credential | Environment Variable | Purpose | Fallback Behavior |
|------------|---------------------|---------|-------------------|
| **Serper API Key** | `SERPER_API_KEY` | Web search functionality | Web search disabled |
| **Mem0 API Key** | `MEM0_API_KEY` | Cloud memory service | Falls back to local memory |
| **HuggingFace Token** | `HF_TOKEN` | Access to HuggingFace models | HuggingFace features disabled |

### AWS Credential Methods

The system supports multiple AWS credential methods (in order of precedence):

1. **Environment Variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
2. **AWS Profile**: Configured via `aws configure` or `~/.aws/credentials`
3. **IAM Instance Profile**: For EC2 instances with attached IAM roles
4. **AWS Vault**: For temporary credential sessions

## 🔄 Automatic Model Selection

The system automatically selects the best available model based on credential availability:

### Priority Order:
1. **OpenAI** (if `OPENAI_API_KEY` present) → `gpt-4o-mini`
2. **Anthropic** (if `ANTHROPIC_API_KEY` present) → `claude-3-5-sonnet-20241022`
3. **AWS Bedrock** (if AWS credentials present) → `us.anthropic.claude-3-5-sonnet-20241022-v2:0`

### Example Scenarios:

```bash
# Scenario 1: OpenAI + Anthropic available
OPENAI_API_KEY=sk-... 
ANTHROPIC_API_KEY=sk-ant-...
# Result: Uses OpenAI GPT-4o-mini (highest priority)

# Scenario 2: Only Anthropic available
ANTHROPIC_API_KEY=sk-ant-...
# Result: Uses Anthropic Claude-3-5-Sonnet

# Scenario 3: Only AWS available
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
# Result: Uses AWS Bedrock Claude-3-5-Sonnet

# Scenario 4: No model credentials
# Result: System fails to initialize with clear error message
```

## 🚫 Feature Disabling Matrix

| Missing Credential | Disabled Features | Impact |
|-------------------|------------------|---------|
| `OPENAI_API_KEY` | OpenAI models | Falls back to Anthropic or Bedrock |
| `ANTHROPIC_API_KEY` | Anthropic models | Falls back to OpenAI or Bedrock |
| AWS Credentials | AWS Bedrock models | Falls back to OpenAI or Anthropic |
| `SERPER_API_KEY` | Web search, Web scraping | No real-time web data |
| `MEM0_API_KEY` | Cloud memory | Uses local memory container |
| `HF_TOKEN` | HuggingFace models, LlamaServer | No local model support |

## 🔧 Configuration Examples

### Minimal Configuration (OpenAI only)
```bash
# .env
OPENAI_API_KEY=sk-proj-...
```

### Full Configuration (All features)
```bash
# .env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
SERPER_API_KEY=...
MEM0_API_KEY=...
HF_TOKEN=hf_...
```

### Production Configuration (AWS + OpenAI)
```bash
# .env
OPENAI_API_KEY=sk-proj-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
SERPER_API_KEY=...
MEM0_API_KEY=...
```

## 🔍 Checking Credential Status

### Programmatic Check
```python
from sdk.utils.env import ENV

env = ENV()

# Check available credentials
credentials = env.get_available_credentials()
print(f"OpenAI available: {credentials['openai_api_key']}")
print(f"Anthropic available: {credentials['anthropic_api_key']}")
print(f"AWS available: {credentials['aws_access_key_id']}")

# Check disabled features
disabled = env.get_disabled_features()
print(f"Disabled features: {disabled}")

# Get best available model
model_config = env.get_best_available_model_config()
print(f"Best model: {model_config['provider']}:{model_config['model']}")
```

### FastAPI Endpoints
```bash
# Check credential status
curl http://localhost:8000/credentials

# Check available models
curl http://localhost:8000/models

# Check workflow status
curl http://localhost:8000/workflow/status
```

## 🏗️ Workflow Behavior

### Automatic Configuration
The `BaseWorkflow` class automatically configures itself based on available credentials:

```python
from sdk.workflow import BaseWorkflow

# Automatically detects credentials and configures components
workflow = BaseWorkflow()

# Check what's available
status = workflow.get_component_status()
print(f"Model client: {status['model_client']}")
print(f"Memory client: {status['memory_client']}")
print(f"Vector client: {status['vector_client']}")
print(f"Search client: {status['search_client']}")
```

### Manual Override
You can still manually specify model configuration:

```python
from sdk.workflow import BaseWorkflow

# Force specific model (will fail if credentials not available)
workflow = BaseWorkflow(model_config={
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "api_key": "sk-ant-..."
})
```

## 📊 Model Registry Integration

The model registry automatically filters available models based on credentials:

```python
from sdk.model.registry import get_model_registry

registry = get_model_registry()

# Get only available models
available = registry.get_available_models_by_credentials()
print(f"Total available models: {available['total_models']}")

# Check provider status
for provider, info in available['providers'].items():
    print(f"{provider}: {info['available']} ({len(info['models'])} models)")
```

## 🚨 Error Handling

### No Model Credentials
```python
# This will raise ValueError with clear message
try:
    workflow = BaseWorkflow()
except ValueError as e:
    print(f"Error: {e}")
    # Output: "No model provider credentials available. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or AWS credentials."
```

### Graceful Degradation
```python
# System continues to work with limited functionality
workflow = BaseWorkflow()  # Only OpenAI key available

# Web search will be disabled but workflow still works
response = workflow.generate_response("What is the capital of France?")
# Works fine, just without web search capability
```

## 🔐 Security Best Practices

### Environment Variables
```bash
# Use .env file (never commit to git)
echo "OPENAI_API_KEY=sk-proj-..." > .env
echo ".env" >> .gitignore
```

### AWS Credentials
```bash
# Preferred: Use AWS profiles
aws configure --profile coreflow
export AWS_PROFILE=coreflow

# Or: Use IAM roles in production
# No need to set credentials when running on EC2 with IAM role
```

### Production Deployment
```bash
# Use environment-specific configurations
# .env.production
OPENAI_API_KEY=${OPENAI_API_KEY}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
SERPER_API_KEY=${SERPER_API_KEY}
MEM0_API_KEY=${MEM0_API_KEY}
```

## 🧪 Testing Credential Scenarios

### Test Different Configurations
```python
import os
from unittest.mock import patch

# Test with only OpenAI
with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
    workflow = BaseWorkflow()
    # Will use OpenAI, disable web search

# Test with only Anthropic  
with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
    workflow = BaseWorkflow()
    # Will use Anthropic, disable web search

# Test with no credentials
with patch.dict(os.environ, {}, clear=True):
    try:
        workflow = BaseWorkflow()
    except ValueError:
        print("Expected error: No model credentials")
```

## 📈 Monitoring and Observability

### Logging
The system provides detailed logging about credential status:

```
[INFO] ENV initialized with graceful degradation mode - no required keys
[WARNING] Disabled features due to missing credentials: AWS Bedrock models, Web search
[INFO] Using best available model: openai:gpt-4o-mini
[INFO] Memory client initialized with local fallback (MEM0_API_KEY not available)
```

### Health Checks
```python
# Check system health
validation = workflow.validate_workflow()
print(f"System valid: {validation['valid']}")
print(f"Warnings: {validation['warnings']}")
print(f"Errors: {validation['errors']}")
```

## 🔄 Migration Guide

### From Static Configuration
```python
# Old way (will break if credentials missing)
workflow = BaseWorkflow(model_config=openai_config("gpt-4"))

# New way (automatic detection)
workflow = BaseWorkflow()  # Automatically selects best available
```

### Environment Setup
```bash
# Check current status
python -c "from sdk.utils.env import ENV; env = ENV(); print(env.get_disabled_features())"

# Add missing credentials
echo "SERPER_API_KEY=your-key" >> .env
echo "MEM0_API_KEY=your-key" >> .env
```

## 🆘 Troubleshooting

### Common Issues

1. **"No model provider credentials available"**
   - Solution: Set at least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS credentials

2. **"Web search disabled"**
   - Solution: Set `SERPER_API_KEY` environment variable

3. **"Memory client not available"**
   - Check: `MEM0_API_KEY` is set for cloud memory
   - Fallback: Local memory container should work automatically

4. **"AWS Bedrock models disabled"**
   - Solution: Set AWS credentials via environment variables, AWS profile, or IAM role

### Debug Commands
```bash
# Check credential detection
python -c "from sdk.utils.env import ENV; env = ENV(); print(env.get_available_credentials())"

# Test workflow initialization
python -c "from sdk.workflow import BaseWorkflow; w = BaseWorkflow(); print(w.get_component_status())"

# Check model registry
python -c "from sdk.model.registry import get_model_registry; r = get_model_registry(); print(r.get_available_models_by_credentials())"
```

## 📚 Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Detailed environment variable setup
- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Advanced workflow configuration
- [FastAPI Integration](FASTAPI_INTEGRATION.md) - API endpoint documentation
- [API Reference](API_REFERENCE.md) - Complete API documentation

---

**Remember**: The credential-aware system is designed to be helpful, not restrictive. It will always tell you what's missing and continue to work with whatever credentials you have available. 