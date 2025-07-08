# API Reference

Complete reference documentation for the CoreFlow FastAPI endpoints with **credential awareness** and **graceful degradation**.

## Base URL

```
http://localhost:8000  # Development
https://api.yourdomain.com  # Production
```

## Authentication

Currently, the API uses user-based isolation through the `user_id` parameter. The system automatically detects available credentials and gracefully disables features when credentials are missing. Future versions will support JWT/OAuth authentication.

## Response Format

All endpoints return JSON responses with consistent error handling:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Optional message",
  "timestamp": 1234567890
}
```

## Error Responses

```json
{
  "detail": "Error description",
  "status_code": 400
}
```

---

## Chat Endpoints

### POST `/chat`

Process a chat query using the complete AI workflow.

**Request Body:**
```json
{
  "query": "string",
  "user_id": "string (optional, default: 'default_user')",
  "model_config": "object (optional)"
}
```

**Response:**
```json
{
  "response": "string",
  "user_id": "string",
  "model_used": "string"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "user_id": "user123"
  }'
```

### POST `/chat/stream`

Streaming chat endpoint for real-time responses.

**Request Body:**
```json
{
  "query": "string",
  "user_id": "string (optional, default: 'default_user')",
  "model_config": "object (optional)"
}
```

**Response:** Server-Sent Events (SSE)
```
data: {"content": "partial response chunk"}
data: {"content": "next chunk"}
data: {"error": "error message if any"}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about AI", "user_id": "user123"}'
```

---

## Workflow Endpoints

### GET `/workflows`

List all available workflows and their status.

**Response:**
```json
[
  {
    "workflow_id": "string",
    "name": "string",
    "description": "string",
    "status": "active|degraded|error",
    "components": {
      "model_client": "boolean",
      "memory_client": "boolean",
      "vector_client": "boolean",
      "search_client": "boolean",
      "scrape_client": "boolean"
    },
    "validation": {
      "valid": "boolean",
      "errors": ["string"],
      "warnings": ["string"]
    }
  }
]
```

### GET `/workflows/{workflow_id}`

Get detailed information about a specific workflow.

**Parameters:**
- `workflow_id` (path): Workflow identifier

**Response:**
```json
{
  "workflow_id": "string",
  "info": {
    "class_name": "string",
    "model_config": "object",
    "use_docker_qdrant": "boolean",
    "user_collection_prefix": "string",
    "system_collection": "string"
  },
  "status": "object",
  "validation": "object"
}
```

### POST `/workflows/{workflow_id}/execute`

Execute a specific workflow with given parameters.

**Parameters:**
- `workflow_id` (path): Workflow identifier

**Request Body:**
```json
{
  "query": "string",
  "user_id": "string (optional, default: 'default_user')",
  "parameters": "object (optional)"
}
```

**Response:**
```json
{
  "workflow_id": "string",
  "execution_id": "string",
  "status": "completed|failed|running",
  "response": "string",
  "user_id": "string"
}
```

---

## RAG Endpoints

### GET `/rag/collections`

List all available RAG collections for a user.

**Query Parameters:**
- `user_id` (optional): User identifier (default: "default_user")

**Response:**
```json
{
  "user_collections": [
    {
      "name": "string",
      "type": "user",
      "user_id": "string"
    }
  ],
  "system_collections": [
    {
      "name": "string",
      "type": "system"
    }
  ],
  "total": "number"
}
```

### POST `/rag/collections/{collection_name}/store`

Store documents in a specific RAG collection.

**Parameters:**
- `collection_name` (path): Collection name

**Query Parameters:**
- `user_id` (optional): User identifier (default: "default_user")

**Request Body:**
```json
{
  "text": "string",
  "metadata": "object (optional)",
  "chunk_size": "number (optional, default: 1000)",
  "overlap": "number (optional, default: 200)"
}
```

**Response:**
```json
{
  "collection": "string",
  "chunks_stored": "number",
  "total_chunks": "number",
  "chunk_ids": ["string"],
  "success": "boolean"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/rag/collections/docs/store?user_id=user123" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
    "metadata": {"topic": "ML", "source": "textbook"}
  }'
```

### POST `/rag/collections/{collection_name}/upload`

Upload and process a file into a RAG collection.

**Parameters:**
- `collection_name` (path): Collection name

**Query Parameters:**
- `user_id` (optional): User identifier (default: "default_user")

**Request Body:** `multipart/form-data`
- `file`: File to upload (PDF, DOCX, TXT, etc.)

**Response:**
```json
{
  "collection": "string",
  "filename": "string",
  "file_size": "number",
  "success": "boolean",
  "metadata": {
    "file_type": "string",
    "pages": "number",
    "word_count": "number"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/rag/collections/docs/upload?user_id=user123" \
  -F "file=@document.pdf"
```

### GET `/rag/collections/{collection_name}/search`

Search for similar documents in a RAG collection.

**Parameters:**
- `collection_name` (path): Collection name

**Query Parameters:**
- `query` (required): Search query
- `user_id` (optional): User identifier (default: "default_user")
- `limit` (optional): Maximum results (default: 10)

**Response:**
```json
{
  "collection": "string",
  "query": "string",
  "results": [
    {
      "text": "string",
      "metadata": "object",
      "score": "number"
    }
  ],
  "count": "number",
  "user_id": "string"
}
```

**Example:**
```bash
curl "http://localhost:8000/rag/collections/docs/search?query=machine%20learning&user_id=user123&limit=5"
```

---

## Model Endpoints

### GET `/models`

List all available models and providers.

**Query Parameters:**
- `provider` (optional): Filter by provider (openai, anthropic, bedrock)

**Response:**
```json
{
  "providers": ["string"],
  "models": [
    {
      "model_id": "string",
      "provider": "string",
      "display_name": "string",
      "description": "string",
      "max_tokens": "number",
      "context_window": "number",
      "supports_streaming": "boolean",
      "supports_functions": "boolean",
      "input_price_per_million": "number",
      "output_price_per_million": "number",
      "is_available": "boolean"
    }
  ],
  "total_models": "number"
}
```

### GET `/models/{model_name}`

Get detailed information about a specific model.

**Parameters:**
- `model_name` (path): Model identifier

**Response:**
```json
{
  "model_id": "string",
  "provider": "string",
  "model_type": "string",
  "display_name": "string",
  "description": "string",
  "max_tokens": "number",
  "context_window": "number",
  "supports_streaming": "boolean",
  "supports_functions": "boolean",
  "supports_vision": "boolean",
  "supports_json_mode": "boolean",
  "input_price_per_million": "number",
  "output_price_per_million": "number",
  "is_available": "boolean",
  "is_deprecated": "boolean",
  "aliases": ["string"],
  "release_date": "string"
}
```

### POST `/models/{model_name}/health`

Perform a health check on a specific model.

**Parameters:**
- `model_name` (path): Model identifier

**Response:**
```json
{
  "model_name": "string",
  "provider": "string",
  "healthy": "boolean",
  "status": "operational|unavailable|error",
  "error": "string (if error)",
  "timestamp": "number"
}
```

---

## Health & Credential Endpoints

### GET `/health`

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "coreflow-api",
  "timestamp": "number"
}
```

### GET `/credentials`

Get detailed credential availability and feature status.

**Response:**
```json
{
  "timestamp": "number",
  "credentials": {
    "available": {
      "openai_api_key": "boolean",
      "anthropic_api_key": "boolean",
      "aws_access_key_id": "boolean",
      "aws_profile_available": "boolean",
      "aws_instance_profile": "boolean",
      "serper_api_key": "boolean",
      "mem0_api_key": "boolean",
      "hf_token": "boolean"
    },
    "disabled_features": ["string"],
    "selected_model": {
      "provider": "string",
      "model": "string"
    }
  },
  "providers": {
    "openai": {
      "available": "boolean",
      "required": ["string"],
      "missing": ["string"]
    },
    "anthropic": {
      "available": "boolean",
      "required": ["string"],
      "missing": ["string"]
    },
    "bedrock": {
      "available": "boolean",
      "required": ["string"],
      "missing": ["string"],
      "methods": {
        "environment_vars": "boolean",
        "aws_profile": "boolean",
        "instance_profile": "boolean"
      }
    },
    "websearch": {
      "available": "boolean",
      "required": ["string"],
      "missing": ["string"]
    },
    "memory": {
      "available": "boolean",
      "cloud_available": "boolean",
      "required": ["string"],
      "missing": ["string"]
    },
    "huggingface": {
      "available": "boolean",
      "required": ["string"],
      "missing": ["string"]
    }
  },
  "recommendations": {
    "critical": "string",
    "optional": "string"
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/credentials"
```

### GET `/health/detailed`

Comprehensive health check showing component status and credential availability.

**Response:**
```json
{
  "status": "healthy|degraded|error",
  "timestamp": "number",
  "components": {
    "model": {
      "available": "boolean",
      "healthy": "boolean",
      "provider": "string",
      "selected_model": "string"
    },
    "memory": {
      "available": "boolean",
      "status": "object",
      "cloud_available": "boolean"
    },
    "vector": {
      "available": "boolean"
    },
    "search": {
      "available": "boolean",
      "enabled": "boolean"
    },
    "scrape": {
      "available": "boolean"
    }
  },
  "credentials": {
    "available": "object",
    "disabled_features": ["string"],
    "model_priority": ["string"]
  },
  "environment": {
    "current_env": "string",
    "project_root": "string",
    "dotenv_files": ["string"]
  },
  "validation": {
    "valid": "boolean",
    "errors": ["string"],
    "warnings": ["string"]
  }
}
```

### GET `/health/components`

Individual component health checks.

**Response:**
```json
{
  "components": {
    "model": {
      "status": "healthy|unhealthy|error",
      "provider": "string",
      "model": "string",
      "error": "string (if error)"
    },
    "memory": {
      "status": "healthy|unhealthy|error",
      "stats": "object",
      "error": "string (if error)"
    },
    "vector": {
      "status": "healthy|unhealthy|error",
      "collections_count": "number",
      "error": "string (if error)"
    }
  },
  "timestamp": "number"
}
```

---

## Status Codes

| Code | Description |
|------|-------------|
| 200  | OK - Request successful |
| 201  | Created - Resource created successfully |
| 400  | Bad Request - Invalid request parameters |
| 401  | Unauthorized - Authentication required |
| 403  | Forbidden - Access denied |
| 404  | Not Found - Resource not found |
| 422  | Unprocessable Entity - Validation error |
| 500  | Internal Server Error - Server error |

## Rate Limiting

Rate limiting is not currently implemented but can be added using FastAPI middleware:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/chat")
@limiter.limit("10/minute")
def chat_endpoint(request: Request):
    # endpoint implementation
```

## WebSocket Support

Future versions will support WebSocket connections for real-time chat:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## SDK Integration

The API endpoints are designed to work seamlessly with the CoreFlow SDK:

```python
from sdk.workflow import BaseWorkflow
from sdk.model.utils import gpt4o_mini_config

# Create workflow instance
workflow = BaseWorkflow(
    model_config=gpt4o_mini_config(),
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True
)

# Use in FastAPI endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = workflow.generate_response(
        query=request.query,
        user_id=request.user_id
    )
    return ChatResponse(response=response, user_id=request.user_id)
```

## Environment Variables

Environment variables with **credential awareness** - only model provider credentials are required:

### Required (At least one model provider)
```bash
# Model Providers - Choose at least one
OPENAI_API_KEY=your_openai_key           # Priority 1 - Recommended
ANTHROPIC_API_KEY=your_anthropic_key     # Priority 2 - Alternative
AWS_ACCESS_KEY_ID=your_aws_key           # Priority 3 - For Bedrock
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_SESSION_TOKEN=your_aws_token         # Optional for temporary credentials
```

### Optional (Enable additional features)
```bash
# Additional Features
SERPER_API_KEY=your_serper_key          # Enables web search
MEM0_API_KEY=your_mem0_key              # Enables cloud memory (falls back to local)
HF_TOKEN=your_huggingface_token         # Enables HuggingFace models

# Environment configuration
NODE_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Vector store configuration
VECTOR_STORE_PROVIDER=qdrant
EMBEDDING_PROVIDER=openai
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**⚠️ Important**: The system will automatically detect available credentials and gracefully disable features when credentials are missing.

## Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Credential Awareness](CREDENTIAL_AWARENESS.md) - Complete guide to credential detection and graceful degradation
- [FastAPI Integration Guide](FASTAPI_INTEGRATION.md) - Complete implementation guide
- [Workflow Configuration](WORKFLOW_CONFIGURATION.md) - Configuring workflows
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Environment setup
- [Vector Configuration](VECTOR_CONFIGURATION.md) - RAG and vector store setup

## OpenAPI/Swagger Documentation

The API automatically generates OpenAPI documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json` 