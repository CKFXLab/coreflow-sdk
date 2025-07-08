# FastAPI Integration Guide

The CoreFlow SDK provides a comprehensive set of tools for building AI-powered applications with support for multiple LLM providers, RAG systems, memory management, and web search capabilities. This guide shows exact implementations for FastAPI endpoints with **credential awareness** and **graceful degradation**.

## 🚀 Quick Start

```python
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
import time

from sdk.workflow import BaseWorkflow
from sdk.vector import RAG
from sdk.model import get_available_providers
from sdk.model.registry import get_model_info
from sdk.utils.env import ENV

app = FastAPI(title="CoreFlow API", version="1.0.0")

# Initialize environment with credential detection
env = ENV()

# Initialize global workflow instance with automatic credential detection
workflow = BaseWorkflow(
    # No model_config needed - automatically selects best available model
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True  # Will be disabled if SERPER_API_KEY missing
)

# Check what features are available
print("Available features:", workflow.get_component_status())
print("Disabled features:", env.get_disabled_features())
```

## 📋 API Endpoints Implementation

### 1. **Chat Endpoints** - `chat/`

#### POST `/chat`
```python
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    model_config: dict = None

class ChatResponse(BaseModel):
    response: str
    user_id: str
    model_used: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat query using the SingleAgentWorkflow (BaseWorkflow).
    Integrates memory, RAG, web search, and LLM capabilities.
    """
    try:
        # Initialize workflow with custom model config if provided
        if request.model_config:
            chat_workflow = BaseWorkflow(
                model_config=request.model_config,
                enable_memory=True,
                enable_rag=True,
                enable_websearch=True
            )
        else:
            chat_workflow = workflow  # Use global instance
        
        # Generate response using the complete workflow
        response = chat_workflow.generate_response(
            query=request.query,
            user_id=request.user_id
        )
        
        # Get model info for response
        model_used = getattr(chat_workflow.model_client, 'model', 'unknown')
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            model_used=model_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
```

#### POST `/chat/stream`
```python
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    """
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_stream():
        try:
            # Process query to get formatted prompt
            prompt = workflow.process_query(request.query, request.user_id)
            
            # Stream response from model
            messages = [{"role": "user", "content": prompt}]
            
            # Note: Streaming implementation depends on model provider
            # This is a simplified example
            response = workflow.model_client(messages=messages, stream=True)
            
            for chunk in response:
                yield f"data: {json.dumps({'content': chunk})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")
```

### 2. **Workflow Endpoints** - `workflow/`

#### GET `/workflows`
```python
class WorkflowInfo(BaseModel):
    workflow_id: str
    name: str
    description: str
    status: str
    components: dict
    validation: dict

@app.get("/workflows", response_model=List[WorkflowInfo])
async def list_workflows():
    """
    List all available workflows and their status.
    """
    try:
        # Get workflow information
        workflow_info = workflow.get_workflow_info()
        component_status = workflow.get_component_status()
        validation = workflow.validate_workflow()
        
        # In a real implementation, you might have multiple workflow types
        workflows = [
            WorkflowInfo(
                workflow_id="single_agent",
                name="Single Agent Workflow",
                description="Complete AI workflow with memory, RAG, and web search",
                status="active" if validation["valid"] else "degraded",
                components=component_status,
                validation=validation
            )
        ]
        
        return workflows
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")
```

#### POST `/workflows/{workflow_id}/execute`
```python
class WorkflowExecuteRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    parameters: dict = {}

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: WorkflowExecuteRequest):
    """
    Execute a specific workflow with given parameters.
    """
    if workflow_id != "single_agent":
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        # Execute the workflow
        response = workflow.generate_response(
            query=request.query,
            user_id=request.user_id,
            **request.parameters
        )
        
        return {
            "workflow_id": workflow_id,
            "execution_id": f"{workflow_id}_{request.user_id}_{int(time.time())}",
            "status": "completed",
            "response": response,
            "user_id": request.user_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")
```

### 3. **RAG Endpoints** - `rag/`

#### GET `/rag/collections`
```python
from sdk.vector import parse_collection_name, is_user_collection

@app.get("/rag/collections")
async def list_collections(user_id: str = "default_user"):
    """
    List all available RAG collections for a user.
    """
    try:
        rag = RAG()
        all_collections = rag.list_collections()
        
        # Parse collections and filter by user access
        user_collections = []
        system_collections = []
        
        for collection in all_collections:
            parsed = parse_collection_name(collection)
            if is_user_collection(collection):
                if parsed.get("user_id") == user_id:
                    user_collections.append({
                        "name": collection,
                        "type": "user",
                        "user_id": parsed.get("user_id")
                    })
            else:
                system_collections.append({
                    "name": collection,
                    "type": "system"
                })
        
        return {
            "user_collections": user_collections,
            "system_collections": system_collections,
            "total": len(user_collections) + len(system_collections)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
```

#### POST `/rag/collections/{collection_name}/store`
```python
class StoreDocumentRequest(BaseModel):
    text: str
    metadata: dict = {}
    chunk_size: int = 1000
    overlap: int = 200

@app.post("/rag/collections/{collection_name}/store")
async def store_to_collection(collection_name: str, request: StoreDocumentRequest, user_id: str = "default_user"):
    """
    Store documents in a specific RAG collection.
    """
    try:
        from sdk.vector import FileOperations
        
        # Initialize RAG with specific collection
        rag = RAG(collection_name=collection_name)
        file_ops = FileOperations()
        
        # Chunk the text
        chunks = file_ops.chunk_text(
            text=request.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            source_file=f"api_upload_{collection_name}",
            metadata=request.metadata
        )
        
        # Store each chunk
        stored_chunks = []
        for chunk in chunks:
            success = rag.store(
                text=chunk.text,
                metadata={**chunk.metadata, **request.metadata},
                collection_type="user",
                user_id=user_id
            )
            if success:
                stored_chunks.append(chunk.chunk_id)
        
        return {
            "collection": collection_name,
            "chunks_stored": len(stored_chunks),
            "total_chunks": len(chunks),
            "chunk_ids": stored_chunks,
            "success": len(stored_chunks) == len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store documents: {str(e)}")
```

#### POST `/rag/collections/{collection_name}/upload`
```python
@app.post("/rag/collections/{collection_name}/upload")
async def upload_file_to_collection(collection_name: str, file: UploadFile, user_id: str = "default_user"):
    """
    Upload and process a file into a RAG collection.
    """
    try:
        from sdk.vector import FileOperations
        import tempfile
        import os
        
        file_ops = FileOperations()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file
            is_valid, error_msg = file_ops.validate_file(tmp_file_path)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Process file
            text_content, file_metadata = file_ops.process_file(tmp_file_path)
            
            # Store in collection
            rag = RAG(collection_name=collection_name)
            success = rag.store(
                text=text_content,
                metadata={
                    **file_metadata,
                    "filename": file.filename,
                    "content_type": file.content_type
                },
                collection_type="user",
                user_id=user_id
            )
            
            return {
                "collection": collection_name,
                "filename": file.filename,
                "file_size": len(content),
                "success": success,
                "metadata": file_metadata
            }
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
```

#### GET `/rag/collections/{collection_name}/search`
```python
@app.get("/rag/collections/{collection_name}/search")
async def search_collection(collection_name: str, query: str, user_id: str = "default_user", limit: int = 10):
    """
    Search for similar documents in a RAG collection.
    """
    try:
        rag = RAG(collection_name=collection_name)
        
        # Perform similarity search
        results = rag.search_similar(
            query=query,
            collection_type="user",
            user_id=user_id,
            limit=limit
        )
        
        return {
            "collection": collection_name,
            "query": query,
            "results": results,
            "count": len(results),
            "user_id": user_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
```

### 4. **Models Endpoints** - `models/`

#### GET `/models`
```python
@app.get("/models")
async def list_models(provider: str = None):
    """
    List all available models and providers.
    """
    try:
        from sdk.model.registry import get_available_models, get_model_registry
        
        # Get available providers
        providers = get_available_providers()
        
        # Get available models
        if provider:
            models = get_available_models(provider)
        else:
            models = get_available_models()
        
        # Get detailed model information
        registry = get_model_registry()
        model_details = []
        
        for model_id in models:
            model_info = registry.get_model(model_id)
            if model_info:
                model_details.append({
                    "model_id": model_id,
                    "provider": model_info.provider.value,
                    "display_name": model_info.display_name,
                    "description": model_info.description,
                    "max_tokens": model_info.max_tokens,
                    "context_window": model_info.context_window,
                    "supports_streaming": model_info.supports_streaming,
                    "supports_functions": model_info.supports_functions,
                    "input_price_per_million": model_info.input_price_per_million,
                    "output_price_per_million": model_info.output_price_per_million,
                    "is_available": model_info.is_available
                })
        
        return {
            "providers": providers,
            "models": model_details,
            "total_models": len(model_details)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
```

#### GET `/models/{model_name}`
```python
@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """
    Get detailed information about a specific model.
    """
    try:
        model_info = get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return {
            "model_id": model_info.model_id,
            "provider": model_info.provider.value,
            "model_type": model_info.model_type.value,
            "display_name": model_info.display_name,
            "description": model_info.description,
            "max_tokens": model_info.max_tokens,
            "context_window": model_info.context_window,
            "supports_streaming": model_info.supports_streaming,
            "supports_functions": model_info.supports_functions,
            "supports_vision": model_info.supports_vision,
            "supports_json_mode": model_info.supports_json_mode,
            "input_price_per_million": model_info.input_price_per_million,
            "output_price_per_million": model_info.output_price_per_million,
            "is_available": model_info.is_available,
            "is_deprecated": model_info.is_deprecated,
            "aliases": model_info.aliases,
            "release_date": model_info.release_date
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
```

### 5. **Health & Credential Endpoints** - `health/` & `credentials/`

#### GET `/health`
```python
@app.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "coreflow-api",
        "timestamp": time.time()
    }
```

#### GET `/credentials`
```python
@app.get("/credentials")
async def get_credentials_status():
    """
    Get detailed credential availability and feature status.
    """
    try:
        from sdk.utils.env import ENV
        
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()
        provider_availability = env.get_provider_availability()
        model_config = env.get_best_available_model_config()
        
        return {
            "timestamp": time.time(),
            "credentials": {
                "available": credentials,
                "disabled_features": disabled_features,
                "selected_model": {
                    "provider": model_config.get('provider') if model_config else None,
                    "model": model_config.get('model') if model_config else None
                } if model_config else None
            },
            "providers": provider_availability,
            "recommendations": {
                "critical": "Ensure at least one model provider (OpenAI, Anthropic, or AWS) is configured",
                "optional": "Add SERPER_API_KEY for web search, MEM0_API_KEY for cloud memory"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get credential status: {str(e)}")
```

#### GET `/health/detailed`
```python
@app.get("/health/detailed")
async def detailed_health_check():
    """
    Comprehensive health check showing component status and credential availability.
    """
    try:
        from sdk.utils.env import ENV
        
        # Get workflow status
        component_status = workflow.get_component_status()
        validation = workflow.validate_workflow()
        
        # Get environment info with credential awareness
        env = ENV()
        env_info = env.get_env_info()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()
        model_config = env.get_best_available_model_config()
        
        # Test model connection
        model_healthy = False
        if workflow.model_client:
            try:
                model_healthy = workflow.model_client.validate_connection()
            except:
                model_healthy = False
        
        # Get memory status
        memory_status = {}
        if workflow.memory_client:
            try:
                memory_status = workflow.memory_client.get_client_status()
            except Exception as e:
                memory_status = {"error": str(e)}
        
        overall_status = "healthy" if validation["valid"] and model_healthy else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "components": {
                "model": {
                    "available": component_status["model_client"],
                    "healthy": model_healthy,
                    "provider": getattr(workflow.model_client, 'provider', 'unknown') if workflow.model_client else None,
                    "selected_model": model_config.get('model') if model_config else None
                },
                "memory": {
                    "available": component_status["memory_client"],
                    "status": memory_status,
                    "cloud_available": credentials.get('mem0_api_key', False)
                },
                "vector": {
                    "available": component_status["vector_client"]
                },
                "search": {
                    "available": component_status["search_client"],
                    "enabled": credentials.get('serper_api_key', False)
                },
                "scrape": {
                    "available": component_status["scrape_client"]
                }
            },
            "credentials": {
                "available": credentials,
                "disabled_features": disabled_features,
                "model_priority": ["OpenAI", "Anthropic", "AWS Bedrock"]
            },
            "environment": {
                "current_env": env_info["environment"],
                "project_root": env_info["project_root"],
                "dotenv_files": env_info["dotenv_files"]
            },
            "validation": validation
        }
    
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }
```

## 🔧 Environment Setup

### Environment Variables with Credential Awareness

#### Required (At least one model provider)
```bash
# Model Providers - Choose at least one
OPENAI_API_KEY=your-openai-key           # Priority 1 - Recommended
ANTHROPIC_API_KEY=your-anthropic-key     # Priority 2 - Alternative
AWS_ACCESS_KEY_ID=your-aws-key           # Priority 3 - For Bedrock
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_SESSION_TOKEN=your-aws-token         # Optional for temporary credentials
```

#### Optional (Enable additional features)
```bash
# Additional Features
SERPER_API_KEY=your-serper-key          # Enables web search
MEM0_API_KEY=your-mem0-key              # Enables cloud memory (falls back to local)
HF_TOKEN=your-huggingface-token         # Enables HuggingFace models

# Environment
NODE_ENV=development  # or production, test, local

# Optional: Custom settings
DEBUG=true
LOG_LEVEL=DEBUG
API_BASE_URL=http://localhost:8000
```

**⚠️ Important**: The system will automatically detect available credentials and gracefully disable features when credentials are missing. Only model provider credentials are required for basic functionality.

### Dotenv File Support

Create environment-specific files:
- `.env.development` - Development settings
- `.env.production` - Production settings  
- `.env.local` - Personal overrides (gitignored)

## 🚀 Running the API

```python
import uvicorn
from fastapi import FastAPI
import time

# Add all the endpoints above to your FastAPI app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

## 📊 Example Usage

### Chat with AI
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "user_id": "user123"
  }'
```

### Store Documents
```bash
curl -X POST "http://localhost:8000/rag/collections/docs/store" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of AI...",
    "metadata": {"topic": "ML", "source": "textbook"}
  }' \
  -G -d "user_id=user123"
```

### Search Documents
```bash
curl "http://localhost:8000/rag/collections/docs/search?query=machine%20learning&user_id=user123&limit=5"
```

### Check Health
```bash
curl "http://localhost:8000/health/detailed"
```

### Check Credentials
```bash
curl "http://localhost:8000/credentials"
```

## 🔒 Security Considerations

1. **API Keys**: Use environment variables, never hardcode
2. **User Isolation**: All RAG collections are user-scoped
3. **Input Validation**: All endpoints include request validation
4. **Error Handling**: Comprehensive error responses
5. **Rate Limiting**: Implement using FastAPI middleware
6. **Authentication**: Add JWT/OAuth as needed

## 🎯 Key Features

- ✅ **Complete Workflow Integration**: Chat, memory, RAG, web search
- ✅ **Multi-Model Support**: OpenAI, Anthropic, Bedrock providers
- ✅ **User Isolation**: Separate collections and memory per user
- ✅ **File Upload Support**: Process PDFs, DOCX, TXT, etc.
- ✅ **Health Monitoring**: Comprehensive component status
- ✅ **Environment Management**: Dotenv support with precedence
- ✅ **Error Handling**: Detailed error responses
- ✅ **Async Support**: Full async/await compatibility

## 📖 Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Credential Awareness](CREDENTIAL_AWARENESS.md) - Complete guide to credential detection and graceful degradation
- [Workflow Configuration Guide](WORKFLOW_CONFIGURATION.md) - Creating and configuring workflows
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Setting up environment variables
- [Vector Configuration](VECTOR_CONFIGURATION.md) - RAG and vector store setup
- [API Reference](API_REFERENCE.md) - Complete API documentation

The CoreFlow SDK provides everything needed to build production-ready AI APIs with FastAPI! 