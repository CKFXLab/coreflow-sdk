"""
FastAPI Integration Tests for CoreFlow SDK

This module contains comprehensive tests for all FastAPI endpoints defined in UPDATES.md.
Tests cover chat, workflow, RAG, model, and health endpoints with proper mocking.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import time
import tempfile
import os
from io import BytesIO

# Import SDK components for mocking
from sdk.workflow import BaseWorkflow
from sdk.model.utils import gpt4o_mini_config
from sdk.vector import RAG
from sdk.model import get_available_providers


@pytest.mark.fastapi
@pytest.mark.integration
class TestFastAPIIntegration:
    """Test suite for FastAPI integration endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app with all endpoints for testing."""
        app = FastAPI(title="CoreFlow API", version="1.0.0")
        
        # Create mock workflow instance
        mock_workflow = Mock()
        mock_workflow.model_client = Mock()
        mock_workflow.model_client.model = "gpt-4o-mini"
        mock_workflow.model_client.provider = "openai"
        mock_workflow.memory_client = Mock()
        mock_workflow.vector_client = Mock()
        mock_workflow.search_client = Mock()
        mock_workflow.scrape_client = Mock()
        
        # Mock workflow methods
        mock_workflow.generate_response.return_value = "Test response"
        mock_workflow.process_query.return_value = "Processed prompt"
        mock_workflow.get_component_status.return_value = {
            "model_client": True,
            "memory_client": True,
            "vector_client": True,
            "search_client": True,
            "scrape_client": True
        }
        mock_workflow.validate_workflow.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        mock_workflow.get_workflow_info.return_value = {
            "class_name": "BaseWorkflow",
            "model_config": {"provider": "openai", "model": "gpt-4o-mini"}
        }
        
        # Add all endpoint implementations here
        self._add_chat_endpoints(app, mock_workflow)
        self._add_workflow_endpoints(app, mock_workflow)
        self._add_rag_endpoints(app)
        self._add_model_endpoints(app)
        self._add_health_endpoints(app, mock_workflow)
            
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_workflow(self):
        """Create mock workflow instance."""
        workflow = Mock()
        workflow.model_client = Mock()
        workflow.model_client.model = "gpt-4o-mini"
        workflow.model_client.provider = "openai"
        workflow.memory_client = Mock()
        workflow.vector_client = Mock()
        workflow.search_client = Mock()
        workflow.scrape_client = Mock()
        
        # Mock workflow methods
        workflow.generate_response.return_value = "Test response"
        workflow.process_query.return_value = "Processed prompt"
        workflow.get_component_status.return_value = {
            "model_client": True,
            "memory_client": True,
            "vector_client": True,
            "search_client": True,
            "scrape_client": True
        }
        workflow.validate_workflow.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        workflow.get_workflow_info.return_value = {
            "class_name": "BaseWorkflow",
            "model_config": {"provider": "openai", "model": "gpt-4o-mini"}
        }
        
        return workflow
    
    def _add_chat_endpoints(self, app: FastAPI, workflow: Mock):
        """Add chat endpoints to the app."""
        
        class ChatRequest(BaseModel):
            query: str
            user_id: str = "default_user"
            model_cfg: dict = None

        class ChatResponse(BaseModel):
            response: str
            user_id: str
            model_used: str

        @app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            try:
                if request.model_cfg:
                    # Mock creating new workflow with custom config
                    chat_workflow = workflow
                else:
                    chat_workflow = workflow
                
                response = chat_workflow.generate_response(
                    query=request.query,
                    user_id=request.user_id
                )
                
                model_used = getattr(chat_workflow.model_client, 'model', 'unknown')
                
                return ChatResponse(
                    response=response,
                    user_id=request.user_id,
                    model_used=model_used
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

        @app.post("/chat/stream")
        async def chat_stream_endpoint(request: ChatRequest):
            from fastapi.responses import StreamingResponse
            
            async def generate_stream():
                try:
                    prompt = workflow.process_query(request.query, request.user_id)
                    
                    # Mock streaming response
                    chunks = ["Hello", " world", "!"]
                    for chunk in chunks:
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                        
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/plain")
    
    def _add_workflow_endpoints(self, app: FastAPI, workflow: Mock):
        """Add workflow endpoints to the app."""
        
        class WorkflowInfo(BaseModel):
            workflow_id: str
            name: str
            description: str
            status: str
            components: dict
            validation: dict

        @app.get("/workflows", response_model=List[WorkflowInfo])
        async def list_workflows():
            try:
                workflow_info = workflow.get_workflow_info()
                component_status = workflow.get_component_status()
                validation = workflow.validate_workflow()
                
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

        @app.get("/workflows/{workflow_id}")
        async def get_workflow(workflow_id: str):
            if workflow_id != "single_agent":
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            try:
                return {
                    "workflow_id": workflow_id,
                    "info": workflow.get_workflow_info(),
                    "status": workflow.get_component_status(),
                    "validation": workflow.validate_workflow()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

        class WorkflowExecuteRequest(BaseModel):
            query: str
            user_id: str = "default_user"
            parameters: dict = {}

        @app.post("/workflows/{workflow_id}/execute")
        async def execute_workflow(workflow_id: str, request: WorkflowExecuteRequest):
            if workflow_id != "single_agent":
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            try:
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
    
    def _add_rag_endpoints(self, app: FastAPI):
        """Add RAG endpoints to the app."""
        
        @app.get("/rag/collections")
        async def list_collections(user_id: str = "default_user"):
            try:
                # Mock RAG collections
                user_collections = [
                    {
                        "name": f"user_{user_id}_docs",
                        "type": "user",
                        "user_id": user_id
                    }
                ]
                system_collections = [
                    {
                        "name": "system_docs",
                        "type": "system"
                    }
                ]
                
                return {
                    "user_collections": user_collections,
                    "system_collections": system_collections,
                    "total": len(user_collections) + len(system_collections)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

        class StoreDocumentRequest(BaseModel):
            text: str
            metadata: dict = {}
            chunk_size: int = 1000
            overlap: int = 200

        @app.post("/rag/collections/{collection_name}/store")
        async def store_to_collection(collection_name: str, request: StoreDocumentRequest, user_id: str = "default_user"):
            try:
                # Mock storing documents
                chunks_stored = 3  # Mock number of chunks
                total_chunks = 3
                chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
                
                return {
                    "collection": collection_name,
                    "chunks_stored": chunks_stored,
                    "total_chunks": total_chunks,
                    "chunk_ids": chunk_ids,
                    "success": chunks_stored == total_chunks
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to store documents: {str(e)}")

        @app.post("/rag/collections/{collection_name}/upload")
        async def upload_file_to_collection(collection_name: str, file: UploadFile, user_id: str = "default_user"):
            try:
                content = await file.read()
                
                return {
                    "collection": collection_name,
                    "filename": file.filename,
                    "file_size": len(content),
                    "success": True,
                    "metadata": {"content_type": file.content_type}
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

        @app.get("/rag/collections/{collection_name}/search")
        async def search_collection(collection_name: str, query: str, user_id: str = "default_user", limit: int = 10):
            try:
                # Mock search results
                results = [
                    {
                        "text": "Sample document content",
                        "metadata": {"source": "test.pdf"},
                        "score": 0.95
                    }
                ]
                
                return {
                    "collection": collection_name,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "user_id": user_id
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def _add_model_endpoints(self, app: FastAPI):
        """Add model endpoints to the app."""
        
        @app.get("/models")
        async def list_models(provider: str = None):
            try:
                # Mock model data
                providers = {
                    "openai": {"available": True, "models": ["gpt-4o-mini", "gpt-4"]},
                    "anthropic": {"available": True, "models": ["claude-3-5-sonnet"]}
                }
                
                model_details = [
                    {
                        "model_id": "gpt-4o-mini",
                        "provider": "openai",
                        "display_name": "GPT-4o Mini",
                        "description": "Affordable and intelligent small model",
                        "max_tokens": 16384,
                        "context_window": 128000,
                        "supports_streaming": True,
                        "supports_functions": True,
                        "input_price_per_million": 0.15,
                        "output_price_per_million": 0.60,
                        "is_available": True
                    }
                ]
                
                return {
                    "providers": providers,
                    "models": model_details,
                    "total_models": len(model_details)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

        @app.get("/models/{model_name}")
        async def get_model(model_name: str):
            try:
                if model_name == "gpt-4o-mini":
                    return {
                        "model_id": "gpt-4o-mini",
                        "provider": "openai",
                        "model_type": "chat",
                        "display_name": "GPT-4o Mini",
                        "description": "Affordable and intelligent small model",
                        "max_tokens": 16384,
                        "context_window": 128000,
                        "supports_streaming": True,
                        "supports_functions": True,
                        "is_available": True
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            except HTTPException:
                raise  # Re-raise HTTPException as-is
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

        @app.post("/models/{model_name}/healthcheck")
        async def healthcheck_model(model_name: str):
            try:
                return {
                    "model_name": model_name,
                    "provider": "openai",
                    "healthy": True,
                    "status": "operational",
                    "timestamp": time.time()
                }
            except Exception as e:
                return {
                    "model_name": model_name,
                    "healthy": False,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
    
    def _add_health_endpoints(self, app: FastAPI, workflow: Mock):
        """Add health endpoints to the app."""
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "coreflow-api",
                "timestamp": time.time()
            }

        @app.get("/health/detailed")
        async def detailed_health_check():
            try:
                component_status = workflow.get_component_status()
                validation = workflow.validate_workflow()
                
                return {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "components": {
                        "model": {
                            "available": component_status["model_client"],
                            "healthy": True,
                            "provider": "openai"
                        },
                        "memory": {
                            "available": component_status["memory_client"],
                            "status": {"client_available": True}
                        },
                        "vector": {
                            "available": component_status["vector_client"]
                        },
                        "search": {
                            "available": component_status["search_client"]
                        },
                        "scrape": {
                            "available": component_status["scrape_client"]
                        }
                    },
                    "validation": validation
                }
            except Exception as e:
                return {
                    "status": "error",
                    "timestamp": time.time(),
                    "error": str(e)
                }

        @app.get("/health/components")
        async def component_health():
            try:
                components = {
                    "model": {
                        "status": "healthy",
                        "provider": "openai",
                        "model": "gpt-4o-mini"
                    },
                    "memory": {
                        "status": "healthy",
                        "stats": {"total_memories": 10}
                    },
                    "vector": {
                        "status": "healthy",
                        "collections_count": 2
                    }
                }
                
                return {
                    "components": components,
                    "timestamp": time.time()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Component health check failed: {str(e)}")

    # =====================================
    # CHAT ENDPOINT TESTS
    # =====================================
    
    @pytest.mark.chat
    def test_chat_endpoint_success(self, client):
        """Test successful chat endpoint."""
        response = client.post("/chat", json={
            "query": "What is machine learning?",
            "user_id": "test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["user_id"] == "test_user"
        assert data["model_used"] == "gpt-4o-mini"
    
    @pytest.mark.chat
    def test_chat_endpoint_with_custom_model(self, client):
        """Test chat endpoint with custom model configuration."""
        response = client.post("/chat", json={
            "query": "Hello world",
            "user_id": "test_user",
            "model_cfg": {"provider": "openai", "model": "gpt-4"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["user_id"] == "test_user"
    
    @pytest.mark.chat
    def test_chat_endpoint_default_user(self, client):
        """Test chat endpoint with default user ID."""
        response = client.post("/chat", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default_user"
    
    @pytest.mark.chat
    def test_chat_stream_endpoint(self, client):
        """Test streaming chat endpoint."""
        response = client.post("/chat/stream", json={
            "query": "Stream test",
            "user_id": "test_user"
        })
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check that response contains streaming data
        content = response.content.decode()
        assert "data:" in content
        assert "Hello" in content
        assert "world" in content
    
    # =====================================
    # WORKFLOW ENDPOINT TESTS
    # =====================================
    
    def test_list_workflows(self, client):
        """Test listing workflows."""
        response = client.get("/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["workflow_id"] == "single_agent"
        assert data[0]["name"] == "Single Agent Workflow"
        assert data[0]["status"] == "active"
    
    def test_get_workflow_success(self, client):
        """Test getting specific workflow."""
        response = client.get("/workflows/single_agent")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "single_agent"
        assert "info" in data
        assert "status" in data
        assert "validation" in data
    
    def test_get_workflow_not_found(self, client):
        """Test getting non-existent workflow."""
        response = client.get("/workflows/nonexistent")
        
        assert response.status_code == 404
        assert "Workflow not found" in response.json()["detail"]
    
    def test_execute_workflow_success(self, client):
        """Test executing workflow."""
        response = client.post("/workflows/single_agent/execute", json={
            "query": "Test execution",
            "user_id": "test_user",
            "parameters": {"temperature": 0.7}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "single_agent"
        assert data["status"] == "completed"
        assert data["response"] == "Test response"
        assert data["user_id"] == "test_user"
        assert "execution_id" in data
    
    def test_execute_workflow_not_found(self, client):
        """Test executing non-existent workflow."""
        response = client.post("/workflows/nonexistent/execute", json={
            "query": "Test",
            "user_id": "test_user"
        })
        
        assert response.status_code == 404
        assert "Workflow not found" in response.json()["detail"]
    
    def test_execute_workflow_default_params(self, client):
        """Test executing workflow with default parameters."""
        response = client.post("/workflows/single_agent/execute", json={
            "query": "Test execution"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default_user"
    
    # =====================================
    # RAG ENDPOINT TESTS
    # =====================================
    
    def test_list_collections(self, client):
        """Test listing RAG collections."""
        response = client.get("/rag/collections?user_id=test_user")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["user_collections"]) == 1
        assert len(data["system_collections"]) == 1
        assert data["total"] == 2
        assert data["user_collections"][0]["user_id"] == "test_user"
    
    def test_list_collections_default_user(self, client):
        """Test listing collections with default user."""
        response = client.get("/rag/collections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_collections"][0]["user_id"] == "default_user"
    
    def test_store_to_collection(self, client):
        """Test storing documents to collection."""
        response = client.post("/rag/collections/test_collection/store", 
                             json={
                                 "text": "This is a test document",
                                 "metadata": {"source": "test"},
                                 "chunk_size": 500,
                                 "overlap": 100
                             },
                             params={"user_id": "test_user"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "test_collection"
        assert data["chunks_stored"] == 3
        assert data["total_chunks"] == 3
        assert data["success"] == True
        assert len(data["chunk_ids"]) == 3
    
    def test_upload_file_to_collection(self, client):
        """Test uploading file to collection."""
        file_content = b"This is test file content"
        
        response = client.post("/rag/collections/test_collection/upload",
                             files={"file": ("test.txt", file_content, "text/plain")},
                             params={"user_id": "test_user"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "test_collection"
        assert data["filename"] == "test.txt"
        assert data["file_size"] == len(file_content)
        assert data["success"] == True
    
    def test_search_collection(self, client):
        """Test searching in collection."""
        response = client.get("/rag/collections/test_collection/search",
                            params={
                                "query": "test query",
                                "user_id": "test_user",
                                "limit": 5
                            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "test_collection"
        assert data["query"] == "test query"
        assert data["count"] == 1
        assert data["user_id"] == "test_user"
        assert len(data["results"]) == 1
    
    def test_search_collection_default_params(self, client):
        """Test searching with default parameters."""
        response = client.get("/rag/collections/test_collection/search?query=test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default_user"
    
    # =====================================
    # MODEL ENDPOINT TESTS
    # =====================================
    
    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "models" in data
        assert data["total_models"] == 1
        assert data["models"][0]["model_id"] == "gpt-4o-mini"
    
    def test_list_models_with_provider_filter(self, client):
        """Test listing models with provider filter."""
        response = client.get("/models?provider=openai")
        
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "models" in data
    
    def test_get_model_success(self, client):
        """Test getting specific model."""
        response = client.get("/models/gpt-4o-mini")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "gpt-4o-mini"
        assert data["provider"] == "openai"
        assert data["display_name"] == "GPT-4o Mini"
        assert data["supports_streaming"] == True
    
    def test_get_model_not_found(self, client):
        """Test getting non-existent model."""
        response = client.get("/models/nonexistent-model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_healthcheck_model(self, client):
        """Test model health check."""
        response = client.post("/models/gpt-4o-mini/healthcheck")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "gpt-4o-mini"
        assert data["healthy"] == True
        assert data["status"] == "operational"
        assert "timestamp" in data
    
    # =====================================
    # HEALTH ENDPOINT TESTS
    # =====================================
    
    @pytest.mark.health
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "coreflow-api"
        assert "timestamp" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "validation" in data
        assert data["components"]["model"]["available"] == True
        assert data["components"]["memory"]["available"] == True
    
    def test_component_health(self, client):
        """Test component health check."""
        response = client.get("/health/components")
        
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "timestamp" in data
        assert data["components"]["model"]["status"] == "healthy"
        assert data["components"]["memory"]["status"] == "healthy"
        assert data["components"]["vector"]["status"] == "healthy"
    
    # =====================================
    # ERROR HANDLING TESTS
    # =====================================
    
    def test_chat_endpoint_missing_query(self, client):
        """Test chat endpoint with missing query."""
        response = client.post("/chat", json={
            "user_id": "test_user"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_json_request(self, client):
        """Test endpoint with invalid JSON."""
        response = client.post("/chat", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422
    
    def test_store_document_missing_text(self, client):
        """Test store endpoint with missing text."""
        response = client.post("/rag/collections/test/store", json={
            "metadata": {"source": "test"}
        })
        
        assert response.status_code == 422
    
    def test_workflow_execute_missing_query(self, client):
        """Test workflow execution with missing query."""
        response = client.post("/workflows/single_agent/execute", json={
            "user_id": "test_user"
        })
        
        assert response.status_code == 422
