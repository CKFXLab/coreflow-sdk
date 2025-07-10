"""
Pytest configuration for CoreFlow SDK Streamlined Test Suite.

Provides essential fixtures and environment setup for CI-focused integration tests.
"""

import pytest
import os
import asyncio
from unittest.mock import Mock, AsyncMock

# Set test environment variables to avoid real API calls
os.environ.update(
    {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "SERPER_API_KEY": "test-serper-key",
        "HF_TOKEN": "test-hf-token",
        "MEM0_API_KEY": "test-mem0-key",
        "NODE_ENV": "test",
    }
)


@pytest.fixture(scope="session")
def test_config():
    """Base test configuration for all tests."""
    return {"test_mode": True, "mock_apis": True, "ci_environment": True}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_workflow():
    """Create a mock workflow instance for testing."""
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
        "scrape_client": True,
    }
    workflow.validate_workflow.return_value = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    workflow.get_workflow_info.return_value = {
        "class_name": "BaseWorkflow",
        "model_config": {"provider": "openai", "model": "gpt-4o-mini"},
    }

    # Mock streaming methods
    workflow.supports_streaming.return_value = True

    async def mock_stream_response(query, user_id="default_user"):
        """Mock streaming response generator."""
        events = [
            {"type": "start", "data": {"query": query, "user_id": user_id}},
            {"type": "context", "data": {"context_gathered": True}},
            {"type": "chunk", "data": {"content": "Test "}},
            {"type": "chunk", "data": {"content": "streaming "}},
            {"type": "chunk", "data": {"content": "response"}},
            {"type": "complete", "data": {"response": "Test streaming response"}},
        ]
        for event in events:
            yield event

    workflow.stream_response = mock_stream_response

    return workflow


@pytest.fixture
def mock_streaming_workflow():
    """Create a mock workflow with streaming capabilities."""
    workflow = Mock()
    workflow.model_client = Mock()
    workflow.model_client.model = "gpt-4o-mini"
    workflow.model_client.provider = "openai"
    workflow.model_client.supports_streaming = True

    # Mock streaming methods
    workflow.supports_streaming.return_value = True

    async def mock_stream_response(query, user_id="default_user"):
        """Mock streaming response generator."""
        events = [
            {"type": "start", "data": {"query": query, "user_id": user_id}},
            {"type": "context", "data": {"context_gathered": True}},
            {"type": "chunk", "data": {"content": "Hello "}},
            {"type": "chunk", "data": {"content": "world!"}},
            {"type": "complete", "data": {"response": "Hello world!"}},
        ]
        for event in events:
            yield event

    workflow.stream_response = mock_stream_response
    return workflow


@pytest.fixture
def mock_non_streaming_workflow():
    """Create a mock workflow without streaming capabilities."""
    workflow = Mock()
    workflow.model_client = Mock()
    workflow.model_client.model = "gpt-3.5-turbo"
    workflow.model_client.provider = "openai"
    workflow.model_client.supports_streaming = False

    # Mock non-streaming methods
    workflow.supports_streaming.return_value = False
    workflow.generate_response.return_value = "Non-streaming response"

    return workflow


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket for testing."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_streaming_response():
    """Create a mock StreamingResponse for testing."""
    from unittest.mock import Mock, AsyncMock

    streaming_response = Mock()
    streaming_response.to_websocket = AsyncMock()
    streaming_response.to_sse = AsyncMock()
    streaming_response.to_generator = AsyncMock()

    async def mock_to_websocket(websocket, query, user_id="default_user"):
        """Mock WebSocket streaming."""
        events = [
            {"type": "start", "data": {"query": query, "user_id": user_id}},
            {"type": "context", "data": {"context_gathered": True}},
            {"type": "chunk", "data": {"content": "Test "}},
            {"type": "chunk", "data": {"content": "response"}},
            {"type": "complete", "data": {"response": "Test response"}},
        ]
        for event in events:
            await websocket.send_text(str(event))

    streaming_response.to_websocket = mock_to_websocket
    return streaming_response


@pytest.fixture
def mock_websocket_client():
    """Create a mock WebSocket client for testing."""
    from fastapi.testclient import TestClient
    from unittest.mock import Mock

    client = Mock(spec=TestClient)

    # Mock WebSocket connection
    mock_websocket = Mock()
    mock_websocket.send_text = Mock()
    mock_websocket.receive_text = Mock()
    mock_websocket.close = Mock()

    client.websocket_connect.return_value.__enter__.return_value = mock_websocket
    client.websocket_connect.return_value.__exit__.return_value = None

    return client


@pytest.fixture
def mock_workflow_factory():
    """Create a mock WorkflowFactory for testing."""
    factory = Mock()
    factory.create_default_workflow.return_value = Mock()
    factory.create_workflow.return_value = Mock()
    factory.get_available_workflow_types.return_value = {
        "single_agent": {"class_name": "BaseWorkflow", "available": True},
        "multi_agent": {"class_name": "MultiAgentWorkflow", "available": False},
        "api_enhanced": {"class_name": "APIWorkflow", "available": False},
    }
    return factory


@pytest.fixture
def mock_env_all_credentials():
    """Mock environment with all credentials available."""
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "AWS_ACCESS_KEY_ID": "test-aws-key",
        "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
        "SERPER_API_KEY": "test-serper-key",
        "MEM0_API_KEY": "test-mem0-key",
        "HF_TOKEN": "test-hf-token",
    }


@pytest.fixture
def mock_env_openai_only():
    """Mock environment with only OpenAI credentials."""
    return {"OPENAI_API_KEY": "test-openai-key"}


@pytest.fixture
def mock_env_anthropic_only():
    """Mock environment with only Anthropic credentials."""
    return {"ANTHROPIC_API_KEY": "test-anthropic-key"}


@pytest.fixture
def mock_env_aws_only():
    """Mock environment with only AWS credentials."""
    return {
        "AWS_ACCESS_KEY_ID": "test-aws-key",
        "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
    }


@pytest.fixture
def mock_env_no_credentials():
    """Mock environment with no credentials."""
    return {}


@pytest.fixture
def mock_env_limited_credentials():
    """Mock environment with limited credentials (OpenAI + Serper)."""
    return {"OPENAI_API_KEY": "test-openai-key", "SERPER_API_KEY": "test-serper-key"}


@pytest.fixture
def mock_fastapi_client():
    """Create a mock FastAPI test client for credential-aware endpoint testing."""
    from fastapi.testclient import TestClient
    from unittest.mock import Mock

    # Mock the FastAPI app and client
    Mock()
    mock_client = Mock(spec=TestClient)

    # Mock common endpoint responses
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = {
        "models": [],
        "credentials": {},
        "status": "ok",
    }

    return mock_client


# Configure pytest markers for the streamlined test suite
def pytest_configure(config):
    """Configure custom pytest markers for CI testing."""
    config.addinivalue_line("markers", "integration: CI-focused integration tests")
    config.addinivalue_line("markers", "core: Core functionality tests")
    config.addinivalue_line("markers", "api: API connection tests")
    config.addinivalue_line("markers", "rag: RAG system tests")
    config.addinivalue_line("markers", "memory: Memory system tests")
    config.addinivalue_line("markers", "workflow: Workflow integration tests")
    config.addinivalue_line("markers", "factory: WorkflowFactory tests")
    config.addinivalue_line("markers", "fastapi: FastAPI endpoint tests")
    config.addinivalue_line("markers", "asyncio_test: Async test support")
    config.addinivalue_line("markers", "chat: Chat endpoint tests")
    config.addinivalue_line("markers", "health: Health check tests")
    config.addinivalue_line("markers", "credentials: Credential awareness tests")
    config.addinivalue_line("markers", "env: Environment configuration tests")
    config.addinivalue_line("markers", "registry: Model registry tests")
    config.addinivalue_line("markers", "degradation: Graceful degradation tests")
    config.addinivalue_line("markers", "edge_cases: Edge case and error handling tests")
    config.addinivalue_line("markers", "websocket: WebSocket streaming tests")
    config.addinivalue_line("markers", "streaming: Streaming functionality tests")
    config.addinivalue_line("markers", "realtime: Real-time communication tests")
    config.addinivalue_line("markers", "concurrent: Concurrent connection tests")
    config.addinivalue_line("markers", "performance: Performance and load tests")
