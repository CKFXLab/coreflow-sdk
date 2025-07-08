"""
Pytest configuration for CoreFlow SDK Streamlined Test Suite.

Provides essential fixtures and environment setup for CI-focused integration tests.
"""

import pytest
import os
import asyncio
from unittest.mock import Mock, patch

# Set test environment variables to avoid real API calls
os.environ.update({
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key", 
    "SERPER_API_KEY": "test-serper-key",
    "HF_TOKEN": "test-hf-token",
    "MEM0_API_KEY": "test-mem0-key",
    "NODE_ENV": "test"
})

@pytest.fixture(scope="session")
def test_config():
    """Base test configuration for all tests."""
    return {
        "test_mode": True,
        "mock_apis": True,
        "ci_environment": True
    }

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

@pytest.fixture
def mock_workflow_factory():
    """Create a mock WorkflowFactory for testing."""
    factory = Mock()
    factory.create_default_workflow.return_value = Mock()
    factory.create_workflow.return_value = Mock()
    factory.get_available_workflow_types.return_value = {
        "single_agent": {"class_name": "BaseWorkflow", "available": True},
        "multi_agent": {"class_name": "MultiAgentWorkflow", "available": False},
        "api_enhanced": {"class_name": "APIWorkflow", "available": False}
    }
    return factory

@pytest.fixture
def mock_env_all_credentials():
    """Mock environment with all credentials available."""
    return {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'AWS_ACCESS_KEY_ID': 'test-aws-key',
        'AWS_SECRET_ACCESS_KEY': 'test-aws-secret',
        'SERPER_API_KEY': 'test-serper-key',
        'MEM0_API_KEY': 'test-mem0-key',
        'HF_TOKEN': 'test-hf-token'
    }

@pytest.fixture
def mock_env_openai_only():
    """Mock environment with only OpenAI credentials."""
    return {
        'OPENAI_API_KEY': 'test-openai-key'
    }

@pytest.fixture
def mock_env_anthropic_only():
    """Mock environment with only Anthropic credentials."""
    return {
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    }

@pytest.fixture
def mock_env_aws_only():
    """Mock environment with only AWS credentials."""
    return {
        'AWS_ACCESS_KEY_ID': 'test-aws-key',
        'AWS_SECRET_ACCESS_KEY': 'test-aws-secret'
    }

@pytest.fixture
def mock_env_no_credentials():
    """Mock environment with no credentials."""
    return {}

@pytest.fixture
def mock_env_limited_credentials():
    """Mock environment with limited credentials (OpenAI + Serper)."""
    return {
        'OPENAI_API_KEY': 'test-openai-key',
        'SERPER_API_KEY': 'test-serper-key'
    }

@pytest.fixture
def mock_fastapi_client():
    """Create a mock FastAPI test client for credential-aware endpoint testing."""
    from fastapi.testclient import TestClient
    from unittest.mock import Mock
    
    # Mock the FastAPI app and client
    mock_app = Mock()
    mock_client = Mock(spec=TestClient)
    
    # Mock common endpoint responses
    mock_client.get.return_value.status_code = 200
    mock_client.get.return_value.json.return_value = {
        "models": [],
        "credentials": {},
        "status": "ok"
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