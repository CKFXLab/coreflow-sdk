"""
WebSocket Streaming Integration Tests for CoreFlow SDK

This module tests streaming functionality for all model providers and workflow integration.
Tests cover OpenAI, Anthropic, and Bedrock streaming capabilities with BaseWorkflow.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncGenerator, Dict, Any

# Import SDK components
from coreflow_sdk.workflow import BaseWorkflow
from coreflow_sdk.model.api.openai import OpenAIClient
from coreflow_sdk.utils.streaming import StreamingResponse, create_streaming_response

# Handle optional dependencies
try:
    from coreflow_sdk.model.api.anthropic import AnthropicClient

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AnthropicClient = None
    ANTHROPIC_AVAILABLE = False

try:
    from coreflow_sdk.model.bedrock.anthropic import BedrockAnthropicClient

    BEDROCK_AVAILABLE = True
except ImportError:
    BedrockAnthropicClient = None
    BEDROCK_AVAILABLE = False


@pytest.mark.asyncio
@pytest.mark.websocket
class TestModelStreaming:
    """Test streaming capabilities for individual model providers."""

    async def test_openai_stream_forward(self):
        """Test OpenAI streaming functionality."""
        with patch("coreflow_sdk.model.api.openai.AsyncOpenAI") as mock_openai:
            # Mock streaming response
            mock_chunks = [
                Mock(choices=[Mock(delta=Mock(content="Hello"))]),
                Mock(choices=[Mock(delta=Mock(content=" world"))]),
                Mock(choices=[Mock(delta=Mock(content="!"))]),
            ]

            async def mock_stream():
                for chunk in mock_chunks:
                    yield chunk

            mock_client = Mock()
            # Create a mock that returns the async generator directly
            mock_create = AsyncMock()
            mock_create.return_value = mock_stream()
            mock_client.chat.completions.create = mock_create
            mock_openai.return_value = mock_client

            # Test streaming
            client = OpenAIClient(api_key="test-key")
            client.async_client = mock_client

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream_forward(prompt="Test prompt"):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

            # Verify API call
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["stream"] is True
            assert call_args[1]["model"] == client.model

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic not available")
    async def test_anthropic_stream_forward(self):
        """Test Anthropic streaming functionality."""
        with patch("coreflow_sdk.model.api.anthropic.AsyncAnthropic") as mock_anthropic:
            # Mock streaming response
            mock_chunks = ["Hello", " world", "!"]

            async def mock_text_stream():
                for chunk in mock_chunks:
                    yield chunk

            # Mock the stream context manager
            mock_stream = Mock()
            mock_stream.text_stream = mock_text_stream()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            mock_client = Mock()
            mock_client.messages.stream.return_value = mock_stream
            mock_anthropic.return_value = mock_client

            # Test streaming
            client = AnthropicClient(api_key="test-key")
            client.async_client = mock_client

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream_forward(prompt="Test prompt"):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.skipif(not BEDROCK_AVAILABLE, reason="Bedrock not available")
    async def test_bedrock_stream_forward(self):
        """Test Bedrock streaming functionality."""
        with patch("coreflow_sdk.model.bedrock.anthropic.boto3") as mock_boto3:
            # Mock streaming response
            mock_events = [
                {"chunk": {"bytes": json.dumps({"delta": {"text": "Hello"}})}},
                {"chunk": {"bytes": json.dumps({"delta": {"text": " world"}})}},
                {"chunk": {"bytes": json.dumps({"delta": {"text": "!"}})}},
            ]

            mock_response = {"body": mock_events}
            mock_client = Mock()
            mock_client.invoke_model_with_response_stream.return_value = mock_response
            mock_boto3.client.return_value = mock_client

            # Test streaming
            client = BedrockAnthropicClient(
                aws_access_key_id="test-key", aws_secret_access_key="test-secret"
            )
            client.bedrock_runtime_client = mock_client

            # Collect streamed chunks
            chunks = []
            async for chunk in client.stream_forward(prompt="Test prompt"):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

            # Verify API call
            mock_client.invoke_model_with_response_stream.assert_called_once()

    async def test_model_without_streaming(self):
        """Test fallback behavior for models without streaming support."""
        # Create a mock model without stream_forward method
        mock_model = Mock()
        mock_model.forward.return_value = {
            "choices": [{"message": {"content": "Complete response"}}]
        }
        mock_model._process_response.return_value = "Complete response"

        # Remove stream_forward method to simulate non-streaming model
        del mock_model.stream_forward

        # Test that it doesn't have stream_forward
        assert not hasattr(mock_model, "stream_forward")

        # This would be handled in the workflow layer
        assert mock_model.forward("test") is not None


@pytest.mark.asyncio
@pytest.mark.websocket
class TestWorkflowStreaming:
    """Test streaming capabilities in BaseWorkflow."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow with streaming support."""
        workflow = Mock(spec=BaseWorkflow)

        # Mock model client with streaming support
        mock_model = Mock()
        mock_model.stream_forward = AsyncMock()
        workflow.model_client = mock_model

        # Mock other components
        workflow.memory_client = Mock()
        workflow.vector_client = Mock()
        workflow.search_client = Mock()
        workflow.scrape_client = Mock()

        # Mock context gathering
        workflow._gather_context.return_value = {
            "memory_context": "User prefers technical explanations",
            "vector_context": "ML is a subset of AI",
            "web_context": {"results": []},
        }

        # Mock other methods
        workflow._format_context_for_streaming.return_value = "Formatted context"
        workflow.process_response.return_value = "Processed response"
        workflow._store_interaction.return_value = None

        return workflow

    async def test_workflow_stream_response_with_streaming_model(self, mock_workflow):
        """Test workflow streaming with a model that supports streaming."""

        # Mock streaming chunks as proper async generator
        async def mock_stream_chunks():
            yield "Hello"
            yield " world"
            yield "!"

        # Mock the model client properly
        mock_model_client = Mock()
        mock_model_client.stream_forward = AsyncMock(return_value=mock_stream_chunks())

        # Create real workflow instance and replace methods
        workflow = BaseWorkflow()
        workflow.model_client = mock_model_client
        workflow._gather_context = Mock(return_value={"memory_context": "test"})
        workflow._format_context_for_streaming = Mock(return_value="formatted context")
        workflow.process_response = Mock(return_value="processed response")
        workflow._store_interaction = Mock()

        # Test streaming
        events = []
        async for event in workflow.stream_response("Test query", "test_user"):
            events.append(event)

        # Verify event sequence: start, context, 3 chunks, complete
        assert len(events) >= 5  # start, context, 3 chunks, complete
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "context"
        assert events[2]["type"] == "chunk"
        assert events[3]["type"] == "chunk"
        assert events[4]["type"] == "chunk"
        assert events[5]["type"] == "complete"

    async def test_workflow_stream_response_without_streaming_model(
        self, mock_workflow
    ):
        """Test workflow streaming fallback for non-streaming models."""
        # Mock regular async forward method
        mock_model_client = Mock()
        mock_model_client.aforward = AsyncMock(
            return_value={"choices": [{"message": {"content": "Complete response"}}]}
        )

        # Remove streaming support
        del mock_model_client.stream_forward

        # Create real workflow instance
        workflow = BaseWorkflow()
        workflow.model_client = mock_model_client
        workflow._gather_context = Mock(return_value={"memory_context": "test"})
        workflow._format_context_for_streaming = Mock(return_value="formatted context")
        workflow._process_response = Mock(return_value="Complete response")
        workflow.process_response = Mock(return_value="processed response")
        workflow._store_interaction = Mock()

        # Test streaming fallback
        events = []
        async for event in workflow.stream_response("Test query", "test_user"):
            events.append(event)

        # Verify fallback behavior: start, context, chunk, complete
        assert len(events) == 4  # start, context, chunk, complete
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "context"
        assert events[2]["type"] == "chunk"
        assert events[3]["type"] == "complete"

    async def test_workflow_stream_response_error_handling(self, mock_workflow):
        """Test error handling in workflow streaming."""

        # Mock error in streaming
        async def mock_error_stream():
            yield "Hello"
            raise Exception("Streaming error")

        mock_model_client = Mock()
        mock_model_client.stream_forward = AsyncMock(return_value=mock_error_stream())

        # Create real workflow instance
        workflow = BaseWorkflow()
        workflow.model_client = mock_model_client
        workflow._gather_context = Mock(return_value={"memory_context": "test"})
        workflow._format_context_for_streaming = Mock(return_value="formatted context")

        # Test error handling
        events = []
        async for event in workflow.stream_response("Test query", "test_user"):
            events.append(event)

        # Verify error event
        assert events[-1]["type"] == "error"
        assert "error" in events[-1]["data"]
        assert "Streaming error" in events[-1]["data"]["error"]

    def test_workflow_supports_streaming(self):
        """Test supports_streaming method."""
        # Test with streaming model
        workflow = BaseWorkflow()
        workflow.model_client = Mock()
        workflow.model_client.stream_forward = Mock()

        assert workflow.supports_streaming() is True

        # Test without streaming model - properly delete attribute
        workflow.model_client = Mock()
        if hasattr(workflow.model_client, "stream_forward"):
            del workflow.model_client.stream_forward

        assert workflow.supports_streaming() is False


@pytest.mark.asyncio
@pytest.mark.websocket
class TestStreamingUtilities:
    """Test streaming utility functions."""

    @pytest.fixture
    def mock_workflow_with_streaming(self):
        """Create a mock workflow with streaming support."""
        mock_workflow = Mock()

        async def mock_stream(query, user_id):
            events = [
                {"type": "start", "data": {"query": query, "user_id": user_id}},
                {"type": "context", "data": {"context_gathered": True}},
                {"type": "chunk", "data": {"content": "Hello"}},
                {"type": "chunk", "data": {"content": " world"}},
                {"type": "complete", "data": {"response": "Hello world"}},
            ]
            for event in events:
                yield event

        mock_workflow.stream_response = mock_stream
        mock_workflow.supports_streaming.return_value = True
        return mock_workflow

    async def test_streaming_response_to_generator(self, mock_workflow_with_streaming):
        """Test streaming response as async generator."""
        streaming = StreamingResponse(mock_workflow_with_streaming)

        events = []
        async for event in streaming.to_generator("test query", "test_user"):
            events.append(event)

        assert len(events) == 5  # start, context, 2 chunks, complete
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "context"
        assert events[2]["type"] == "chunk"
        assert events[3]["type"] == "chunk"
        assert events[4]["type"] == "complete"

    async def test_streaming_response_to_sse(self, mock_workflow_with_streaming):
        """Test streaming response as Server-Sent Events."""
        streaming = StreamingResponse(mock_workflow_with_streaming)

        sse_chunks = []
        async for chunk in streaming.to_sse("test query", "test_user"):
            sse_chunks.append(chunk)

        assert len(sse_chunks) == 5  # start, context, 2 chunks, complete
        assert all(chunk.startswith("data: ") for chunk in sse_chunks)
        assert all(chunk.endswith("\n\n") for chunk in sse_chunks)
        assert '"type": "start"' in sse_chunks[0]
        assert '"type": "context"' in sse_chunks[1]
        assert '"type": "chunk"' in sse_chunks[2]
        assert '"type": "complete"' in sse_chunks[4]

    async def test_streaming_response_to_websocket(self, mock_workflow_with_streaming):
        """Test streaming response to WebSocket."""
        streaming = StreamingResponse(mock_workflow_with_streaming)

        # Mock WebSocket
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()

        await streaming.to_websocket(mock_websocket, "test query", "test_user")

        # Verify WebSocket calls
        assert (
            mock_websocket.send_text.call_count == 5
        )  # start, context, 2 chunks, complete

        # Verify sent data
        calls = mock_websocket.send_text.call_args_list
        assert '"type": "start"' in calls[0][0][0]
        assert '"type": "context"' in calls[1][0][0]
        assert '"type": "chunk"' in calls[2][0][0]
        assert '"type": "chunk"' in calls[3][0][0]
        assert '"type": "complete"' in calls[4][0][0]

    def test_create_streaming_response(self, mock_workflow_with_streaming):
        """Test streaming response factory function."""
        streaming = create_streaming_response(mock_workflow_with_streaming)

        assert isinstance(streaming, StreamingResponse)
        assert streaming.workflow == mock_workflow_with_streaming


@pytest.mark.asyncio
@pytest.mark.websocket
class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    async def test_fastapi_websocket_integration(self):
        """Test FastAPI WebSocket integration."""
        mock_workflow = Mock()

        async def mock_stream(query, user_id):
            events = [
                {"type": "start", "data": {"query": query, "user_id": user_id}},
                {"type": "chunk", "data": {"content": "Hello"}},
                {"type": "chunk", "data": {"content": " world"}},
                {"type": "complete", "data": {"response": "Hello world"}},
            ]
            for event in events:
                yield event

        mock_workflow.stream_response = mock_stream

        streaming = StreamingResponse(mock_workflow)

        # Mock WebSocket
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()

        # Test streaming to WebSocket
        await streaming.to_websocket(mock_websocket, "test message", "user123")

        # Verify WebSocket calls
        assert mock_websocket.send_text.call_count == 4

        # Verify sent data
        calls = mock_websocket.send_text.call_args_list
        assert '"type": "start"' in calls[0][0][0]
        assert '"type": "chunk"' in calls[1][0][0]
        assert '"type": "chunk"' in calls[2][0][0]
        assert '"type": "complete"' in calls[3][0][0]

    async def test_fastapi_sse_integration(self):
        """Test FastAPI Server-Sent Events integration."""
        mock_workflow = Mock()

        async def mock_stream(query, user_id):
            events = [
                {"type": "start", "data": {"query": query, "user_id": user_id}},
                {"type": "chunk", "data": {"content": "Hello"}},
                {"type": "chunk", "data": {"content": " world"}},
                {"type": "complete", "data": {"response": "Hello world"}},
            ]
            for event in events:
                yield event

        mock_workflow.stream_response = mock_stream

        streaming = StreamingResponse(mock_workflow)

        # Test streaming to SSE
        chunks = []
        async for chunk in streaming.to_sse("test query", "user123"):
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0].startswith("data: ")
        assert '"type": "start"' in chunks[0]
        assert '"type": "chunk"' in chunks[1]
        assert '"type": "complete"' in chunks[3]

    async def test_error_recovery_in_streaming(self):
        """Test error recovery in streaming scenarios."""
        mock_workflow = Mock()

        async def mock_stream_with_error(query, user_id):
            yield {"type": "start", "data": {"query": query, "user_id": user_id}}
            yield {"type": "chunk", "data": {"content": "Hello"}}
            yield {"type": "error", "data": {"error": "Network timeout"}}

        mock_workflow.stream_response = mock_stream_with_error

        streaming = StreamingResponse(mock_workflow)

        events = []
        async for event in streaming.to_generator("test query", "user123"):
            events.append(event)

        assert len(events) == 3
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "chunk"
        assert events[2]["type"] == "error"
        assert "Network timeout" in events[2]["data"]["error"]

    async def test_concurrent_streaming_sessions(self):
        """Test concurrent streaming sessions."""
        mock_workflow = Mock()

        async def mock_stream(query, user_id):
            events = [
                {"type": "start", "data": {"query": query, "user_id": user_id}},
                {"type": "chunk", "data": {"content": f"Response for {user_id}"}},
                {"type": "complete", "data": {"response": f"Complete for {user_id}"}},
            ]
            for event in events:
                yield event

        mock_workflow.stream_response = mock_stream

        streaming = StreamingResponse(mock_workflow)

        # Create multiple concurrent sessions
        async def _collect_stream_events(user_id):
            events = []
            async for event in streaming.to_generator("test query", user_id):
                events.append(event)
            return events

        # Run concurrent sessions
        tasks = [
            _collect_stream_events("user1"),
            _collect_stream_events("user2"),
            _collect_stream_events("user3"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify each session got its own responses
        assert len(results) == 3
        for i, events in enumerate(results):
            user_id = f"user{i+1}"
            assert len(events) == 3
            assert events[0]["data"]["user_id"] == user_id
            assert user_id in events[1]["data"]["content"]


@pytest.mark.asyncio
@pytest.mark.websocket
class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""

    async def test_large_response_streaming(self):
        """Test streaming with large responses."""
        mock_workflow = Mock()

        async def mock_large_stream(query, user_id):
            # Simulate large response with many chunks
            yield {"type": "start", "data": {"query": query, "user_id": user_id}}

            # Generate 100 chunks
            for i in range(100):
                yield {"type": "chunk", "data": {"content": f"Chunk {i} "}}

            yield {"type": "complete", "data": {"response": "Large response complete"}}

        mock_workflow.stream_response = mock_large_stream

        streaming = StreamingResponse(mock_workflow)

        # Test large response streaming
        events = []
        async for event in streaming.to_generator("large response test", "user123"):
            events.append(event)

        assert len(events) == 102  # start + 100 chunks + complete
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "complete"

        # Verify all chunks are present
        chunk_events = [e for e in events if e["type"] == "chunk"]
        assert len(chunk_events) == 100

    async def test_streaming_timeout_handling(self):
        """Test streaming with timeout scenarios."""
        mock_workflow = Mock()

        async def mock_slow_stream(query, user_id):
            yield {"type": "start", "data": {"query": query, "user_id": user_id}}

            # Simulate slow response
            await asyncio.sleep(0.1)
            yield {"type": "chunk", "data": {"content": "Slow response"}}

            # Simulate timeout
            yield {"type": "error", "data": {"error": "Request timeout"}}

        mock_workflow.stream_response = mock_slow_stream

        streaming = StreamingResponse(mock_workflow)

        # Test timeout handling
        events = []
        async for event in streaming.to_generator("slow test", "user123"):
            events.append(event)

        assert len(events) == 3
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "chunk"
        assert events[2]["type"] == "error"
        assert "timeout" in events[2]["data"]["error"]
