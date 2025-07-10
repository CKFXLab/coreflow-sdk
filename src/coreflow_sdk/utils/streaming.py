"""Streaming utilities for WebSocket integration."""

import json
from typing import AsyncGenerator, Dict, Any
from fastapi import WebSocket


class StreamingResponse:
    """Utility class for handling streaming responses."""

    def __init__(self, workflow):
        self.workflow = workflow

    async def to_websocket(
        self, websocket: WebSocket, query: str, user_id: str = "default_user"
    ):
        """Stream workflow response to WebSocket."""
        async for event in self.workflow.stream_response(query, user_id):
            await websocket.send_text(json.dumps(event))

    async def to_sse(
        self, query: str, user_id: str = "default_user"
    ) -> AsyncGenerator[str, None]:
        """Stream workflow response as Server-Sent Events."""
        async for event in self.workflow.stream_response(query, user_id):
            yield f"data: {json.dumps(event)}\n\n"

    async def to_generator(
        self, query: str, user_id: str = "default_user"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream workflow response as async generator."""
        async for event in self.workflow.stream_response(query, user_id):
            yield event


def create_streaming_response(workflow) -> StreamingResponse:
    """Create a streaming response handler for a workflow."""
    return StreamingResponse(workflow)
