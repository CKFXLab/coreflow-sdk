"""
Chat endpoint for AI conversation functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ....workflow import BaseWorkflow

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    model_cfg: Optional[Dict[str, Any]] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    user_id: str
    model_used: str


@router.post("/")
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat query using the complete AI workflow.

    Args:
        request: Chat request with query, optional configuration, and stream flag

    Returns:
        Chat response (JSON or streaming based on stream parameter)
    """
    try:
        # Initialize workflow with custom model config if provided
        if request.model_cfg:
            chat_workflow = BaseWorkflow(
                model_config=request.model_cfg,
                enable_memory=True,
                enable_rag=True,
                enable_websearch=True,
            )
        else:
            chat_workflow = BaseWorkflow()

        # Handle streaming response
        if request.stream:
            from fastapi.responses import StreamingResponse
            import json

            async def generate_stream():
                try:
                    # Process query and stream response
                    async for chunk in chat_workflow.stream_response(
                        request.query, request.user_id
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(generate_stream(), media_type="text/plain")

        # Handle regular JSON response
        else:
            # Generate response using the complete workflow
            response = chat_workflow.generate_response(
                query=request.query, user_id=request.user_id
            )

            # Get model info for response
            model_used = getattr(chat_workflow.model_client, "model", "unknown")

            return ChatResponse(
                response=response, user_id=request.user_id, model_used=model_used
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
