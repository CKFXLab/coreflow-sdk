import json
import copy
import time
import asyncio
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from ._mabc import BedrockModel, BedrockTrainingJob, BedrockReinforceJob
from ...utils import AppLogger


class BedrockAnthropicClient(BedrockModel):
    """
    Bedrock wrapper for Anthropic Claude models that makes them DSPy-compatible.

    This wrapper translates DSPy interface calls to Bedrock API calls and back,
    allowing Claude models on Bedrock to work seamlessly with DSPy modules.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        **kwargs,
    ):
        """
        Initialize Bedrock Anthropic wrapper.

        Args:
            model_id: Bedrock model identifier for Claude
            region_name: AWS region
            aws_access_key_id: AWS credentials (optional - uses AWS credential chain if not provided)
            aws_secret_access_key: AWS credentials (optional - uses AWS credential chain if not provided)
            aws_session_token: AWS session token (optional - uses AWS credential chain if not provided)
            **kwargs: Additional DSPy parameters
        """
        self.logger = AppLogger(__name__)

        # Set DSPy required properties
        self.model = model_id
        self.model_type = kwargs.get("model_type", "chat")
        self.provider = "bedrock-anthropic"
        self.cache = kwargs.get("cache", True)
        self.num_retries = kwargs.get("num_retries", 3)
        self.kwargs = kwargs
        self.history = []

        # Set Bedrock-specific properties
        # Only set these if explicitly provided, otherwise rely on AWS credential chain
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.region_name = region_name

        # Initialize Bedrock clients
        self._setup_boto3_clients()

        self.logger.info(f"Bedrock Anthropic wrapper initialized for model: {model_id}")

    def _setup_boto3_clients(self) -> None:
        """
        Initialize boto3 clients for Bedrock.

        This method respects the AWS credential chain:
        1. Explicitly passed credentials (aws_access_key_id, etc.)
        2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
        3. AWS profiles (AWS_PROFILE)
        4. IAM roles (for EC2 instances)
        5. AWS SSO
        6. aws-vault and other credential helpers
        """
        try:
            session_kwargs = {"region_name": self.region_name}

            # Only set credentials if explicitly provided
            # Otherwise, boto3 will use the standard AWS credential chain
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": self.aws_access_key_id,
                        "aws_secret_access_key": self.aws_secret_access_key,
                    }
                )
                if self.aws_session_token:
                    session_kwargs["aws_session_token"] = self.aws_session_token
                self.logger.info("Using explicitly provided AWS credentials")
            else:
                self.logger.info(
                    "Using AWS credential chain (environment, profile, IAM role, etc.)"
                )

            session = boto3.Session(**session_kwargs)

            self.bedrock_client = session.client("bedrock")
            self.bedrock_runtime_client = session.client("bedrock-runtime")

            self.logger.info("Bedrock clients initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Bedrock clients: {e}")
            raise

    # === CORE DSPy INTERFACE (Wrapper Implementation) ===

    def __call__(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Any:
        """DSPy callable interface - translates to Bedrock call."""
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_bedrock_response_to_dspy_format(response)

    def forward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        DSPy forward method - translates DSPy call to Bedrock InvokeModel.

        This is where the magic happens: DSPy thinks it's calling a standard model,
        but we're actually translating to Bedrock's specific API.
        """
        # Convert DSPy format to Bedrock format
        bedrock_payload = self._prepare_bedrock_payload(
            messages or [{"role": "user", "content": prompt}], **kwargs
        )

        # Retry logic for Bedrock throttling
        for attempt in range(self.num_retries + 1):
            try:
                # Call Bedrock InvokeModel API
                response = self.bedrock_runtime_client.invoke_model(
                    modelId=self.model,
                    body=json.dumps(bedrock_payload),
                    contentType="application/json",
                    accept="application/json",
                )

                # Parse Bedrock response
                response_body = json.loads(response["body"].read())

                # Convert back to DSPy-compatible format
                dspy_response = self._parse_bedrock_response(response_body)

                # Update history for DSPy debugging
                history_entry = {
                    "prompt": prompt,
                    "messages": messages,
                    "bedrock_payload": bedrock_payload,
                    "bedrock_response": response_body,
                    "dspy_response": dspy_response,
                    "kwargs": kwargs,
                }
                self.history.append(history_entry)
                self.update_global_history(history_entry)

                return dspy_response

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ThrottlingException" and attempt < self.num_retries:
                    self._handle_bedrock_throttling(attempt)
                else:
                    self.logger.error(f"Bedrock API error: {e}")
                    raise
            except Exception as e:
                if attempt < self.num_retries:
                    time.sleep(2**attempt)
                else:
                    self.logger.error(f"Bedrock request failed: {e}")
                    raise

    async def aforward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Async version - Bedrock doesn't have native async, so we use asyncio.to_thread."""
        return await asyncio.to_thread(self.forward, prompt, messages, **kwargs)

    async def acall(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ) -> Any:
        """Asynchronous callable interface."""
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_bedrock_response_to_dspy_format(response)

    async def stream_forward(
        self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs
    ):
        """Stream model response chunks via Bedrock."""
        # Format messages using the same logic as forward method
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            messages = [{"role": "user", "content": ""}]

        payload = self._prepare_bedrock_payload(messages, **kwargs)

        response = await asyncio.to_thread(
            self.bedrock_runtime_client.invoke_model_with_response_stream,
            modelId=self.model,
            body=json.dumps(payload),
        )

        for event in response["body"]:
            if "chunk" in event:
                chunk_data = json.loads(event["chunk"]["bytes"])
                if "delta" in chunk_data and "text" in chunk_data["delta"]:
                    yield chunk_data["delta"]["text"]

    # === TRANSLATION METHODS (The Core Wrapper Logic) ===

    def _prepare_bedrock_payload(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Translate DSPy message format to Bedrock Anthropic format.

        DSPy format: [{"role": "user", "content": "Hello"}]
        Bedrock format: {"messages": [...], "max_tokens": 1000, "anthropic_version": "..."}
        """
        # Extract parameters with defaults
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        # Convert DSPy messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build Bedrock-specific payload for Anthropic models
        payload = {
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "anthropic_version": "bedrock-2023-05-31",
        }

        return payload

    def _parse_bedrock_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate Bedrock Anthropic response to DSPy-compatible format.

        Bedrock format: {"content": [{"type": "text", "text": "..."}], "usage": {...}}
        DSPy format: {"choices": [{"message": {"content": "..."}}], "usage": {...}}
        """
        try:
            # Extract content from Bedrock Anthropic response
            content = ""
            if "content" in response and response["content"]:
                for content_block in response["content"]:
                    if content_block.get("type") == "text":
                        content += content_block.get("text", "")

            # Convert to DSPy-compatible format (mimics OpenAI structure)
            dspy_response = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": response.get("usage", {}),
                "model": self.model,
                "object": "chat.completion",
            }

            return dspy_response

        except Exception as e:
            self.logger.error(f"Failed to parse Bedrock response: {e}")
            raise

    def _process_bedrock_response_to_dspy_format(self, response: Dict[str, Any]) -> str:
        """Extract text content for DSPy modules."""
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        return ""

    # === DSPy FRAMEWORK METHODS ===

    def copy(self, **kwargs) -> "BedrockAnthropicClient":
        """Create a copy with updated parameters."""
        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in new_instance.kwargs or not hasattr(new_instance, key):
                new_instance.kwargs[key] = value

        return new_instance

    def inspect_history(self, n: int = 1) -> str:
        """Inspect recent interactions with Bedrock translation details."""
        if not self.history:
            return "No history available"

        recent_entries = self.history[-n:]
        formatted_history = []

        for i, entry in enumerate(recent_entries, 1):
            formatted_entry = (
                f"--- Bedrock Translation Entry {len(self.history) - n + i} ---\n"
            )
            formatted_entry += (
                f"DSPy Input: {entry.get('messages', entry.get('prompt'))}\n"
            )
            formatted_entry += f"Bedrock Payload: {entry.get('bedrock_payload')}\n"
            formatted_entry += f"Bedrock Response: {entry.get('bedrock_response')}\n"
            formatted_entry += f"DSPy Output: {entry.get('dspy_response')}\n"
            formatted_history.append(formatted_entry)

        return "\n".join(formatted_history)

    def update_global_history(self, entry: Dict[str, Any]) -> None:
        """Update global DSPy history."""

    def dump_state(self) -> Dict[str, Any]:
        """Serialize wrapper state."""
        return {
            "model": self.model,
            "model_type": self.model_type,
            "provider": self.provider,
            "cache": self.cache,
            "num_retries": self.num_retries,
            "region_name": self.region_name,
            "kwargs": self.kwargs,
        }

    # === PROVIDER MANAGEMENT ===

    def infer_provider(self) -> str:
        """Infer provider."""
        return "bedrock-anthropic"

    def launch(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Bedrock models don't require server launch."""
        self.logger.info("Bedrock models are managed by AWS - no launch required")

    def kill(self, launch_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Bedrock models don't require termination."""
        self.logger.info("Bedrock models are managed by AWS - no termination required")

    # === BEDROCK-SPECIFIC METHODS ===

    def get_foundation_models(self) -> List[Dict[str, Any]]:
        """Get available foundation models from Bedrock."""
        try:
            response = self.bedrock_client.list_foundation_models()
            return response.get("modelSummaries", [])
        except Exception as e:
            self.logger.error(f"Failed to get foundation models: {e}")
            return []

    def get_custom_models(self) -> List[Dict[str, Any]]:
        """Get custom models from Bedrock."""
        try:
            response = self.bedrock_client.list_custom_models()
            return response.get("modelSummaries", [])
        except Exception as e:
            self.logger.error(f"Failed to get custom models: {e}")
            return []

    def _handle_bedrock_throttling(self, retry_count: int) -> None:
        """Handle Bedrock throttling with exponential backoff."""
        wait_time = (2**retry_count) + 1  # Bedrock-specific backoff
        self.logger.warning(f"Bedrock throttling, waiting {wait_time} seconds")
        time.sleep(wait_time)

    def get_model_invoke_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate Bedrock invocation cost."""
        # Import here to avoid circular imports
        from ..registry import estimate_cost

        cost = estimate_cost(model_id, input_tokens, output_tokens)
        if cost > 0:
            return cost

        # Fallback for unknown models
        self.logger.warning(f"Unknown model for cost estimation: {model_id}")
        return 0.0

    def _validate_model_access(self, model_id: str) -> bool:
        """Validate access to Bedrock model."""
        try:
            # Try to get model info
            self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            return True
        except Exception:
            return False

    # === REQUIRED ABSTRACT METHODS ===

    def finetune(
        self,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[str] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BedrockTrainingJob:
        """Start Bedrock fine-tuning job."""
        return BedrockTrainingJob(
            bedrock_client=self,
            model=self.model,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs or {},
        )

    def reinforce(self, train_kwargs: Dict[str, Any]) -> BedrockReinforceJob:
        """Start Bedrock RL job."""
        return BedrockReinforceJob(bedrock_client=self, train_kwargs=train_kwargs)

    def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embeddings using Bedrock Titan or other embedding models."""
        embedding_model = model or "amazon.titan-embed-text-v1"

        try:
            # Different payload formats for different embedding models
            if "titan-embed" in embedding_model:
                payload = {"inputText": text}
            elif "cohere.embed" in embedding_model:
                payload = {"texts": [text], "input_type": "search_document"}
            else:
                # Default to Titan format
                payload = {"inputText": text}

            response = self.bedrock_runtime_client.invoke_model(
                modelId=embedding_model,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Extract embedding based on model type
            if "titan-embed" in embedding_model:
                return response_body.get("embedding", [])
            elif "cohere.embed" in embedding_model:
                embeddings = response_body.get("embeddings", [])
                return embeddings[0] if embeddings else []
            else:
                # Default extraction
                return response_body.get("embedding", [])

        except Exception as e:
            self.logger.error(
                f"Failed to generate embedding with model {embedding_model}: {e}"
            )
            raise

    def validate_connection(self) -> bool:
        """Test Bedrock connection."""
        try:
            self.bedrock_client.list_foundation_models()
            return True
        except Exception as e:
            self.logger.error(f"Bedrock connection validation failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available Bedrock models."""
        try:
            models = self.get_foundation_models()
            return [
                model.get("modelId", "") for model in models if model.get("modelId")
            ]
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            # Return known models from registry as fallback
            from ..registry import get_available_models

            return get_available_models(provider="bedrock")

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific Bedrock model."""
        try:
            response = self.bedrock_client.get_foundation_model(
                modelIdentifier=model_id
            )
            return response.get("modelDetails", {})
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_id}: {e}")
            # Fallback to registry information
            from ..registry import get_model_info

            model_info = get_model_info(model_id)
            if model_info:
                return {
                    "modelId": model_info.model_id,
                    "modelName": model_info.display_name,
                    "providerName": model_info.provider.value,
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "maxTokens": model_info.max_tokens,
                    "contextWindow": model_info.context_window,
                    "description": model_info.description,
                    "supports_streaming": model_info.supports_streaming,
                    "supports_functions": model_info.supports_functions,
                    "supports_vision": model_info.supports_vision,
                    "input_price_per_million": model_info.input_price_per_million,
                    "output_price_per_million": model_info.output_price_per_million,
                }
            return {}


# Example usage showing how the wrapper makes Bedrock work with DSPy
def example_usage():
    """
    Example showing how Bedrock wrapper integrates with DSPy.
    """
    example_code = """
    # Initialize Bedrock wrapper
    bedrock_model = BedrockAnthropicClient(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1"
    )

    # Use with DSPy - DSPy has no idea this is Bedrock!
    import dspy
    dspy.settings.configure(lm=bedrock_model)

    # DSPy signature works normally
    class QASignature(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    # DSPy module works normally
    qa_module = dspy.Predict(QASignature)

    # DSPy call gets translated to Bedrock automatically
    result = qa_module(question="What is machine learning?")

    # DSPy optimization works too
    optimized_model = bedrock_model.copy(temperature=0.2)

    # Debugging shows the translation process
    print(bedrock_model.inspect_history())
    """
    return example_code
