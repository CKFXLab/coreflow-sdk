from abc import abstractmethod
from typing import Dict, Any, Optional, List
import boto3

from .._mabc import Model, TrainingJob, ReinforceJob


class BedrockModel(Model):
    """
    Abstract base class for AWS Bedrock language models.

    This class inherits all DSPy requirements from Model and adds
    Bedrock-specific mandatory methods that all Bedrock providers must implement.
    """

    # Bedrock-specific properties that must be set
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_session_token: Optional[str]
    region_name: str
    bedrock_client: boto3.client
    bedrock_runtime_client: boto3.client

    # === BEDROCK-SPECIFIC REQUIREMENTS ===

    @abstractmethod
    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        **kwargs,
    ):
        """
        Initialize Bedrock model client.

        Args:
            region_name: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token for temporary credentials
            **kwargs: Additional DSPy parameters
        """

    @abstractmethod
    def _setup_boto3_clients(self) -> None:
        """
        Set up boto3 clients for Bedrock and Bedrock Runtime.
        Must handle authentication and region configuration.
        """

    @abstractmethod
    def _prepare_bedrock_payload(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Convert DSPy format to Bedrock-specific payload format.
        Each Bedrock model may have different payload requirements.

        Args:
            messages: DSPy-style messages
            **kwargs: Additional parameters

        Returns:
            Bedrock-compatible payload dictionary
        """

    @abstractmethod
    def _parse_bedrock_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Bedrock response to DSPy-compatible format.

        Args:
            response: Raw Bedrock response

        Returns:
            DSPy-compatible response dictionary
        """

    @abstractmethod
    def get_foundation_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available foundation models from Bedrock.

        Returns:
            List of model information dictionaries
        """

    @abstractmethod
    def get_custom_models(self) -> List[Dict[str, Any]]:
        """
        Get list of custom models from Bedrock.

        Returns:
            List of custom model information dictionaries
        """

    @abstractmethod
    def get_model_invoke_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate the cost for invoking a Bedrock model.

        Args:
            model_id: Bedrock model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """


class BedrockTrainingJob(TrainingJob):
    """Training job implementation for Bedrock models."""

    bedrock_client: BedrockModel

    @abstractmethod
    def __init__(
        self,
        bedrock_client: BedrockModel,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[str] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Bedrock training job."""


class BedrockReinforceJob(ReinforceJob):
    """Reinforcement learning job implementation for Bedrock models."""

    bedrock_client: BedrockModel

    @abstractmethod
    def __init__(self, bedrock_client: BedrockModel, train_kwargs: Dict[str, Any]):
        """Initialize Bedrock reinforcement learning job."""
