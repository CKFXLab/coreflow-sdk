from abc import abstractmethod
from typing import Dict, Any, Optional, List
import subprocess

from .._mabc import Model, TrainingJob, ReinforceJob


class LlamaServerModel(Model):
    """
    Abstract base class for LlamaServer/local language models.

    This class inherits all DSPy requirements from Model and adds
    LlamaServer-specific mandatory methods for local model deployment.
    """

    # LlamaServer-specific properties that must be set
    server_url: str
    server_port: int
    model_path: str
    server_process: Optional[subprocess.Popen]
    context_length: int
    gpu_layers: int

    # === LLAMASERVER-SPECIFIC REQUIREMENTS ===

    @abstractmethod
    def __init__(
        self,
        model_path: str,
        server_port: int = 8080,
        context_length: int = 4096,
        gpu_layers: int = -1,
        **kwargs,
    ):
        """
        Initialize LlamaServer model client.

        Args:
            model_path: Path to the model file (.gguf, .bin, etc.)
            server_port: Port for the local server
            context_length: Maximum context length for the model
            gpu_layers: Number of layers to offload to GPU (-1 for all)
            **kwargs: Additional DSPy parameters
        """

    @abstractmethod
    def _start_server(self) -> None:
        """
        Start the LlamaServer process.
        Must handle server startup and validation.
        """

    @abstractmethod
    def _stop_server(self) -> None:
        """
        Stop the LlamaServer process.
        Must handle graceful shutdown.
        """

    @abstractmethod
    def _health_check(self) -> bool:
        """
        Check if the server is running and responding.

        Returns:
            True if server is healthy, False otherwise
        """

    @abstractmethod
    def _prepare_llamaserver_payload(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Convert DSPy format to LlamaServer API format.

        Args:
            messages: DSPy-style messages
            **kwargs: Additional parameters

        Returns:
            LlamaServer-compatible payload dictionary
        """

    @abstractmethod
    def _parse_llamaserver_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LlamaServer response to DSPy-compatible format.

        Args:
            response: Raw LlamaServer response

        Returns:
            DSPy-compatible response dictionary
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """

    @abstractmethod
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get server performance statistics.

        Returns:
            Dictionary with memory usage, GPU utilization, etc.
        """


class LlamaServerTrainingJob(TrainingJob):
    """Training job implementation for LlamaServer models."""

    llamaserver_client: LlamaServerModel

    @abstractmethod
    def __init__(
        self,
        llamaserver_client: LlamaServerModel,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[str] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LlamaServer training job."""


class LlamaServerReinforceJob(ReinforceJob):
    """Reinforcement learning job implementation for LlamaServer models."""

    llamaserver_client: LlamaServerModel

    @abstractmethod
    def __init__(
        self, llamaserver_client: LlamaServerModel, train_kwargs: Dict[str, Any]
    ):
        """Initialize LlamaServer reinforcement learning job."""
