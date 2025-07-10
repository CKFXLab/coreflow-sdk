__version__ = "0.1.0"

from .utils import AppLogger, ENV
from .vector import (
    RAG,
    Mem0,
)
from .websearch import Search, Scrape, KEYWORDS


# Lazy import for workflow to avoid RuntimeWarning with python -m
def __getattr__(name):
    if name == "BaseWorkflow":
        from .workflow import BaseWorkflow

        return BaseWorkflow
    elif name == "WorkflowFactory":
        from .workflow import WorkflowFactory

        return WorkflowFactory
    elif name == "CustomWorkflow":
        from .workflow import CustomWorkflow

        return CustomWorkflow
    elif name == "create_workflow":
        from .workflow import create_workflow

        return create_workflow
    elif name == "get_available_workflow_types":
        from .workflow import get_available_workflow_types

        return get_available_workflow_types
    elif name == "create_default_workflow":
        from .workflow import create_default_workflow

        return create_default_workflow
    elif name == "single_agent_config":
        from .workflow import single_agent_config

        return single_agent_config
    elif name == "multi_agent_config":
        from .workflow import multi_agent_config

        return multi_agent_config
    elif name == "api_enhanced_config":
        from .workflow import api_enhanced_config

        return api_enhanced_config
    elif name == "custom_config":
        from .workflow._factory import custom_config

        return custom_config
    elif name == "WorkflowType":
        from .workflow._default import WorkflowType

        return WorkflowType
    elif name == "Model":
        from .model import Model

        return Model
    elif name == "create_model":
        from .model import create_model

        return create_model
    elif name == "get_available_providers":
        from .model import get_available_providers

        return get_available_providers
    elif name == "get_model_info":
        from .model import get_model_info

        return get_model_info
    elif name == "ModelRegistry":
        from .model import ModelRegistry

        return ModelRegistry
    elif name == "StreamingResponse":
        from .utils.streaming import StreamingResponse

        return StreamingResponse
    elif name == "create_streaming_response":
        from .utils.streaming import create_streaming_response

        return create_streaming_response
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core model components
    "Model",
    "create_model",
    "get_available_providers",
    "get_model_info",
    "ModelRegistry",
    # Vector and memory components
    "RAG",
    "Mem0",
    # Web search components
    "Search",
    "Scrape",
    "KEYWORDS",
    # Workflow components
    "BaseWorkflow",
    "WorkflowFactory",
    "CustomWorkflow",
    "create_workflow",
    "get_available_workflow_types",
    "create_default_workflow",
    # Configuration helpers
    "single_agent_config",
    "multi_agent_config",
    "api_enhanced_config",
    "custom_config",
    # Types
    "WorkflowType",
    # Streaming components
    "StreamingResponse",
    "create_streaming_response",
    # Utilities
    "AppLogger",
    "ENV",
]
