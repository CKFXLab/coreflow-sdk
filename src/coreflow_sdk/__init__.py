__version__ = "0.1.0"

from .utils import AppLogger, ENV
from .vector import (
    RAG, Mem0, CollectionType, SYSTEM_COLLECTION_NAME, USER_COLLECTION_PREFIX,
    sanitize_username, create_user_collection_name, create_system_collection_name,
    parse_collection_name, is_user_collection, is_system_collection, validate_collection_name
)
from .websearch import Search, Scrape
from .model import OpenAIClient

# Optional HuggingFace utilities (graceful fallback)
try:
    from .model import HuggingFace, ModelInfo
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HuggingFace = None
    ModelInfo = None
    HUGGINGFACE_AVAILABLE = False

# Lazy import for workflow to avoid RuntimeWarning with python -m
def __getattr__(name):
    if name == 'BaseWorkflow':
        from .workflow import BaseWorkflow
        return BaseWorkflow
    elif name == 'WorkflowFactory':
        from .workflow import WorkflowFactory
        return WorkflowFactory
    elif name == 'CustomWorkflow':
        from .workflow import CustomWorkflow
        return CustomWorkflow
    elif name == 'create_workflow':
        from .workflow import create_workflow
        return create_workflow
    elif name == 'get_available_workflow_types':
        from .workflow import get_available_workflow_types
        return get_available_workflow_types
    elif name == 'single_agent_config':
        from .workflow import single_agent_config
        return single_agent_config
    elif name == 'multi_agent_config':
        from .workflow import multi_agent_config
        return multi_agent_config
    elif name == 'api_enhanced_config':
        from .workflow import api_enhanced_config
        return api_enhanced_config
    elif name == 'WorkflowType':
        from .workflow import WorkflowType
        return WorkflowType
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core utilities
    'AppLogger',
    'ENV', 
    
    # Vector and memory
    'RAG',
    'Mem0',
    'CollectionType',
    'SYSTEM_COLLECTION_NAME', 
    'USER_COLLECTION_PREFIX',
    
    # Collection utilities
    'sanitize_username',
    'create_user_collection_name',
    'create_system_collection_name',
    'parse_collection_name',
    'is_user_collection',
    'is_system_collection',
    'validate_collection_name',
    
    # Web search and scraping
    'Search',
    'Scrape',
    
    # Model clients
    'OpenAIClient',
    
    # Workflows
    'BaseWorkflow',
    
    # Workflow Factory and creation
    'WorkflowFactory',
    'CustomWorkflow',
    'create_workflow',
    'get_available_workflow_types',
    'single_agent_config',
    'multi_agent_config',
    'api_enhanced_config',
    'WorkflowType',
]

# Add HuggingFace exports if available
if HUGGINGFACE_AVAILABLE:
    __all__.extend([
        'HuggingFace',
        'ModelInfo',
    ])
