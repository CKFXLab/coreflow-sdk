"""
WorkflowFactory for creating workflow instances from configuration dictionaries.

This factory supports creating workflows from:
- Configuration dictionaries: {"workflow_type": "single_agent", "model_config": {...}, ...}
- Already instantiated Workflow objects (pass-through)

For type safety and better IDE support, use configuration helper functions:
- single_agent_config(model_config, enable_memory=True)
- multi_agent_config(agents=[...], coordination_strategy="sequential")
- api_enhanced_config(external_apis=[...], model_config={...})

Method Override Support:
- Override public methods via configuration: {"method_overrides": {"process_query": custom_function}}
- Use CustomWorkflow for easy subclassing
- Pass callable functions or method names
"""

import importlib
import os
from typing import Union, Dict, Any, Optional, List, Callable
from ._default import (
    DEFAULT_WORKFLOW_CONFIGS, WORKFLOW_CLASS_MAP, WORKFLOW_IMPORT_PATHS,
    WORKFLOW_DEPENDENCIES, DEFAULT_FALLBACK_CONFIG, WorkflowType,
    COORDINATION_STRATEGIES, DEFAULT_API_CONFIGS
)

try:
    from ..utils import AppLogger
    from ..model import create_model
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ..utils import AppLogger
    from ..model import create_model


class CustomWorkflow:
    """
    Wrapper class that allows easy method overriding for BaseWorkflow.
    
    This class provides a convenient way to customize the three public methods:
    - process_query()
    - generate_response() 
    - process_response()
    
    Usage:
        # Method 1: Subclass CustomWorkflow
        class MyWorkflow(CustomWorkflow):
            def process_query(self, query, user_id="default_user"):
                # Custom query processing
                return super().process_query(query, user_id)
        
        # Method 2: Pass method overrides to factory
        config = single_agent_config(method_overrides={
            "process_query": my_custom_process_query
        })
    """
    
    def __init__(self, base_workflow, method_overrides: Optional[Dict[str, Callable]] = None):
        """
        Initialize CustomWorkflow with a base workflow and optional method overrides.
        
        Args:
            base_workflow: The underlying BaseWorkflow instance
            method_overrides: Dictionary of method names to override functions
        """
        self.base_workflow = base_workflow
        self.method_overrides = method_overrides or {}
        
        # Apply method overrides
        self._apply_method_overrides()
    
    def _apply_method_overrides(self):
        """Apply method overrides to this instance."""
        for method_name, override_func in self.method_overrides.items():
            if hasattr(self, method_name):
                # Bind the override function to this instance
                bound_method = self._bind_override_method(override_func, method_name)
                setattr(self, method_name, bound_method)
    
    def _bind_override_method(self, override_func: Callable, method_name: str) -> Callable:
        """
        Bind an override function to this instance with access to the original method.
        
        Args:
            override_func: The function to use as override
            method_name: Name of the method being overridden
            
        Returns:
            Bound method that can call the original via self.super_method()
        """
        original_method = getattr(self, method_name)
        
        def bound_override(*args, **kwargs):
            # Inject access to original method
            if hasattr(override_func, '__self__'):
                # Already bound method
                return override_func(*args, **kwargs)
            else:
                # Unbound function - bind it and provide super access
                return override_func(self, *args, **kwargs)
        
        # Store reference to original for super() calls
        setattr(self, f"_original_{method_name}", original_method)
        
        return bound_override
    
    def super_method(self, method_name: str, *args, **kwargs):
        """
        Call the original (super) method.
        
        Args:
            method_name: Name of the original method to call
            *args, **kwargs: Arguments to pass to the original method
            
        Returns:
            Result from the original method
        """
        original_method = getattr(self, f"_original_{method_name}", None)
        if original_method:
            return original_method(*args, **kwargs)
        else:
            # Fall back to base workflow
            return getattr(self.base_workflow, method_name)(*args, **kwargs)
    
    # === DELEGATED PUBLIC INTERFACE ===
    
    def process_query(self, query: str, user_id: str = "default_user") -> str:
        """Process query - can be overridden."""
        return self.base_workflow.process_query(query, user_id)
    
    def generate_response(self, query: str, user_id: str = "default_user", **kwargs) -> str:
        """Generate response - can be overridden."""
        return self.base_workflow.generate_response(query, user_id, **kwargs)
    
    def process_response(self, model_response: str, query: str, user_id: str) -> str:
        """Process response - can be overridden."""
        return self.base_workflow.process_response(model_response, query, user_id)
    
    # === DELEGATE ALL OTHER METHODS TO BASE WORKFLOW ===
    
    def __getattr__(self, name):
        """Delegate any other method calls to the base workflow."""
        return getattr(self.base_workflow, name)


class WorkflowFactory:
    """
    Factory for creating workflow instances from configuration dictionaries.
    
    Supports these input formats:
    1. Dictionary format: {"workflow_type": "single_agent", "model_config": {...}, ...}
    2. Workflow instance: Returns as-is (pass-through)
    
    For better type safety and IDE support, use configuration helper functions
    from sdk.workflow.utils.config instead of raw dictionaries.
    """
    
    def __init__(self):
        """Initialize the WorkflowFactory."""
        self.logger = AppLogger(__name__)
        self._workflow_cache = {}  # Cache for imported workflow classes
    
    def create_workflow(self, 
                       workflow_config: Union[Dict[str, Any], Any],
                       **override_kwargs) -> Any:
        """
        Create a workflow instance from configuration dictionary or pre-instantiated workflow.
        
        Args:
            workflow_config: Workflow configuration in one of these formats:
                - Dict: {"workflow_type": "single_agent", "model_config": {...}, ...}
                - Workflow: Already instantiated workflow (returned as-is)
            **override_kwargs: Additional parameters to override defaults
            
        Returns:
            Instantiated Workflow object
            
        Raises:
            ValueError: If configuration is invalid or workflow type not supported
            ImportError: If workflow dependencies are not available
        """
        try:
            # Handle already instantiated workflows
            if hasattr(workflow_config, 'generate_response'):  # Duck typing for workflow
                self.logger.info(f"Using pre-instantiated workflow: {workflow_config.__class__.__name__}")
                return workflow_config
            
            # Validate that we have a dictionary
            if not isinstance(workflow_config, dict):
                raise ValueError(f"workflow_config must be a dictionary, got {type(workflow_config)}")
            
            # Copy config and merge with override kwargs
            config = workflow_config.copy()
            config.update(override_kwargs)
            
            # Validate configuration
            self._validate_config(config)
            
            # Create and return workflow instance
            return self._instantiate_workflow(config)
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow from config {workflow_config}: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate and normalize the configuration dictionary.
        
        Args:
            config: Configuration to validate and normalize
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Ensure required fields
        workflow_type = config.get("workflow_type")
        if not workflow_type:
            # Default to single agent if not specified
            workflow_type = WorkflowType.SINGLE_AGENT.value
            config["workflow_type"] = workflow_type
        
        if workflow_type not in WORKFLOW_CLASS_MAP:
            available = list(WORKFLOW_CLASS_MAP.keys())
            raise ValueError(f"Unsupported workflow type: {workflow_type}. Available: {available}")
        
        # Merge with workflow type defaults (user config takes precedence)
        if workflow_type in DEFAULT_WORKFLOW_CONFIGS:
            defaults = DEFAULT_WORKFLOW_CONFIGS[workflow_type].copy()
            defaults.update(config)
            config.clear()
            config.update(defaults)
        
        # Validate method overrides if present
        method_overrides = config.get("method_overrides")
        if method_overrides:
            self._validate_method_overrides(method_overrides)
        
        # Validate workflow-specific requirements
        self._validate_workflow_specific_config(config, workflow_type)
    
    def _validate_method_overrides(self, method_overrides: Dict[str, Any]) -> None:
        """
        Validate method override configurations.
        
        Args:
            method_overrides: Dictionary of method names to override functions
        """
        valid_methods = {"process_query", "generate_response", "process_response"}
        
        for method_name, override_func in method_overrides.items():
            if method_name not in valid_methods:
                raise ValueError(f"Invalid method override: {method_name}. Valid methods: {valid_methods}")
            
            if not callable(override_func):
                raise ValueError(f"Method override for {method_name} must be callable, got {type(override_func)}")
    
    def _validate_workflow_specific_config(self, config: Dict[str, Any], workflow_type: str) -> None:
        """
        Validate workflow-specific configuration requirements.
        
        Args:
            config: Configuration dictionary
            workflow_type: Type of workflow being created
        """
        if workflow_type == WorkflowType.MULTI_AGENT.value:
            self._validate_multi_agent_config(config)
        elif workflow_type == WorkflowType.API_ENHANCED.value:
            self._validate_api_enhanced_config(config)
        elif workflow_type == WorkflowType.CUSTOM.value:
            self._validate_custom_config(config)
    
    def _validate_multi_agent_config(self, config: Dict[str, Any]) -> None:
        """Validate multi-agent workflow configuration."""
        agents = config.get("agents", [])
        if not agents:
            # Create default agent configuration
            config["agents"] = [{"role": "assistant", "model_config": None}]
        
        # Validate coordination strategy
        strategy = config.get("coordination_strategy", "sequential")
        if strategy not in COORDINATION_STRATEGIES:
            available = list(COORDINATION_STRATEGIES.keys())
            raise ValueError(f"Invalid coordination strategy: {strategy}. Available: {available}")
        
        # Validate agent configurations
        for i, agent in enumerate(config["agents"]):
            if not isinstance(agent, dict):
                raise ValueError(f"Agent {i} configuration must be a dictionary")
            if "role" not in agent:
                agent["role"] = f"agent_{i}"
    
    def _validate_api_enhanced_config(self, config: Dict[str, Any]) -> None:
        """Validate API-enhanced workflow configuration."""
        external_apis = config.get("external_apis", [])
        
        # Validate API configurations
        for i, api_config in enumerate(external_apis):
            if not isinstance(api_config, dict):
                raise ValueError(f"API configuration {i} must be a dictionary")
            if "url" not in api_config:
                raise ValueError(f"API configuration {i} must have a 'url' field")
            
            # Apply default API configuration
            api_type = api_config.get("type", "rest_api")
            if api_type in DEFAULT_API_CONFIGS:
                defaults = DEFAULT_API_CONFIGS[api_type].copy()
                defaults.update(api_config)
                external_apis[i] = defaults
    
    def _validate_custom_config(self, config: Dict[str, Any]) -> None:
        """Validate custom workflow configuration."""
        custom_components = config.get("custom_components", [])
        
        # Validate custom component configurations
        for i, component in enumerate(custom_components):
            if not isinstance(component, dict):
                raise ValueError(f"Custom component {i} configuration must be a dictionary")
            if "type" not in component:
                raise ValueError(f"Custom component {i} must have a 'type' field")
    
    def _instantiate_workflow(self, config: Dict[str, Any]) -> Any:
        """
        Instantiate the workflow from validated configuration.
        
        Args:
            config: Validated configuration dictionary
            
        Returns:
            Workflow instance
        """
        workflow_type = config["workflow_type"]
        
        # Get the workflow class
        workflow_class = self._get_workflow_class(workflow_type)
        
        # Prepare initialization arguments
        init_kwargs = self._prepare_init_kwargs(config, workflow_type)
        
        # Check for method overrides
        method_overrides = config.get("method_overrides")
        
        # Instantiate the workflow
        self.logger.info(f"Creating {workflow_type} workflow")
        workflow_instance = workflow_class(**init_kwargs)
        
        # Apply method overrides if present
        if method_overrides:
            self.logger.info(f"Applying method overrides: {list(method_overrides.keys())}")
            workflow_instance = CustomWorkflow(workflow_instance, method_overrides)
        
        self.logger.info(f"Successfully created {workflow_type} workflow instance")
        return workflow_instance
    
    def _get_workflow_class(self, workflow_type: str):
        """
        Get the workflow class, using cache for performance.
        
        Args:
            workflow_type: Workflow type name
            
        Returns:
            Workflow class
        """
        if workflow_type in self._workflow_cache:
            return self._workflow_cache[workflow_type]
        
        # Import the workflow module
        if workflow_type not in WORKFLOW_IMPORT_PATHS:
            raise ValueError(f"No import path configured for workflow type: {workflow_type}")
        
        module_path = WORKFLOW_IMPORT_PATHS[workflow_type]
        class_name = WORKFLOW_CLASS_MAP[workflow_type]
        
        try:
            # Dynamic import
            module = importlib.import_module(module_path)
            workflow_class = getattr(module, class_name)
            
            # Cache for future use
            self._workflow_cache[workflow_type] = workflow_class
            
            return workflow_class
            
        except ImportError as e:
            raise ImportError(f"Failed to import {workflow_type} workflow. "
                            f"Make sure dependencies are installed: {e}")
        except AttributeError as e:
            raise ImportError(f"Workflow class {class_name} not found in {module_path}: {e}")
    
    def _prepare_init_kwargs(self, config: Dict[str, Any], workflow_type: str) -> Dict[str, Any]:
        """
        Prepare initialization kwargs for the specific workflow type.
        
        Args:
            config: Full configuration dictionary
            workflow_type: Workflow type name
            
        Returns:
            Kwargs suitable for workflow class initialization
        """
        # Start with all config
        kwargs = config.copy()
        
        # Remove factory-specific keys that shouldn't be passed to workflow
        factory_keys = {"workflow_type", "method_overrides"}
        for key in factory_keys:
            kwargs.pop(key, None)
        
        # Workflow-specific argument preparation
        if workflow_type == WorkflowType.SINGLE_AGENT.value:
            # BaseWorkflow expects these exact parameters
            pass  # No special mapping needed
        
        elif workflow_type == WorkflowType.MULTI_AGENT.value:
            # Prepare agent configurations
            agents = kwargs.get("agents", [])
            for agent in agents:
                if agent.get("model_config") is None:
                    agent["model_config"] = kwargs.get("model_config")
        
        elif workflow_type == WorkflowType.API_ENHANCED.value:
            # Prepare API configurations
            pass  # API configurations are already validated
        
        elif workflow_type == WorkflowType.CUSTOM.value:
            # Custom workflow preparation
            pass  # Custom components are already validated
        
        return kwargs
    
    def get_available_workflow_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available workflow types.
        
        Returns:
            Dictionary with workflow type information
        """
        workflow_types = {}
        
        for workflow_type in WORKFLOW_CLASS_MAP.keys():
            workflow_types[workflow_type] = {
                "class_name": WORKFLOW_CLASS_MAP[workflow_type],
                "default_config": DEFAULT_WORKFLOW_CONFIGS.get(workflow_type, {}),
                "required_dependencies": WORKFLOW_DEPENDENCIES.get(workflow_type, []),
                "available": self._check_workflow_availability(workflow_type),
                "supports_method_overrides": True
            }
        
        return workflow_types
    
    def _check_workflow_availability(self, workflow_type: str) -> bool:
        """
        Check if a workflow type is available (dependencies installed).
        
        Args:
            workflow_type: Workflow type name
            
        Returns:
            True if workflow type is available, False otherwise
        """
        try:
            # Single agent workflow is always available (uses existing BaseWorkflow)
            if workflow_type == WorkflowType.SINGLE_AGENT.value:
                return True
            
            self._get_workflow_class(workflow_type)
            return True
        except (ImportError, AttributeError):
            return False
    
    def create_default_workflow(self, workflow_type: Optional[str] = None) -> Any:
        """
        Create a workflow with default configuration.
        
        Args:
            workflow_type: Workflow type to use (defaults to single_agent)
            
        Returns:
            Workflow instance with default configuration
        """
        if workflow_type is None:
            workflow_type = WorkflowType.SINGLE_AGENT.value
        
        if workflow_type not in DEFAULT_WORKFLOW_CONFIGS:
            raise ValueError(f"No default configuration for workflow type: {workflow_type}")
        
        config = DEFAULT_WORKFLOW_CONFIGS[workflow_type].copy()
        
        return self.create_workflow(config)


# Global factory instance for convenience
_factory = WorkflowFactory()

# Convenience functions
def create_workflow(workflow_config: Union[Dict[str, Any], Any], **kwargs) -> Any:
    """
    Convenience function to create a workflow using the global factory.
    
    Args:
        workflow_config: Workflow configuration dictionary or Workflow instance
        **kwargs: Additional parameters
        
    Returns:
        Workflow instance
    """
    return _factory.create_workflow(workflow_config, **kwargs)

def get_available_workflow_types() -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get available workflow types.
    
    Returns:
        Dictionary with workflow type information
    """
    return _factory.get_available_workflow_types()

def create_default_workflow(workflow_type: Optional[str] = None) -> Any:
    """
    Convenience function to create a default workflow.
    
    Args:
        workflow_type: Workflow type to use
        
    Returns:
        Workflow instance
    """
    return _factory.create_default_workflow(workflow_type)


# Configuration helper functions for type safety and IDE support

def single_agent_config(model_config: Union[Dict[str, Any], Any] = None,
                       enable_memory: bool = True,
                       enable_rag: bool = True,
                       enable_websearch: bool = True,
                       method_overrides: Optional[Dict[str, Callable]] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Create a single-agent workflow configuration.
    
    Args:
        model_config: Model configuration for the agent
        enable_memory: Whether to enable memory/conversation history
        enable_rag: Whether to enable RAG/vector storage
        enable_websearch: Whether to enable web search
        method_overrides: Dictionary of method names to override functions
        **kwargs: Additional configuration parameters
        
    Returns:
        Single-agent workflow configuration dictionary
    """
    config = {
        "workflow_type": WorkflowType.SINGLE_AGENT.value,
        "model_config": model_config,
        "enable_memory": enable_memory,
        "enable_rag": enable_rag,
        "enable_websearch": enable_websearch,
    }
    
    if method_overrides:
        config["method_overrides"] = method_overrides
    
    config.update(kwargs)
    return config

def multi_agent_config(agents: List[Dict[str, Any]],
                      coordination_strategy: str = "sequential",
                      enable_memory: bool = True,
                      enable_rag: bool = True,
                      method_overrides: Optional[Dict[str, Callable]] = None,
                      **kwargs) -> Dict[str, Any]:
    """
    Create a multi-agent workflow configuration.
    
    Args:
        agents: List of agent configurations
        coordination_strategy: How agents coordinate (sequential, parallel, hierarchical, collaborative)
        enable_memory: Whether to enable shared memory
        enable_rag: Whether to enable shared RAG
        method_overrides: Dictionary of method names to override functions
        **kwargs: Additional configuration parameters
        
    Returns:
        Multi-agent workflow configuration dictionary
    """
    config = {
        "workflow_type": WorkflowType.MULTI_AGENT.value,
        "agents": agents,
        "coordination_strategy": coordination_strategy,
        "enable_memory": enable_memory,
        "enable_rag": enable_rag,
    }
    
    if method_overrides:
        config["method_overrides"] = method_overrides
    
    config.update(kwargs)
    return config

def api_enhanced_config(external_apis: List[Dict[str, Any]],
                       model_config: Union[Dict[str, Any], Any] = None,
                       enable_function_calling: bool = True,
                       method_overrides: Optional[Dict[str, Callable]] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Create an API-enhanced workflow configuration.
    
    Args:
        external_apis: List of external API configurations
        model_config: Model configuration for the workflow
        enable_function_calling: Whether to enable function calling
        method_overrides: Dictionary of method names to override functions
        **kwargs: Additional configuration parameters
        
    Returns:
        API-enhanced workflow configuration dictionary
    """
    config = {
        "workflow_type": WorkflowType.API_ENHANCED.value,
        "external_apis": external_apis,
        "model_config": model_config,
        "enable_function_calling": enable_function_calling,
    }
    
    if method_overrides:
        config["method_overrides"] = method_overrides
    
    config.update(kwargs)
    return config

def custom_config(custom_components: List[Dict[str, Any]],
                 method_overrides: Optional[Dict[str, Callable]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Create a custom workflow configuration.
    
    Args:
        custom_components: List of custom component configurations
        method_overrides: Dictionary of method names to override functions
        **kwargs: Additional configuration parameters
        
    Returns:
        Custom workflow configuration dictionary
    """
    config = {
        "workflow_type": WorkflowType.CUSTOM.value,
        "custom_components": custom_components,
    }
    
    if method_overrides:
        config["method_overrides"] = method_overrides
    
    config.update(kwargs)
    return config 