# Lazy import to avoid RuntimeWarning when using python -m
def __getattr__(name):
    if name == 'SingleAgentWorkflow':
        from ._wabc import BaseWorkflow
        return BaseWorkflow
    elif name == 'BaseWorkflow':
        from ._wabc import BaseWorkflow
        return BaseWorkflow
    elif name == 'WorkflowFactory':
        from ._factory import WorkflowFactory
        return WorkflowFactory
    elif name == 'CustomWorkflow':
        from ._factory import CustomWorkflow
        return CustomWorkflow
    elif name == 'create_workflow':
        from ._factory import create_workflow
        return create_workflow
    elif name == 'get_available_workflow_types':
        from ._factory import get_available_workflow_types
        return get_available_workflow_types
    elif name == 'create_default_workflow':
        from ._factory import create_default_workflow
        return create_default_workflow
    elif name == 'single_agent_config':
        from ._factory import single_agent_config
        return single_agent_config
    elif name == 'multi_agent_config':
        from ._factory import multi_agent_config
        return multi_agent_config
    elif name == 'api_enhanced_config':
        from ._factory import api_enhanced_config
        return api_enhanced_config
    elif name == 'custom_config':
        from ._factory import custom_config
        return custom_config
    elif name == 'WorkflowType':
        from ._default import WorkflowType
        return WorkflowType
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Existing workflows
    'SingleAgentWorkflow', 
    'BaseWorkflow',
    
    # Factory and creation functions
    'WorkflowFactory',
    'CustomWorkflow',
    'create_workflow',
    'get_available_workflow_types', 
    'create_default_workflow',
    
    # Configuration helpers
    'single_agent_config',
    'multi_agent_config', 
    'api_enhanced_config',
    'custom_config',
    
    # Types
    'WorkflowType',
]
