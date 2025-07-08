# Workflow Configuration Guide

The CoreFlow SDK provides a powerful `WorkflowFactory` system that follows the same patterns as `ModelFactory` for creating different types of workflows with **credential awareness** and **automatic model selection**. This guide covers all the ways to create, configure, and customize workflows.

## 🚀 Quick Start

```python
from sdk.workflow import create_workflow, single_agent_config

# Create a single-agent workflow with automatic credential detection
workflow = create_workflow(single_agent_config(
    # No model_config needed - automatically selects best available model
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True  # Will be disabled if SERPER_API_KEY missing
))

# Check what's available
status = workflow.get_component_status()
print(f"Model: {status['model_client']}")
print(f"Web search: {status['search_client']}")
```

## 🏗️ Creating Workflows with WorkflowFactory

### Method 1: Using WorkflowFactory Directly

```python
from sdk.workflow import WorkflowFactory, WorkflowType

# Create factory instance
factory = WorkflowFactory()

# Create with default configuration
workflow = factory.create_default_workflow(WorkflowType.SINGLE_AGENT.value)
```

### Method 2: Using Configuration Dictionary

```python
from sdk.workflow import create_workflow

# Define configuration with automatic model selection
config = {
    "workflow_type": "single_agent",
    # No model_config needed - automatically selects best available model
    "enable_memory": True,
    "enable_rag": True,
    "enable_websearch": True
}

# Create workflow
workflow = create_workflow(config)
```

### Method 3: Using Helper Functions (Recommended)

```python
from sdk.workflow import create_workflow, single_agent_config

# Create single-agent workflow with automatic model selection
workflow = create_workflow(single_agent_config(
    # No model_config needed - automatically selects best available model
    enable_memory=True,
    enable_rag=False,  # Disable RAG for this example
    enable_websearch=True  # Will be disabled if SERPER_API_KEY missing
))

# Or specify a specific model if you want to override auto-selection
from sdk.model.utils import claude3_haiku_config
workflow_with_claude = create_workflow(single_agent_config(
    model_config=claude3_haiku_config(),  # Force Claude usage
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True
))
```

## 📋 Workflow Types

### Single Agent Workflow

The most common workflow type - combines all components into a single intelligent agent with automatic credential detection.

```python
from sdk.workflow import single_agent_config
from sdk.utils.env import ENV

# Check what credentials are available
env = ENV()
print("Available credentials:", env.get_available_credentials())
print("Disabled features:", env.get_disabled_features())

# Create workflow with automatic model selection
config = single_agent_config(
    # No model_config needed - automatically selects best available model
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True,  # Will be disabled if SERPER_API_KEY missing
    use_docker_qdrant=True,
    log_format="ascii"
)

workflow = create_workflow(config)

# Check what was actually enabled
status = workflow.get_component_status()
print("Enabled components:", status)
```

### Multi-Agent Workflow Configuration

Define multiple agents with different roles and specializations.

```python
from sdk.workflow import multi_agent_config
from sdk.model.utils import gpt4o_mini_config, claude3_haiku_config

# Define multiple agents with different roles
agents = [
    {
        "role": "researcher",
        "model_config": gpt4o_mini_config(),
        "specialization": "web_search_and_analysis",
        "tools": ["web_search", "rag"]
    },
    {
        "role": "writer", 
        "model_config": claude3_haiku_config(),
        "specialization": "content_generation",
        "tools": ["memory", "rag"]
    },
    {
        "role": "reviewer",
        "model_config": gpt4o_mini_config(), 
        "specialization": "quality_control",
        "tools": ["memory"]
    }
]

# Create multi-agent configuration
config = multi_agent_config(
    agents=agents,
    coordination_strategy="sequential",  # researcher -> writer -> reviewer
    enable_memory=True,
    enable_rag=True,
    max_iterations=3
)

# Note: This creates the configuration - actual MultiAgentWorkflow implementation coming soon
```

### API-Enhanced Workflow Configuration

Configure workflows to call external APIs.

```python
from sdk.workflow import api_enhanced_config
from sdk.model.utils import gpt4o_mini_config

# Define external APIs that the workflow can call
external_apis = [
    {
        "name": "weather_api",
        "url": "https://api.openweathermap.org/data/2.5/weather",
        "type": "rest_api",
        "method": "GET",
        "headers": {"Content-Type": "application/json"},
        "auth_type": "api_key",
        "auth_header": "X-API-Key"
    },
    {
        "name": "stock_api", 
        "url": "https://api.polygon.io/v1/open-close",
        "type": "rest_api",
        "method": "GET",
        "timeout": 15
    },
    {
        "name": "database_api",
        "url": "https://my-company-api.com/graphql",
        "type": "graphql_api",
        "headers": {"Authorization": "Bearer {token}"}
    }
]

# Create API-enhanced configuration
config = api_enhanced_config(
    external_apis=external_apis,
    model_config=gpt4o_mini_config(),
    enable_function_calling=True,
    api_timeout=30,
    max_api_calls_per_query=5
)

# Note: This creates the configuration - actual APIWorkflow implementation coming soon
```

## 🔧 WorkflowFactory Features

### Available Workflow Types

```python
from sdk.workflow import get_available_workflow_types

# Get available workflow types
available_types = get_available_workflow_types()
print("Available workflow types:")
for workflow_type, info in available_types.items():
    print(f"  - {workflow_type}: {info['class_name']} (available: {info['available']})")
```

### Pass-through Existing Workflow

```python
from sdk.workflow import create_workflow, single_agent_config

# Pass-through existing workflow instance
existing_workflow = create_workflow(single_agent_config())
passed_through = create_workflow(existing_workflow)
print(f"Pass-through test: {passed_through is existing_workflow}")  # True
```

### Override Configuration Parameters

```python
# Override configuration parameters
base_config = single_agent_config(enable_memory=True)
overridden_workflow = create_workflow(base_config, enable_memory=False, log_format="json")
```

## 🎨 Method Override Examples

The CoreFlow SDK provides multiple ways to override the three public methods of `BaseWorkflow`:
- `process_query()` - Context gathering and prompt formatting
- `generate_response()` - Main orchestration function  
- `process_response()` - Post-processing of model responses

### Method 1: Configuration-Based Overrides

```python
from sdk.workflow import create_workflow, single_agent_config
from sdk.model.utils import gpt4o_mini_config
import time

def custom_process_query(workflow_instance, query: str, user_id: str = "default_user") -> str:
    """Custom query processing with pre-processing steps."""
    print(f"🔧 Custom process_query called for: {query[:50]}...")
    
    # Add custom pre-processing
    if "urgent" in query.lower():
        query = f"[URGENT REQUEST] {query}"
    
    # Add query enrichment
    enriched_query = f"{query}\n\nAdditional context: User {user_id} preferences should be prioritized."
    
    # Call the original method with enriched query
    original_prompt = workflow_instance.super_method("process_query", enriched_query, user_id)
    
    # Add custom post-processing to the prompt
    custom_prompt = f"SYSTEM: Enhanced query processing enabled.\n\n{original_prompt}"
    
    return custom_prompt

def custom_generate_response(workflow_instance, query: str, user_id: str = "default_user", **kwargs) -> str:
    """Custom response generation with external API calls."""
    print(f"🔧 Custom generate_response called for: {query[:50]}...")
    
    # Check if query needs external API calls
    needs_external_data = any(keyword in query.lower() for keyword in ["weather", "stock", "news", "current"])
    
    if needs_external_data:
        print("📡 Query needs external data - simulating API call...")
        external_data = f"[External API Data] Current timestamp: {time.time()}, Status: Active"
        enhanced_query = f"{query}\n\nExternal Data: {external_data}"
        response = workflow_instance.super_method("generate_response", enhanced_query, user_id, **kwargs)
    else:
        response = workflow_instance.super_method("generate_response", query, user_id, **kwargs)
    
    # Add custom metadata to response
    enhanced_response = f"{response}\n\n---\n[Enhanced by CustomWorkflow at {time.strftime('%Y-%m-%d %H:%M:%S')}]"
    
    return enhanced_response

def custom_process_response(workflow_instance, model_response: str, query: str, user_id: str) -> str:
    """Custom response processing with filtering and formatting."""
    print(f"🔧 Custom process_response called")
    
    # Custom response filtering
    if "error" in model_response.lower():
        model_response = f"⚠️ Note: Response contains error information.\n\n{model_response}"
    
    # Call original processing first
    processed_response = workflow_instance.super_method("process_response", model_response, query, user_id)
    
    # Add custom post-processing - format as JSON if requested
    if "json" in query.lower():
        try:
            import json
            structured_response = {
                "query": query,
                "response": processed_response,
                "user_id": user_id,
                "timestamp": time.time(),
                "custom_processed": True
            }
            return json.dumps(structured_response, indent=2)
        except Exception:
            pass  # Fall back to original response
    
    return processed_response

# Create workflow with method overrides
config = single_agent_config(
    model_config=gpt4o_mini_config(),
    enable_memory=True,
    enable_rag=True,
    method_overrides={
        "process_query": custom_process_query,
        "generate_response": custom_generate_response,
        "process_response": custom_process_response
    }
)

workflow = create_workflow(config)

# Test the overridden methods
response = workflow.generate_response("What's the current weather? Please respond in JSON format.", "test_user")
```

### Method 2: Subclassing CustomWorkflow

```python
from sdk.workflow import CustomWorkflow, create_workflow, single_agent_config
import json
import time

class APIEnhancedWorkflow(CustomWorkflow):
    """Example of subclassing CustomWorkflow for API-enhanced functionality."""
    
    def __init__(self, base_workflow, api_endpoints=None):
        super().__init__(base_workflow)
        self.api_endpoints = api_endpoints or {}
    
    def process_query(self, query: str, user_id: str = "default_user") -> str:
        """Override process_query to add API context detection."""
        print(f"🔧 APIEnhancedWorkflow.process_query called")
        
        # Detect if query needs API calls
        api_needed = self._detect_api_needs(query)
        
        if api_needed:
            # Add API context to query
            api_context = f"\n[API Context] Available APIs: {list(self.api_endpoints.keys())}"
            enhanced_query = f"{query}{api_context}"
            return super().process_query(enhanced_query, user_id)
        else:
            return super().process_query(query, user_id)
    
    def generate_response(self, query: str, user_id: str = "default_user", **kwargs) -> str:
        """Override generate_response to include API calls."""
        print(f"🔧 APIEnhancedWorkflow.generate_response called")
        
        # Check for API calls needed
        api_calls = self._extract_api_calls(query)
        
        if api_calls:
            # Make API calls (simulated)
            api_results = {}
            for api_name in api_calls:
                if api_name in self.api_endpoints:
                    api_results[api_name] = self._call_api(api_name, query)
            
            # Add API results to context
            api_context = f"\nAPI Results: {json.dumps(api_results, indent=2)}"
            enhanced_query = f"{query}{api_context}"
            
            return super().generate_response(enhanced_query, user_id, **kwargs)
        else:
            return super().generate_response(query, user_id, **kwargs)
    
    def process_response(self, model_response: str, query: str, user_id: str) -> str:
        """Override process_response to add API call metadata."""
        print(f"🔧 APIEnhancedWorkflow.process_response called")
        
        # Call original processing
        processed = super().process_response(model_response, query, user_id)
        
        # Add API metadata if APIs were used
        if any(api in query for api in self.api_endpoints):
            metadata = f"\n\n---\n[API Enhanced Response] External data sources used: {list(self.api_endpoints.keys())}"
            processed += metadata
        
        return processed
    
    def _detect_api_needs(self, query: str) -> bool:
        """Detect if query needs API calls."""
        api_keywords = ["weather", "stock", "price", "current", "latest", "real-time"]
        return any(keyword in query.lower() for keyword in api_keywords)
    
    def _extract_api_calls(self, query: str) -> list:
        """Extract which APIs are needed for the query."""
        needed_apis = []
        query_lower = query.lower()
        
        if "weather" in query_lower:
            needed_apis.append("weather_api")
        if "stock" in query_lower or "price" in query_lower:
            needed_apis.append("stock_api")
        if "news" in query_lower:
            needed_apis.append("news_api")
        
        return needed_apis
    
    def _call_api(self, api_name: str, query: str) -> dict:
        """Simulate API call."""
        return {
            "api": api_name,
            "query": query,
            "result": f"Simulated {api_name} result",
            "timestamp": time.time()
        }

# Create API-enhanced workflow
base_workflow = create_workflow(single_agent_config(
    model_config=gpt4o_mini_config(),
    enable_memory=True
))

api_endpoints = {
    "weather_api": "https://api.weather.com",
    "stock_api": "https://api.stocks.com",
    "news_api": "https://api.news.com"
}

api_workflow = APIEnhancedWorkflow(base_workflow, api_endpoints)

# Test the API-enhanced workflow
response = api_workflow.generate_response("What's the current stock price of Apple and the weather in New York?", "api_user")
```

### Method 3: Multi-Agent Coordination

```python
from sdk.workflow import CustomWorkflow, create_workflow, single_agent_config
from typing import List

class MultiAgentCoordinator(CustomWorkflow):
    """Example of overriding for multi-agent coordination."""
    
    def __init__(self, base_workflow, agents=None):
        super().__init__(base_workflow)
        self.agents = agents or []
    
    def generate_response(self, query: str, user_id: str = "default_user", **kwargs) -> str:
        """Override to coordinate multiple agents."""
        print(f"🔧 MultiAgentCoordinator.generate_response called")
        
        if len(self.agents) <= 1:
            # Single agent - use original method
            return super().generate_response(query, user_id, **kwargs)
        
        # Multi-agent coordination
        agent_responses = []
        
        for i, agent in enumerate(self.agents):
            print(f"🤖 Agent {i+1} ({agent.get('role', 'assistant')}) processing...")
            
            # Each agent processes the query
            agent_response = agent['workflow'].generate_response(query, user_id, **kwargs)
            agent_responses.append({
                "agent": agent.get('role', f'agent_{i+1}'),
                "response": agent_response
            })
        
        # Coordinate responses
        coordinated_response = self._coordinate_responses(agent_responses, query)
        
        # Process the final coordinated response
        return self.process_response(coordinated_response, query, user_id)
    
    def _coordinate_responses(self, agent_responses: List[dict], query: str) -> str:
        """Coordinate multiple agent responses."""
        coordination_prompt = f"""
Original Query: {query}

Agent Responses:
"""
        
        for response in agent_responses:
            coordination_prompt += f"\n{response['agent']}: {response['response']}\n"
        
        coordination_prompt += """
Please provide a coordinated final response that synthesizes the best insights from all agents.
"""
        
        # Use the base workflow to coordinate (simulated)
        return f"[Coordinated Response] Based on {len(agent_responses)} agents: " + agent_responses[0]['response']

# Create multiple agent workflows
researcher_workflow = create_workflow(single_agent_config(
    model_config=gpt4o_mini_config(),
    enable_websearch=True,
    enable_rag=True
))

writer_workflow = create_workflow(single_agent_config(
    model_config=gpt4o_mini_config(),
    enable_memory=True,
    enable_rag=True
))

# Create multi-agent coordinator
agents = [
    {"role": "researcher", "workflow": researcher_workflow},
    {"role": "writer", "workflow": writer_workflow}
]

base_workflow = create_workflow(single_agent_config())
multi_agent = MultiAgentCoordinator(base_workflow, agents)

# Test coordination
response = multi_agent.generate_response("Research and write a summary about AI workflow patterns.", "multi_user")
```

### Method 4: Simple Function-Based Overrides

```python
from sdk.workflow import create_workflow, single_agent_config
import time

def simple_custom_query_processor(workflow_instance, query: str, user_id: str = "default_user") -> str:
    """Simple example of query preprocessing."""
    # Add timestamp to all queries
    timestamped_query = f"[{time.strftime('%H:%M:%S')}] {query}"
    
    # Call original method
    return workflow_instance.super_method("process_query", timestamped_query, user_id)

def simple_response_formatter(workflow_instance, model_response: str, query: str, user_id: str) -> str:
    """Simple example of response formatting."""
    # Call original processing
    processed = workflow_instance.super_method("process_response", model_response, query, user_id)
    
    # Add simple formatting
    formatted = f"📝 Response for {user_id}:\n{processed}\n\n✅ Processed by CustomWorkflow"
    
    return formatted

# Create workflow with simple overrides
workflow = create_workflow(single_agent_config(
    method_overrides={
        "process_query": simple_custom_query_processor,
        "process_response": simple_response_formatter
    }
))

# Test the overrides
response = workflow.generate_response("Hello, how are you?", "simple_user")
```

## 🔧 Extending Workflows with Custom Functions

```python
from sdk.workflow import create_workflow, single_agent_config
from sdk.model.utils import gpt4o_mini_config

# Create a single-agent workflow
workflow = create_workflow(single_agent_config(
    model_config=gpt4o_mini_config(),
    enable_memory=True
))

# Example of how you could extend the workflow with custom capabilities
class ExtendedWorkflow:
    def __init__(self, base_workflow):
        self.base_workflow = base_workflow
        
    def call_external_api(self, api_url, params):
        """Custom function to call external APIs"""
        import requests
        try:
            response = requests.get(api_url, params=params, timeout=30)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def multi_agent_coordination(self, query, agents):
        """Custom function for multi-agent coordination"""
        results = []
        for agent in agents:
            # Each agent could be a separate workflow instance
            agent_response = agent.generate_response(query)
            results.append({
                "agent": agent.__class__.__name__,
                "response": agent_response
            })
        return results
    
    def generate_response(self, query, user_id="default_user", **kwargs):
        """Enhanced response generation with custom capabilities"""
        # Use base workflow
        base_response = self.base_workflow.generate_response(query, user_id, **kwargs)
        
        # Add custom processing here
        # - Call external APIs based on query content
        # - Coordinate with other agents
        # - Apply custom post-processing
        
        return base_response

# Create extended workflow
extended_workflow = ExtendedWorkflow(workflow)
```

## 📊 Override Methods Summary

1. **⚙️ Configuration-based**: Pass `method_overrides` to `single_agent_config()`
2. **🏗️ Subclassing**: Extend `CustomWorkflow` class with full OOP patterns
3. **🤖 Multi-agent**: Coordinate multiple workflow instances
4. **📝 Simple functions**: Override individual methods with functions

All methods provide access to original functionality via `super_method()` or `super()`.

## 🎯 Configuration Options

### Single Agent Configuration

```python
single_agent_config(
    model_config=None,           # Model configuration
    enable_memory=True,          # Enable conversation memory
    enable_rag=True,            # Enable RAG document retrieval
    enable_websearch=True,      # Enable web search
    use_docker_qdrant=True,     # Use Docker for Qdrant
    log_format="ascii",         # Log format: "ascii" or "json"
    user_collection_prefix="user_",  # Prefix for user collections
    system_collection="system", # System collection name
    method_overrides={}         # Method override functions
)
```

### Multi-Agent Configuration

```python
multi_agent_config(
    agents=[],                  # List of agent configurations
    coordination_strategy="sequential",  # "sequential", "parallel", "hierarchical"
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True,
    max_iterations=5,
    enable_agent_communication=True
)
```

### API-Enhanced Configuration

```python
api_enhanced_config(
    external_apis=[],           # List of API configurations
    enable_function_calling=True,
    api_timeout=30,
    max_api_calls_per_query=10,
    enable_memory=True,
    enable_rag=True,
    enable_websearch=True
)
```

## 📖 Related Documentation

- [UPDATES.md](UPDATES.md) - Main project overview and getting started guide
- [Credential Awareness](CREDENTIAL_AWARENESS.md) - Complete guide to credential detection and graceful degradation
- [FastAPI Integration](FASTAPI_INTEGRATION.md) - Building APIs with workflows
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md) - Setting up environment variables
- [Vector Configuration](VECTOR_CONFIGURATION.md) - RAG and vector store setup
- [Model Configuration](MODEL_CONFIGURATION.md) - Configuring LLM providers

## 🔄 Migration from Legacy Configuration

### Before (BaseWorkflow directly)
```python
from sdk.workflow import BaseWorkflow

workflow = BaseWorkflow(
    model_config=config,
    enable_memory=True
)
```

### After (using factory)
```python
from sdk.workflow import create_workflow, single_agent_config

workflow = create_workflow(single_agent_config(
    model_config=config,
    enable_memory=True
))
```

The new factory system provides better type safety, validation, and extensibility while maintaining backward compatibility. 