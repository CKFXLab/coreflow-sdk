from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging
import time

# Import the concrete implementations
from ..vector import RAG, Mem0
from ..websearch import Search, Scrape, KEYWORDS
from ..model import create_model, Model
from ..utils import AppLogger, ENV
from ._default import AGENT_ROLE, DEFAULT_INSTRUCTION, WEB_DATA_INSTRUCTION, PROMPT_STRUCTURE


class BaseWorkflow:
    """
    Complete workflow class that combines web search, memory, RAG, and LLM capabilities.
    
    This class provides ready-to-use implementations for:
    - Memory management (Mem0)
    - Vector storage and retrieval (RAG) 
    - Web search (Search)
    - Web scraping (Scrape)
    - Model client (any provider via ModelFactory)
    
    Supports multiple model providers via dictionary configuration:
    - OpenAI: openai_config("gpt-4"), gpt4o_mini_config()
    - Bedrock: bedrock_config("claude-3-sonnet"), claude3_haiku_config()
    - Custom configurations via dictionaries
    - Pre-instantiated model objects
    
    Public Interface (only these 3 methods):
    - process_query() - main workflow execution entry point
    - generate_response() - response generation using standardized prompts
    - process_response() - post-processing of generated responses
    """
    
    def __init__(self, 
                 model_config: Union[Dict[str, Any], Model] = None,
                 use_docker_qdrant: bool = True,
                 enable_websearch: bool = True,
                 enable_rag: bool = True,
                 enable_memory: bool = True,
                 log_format: str = "ascii",
                 user_collection_prefix: str = "user_",
                 system_collection: str = "system"):
        """
        Initialize workflow with all concrete components.
        
        Args:
            model_config: Model configuration in one of these formats:
                - Dict: {"provider": "openai", "model": "gpt-4", "api_key": "..."}
                - Model: Already instantiated model object
                - None: Uses best available model based on credentials
            use_docker_qdrant: Whether to use Docker Qdrant or in-memory
            enable_websearch: Whether to enable web search and scraping capabilities
            enable_rag: Whether to enable RAG/vector storage and retrieval capabilities
            enable_memory: Whether to enable memory/conversation history capabilities
            log_format: Logging format ("json" or "ascii")
            user_collection_prefix: Prefix for user-specific collections
            system_collection: Name of system-wide collection
        """
        # Initialize logging
        self.logger = AppLogger(__name__, output_format=log_format)
        self.env = ENV()
        
        # Check available credentials and adjust feature flags
        self.available_credentials = self.env.get_available_credentials()
        self.disabled_features = self.env.get_disabled_features()
        
        # Log credential status
        if self.disabled_features:
            self.logger.warning(f"Disabled features due to missing credentials: {', '.join(self.disabled_features)}")
        
        # Auto-adjust feature flags based on credentials
        enable_websearch = enable_websearch and self.available_credentials.get('serper_api_key', False)
        if not self.available_credentials.get('serper_api_key', False) and enable_websearch:
            self.logger.info("Web search disabled: SERPER_API_KEY not available")
        
        # Memory can always be enabled (falls back to local)
        # RAG can always be enabled (falls back to in-memory)
        
        # Store configuration
        # Use best available model if none provided
        if model_config is None:
            model_config = self.env.get_best_available_model_config()
            if model_config is None:
                self.logger.error("No model provider credentials available")
                raise ValueError("No model provider credentials available. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or AWS credentials.")
            else:
                self.logger.info(f"Using best available model: {model_config['provider']}:{model_config['model']}")
        
        self.model_config = model_config
        self.use_docker_qdrant = use_docker_qdrant
        self.enable_websearch = enable_websearch
        self.enable_rag = enable_rag
        self.enable_memory = enable_memory
        self.user_collection_prefix = user_collection_prefix
        self.system_collection = system_collection
        
        # Initialize all components
        self.model_client = None
        self.memory_client = None
        self.vector_client = None
        self.search_client = None
        self.scrape_client = None
        
        # Track initialization status
        self._component_status = {}
        
        # Initialize components with error handling
        self._init_all_components()
        
        self.logger.info("BaseWorkflow initialized successfully")
    
    def _init_all_components(self):
        """Initialize all workflow components with proper error handling."""
        self._init_model_client()
        self._init_memory_client()
        self._init_vector_client()
        self._init_web_clients()
    
    def _init_model_client(self):
        """Initialize the model client using ModelFactory."""
        try:
            self.model_client = create_model(self.model_config)
            self._component_status['model_client'] = True
            
            # Get model info for logging
            model_info = getattr(self.model_client, 'model', 'unknown')
            provider_info = getattr(self.model_client, 'provider', 'unknown')
            self.logger.info(f"Model client initialized: {provider_info}:{model_info}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model client: {e}")
            self._component_status['model_client'] = False
            # Don't raise - model is critical but we want to show status
    
    def _init_memory_client(self):
        """
        Initialize memory client (Mem0) with fallback support.
        
        This method attempts to initialize the memory client with the configured
        settings, with appropriate error handling for optional functionality.
        """
        if not self.enable_memory:
            self.logger.info("Memory disabled - skipping memory client initialization")
            self._component_status['memory_client'] = False
            return
        
        try:
            # Check if Mem0 cloud is available
            if self.available_credentials.get('mem0_api_key', False):
                self.memory_client = Mem0(env=self.env)  # Use cloud Mem0
                self.logger.info("Memory client initialized with Mem0 cloud")
            else:
                # Try local memory fallback
                self.memory_client = Mem0(env=self.env)  # This will fall back to local
                self.logger.info("Memory client initialized with local fallback (MEM0_API_KEY not available)")
            
            self._component_status['memory_client'] = True
        except Exception as e:
            self.logger.error(f"Failed to initialize memory client: {e}")
            self._component_status['memory_client'] = False
            self.memory_client = None
            # Don't raise - memory is optional
    
    def _init_vector_client(self):
        """
        Initialize vector client (RAG) with fallback support.
        
        This method attempts to initialize the vector client with the configured
        settings, falling back to in-memory mode if Docker connection fails.
        """
        if not self.enable_rag:
            self.logger.info("RAG disabled - skipping vector client initialization")
            self._component_status['vector_client'] = False
            return
            
        try:
            self.vector_client = RAG(
                collection_name="docs",
                purge=False,  # Don't purge by default
                env=self.env  # Pass environment configuration
            )
            self._component_status['vector_client'] = True
            self.logger.info("Vector client initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize vector client with environment config: {e}")
            try:
                # Fallback to in-memory
                self.vector_client = RAG(
                    collection_name="docs",
                    purge=True,
                    use_docker_qdrant=False,
                    env=self.env
                )
                self._component_status['vector_client'] = True
                self.logger.info("Vector client initialized with in-memory fallback")
            except Exception as fallback_error:
                self.logger.error(f"Failed to initialize vector client: {fallback_error}")
                self._component_status['vector_client'] = False
                self.vector_client = None
                # Don't raise - vector storage is optional
    
    def _init_web_clients(self):
        """Initialize web search and scraping clients (concrete implementation)."""
        if not self.enable_websearch:
            self.logger.info("Web search disabled - skipping web client initialization")
            self.search_client = None
            self.scrape_client = None
            self._component_status['search_client'] = False
            self._component_status['scrape_client'] = False
            return
        
        try:
            serper_key = self.env.get_serper_key()
            if not serper_key:
                self.logger.warning("SERPER_API_KEY not available - web search disabled")
                self.search_client = None
                self._component_status['search_client'] = False
            else:
                self.search_client = Search(api_key=serper_key)
                self._component_status['search_client'] = True
                self.logger.info("Search client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize search client: {e}")
            self._component_status['search_client'] = False
            self.search_client = None
        
        try:
            self.scrape_client = Scrape()
            self._component_status['scrape_client'] = True
            self.logger.info("Scrape client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize scrape client: {e}")
            self._component_status['scrape_client'] = False
            self.scrape_client = None
    
    # === PUBLIC INTERFACE (Only these 3 methods) ===
    
    def process_query(self, query: str, user_id: str = "default_user") -> str:
        """
        Context gathering and prompt formatting.
        
        Args:
            query: User's input query
            user_id: User identifier for personalization
            
        Returns:
            Formatted prompt ready for model call
        """
        try:
            self.logger.info(f"Processing query for user {user_id}: {query[:100]}...")
            
            # Gather context from all enabled sources
            context = self._gather_context(query, user_id)
            
            # Build context sections dynamically based on enabled components
            context_sections = []
            
            # Include memory context if enabled
            if self.enable_memory:
                context_sections.append(context.get('memory_section', '[Memory Context from Previous Conversations]\nNo memory context available'))
            
            # Include vector context if enabled
            if self.enable_rag:
                context_sections.append(context.get('vector_section', '[Document Context from Knowledge Base]\nNo document context available'))
            
            # Include web context only if enabled and available
            if self.enable_websearch:
                web_section = context.get('web_section', '')
                if web_section:
                    context_sections.append(web_section)
            
            # Build context text
            context_text = '\n\n'.join(context_sections)
            
            # Choose instruction based on web data availability
            instruction = WEB_DATA_INSTRUCTION if context.get('has_web_data', False) else DEFAULT_INSTRUCTION
            
            # Use standardized prompt structure from _default.py
            prompt = PROMPT_STRUCTURE.format(
                agent_role=AGENT_ROLE,
                query=query,
                context=context_text
            ) + f"\n{instruction}"
            
            self.logger.info("Query processed and prompt formatted successfully")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            return f"Query processing failed: {str(e)}"
    
    def generate_response(self, query: str, user_id: str = "default_user", **kwargs) -> str:
        """
        Main orchestration function - coordinates the entire workflow.
        
        Args:
            query: User's input query
            user_id: User identifier for personalization
            **kwargs: Additional workflow-specific parameters
            
        Returns:
            Final processed response string
        """
        try:
            # Step 1: Process query and get formatted prompt
            prompt = self.process_query(query, user_id)
            
            # Step 2: Make model call with formatted prompt
            if not self.model_client:
                return "Model client unavailable"
            
            messages = [{"role": "user", "content": prompt}]
            model_response = self.model_client(messages=messages, temperature=0.7, **kwargs)
            
            # Step 3: Post-process the model response
            final_response = self.process_response(model_response, query, user_id)
            
            self.logger.info("Response generation completed successfully")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return f"Response generation failed: {str(e)}"
    
    def process_response(self, model_response: str, query: str, user_id: str) -> str:
        """
        Post-process the model response (storage, cleanup, formatting, etc.).
        
        Args:
            model_response: Raw response from the model
            query: Original user query
            user_id: User identifier
            
        Returns:
            Final processed response
        """
        try:
            # Store the interaction for future reference
            self._store_interaction(query, model_response, user_id)
            
            # For now, just return the response as-is
            # This can be extended for post-processing like:
            # - Response filtering/safety checks
            # - Format conversion (markdown, etc.)
            # - Response enhancement
            # - Analytics/logging
            return model_response
            
        except Exception as e:
            self.logger.warning(f"Failed to post-process response: {e}")
            # Return original response even if post-processing fails
            return model_response
    
    # === PRIVATE HELPER METHODS ===
    
    def _gather_memory_context(self, query: str, user_id: str) -> str:
        """
        Gather context from memory (private implementation).
        
        Args:
            query: User's query
            user_id: User identifier
            
        Returns:
            Memory context string
        """
        if not self.enable_memory or not self.memory_client:
            return "Memory disabled" if not self.enable_memory else "No memory context available"
        
        try:
            return self.memory_client.get_memory_context(query, user_id)
        except Exception as e:
            self.logger.error(f"Failed to get memory context: {e}")
            return "Memory context unavailable"
    
    def _gather_vector_context(self, query: str, collection_type: str = "user", user_id: str = "default_user") -> str:
        """
        Gather context from vector storage (private implementation).
        
        Args:
            query: User's query
            collection_type: "user" or "system" collection
            user_id: User identifier for user collections
            
        Returns:
            Vector context string
        """
        if not self.enable_rag or not self.vector_client:
            return "RAG disabled" if not self.enable_rag else "No vector context available"
        
        try:
            if collection_type == "user":
                collection_name = f"{self.user_collection_prefix}{user_id}"
            else:
                collection_name = self.system_collection
            
            # Search in appropriate collection
            if collection_type == "user":
                results = self.vector_client.search_similar(query, "user", user_id)
            else:
                results = self.vector_client.search_similar(query, "system")
            return str(results) if results else "No relevant documents found"
        except Exception as e:
            self.logger.error(f"Failed to get vector context: {e}")
            return "Vector context unavailable"
    
    def _should_trigger_web_search(self, query: str) -> bool:
        """
        Determine if web search should be triggered based on query keywords.
        
        Args:
            query: User's query
            
        Returns:
            True if web search should be triggered, False otherwise
        """
        if not self.enable_websearch:
            return False
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in KEYWORDS)
    
    def _gather_web_context(self, query: str, max_results: int = 3) -> Dict[str, str]:
        """
        Gather context from web search and scraping (private implementation).
        
        Args:
            query: Search query
            max_results: Maximum number of results to process
            
        Returns:
            Dictionary with search_results and webpage_content
        """
        context = {
            "search_results": "No search results available",
            "webpage_content": "No webpage content available"
        }
        
        # Check if web search is enabled
        if not self.enable_websearch:
            context["search_results"] = "Web search disabled"
            context["webpage_content"] = "Web search disabled"
            return context
        
        # Check if query needs web search based on keywords
        if not self._should_trigger_web_search(query):
            context["search_results"] = "No time-sensitive keywords detected - using knowledge base"
            context["webpage_content"] = "No web search needed for this query"
            return context
        
        # Get web search results
        if self.search_client:
            try:
                search_results = self.search_client.google_query(query)
                context["search_results"] = search_results
                
                # Try to scrape the first URL
                if self.scrape_client and search_results and ":" in search_results:
                    try:
                        first_line = search_results.split("\n")[0]
                        if ": " in first_line:
                            first_url = first_line.split(": ", 1)[1]
                            webpage_content = self.scrape_client.browse_url(first_url)
                            context["webpage_content"] = webpage_content
                    except Exception as scrape_error:
                        self.logger.warning(f"Failed to scrape URL: {scrape_error}")
                        
            except Exception as e:
                self.logger.error(f"Failed to get web search results: {e}")
        
        return context
    
    def _store_memory(self, query: str, response: str, user_id: str, memory_type: str = "user") -> bool:
        """
        Store interaction in memory (private implementation).
        
        Args:
            query: User's query
            response: Generated response
            user_id: User identifier
            memory_type: "user" or "system" memory
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enable_memory or not self.memory_client:
            return False
        
        try:
            if memory_type == "system":
                return self.memory_client.store_system_memory(f"Q: {query} | A: {response}")
            else:
                return self.memory_client.store_conversation(query, response, user_id)
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False
    
    def _store_vector_data(self, text: str, metadata: Dict[str, Any], collection_type: str = "user", user_id: str = "default_user") -> bool:
        """
        Store data in vector storage (private implementation).
        
        Args:
            text: Text to store
            metadata: Associated metadata
            collection_type: "user" or "system" collection
            user_id: User identifier for user collections
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enable_rag or not self.vector_client:
            return False
        
        try:
            if collection_type == "user":
                collection_name = f"{self.user_collection_prefix}{user_id}"
            else:
                collection_name = self.system_collection
            
            return self.vector_client.store(text, metadata, collection_type, user_id)
        except Exception as e:
            self.logger.error(f"Failed to store vector data: {e}")
            return False
    

    
    # === WORKFLOW UTILITIES (Concrete implementations) ===
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status of all workflow components.
        
        Returns:
            Dictionary with component status information
        """
        status = {
            "model_client": self.model_client is not None,
            "memory_client": self.memory_client is not None,
            "vector_client": self.vector_client is not None,
            "search_client": self.search_client is not None,
            "scrape_client": self.scrape_client is not None,
            "component_details": self._component_status,
            "credentials": self.available_credentials,
            "disabled_features": self.disabled_features
        }
        
        # Add memory status if available
        if self.memory_client:
            try:
                status["memory_status"] = self.memory_client.get_client_status()
            except Exception as e:
                status["memory_status"] = {"error": str(e)}
        
        return status
    
    def validate_workflow(self) -> Dict[str, Any]:
        """
        Validate that the workflow is properly configured.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "component_status": self.get_component_status(),
            "credential_status": self.env.get_provider_availability()
        }
        
        # Check critical components
        if not self.model_client:
            validation["valid"] = False
            validation["errors"].append("Model client not initialized - workflow cannot generate responses")
        
        # Check optional components and add warnings for missing credentials
        if not self.memory_client:
            validation["warnings"].append("Memory client not available - no conversation history")
        
        if not self.vector_client:
            validation["warnings"].append("Vector client not available - no document retrieval")
        
        if not self.search_client:
            if not self.available_credentials.get('serper_api_key', False):
                validation["warnings"].append("Search client not available - SERPER_API_KEY missing")
            else:
                validation["warnings"].append("Search client not available - initialization failed")
        
        if not self.scrape_client:
            validation["warnings"].append("Scrape client not available - no webpage content")
        
        # Add credential-specific warnings
        for feature in self.disabled_features:
            validation["warnings"].append(f"Feature disabled: {feature}")
        
        return validation
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get comprehensive workflow information.
        
        Returns:
            Dictionary with workflow details
        """
        return {
            "class_name": self.__class__.__name__,
            "model_config": self.model_config,
            "use_docker_qdrant": self.use_docker_qdrant,
            "user_collection_prefix": self.user_collection_prefix,
            "system_collection": self.system_collection,
            "component_status": self.get_component_status(),
            "validation": self.validate_workflow(),
            "available_credentials": self.available_credentials,
            "disabled_features": self.disabled_features
        }
    
    # === OPTIONAL OVERRIDE METHODS ===
    
    def _gather_context(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Context gathering implementation with formatted prompt sections.
        
        Args:
            query: User's query
            user_id: User identifier
            
        Returns:
            Dictionary with formatted context sections ready for prompt building
        """
        # Gather raw context data
        memory_context = self._gather_memory_context(query, user_id)
        vector_context = self._gather_vector_context(query, "user", user_id)
        web_context = self._gather_web_context(query)
        
        # Build formatted context sections
        context = {
            "memory_section": f"[Memory Context from Previous Conversations]\n{memory_context}",
            "vector_section": f"[Document Context from Knowledge Base]\n{vector_context}",
            "web_section": self._build_web_section(web_context),
            "has_web_data": (self.enable_websearch and 
                           web_context.get("search_results") not in ["Web search disabled", "No time-sensitive keywords detected - using knowledge base"])
        }
        
        return context
    
    def _build_web_section(self, web_context: Dict[str, str]) -> str:
        """
        Build formatted web context section for prompts.
        
        Args:
            web_context: Dictionary with search_results and webpage_content
            
        Returns:
            Formatted web section string
        """
        search_results = web_context.get("search_results", "No search results available")
        webpage_content = web_context.get("webpage_content", "No webpage content available")
        
        # Only include web section if we have actual search results (not disabled or keyword-filtered)
        if (self.enable_websearch and 
            search_results not in ["Web search disabled", "No time-sensitive keywords detected - using knowledge base"]):
            return f"[Web Search Results]\n{search_results}\n\n[Webpage Content]\n{webpage_content}"
        else:
            return ""
    

    
    def _store_interaction(self, query: str, response: str, user_id: str):
        """
        Interaction storage implementation.
        
        Args:
            query: User's query
            response: Generated response
            user_id: User identifier
        """
        # Store in memory
        self._store_memory(query, response, user_id)
        
        # Optionally store in vector database for future retrieval
        interaction_text = f"User asked: {query}\nAssistant responded: {response}"
        metadata = {
            "type": "conversation",
            "user_id": user_id,
            "timestamp": time.time(),
            "query": query
        }
        self._store_vector_data(interaction_text, metadata, "user", user_id)



