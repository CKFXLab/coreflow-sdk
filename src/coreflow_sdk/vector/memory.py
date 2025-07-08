import logging
from mem0 import Memory
from typing import Optional, Dict, List, Union

from ._default import CONFIG, CollectionType, get_config
from ..utils import AppLogger
from ..utils.env import ENV


class Mem0:
    """Enhanced Mem0 memory management with multi-tenant support for both user and system memories."""

    def __init__(self, config: dict = None, env: Optional[ENV] = None):
        """
        Initialize Mem0 client with multi-tenant support and dynamic configuration.
        
        Args:
            config: Mem0 configuration dict. If None, uses dynamic configuration from environment.
            env: ENV instance for configuration (creates new one if None)
        """
        self.logger = AppLogger(__name__)
        
        # Initialize environment configuration
        self.env = env or ENV()
        
        # Use provided config or create dynamic config
        if config is not None:
            self.config = config
        else:
            self.config = get_config(self.env)
        
        self.client = None
        
        # Special user ID for system-level memories
        self.SYSTEM_USER_ID = "system"
        self.SYSTEM_AGENT_ID = "system_agent"
        
        try:
            self.client = Memory.from_config(self.config)
            provider = self.config.get("embedder", {}).get("provider", "openai")
            model = self.config.get("embedder", {}).get("config", {}).get("model", "unknown")
            self.logger.info(f"Mem0 client initialized with {provider} embeddings (model: {model})")
        except Exception as e:
            self.logger.error(f"Failed to initialize Mem0 client: {e}")
            raise

    def store_memory(self, content: str, user_id: str = "default_user", 
                    memory_type: str = CollectionType.USER) -> bool:
        """
        Store content in memory for a specific user or system-wide.
        
        Args:
            content: Content to store in memory
            user_id: User identifier for memory isolation (ignored for system memory)
            memory_type: Either 'user' or 'system' to determine storage location
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("Mem0 client not available")
            return False
        
        try:
            # Determine target user_id based on memory type
            target_user_id = self.SYSTEM_USER_ID if memory_type == CollectionType.SYSTEM else user_id
            
            # Add metadata to distinguish memory types
            if memory_type == CollectionType.SYSTEM:
                enhanced_content = f"[SYSTEM] {content}"
                self.client.add(enhanced_content, user_id=target_user_id)
                self.logger.info(f"Stored system memory")
            else:
                enhanced_content = f"[USER:{user_id}] {content}"
                self.client.add(enhanced_content, user_id=target_user_id)
                self.logger.info(f"Stored user memory for {user_id}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False

    def store_user_memory(self, content: str, user_id: str) -> bool:
        """
        Convenience method to store user-specific memory.
        
        Args:
            content: Content to store
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.store_memory(content, user_id, CollectionType.USER)

    def store_system_memory(self, content: str) -> bool:
        """
        Convenience method to store system-wide memory.
        
        Args:
            content: Content to store in system memory
            
        Returns:
            True if successful, False otherwise
        """
        return self.store_memory(content, memory_type=CollectionType.SYSTEM)

    def get_memory_context(self, query: str, user_id: str = "default_user", 
                          memory_type: str = CollectionType.USER, limit: int = 5) -> str:
        """
        Retrieve memory context for a query from specified memory type.
        
        Args:
            query: Query to search for relevant memories
            user_id: User identifier for memory isolation (ignored for system memory)
            memory_type: Either 'user' or 'system' to determine search location
            limit: Maximum number of memories to retrieve
            
        Returns:
            Formatted string with relevant memory context
        """
        if not self.client:
            self.logger.warning("Mem0 client not available")
            return "Memory context unavailable"
        
        try:
            # Determine target user_id based on memory type
            target_user_id = self.SYSTEM_USER_ID if memory_type == CollectionType.SYSTEM else user_id
            
            # Search for memories
            memories = self.client.search(query, user_id=target_user_id, limit=limit)
            if memories:
                context_parts = []
                for memory in memories:
                    if isinstance(memory, dict) and 'memory' in memory:
                        # Clean up the memory content (remove our prefixes)
                        clean_memory = self._clean_memory_content(memory['memory'])
                        context_parts.append(clean_memory)
                    else:
                        clean_memory = self._clean_memory_content(str(memory))
                        context_parts.append(clean_memory)
                
                if context_parts:
                    result = " | ".join(context_parts)
                    memory_source = "system" if memory_type == CollectionType.SYSTEM else f"user {user_id}"
                    self.logger.debug(f"Retrieved {len(context_parts)} memory context items from {memory_source}")
                    return result
                else:
                    return "No relevant memories found"
            
            return "No memory context found"
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory context: {e}")
            return f"Memory retrieval failed: {str(e)}"

    def get_user_memory_context(self, query: str, user_id: str, limit: int = 5) -> str:
        """
        Convenience method to get user-specific memory context.
        
        Args:
            query: Query to search for relevant memories
            user_id: User identifier
            limit: Maximum number of memories to retrieve
            
        Returns:
            Formatted string with relevant user memory context
        """
        return self.get_memory_context(query, user_id, CollectionType.USER, limit)

    def get_system_memory_context(self, query: str, limit: int = 5) -> str:
        """
        Convenience method to get system-wide memory context.
        
        Args:
            query: Query to search for relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            Formatted string with relevant system memory context
        """
        return self.get_memory_context(query, memory_type=CollectionType.SYSTEM, limit=limit)

    def get_combined_memory_context(self, query: str, user_id: str = "default_user", 
                                   user_limit: int = 3, system_limit: int = 2) -> Dict[str, str]:
        """
        Get memory context from both user and system memories.
        
        Args:
            query: Query to search for relevant memories
            user_id: User identifier for user memories
            user_limit: Maximum number of user memories to retrieve
            system_limit: Maximum number of system memories to retrieve
            
        Returns:
            Dictionary with 'user' and 'system' memory contexts
        """
        user_context = self.get_user_memory_context(query, user_id, user_limit)
        system_context = self.get_system_memory_context(query, system_limit)
        
        return {
            "user": user_context,
            "system": system_context,
            "combined": f"User Context: {user_context} | System Context: {system_context}"
        }

    def _clean_memory_content(self, memory_content: str) -> str:
        """
        Clean memory content by removing our internal prefixes.
        
        Args:
            memory_content: Raw memory content with prefixes
            
        Returns:
            Cleaned memory content
        """
        # Remove our internal prefixes
        if memory_content.startswith("[SYSTEM] "):
            return memory_content[9:]  # Remove "[SYSTEM] "
        elif memory_content.startswith("[USER:"):
            # Remove "[USER:username] " pattern
            end_bracket = memory_content.find("] ")
            if end_bracket != -1:
                return memory_content[end_bracket + 2:]
        
        return memory_content

    def store_conversation(self, query: str, response: str, user_id: str = "default_user",
                          memory_type: str = CollectionType.USER) -> bool:
        """
        Store a conversation exchange in memory.
        
        Args:
            query: User's question
            response: Assistant's response
            user_id: User identifier (ignored for system memory)
            memory_type: Either 'user' or 'system' to determine storage location
            
        Returns:
            True if successful, False otherwise
        """
        memory_entry = f"Question: {query} | Answer: {response}"
        return self.store_memory(memory_entry, user_id, memory_type)

    def store_user_conversation(self, query: str, response: str, user_id: str) -> bool:
        """Convenience method to store user conversation."""
        return self.store_conversation(query, response, user_id, CollectionType.USER)

    def store_system_conversation(self, query: str, response: str) -> bool:
        """Convenience method to store system-wide conversation pattern."""
        return self.store_conversation(query, response, memory_type=CollectionType.SYSTEM)

    def get_all_memories(self, user_id: str = "default_user", 
                        memory_type: str = CollectionType.USER) -> List[Dict]:
        """
        Get all memories for a specific user or system.
        
        Args:
            user_id: User identifier (ignored for system memory)
            memory_type: Either 'user' or 'system'
            
        Returns:
            List of memory dictionaries
        """
        if not self.client:
            self.logger.error("Mem0 client not available")
            return []
        
        try:
            target_user_id = self.SYSTEM_USER_ID if memory_type == CollectionType.SYSTEM else user_id
            memories = self.client.get_all(user_id=target_user_id)
            
            # Clean the memory content in results
            if isinstance(memories, dict) and 'results' in memories:
                for memory in memories['results']:
                    if 'memory' in memory:
                        memory['memory'] = self._clean_memory_content(memory['memory'])
                        memory['memory_type'] = memory_type
                return memories['results']
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get all memories: {e}")
            return []

    def clear_user_memory(self, user_id: str) -> bool:
        """
        Clear all memories for a specific user.
        
        Args:
            user_id: User identifier whose memories to clear
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("Mem0 client not available")
            return False
        
        try:
            self.client.delete_all(user_id=user_id)
            self.logger.info(f"Cleared all memories for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear memory for user {user_id}: {e}")
            return False

    def clear_system_memory(self) -> bool:
        """
        Clear all system-wide memories.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("Mem0 client not available")
            return False
        
        try:
            self.client.delete_all(user_id=self.SYSTEM_USER_ID)
            self.logger.info("Cleared all system memories")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear system memory: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get statistics about stored memories.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            system_memories = self.get_all_memories(memory_type=CollectionType.SYSTEM)
            
            # Note: Getting all user memories would require knowing all user IDs
            # This is a simplified implementation
            return {
                "system_memories": len(system_memories),
                "client_available": self.client is not None
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    def get_client_status(self) -> dict:
        """
        Get status information about the Mem0 client.
        
        Returns:
            Dictionary with client status information
        """
        stats = self.get_memory_stats()
        return {
            "client_available": self.client is not None,
            "config": self.config,
            "system_user_id": self.SYSTEM_USER_ID,
            "memory_stats": stats
        }