import sys
import time
import uuid
from datetime import datetime
from typing import Optional, Union, Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from ._default import (
    COLLECTION_NAME,
    SYSTEM_COLLECTION_NAME,
    USER_COLLECTION_PREFIX,
    CollectionType,
    create_qdrant_config,
    create_embedding_client,
    get_embedding_client_config,
)
from ..utils.audit import AppLogger
from ..utils.env import ENV
from .utils.file_operations import FileOperations
from .utils.collections import create_user_collection_name

MAX_ATTEMPTS = 3


class RAG:
    """Multi-tenant RAG system supporting both user and system collections."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        purge: bool = True,
        use_docker_qdrant: bool = None,
        qdrant_host: str = None,
        qdrant_port: int = None,
        env: Optional[ENV] = None,
    ):
        """
        Initialize RAG system with dynamic configuration support.

        Args:
            collection_name: Base collection name for backward compatibility
            purge: Whether to purge/setup collections on initialization
            use_docker_qdrant: Override for Docker usage (deprecated, use env vars)
            qdrant_host: Override for Qdrant host (deprecated, use env vars)
            qdrant_port: Override for Qdrant port (deprecated, use env vars)
            env: ENV instance for configuration (creates new one if None)
        """
        self.base_collection_name = collection_name
        self.client = None

        # Initialize environment configuration
        self.env = env or ENV()

        # Get dynamic configuration
        self.qdrant_config = create_qdrant_config(self.env)
        self.embedding_config = get_embedding_client_config(self.env)

        # Apply legacy parameter overrides if provided
        if use_docker_qdrant is not None:
            self.qdrant_config["use_docker"] = use_docker_qdrant
        if qdrant_host is not None:
            self.qdrant_config["host"] = qdrant_host
        if qdrant_port is not None:
            self.qdrant_config["port"] = qdrant_port

        # Store configuration for backward compatibility
        self.use_docker_qdrant = self.qdrant_config.get("use_docker", True)
        self.qdrant_host = self.qdrant_config.get("host", "localhost")
        self.qdrant_port = self.qdrant_config.get("port", 6333)

        # Initialize logger
        self.logger = AppLogger(__name__)

        # Initialize embedding client
        self.embedding_client = None
        self._init_embedding_client()

        # Initialize file operations handler
        self.file_ops = FileOperations()

        # Connect to Qdrant
        try:
            for i in range(MAX_ATTEMPTS):
                connected = self._connect_qdrant()
                if connected:
                    break
                else:
                    if i < MAX_ATTEMPTS - 1:
                        time.sleep(2)
                        connected = self._connect_qdrant()
                    else:
                        connected = self._connect_qdrant(inmemory=True)
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG: {e}")
            sys.exit(1)

        if purge and self.client:
            self._setup_collections()

    def _init_embedding_client(self):
        """Initialize embedding client using dynamic configuration."""
        try:
            self.embedding_client = create_embedding_client(self.env)
            provider = self.embedding_config.get("provider", "openai")
            model = self.embedding_config.get("model", "text-embedding-3-small")
            self.logger.info(
                f"Embedding client initialized: {provider} with model {model}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding client: {e}")
            # Fallback to OpenAI for backward compatibility
            try:
                from ..model.api.openai import OpenAIClient

                self.embedding_client = OpenAIClient()
                self.logger.info("Fallback to OpenAI embedding client")
            except Exception as fallback_error:
                self.logger.error(
                    f"Failed to initialize fallback embedding client: {fallback_error}"
                )
                raise

    def _connect_qdrant(self, inmemory: bool = False) -> bool:
        """Connect to Qdrant using dynamic configuration."""
        try:
            if inmemory:
                self.client = QdrantClient(":memory:")
                self.logger.info("Using in-memory Qdrant instance")
                return True

            # Check if URL-based configuration is provided
            qdrant_url = self.qdrant_config.get("url")
            if qdrant_url:
                # URL-based connection (for Fargate, cloud deployments)
                api_key = self.qdrant_config.get("api_key")
                if api_key:
                    self.client = QdrantClient(url=qdrant_url, api_key=api_key)
                    self.logger.info(
                        f"Connected to Qdrant at {qdrant_url} with API key"
                    )
                else:
                    self.client = QdrantClient(url=qdrant_url)
                    self.logger.info(f"Connected to Qdrant at {qdrant_url}")
                return True

            # Host/port-based connection (for local Docker or direct connection)
            elif self.use_docker_qdrant:
                self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                self.logger.info(
                    f"Connected to Docker Qdrant at {self.qdrant_host}:{self.qdrant_port}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            return False

    def _setup_collections(self) -> None:
        """Setup both system and example user collections."""
        # Get embedding dimensions from configuration
        embedding_dims = self.qdrant_config.get("embedding_dimensions", 1536)

        # Setup system collection
        self._ensure_collection_exists(SYSTEM_COLLECTION_NAME, embedding_dims)

        # Setup base collection for backwards compatibility
        self._ensure_collection_exists(self.base_collection_name, embedding_dims)

        self.logger.info("Multi-tenant RAG collections initialized")

    def _ensure_collection_exists(
        self, collection_name: str, embedding_dims: int = 1536
    ) -> None:
        """Ensure a collection exists, creating it if necessary."""
        try:
            if self.client.collection_exists(collection_name):
                self.logger.info(f"Collection {collection_name} already exists")
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dims, distance=Distance.COSINE
                    ),
                )
                self.logger.info(
                    f"Created collection {collection_name} with {embedding_dims} dimensions"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to ensure collection {collection_name} exists: {e}"
            )

    def _get_collection_name(
        self, collection_type: str, user_id: Optional[str] = None
    ) -> str:
        """Generate collection name based on type and user_id using new :: namespace pattern."""
        if collection_type == CollectionType.SYSTEM:
            return SYSTEM_COLLECTION_NAME
        elif collection_type == CollectionType.USER:
            if not user_id:
                raise ValueError("user_id is required for user collections")
            # Use the new collection utility to sanitize username and create proper collection name
            return create_user_collection_name(user_id)
        else:
            # For backwards compatibility
            return self.base_collection_name

    def _generate_point_id(self, metadata: dict, text: str) -> Union[int, str]:
        """
        Generate a valid Qdrant point ID from metadata or text.

        Qdrant accepts either:
        - Unsigned integers
        - UUIDs (as strings)

        Args:
            metadata: Metadata dict that may contain an "id" field
            text: Text content to use as fallback for ID generation

        Returns:
            Valid point ID (int or UUID string)
        """
        original_id = metadata.get("id")

        if original_id is None:
            # No ID provided, use hash of text (guaranteed to be positive integer)
            return abs(hash(text))

        if isinstance(original_id, int) and original_id >= 0:
            # Already a valid unsigned integer
            return original_id

        if isinstance(original_id, str):
            try:
                # Try to parse as UUID (Qdrant accepts UUID strings)
                uuid.UUID(original_id)
                return original_id
            except ValueError:
                # Not a UUID, convert string to hash (positive integer)
                return abs(hash(original_id))

        # Fallback: convert whatever it is to a hash
        return abs(hash(str(original_id)))

    def get_collection_info(
        self,
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
    ):
        """Get information about a specific collection."""
        if not self.client:
            self.logger.error("Qdrant client not available")
            return None

        collection_name = self._get_collection_name(collection_type, user_id)

        try:
            if self.client.collection_exists(collection_name):
                info = self.client.get_collection(collection_name)
                self.logger.info(
                    f"Collection {collection_name} exists with {info.points_count} points"
                )
                return info
            else:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return None
        except Exception as e:
            self.logger.error(
                f"Failed to get collection info for {collection_name}: {e}"
            )
            return None

    def search_similar(
        self,
        query: str,
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
        limit: int = 5,
    ):
        """Search for similar vectors in specified collection."""
        if not self.client:
            self.logger.error("Qdrant client not available")
            return []

        if not self.embedding_client:
            self.logger.error("Embedding client not available")
            return []

        collection_name = self._get_collection_name(collection_type, user_id)

        # Ensure collection exists
        if collection_type == CollectionType.USER:
            self._ensure_collection_exists(collection_name)

        try:
            # Generate embedding for query using SDK abstraction
            embedding_model = self.embedding_config.get("model")
            query_embedding = self.embedding_client.generate_embedding(
                query, embedding_model
            )

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
            )

            self.logger.info(
                f"Found {len(search_results)} similar results in {collection_name}"
            )
            return search_results

        except Exception as e:
            self.logger.error(f"Failed to search in {collection_name}: {e}")
            return []

    def search_across_collections(
        self, query: str, user_id: Optional[str] = None, limit_per_collection: int = 3
    ):
        """Search across both system and user collections, combining results."""
        all_results = []

        # Search system collection
        system_results = self.search_similar(
            query, CollectionType.SYSTEM, limit=limit_per_collection
        )
        for result in system_results:
            result.payload = result.payload or {}
            result.payload["source_collection"] = "system"
        all_results.extend(system_results)

        # Search user collection if user_id provided
        if user_id:
            user_results = self.search_similar(
                query, CollectionType.USER, user_id, limit=limit_per_collection
            )
            for result in user_results:
                result.payload = result.payload or {}
                result.payload["source_collection"] = f"user_{user_id}"
            all_results.extend(user_results)

        # Sort combined results by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)

        self.logger.info(f"Combined search returned {len(all_results)} results")
        return all_results

    def store(
        self,
        text: str,
        metadata: dict,
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
    ):
        """Store text and metadata to specified collection."""
        if not self.client:
            self.logger.error("Qdrant client not available")
            return False

        if not self.embedding_client:
            self.logger.error("Embedding client not available")
            return False

        collection_name = self._get_collection_name(collection_type, user_id)

        # Ensure collection exists (especially important for user collections)
        if collection_type == CollectionType.USER:
            self._ensure_collection_exists(collection_name)

        try:
            # Generate embedding using OpenAI SDK abstraction
            embedding_model = self.embedding_config.get("model")
            embedding = self.embedding_client.generate_embedding(text, embedding_model)

            # Add collection info to metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update(
                {
                    "collection_type": collection_type,
                    "text": text,  # Store original text for reference
                }
            )
            if user_id:
                enhanced_metadata["user_id"] = user_id

            # Generate valid Qdrant point ID and preserve original ID in metadata
            point_id = self._generate_point_id(enhanced_metadata, text)
            if "id" in metadata:
                enhanced_metadata["original_id"] = metadata[
                    "id"
                ]  # Preserve original string ID

            # Create point and store in Qdrant
            point = PointStruct(
                id=point_id, vector=embedding, payload=enhanced_metadata
            )
            self.client.upsert(collection_name=collection_name, points=[point])
            self.logger.info(f"Successfully stored to {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store to {collection_name}: {e}")
            return False

    def store_to_system(self, text: str, metadata: dict):
        """Convenience method to store to system collection."""
        return self.store(text, metadata, CollectionType.SYSTEM)

    def store_to_user(self, text: str, metadata: dict, user_id: str):
        """Convenience method to store to user collection."""
        if not user_id:
            raise ValueError("user_id is required for user collections")
        return self.store(text, metadata, CollectionType.USER, user_id)

    def list_user_collections(self):
        """List all user collections."""
        try:
            collections = self.client.get_collections()
            user_collections = [
                col.name
                for col in collections.collections
                if col.name.startswith(USER_COLLECTION_PREFIX)
            ]
            self.logger.info(f"Found {len(user_collections)} user collections")
            return user_collections
        except Exception as e:
            self.logger.error(f"Failed to list user collections: {e}")
            return []

    def delete_user_collection(self, user_id: str):
        """Delete a specific user's collection."""
        collection_name = self._get_collection_name(CollectionType.USER, user_id)
        try:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                self.logger.info(f"Deleted user collection: {collection_name}")
                return True
            else:
                self.logger.warning(f"User collection {collection_name} does not exist")
                return False
        except Exception as e:
            self.logger.error(
                f"Failed to delete user collection {collection_name}: {e}"
            )
            return False

    # === FILE PROCESSING INTEGRATION ===

    def process_file(
        self,
        filepath: str,
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
        subject: Optional[str] = None,
        short_description: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunking_method: str = "sliding_window",
    ) -> Dict[str, Any]:
        """
        Process a file and store its chunks in the RAG system with comprehensive metadata.

        This method handles the complete file processing pipeline:
        1. File validation and type detection
        2. Content extraction based on file type
        3. Text chunking with configurable parameters
        4. Metadata enrichment with file info, dates, and user-provided data
        5. Vector embedding generation and storage in Qdrant

        Args:
            filepath: Path to the file to process
            collection_type: Target collection type ('system' or 'user')
            user_id: Required for user collections, ignored for system collections
            subject: Subject or category for the document
            short_description: Brief description of the document content
            chunk_size: Maximum characters per chunk (default: 1000)
            overlap: Characters to overlap between chunks (default: 200)
            chunking_method: Method for chunking ('sliding_window' or 'paragraph')

        Returns:
            Dictionary with processing results including:
            - success: Boolean indicating if processing succeeded
            - file_info: FileInfo object with file metadata
            - chunks_processed: Number of chunks created and stored
            - chunks_failed: Number of chunks that failed to store
            - total_content_length: Total characters processed
            - errors: List of any errors encountered

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or other validation errors
        """
        results = {
            "success": False,
            "file_info": None,
            "chunks_processed": 0,
            "chunks_failed": 0,
            "total_content_length": 0,
            "errors": [],
        }

        try:
            # Validate file
            is_valid, error_msg = self.file_ops.validate_file(filepath)
            if not is_valid:
                results["errors"].append(error_msg)
                return results

            # Get comprehensive file information
            file_info = self.file_ops.get_file_info(filepath)
            results["file_info"] = file_info

            self.logger.info(
                f"Processing file: {file_info.filename} ({file_info.file_type}, {file_info.size_mb}MB)"
            )

            # Extract content from file
            try:
                content = self.file_ops.read_file(filepath)
                results["total_content_length"] = len(content)

                if not content or not content.strip():
                    results["errors"].append(
                        "File content is empty or contains only whitespace"
                    )
                    return results

            except Exception as e:
                error_msg = f"Failed to extract content from file: {e}"
                results["errors"].append(error_msg)
                self.logger.error(error_msg)
                return results

            # Create base metadata
            base_metadata = {
                # User-provided metadata
                "subject": subject or "General",
                "short_description": short_description or file_info.filename,
                # File metadata
                "source_file": file_info.filename,
                "file_path": file_info.filepath,
                "file_type": file_info.file_type,
                "file_extension": file_info.file_extension,
                "file_size_mb": file_info.size_mb,
                "file_hash": file_info.file_hash,
                "mime_type": file_info.mime_type,
                # Processing metadata
                "processed_date": datetime.now().isoformat(),
                "file_created_date": file_info.created_date.isoformat(),
                "file_modified_date": file_info.modified_date.isoformat(),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunking_method": chunking_method,
                "collection_type": collection_type,
                # Content metadata
                "total_content_length": len(content),
                "word_count": len(content.split()) if content else 0,
                "line_count": content.count("\n") + 1 if content else 0,
            }

            # Add user ID to metadata if provided
            if user_id:
                base_metadata["user_id"] = user_id

            # Add auto-generated summary
            base_metadata.update(
                self.file_ops.extract_metadata_from_content(content, filepath)
            )

            # Create chunks based on selected method
            if chunking_method == "paragraph":
                chunks = self.file_ops.chunk_by_paragraphs(
                    content,
                    max_chunk_size=chunk_size,
                    source_file=file_info.filepath,
                    metadata=base_metadata,
                )
            else:  # sliding_window (default)
                chunks = self.file_ops.chunk_text(
                    content,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    source_file=file_info.filepath,
                    metadata=base_metadata,
                )

            if not chunks:
                results["errors"].append("No chunks were created from the content")
                return results

            self.logger.info(
                f"Created {len(chunks)} chunks using {chunking_method} method"
            )

            # Store chunks in vector database
            chunks_processed = 0
            chunks_failed = 0

            for i, chunk in enumerate(chunks):
                try:
                    # Enhance chunk metadata with chunk-specific information
                    chunk_metadata = chunk.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "chunk_size": chunk.chunk_size,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "chunk_text_preview": (
                                chunk.text[:100] + "..."
                                if len(chunk.text) > 100
                                else chunk.text
                            ),
                        }
                    )

                    # Store chunk in RAG system
                    success = self.store(
                        text=chunk.text,
                        metadata=chunk_metadata,
                        collection_type=collection_type,
                        user_id=user_id,
                    )

                    if success:
                        chunks_processed += 1
                    else:
                        chunks_failed += 1
                        self.logger.warning(
                            f"Failed to store chunk {i+1}/{len(chunks)}"
                        )

                except Exception as e:
                    chunks_failed += 1
                    error_msg = f"Error processing chunk {i+1}/{len(chunks)}: {e}"
                    results["errors"].append(error_msg)
                    self.logger.error(error_msg)

            # Update results
            results["chunks_processed"] = chunks_processed
            results["chunks_failed"] = chunks_failed

            # Determine overall success
            if chunks_processed > 0:
                results["success"] = True
                success_rate = (chunks_processed / len(chunks)) * 100
                self.logger.info(
                    f"File processing completed: {chunks_processed}/{len(chunks)} chunks stored "
                    f"({success_rate:.1f}% success rate)"
                )
            else:
                results["errors"].append("No chunks were successfully stored")
                self.logger.error("File processing failed: No chunks were stored")

            return results

        except Exception as e:
            error_msg = f"Unexpected error during file processing: {e}"
            results["errors"].append(error_msg)
            self.logger.error(error_msg)
            return results

    def process_file_to_system(
        self,
        filepath: str,
        subject: Optional[str] = None,
        short_description: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunking_method: str = "sliding_window",
    ) -> Dict[str, Any]:
        """
        Convenience method to process a file into the system collection.

        Args:
            filepath: Path to the file to process
            subject: Subject or category for the document
            short_description: Brief description of the document content
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            chunking_method: Method for chunking ('sliding_window' or 'paragraph')

        Returns:
            Processing results dictionary
        """
        return self.process_file(
            filepath=filepath,
            collection_type=CollectionType.SYSTEM,
            subject=subject,
            short_description=short_description,
            chunk_size=chunk_size,
            overlap=overlap,
            chunking_method=chunking_method,
        )

    def process_file_to_user(
        self,
        filepath: str,
        user_id: str,
        subject: Optional[str] = None,
        short_description: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunking_method: str = "sliding_window",
    ) -> Dict[str, Any]:
        """
        Convenience method to process a file into a user-specific collection.

        Args:
            filepath: Path to the file to process
            user_id: User identifier for the target collection
            subject: Subject or category for the document
            short_description: Brief description of the document content
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            chunking_method: Method for chunking ('sliding_window' or 'paragraph')

        Returns:
            Processing results dictionary
        """
        if not user_id:
            raise ValueError("user_id is required for user collections")

        return self.process_file(
            filepath=filepath,
            collection_type=CollectionType.USER,
            user_id=user_id,
            subject=subject,
            short_description=short_description,
            chunk_size=chunk_size,
            overlap=overlap,
            chunking_method=chunking_method,
        )

    def batch_process_files(
        self,
        filepaths: List[str],
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
        default_subject: Optional[str] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunking_method: str = "sliding_window",
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Process multiple files in batch.

        Args:
            filepaths: List of file paths to process
            collection_type: Target collection type
            user_id: Required for user collections
            default_subject: Default subject for all files
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            chunking_method: Method for chunking
            continue_on_error: Whether to continue processing if a file fails

        Returns:
            Batch processing results with individual file results
        """
        batch_results = {
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks_processed": 0,
            "total_chunks_failed": 0,
            "file_results": {},
            "errors": [],
        }

        self.logger.info(f"Starting batch processing of {len(filepaths)} files")

        for filepath in filepaths:
            try:
                result = self.process_file(
                    filepath=filepath,
                    collection_type=collection_type,
                    user_id=user_id,
                    subject=default_subject,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunking_method=chunking_method,
                )

                batch_results["file_results"][filepath] = result

                if result["success"]:
                    batch_results["files_processed"] += 1
                    batch_results["total_chunks_processed"] += result[
                        "chunks_processed"
                    ]
                    batch_results["total_chunks_failed"] += result["chunks_failed"]
                else:
                    batch_results["files_failed"] += 1
                    if not continue_on_error:
                        self.logger.error(
                            f"Stopping batch processing due to failure with {filepath}"
                        )
                        break

            except Exception as e:
                error_msg = f"Failed to process file {filepath}: {e}"
                batch_results["errors"].append(error_msg)
                batch_results["files_failed"] += 1
                self.logger.error(error_msg)

                if not continue_on_error:
                    break

        success_rate = (
            (batch_results["files_processed"] / len(filepaths)) * 100
            if filepaths
            else 0
        )
        self.logger.info(
            f"Batch processing completed: {batch_results['files_processed']}/{len(filepaths)} files "
            f"({success_rate:.1f}% success rate), {batch_results['total_chunks_processed']} total chunks"
        )

        return batch_results

    def get_file_processing_stats(
        self,
        collection_type: str = CollectionType.SYSTEM,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about processed files in a collection.

        Args:
            collection_type: Collection type to analyze
            user_id: Required for user collections

        Returns:
            Dictionary with file processing statistics
        """
        try:
            collection_info = self.get_collection_info(collection_type, user_id)
            if not collection_info:
                return {"error": "Collection not found or empty"}

            # This is a simplified implementation
            # In a real system, you might want to query the database for more detailed stats
            stats = {
                "collection_type": collection_type,
                "total_points": (
                    collection_info.points_count
                    if hasattr(collection_info, "points_count")
                    else 0
                ),
                "collection_status": "active" if collection_info else "empty",
            }

            if user_id:
                stats["user_id"] = user_id

            return stats

        except Exception as e:
            return {"error": f"Failed to get file processing stats: {e}"}
