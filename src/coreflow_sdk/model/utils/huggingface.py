"""
HuggingFace SDK Integration

This module provides utilities for downloading, managing, and working with
HuggingFace models, tokenizers, and datasets. It includes functionality for
model caching, version management, and integration with the CoreFlow SDK.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

try:
    from huggingface_hub import (
        HfApi, hf_hub_download, snapshot_download, 
        model_info, list_models, login, logout,
        ModelCard, DatasetCard, SpaceCard
    )
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        pipeline, Pipeline
    )
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ...utils.audit import AppLogger


@dataclass
class ModelInfo:
    """Information about a HuggingFace model."""
    model_id: str
    model_type: str
    task: Optional[str] = None
    library: Optional[str] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    tags: Optional[List[str]] = None
    pipeline_tag: Optional[str] = None
    size_mb: Optional[float] = None
    local_path: Optional[str] = None


class HuggingFace:
    """
    HuggingFace SDK for downloading, managing, and working with models.
    
    This class provides a unified interface for:
    - Downloading models and tokenizers
    - Managing local model cache
    - Creating inference pipelines
    - Model information and search
    - Authentication management
    """
    
    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize HuggingFace SDK.
        
        Args:
            cache_dir: Custom cache directory for models (defaults to HF_HOME)
            token: HuggingFace token for authentication
        """
        self.logger = AppLogger(__name__)
        
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace dependencies not available. Install with: "
                "pip install transformers torch huggingface_hub"
            )
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API client
        self.api = HfApi(token=token)
        self.token = token
        
        # Model registry for tracking downloaded models
        self.registry_file = self.cache_dir / "coreflow_model_registry.json"
        self.model_registry = self._load_registry()
        
        self.logger.info(f"HuggingFace SDK initialized with cache dir: {self.cache_dir}")
        
        if token:
            self.login(token)
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load model registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save the model registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    def login(self, token: str):
        """
        Authenticate with HuggingFace Hub.
        
        Args:
            token: HuggingFace authentication token
        """
        try:
            login(token=token)
            self.token = token
            self.logger.info("Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            self.logger.error(f"Failed to authenticate with HuggingFace Hub: {e}")
            raise
    
    def logout(self):
        """Logout from HuggingFace Hub."""
        try:
            logout()
            self.token = None
            self.logger.info("Logged out from HuggingFace Hub")
        except Exception as e:
            self.logger.error(f"Failed to logout: {e}")
    
    def search_models(self, 
                     query: Optional[str] = None,
                     task: Optional[str] = None,
                     library: Optional[str] = None,
                     language: Optional[str] = None,
                     sort: str = "downloads",
                     limit: int = 10) -> List[ModelInfo]:
        """
        Search for models on HuggingFace Hub.
        
        Args:
            query: Search query string
            task: Filter by task (e.g., "text-generation", "text-classification")
            library: Filter by library (e.g., "transformers", "diffusers")
            language: Filter by language (e.g., "en", "zh")
            sort: Sort by "downloads", "trending", "created_at", etc.
            limit: Maximum number of results
            
        Returns:
            List of ModelInfo objects
        """
        try:
            models = list_models(
                search=query,
                task=task,
                library=library,
                language=language,
                sort=sort,
                limit=limit
            )
            
            model_infos = []
            for model in models:
                model_info = ModelInfo(
                    model_id=model.modelId,
                    model_type=getattr(model, 'config', {}).get('model_type', 'unknown'),
                    task=getattr(model, 'pipeline_tag', task),
                    library=getattr(model, 'library_name', library),
                    downloads=getattr(model, 'downloads', 0),
                    likes=getattr(model, 'likes', 0),
                    tags=getattr(model, 'tags', []),
                    pipeline_tag=getattr(model, 'pipeline_tag', None)
                )
                model_infos.append(model_info)
            
            self.logger.info(f"Found {len(model_infos)} models matching search criteria")
            return model_infos
            
        except Exception as e:
            self.logger.error(f"Failed to search models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: Model identifier (e.g., "microsoft/DialoGPT-medium")
            
        Returns:
            ModelInfo object or None if not found
        """
        try:
            info = model_info(model_id)
            
            # Calculate approximate size if possible
            size_mb = None
            if hasattr(info, 'siblings') and info.siblings:
                total_size = sum(getattr(file, 'size', 0) for file in info.siblings if hasattr(file, 'size'))
                if total_size > 0:
                    size_mb = total_size / (1024 * 1024)  # Convert to MB
            
            # Check if model exists locally
            local_path = None
            if model_id in self.model_registry:
                local_path = self.model_registry[model_id].get('local_path')
                if local_path and not Path(local_path).exists():
                    local_path = None
            
            model_info_obj = ModelInfo(
                model_id=model_id,
                model_type=getattr(info, 'config', {}).get('model_type', 'unknown'),
                task=getattr(info, 'pipeline_tag', None),
                library=getattr(info, 'library_name', None),
                downloads=getattr(info, 'downloads', 0),
                likes=getattr(info, 'likes', 0),
                tags=getattr(info, 'tags', []),
                pipeline_tag=getattr(info, 'pipeline_tag', None),
                size_mb=size_mb,
                local_path=local_path
            )
            
            return model_info_obj
            
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_id}: {e}")
            return None
    
    def download_model(self, 
                      model_id: str,
                      revision: str = "main",
                      cache_subdir: Optional[str] = None,
                      force_download: bool = False) -> str:
        """
        Download a model and its associated files.
        
        Args:
            model_id: Model identifier
            revision: Model revision/branch to download
            cache_subdir: Subdirectory within cache for this model
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded model directory
        """
        try:
            self.logger.info(f"Downloading model: {model_id}")
            
            # Determine cache directory for this model
            if cache_subdir:
                model_cache_dir = self.cache_dir / cache_subdir
            else:
                # Create safe directory name from model_id
                safe_name = model_id.replace("/", "--").replace("\\", "--")
                model_cache_dir = self.cache_dir / "models" / safe_name
            
            # Download model using snapshot_download
            local_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=str(model_cache_dir),
                force_download=force_download,
                token=self.token
            )
            
            # Update registry
            self.model_registry[model_id] = {
                "local_path": local_dir,
                "revision": revision,
                "download_date": str(Path().cwd()),  # Current timestamp would be better
                "cache_subdir": cache_subdir
            }
            self._save_registry()
            
            self.logger.info(f"Model {model_id} downloaded to: {local_dir}")
            return local_dir
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_id}: {e}")
            raise
    
    def download_file(self, 
                     model_id: str, 
                     filename: str,
                     revision: str = "main",
                     subfolder: Optional[str] = None) -> str:
        """
        Download a specific file from a model repository.
        
        Args:
            model_id: Model identifier
            filename: Name of file to download
            revision: Model revision/branch
            subfolder: Subfolder within the repository
            
        Returns:
            Path to downloaded file
        """
        try:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                subfolder=subfolder,
                cache_dir=str(self.cache_dir),
                token=self.token
            )
            
            self.logger.info(f"Downloaded file {filename} from {model_id} to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file {filename} from {model_id}: {e}")
            raise
    
    def load_model(self, model_id: str, **kwargs) -> Any:
        """
        Load a model using transformers AutoModel.
        
        Args:
            model_id: Model identifier or local path
            **kwargs: Additional arguments for AutoModel.from_pretrained
            
        Returns:
            Loaded model object
        """
        try:
            # Check if model is in local registry
            local_path = None
            if model_id in self.model_registry:
                local_path = self.model_registry[model_id].get('local_path')
                if local_path and Path(local_path).exists():
                    model_id = local_path
            
            self.logger.info(f"Loading model: {model_id}")
            
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.token,
                **kwargs
            )
            
            self.logger.info(f"Successfully loaded model: {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def load_tokenizer(self, model_id: str, **kwargs) -> Any:
        """
        Load a tokenizer using transformers AutoTokenizer.
        
        Args:
            model_id: Model identifier or local path
            **kwargs: Additional arguments for AutoTokenizer.from_pretrained
            
        Returns:
            Loaded tokenizer object
        """
        try:
            # Check if model is in local registry
            local_path = None
            if model_id in self.model_registry:
                local_path = self.model_registry[model_id].get('local_path')
                if local_path and Path(local_path).exists():
                    model_id = local_path
            
            self.logger.info(f"Loading tokenizer: {model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                token=self.token,
                **kwargs
            )
            
            self.logger.info(f"Successfully loaded tokenizer: {model_id}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer {model_id}: {e}")
            raise
    
    def create_pipeline(self, 
                       task: str, 
                       model_id: Optional[str] = None,
                       **kwargs) -> Any:
        """
        Create a transformers pipeline for a specific task.
        
        Args:
            task: Pipeline task (e.g., "text-generation", "sentiment-analysis")
            model_id: Model identifier (uses default for task if None)
            **kwargs: Additional pipeline arguments
            
        Returns:
            Configured pipeline object
        """
        try:
            # Check if model is in local registry
            if model_id and model_id in self.model_registry:
                local_path = self.model_registry[model_id].get('local_path')
                if local_path and Path(local_path).exists():
                    model_id = local_path
            
            self.logger.info(f"Creating {task} pipeline with model: {model_id or 'default'}")
            
            pipe = pipeline(
                task=task,
                model=model_id,
                token=self.token,
                **kwargs
            )
            
            self.logger.info(f"Successfully created {task} pipeline")
            return pipe
            
        except Exception as e:
            self.logger.error(f"Failed to create {task} pipeline: {e}")
            raise
    
    def list_local_models(self) -> List[Dict[str, Any]]:
        """
        List all locally downloaded models.
        
        Returns:
            List of model information dictionaries
        """
        local_models = []
        for model_id, info in self.model_registry.items():
            local_path = info.get('local_path')
            if local_path and Path(local_path).exists():
                # Get directory size
                size_bytes = sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                
                local_models.append({
                    'model_id': model_id,
                    'local_path': local_path,
                    'revision': info.get('revision', 'unknown'),
                    'size_mb': round(size_mb, 2),
                    'download_date': info.get('download_date', 'unknown')
                })
        
        return local_models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a locally cached model.
        
        Args:
            model_id: Model identifier to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if model_id not in self.model_registry:
            self.logger.warning(f"Model {model_id} not found in registry")
            return False
        
        try:
            local_path = self.model_registry[model_id].get('local_path')
            if local_path and Path(local_path).exists():
                shutil.rmtree(local_path)
                self.logger.info(f"Deleted model directory: {local_path}")
            
            # Remove from registry
            del self.model_registry[model_id]
            self._save_registry()
            
            self.logger.info(f"Successfully deleted model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def clear_cache(self, older_than_days: int = 30) -> Dict[str, Any]:
        """
        Clear old cached models and files.
        
        Args:
            older_than_days: Delete files older than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        from datetime import datetime, timedelta
        
        try:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            deleted_count = 0
            freed_bytes = 0
            
            # Clean up registry entries for non-existent models
            models_to_remove = []
            for model_id, info in self.model_registry.items():
                local_path = info.get('local_path')
                if not local_path or not Path(local_path).exists():
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                del self.model_registry[model_id]
                deleted_count += 1
            
            # Could add more sophisticated cache cleanup here
            # (checking file modification times, etc.)
            
            self._save_registry()
            
            stats = {
                'deleted_entries': deleted_count,
                'freed_mb': round(freed_bytes / (1024 * 1024), 2),
                'registry_cleaned': True
            }
            
            self.logger.info(f"Cache cleanup complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return {'error': str(e)}
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            # Count models in registry
            total_models = len(self.model_registry)
            
            # Count actually existing models
            existing_models = 0
            total_size_bytes = 0
            
            for model_id, info in self.model_registry.items():
                local_path = info.get('local_path')
                if local_path and Path(local_path).exists():
                    existing_models += 1
                    # Calculate directory size
                    size_bytes = sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file())
                    total_size_bytes += size_bytes
            
            cache_info = {
                'cache_dir': str(self.cache_dir),
                'total_models_in_registry': total_models,
                'existing_models': existing_models,
                'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
                'registry_file': str(self.registry_file),
                'authenticated': self.token is not None
            }
            
            return cache_info
            
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}