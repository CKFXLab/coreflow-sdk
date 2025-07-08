import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from .audit import AppLogger

logger = AppLogger(__name__)


def get_environment() -> str:
    """
    Get the current environment (similar to NODE_ENV).
    
    Returns:
        Environment string: 'production', 'development', 'test', or 'local'
    """
    env = os.getenv('NODE_ENV', os.getenv('ENV', os.getenv('ENVIRONMENT', 'development'))).lower()
    
    # Normalize common variations
    if env in ['prod', 'production']:
        return 'production'
    elif env in ['dev', 'development']:
        return 'development'
    elif env in ['test', 'testing']:
        return 'test'
    elif env in ['local']:
        return 'local'
    else:
        return 'development'  # Default fallback


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by looking for common project markers.
    
    Args:
        start_path: Starting directory to search from (defaults to current working directory)
        
    Returns:
        Path to project root
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = Path(start_path).resolve()
    
    # Common project root indicators
    markers = [
        'requirements.txt',
        'pyproject.toml',
        'setup.py',
        'Pipfile',
        'poetry.lock',
        '.git',
        'package.json',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    # Search up the directory tree
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # If no markers found, return current directory
    return current


def find_dotenv_files(project_root: Optional[Path] = None) -> List[Path]:
    """
    Find dotenv files in the proper loading order.
    
    Loading order (highest to lowest precedence):
    1. .env.local (always loaded, should be in .gitignore)
    2. .env.[environment] (e.g., .env.production, .env.development)
    3. .env.[environment].local (e.g., .env.production.local)
    4. System environment variables (handled separately)
    
    Note: .env and .env.example are NOT loaded automatically (following Node.js conventions)
    
    Args:
        project_root: Project root directory (auto-detected if None)
        
    Returns:
        List of dotenv file paths in loading order (lowest to highest precedence)
    """
    if project_root is None:
        project_root = find_project_root()
    
    env = get_environment()
    dotenv_files = []
    
    # Order matters: files loaded later override earlier ones
    potential_files = [
        f'.env.{env}',           # Environment-specific
        f'.env.{env}.local',     # Environment-specific local overrides
        '.env.local'             # Local overrides (highest precedence)
    ]
    
    for filename in potential_files:
        file_path = project_root / filename
        if file_path.exists():
            dotenv_files.append(file_path)
    
    return dotenv_files


def load_dotenv_files(project_root: Optional[Path] = None, verbose: bool = False) -> Dict[str, str]:
    """
    Load environment variables from dotenv files with proper precedence.
    
    Args:
        project_root: Project root directory (auto-detected if None)
        verbose: Whether to log detailed loading information
        
    Returns:
        Dictionary of loaded environment variables
    """
    if project_root is None:
        project_root = find_project_root()
    
    env = get_environment()
    loaded_vars = {}
    
    # Find and load dotenv files
    dotenv_files = find_dotenv_files(project_root)
    
    if verbose or logger.level == "DEBUG":
        logger.info(f"Environment: {env}")
        logger.info(f"Project root: {project_root}")
        if dotenv_files:
            logger.info(f"Loading dotenv files: {[str(f) for f in dotenv_files]}")
        else:
            logger.info("No dotenv files found")
    
    # Load each file (later files override earlier ones)
    for dotenv_file in dotenv_files:
        try:
            # Load without overriding existing environment variables
            # We'll handle precedence manually
            temp_env = {}
            
            # Read the file manually to track what we're loading
            with open(dotenv_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        temp_env[key] = value
            
            # Load using dotenv for proper parsing
            load_dotenv(dotenv_file, override=False)
            
            # Track what we loaded
            for key, value in temp_env.items():
                loaded_vars[key] = value
                if verbose:
                    logger.debug(f"Loaded {key} from {dotenv_file.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {dotenv_file}: {e}")
    
    return loaded_vars


def validate_required_env(required_vars: List[str], project_root: Optional[Path] = None) -> None:
    """
    Validate that required environment variables are set.
    First loads dotenv files, then checks system environment.
    
    Args:
        required_vars: List of required environment variable names
        project_root: Project root directory (auto-detected if None)
        
    Raises:
        ValueError: If any required variables are missing
    """
    # Load dotenv files first
    load_dotenv_files(project_root)
    
    # Check for missing variables
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        env = get_environment()
        root = project_root or find_project_root()
        dotenv_files = find_dotenv_files(root)
        
        error_msg = f"Missing required environment variables: {missing}\n"
        error_msg += f"Current environment: {env}\n"
        error_msg += f"Searched for dotenv files in: {root}\n"
        
        if dotenv_files:
            error_msg += f"Found dotenv files: {[f.name for f in dotenv_files]}\n"
        else:
            error_msg += "No dotenv files found.\n"
        
        error_msg += f"Expected files: .env.{env}, .env.{env}.local, .env.local\n"
        error_msg += "Note: .env and .env.example are not loaded automatically"
        
        raise ValueError(error_msg)


class ENV:
    """
    Enhanced environment variable manager with dotenv support and validation.
    
    Supports Node.js-style environment loading:
    1. .env.[environment] (e.g., .env.production, .env.development)
    2. .env.[environment].local (local overrides for specific environment)
    3. .env.local (local overrides, highest precedence)
    4. System environment variables (always highest precedence)
    
    Note: .env and .env.example are NOT loaded automatically
    """

    def __init__(self, project_root: Optional[Path] = None, required_keys: Optional[List[str]] = None):
        """
        Initialize ENV with dotenv support and validation.
        
        Args:
            project_root: Project root directory (auto-detected if None)
            required_keys: List of required environment variable names (empty by default for graceful degradation)
        """
        # Set up logging
        self.logger = AppLogger(__name__)
        
        # Detect project root and environment
        self.project_root = project_root or find_project_root()
        self.environment = get_environment()
        
        # Define required environment variable names - now empty by default for graceful degradation
        self.required_keys = required_keys or []
        
        # Initialize attribute values to None
        self.anthropic_api_key = None
        self.hf_token = None
        self.mem0_api_key = None
        self.openai_api_key = None
        self.serper_api_key = None
        
        # Vector store and embedding configuration
        self.vector_store_provider = None
        self.embedding_provider = None
        self.embedding_model = None
        self.embedding_dimensions = None
        self.qdrant_host = None
        self.qdrant_port = None
        self.qdrant_collection_name = None
        self.qdrant_use_docker = None
        self.qdrant_url = None
        self.qdrant_api_key = None
        self.qdrant_deployment_mode = None
        self.aws_region = None
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_session_token = None
        
        # Load dotenv files first
        self.loaded_dotenv_vars = self._load_environment()
        
        # Load all environment variables (both required and optional)
        self.missing = []
        self._load_all_variables()
        
        # Only validate required keys if any were specified
        if self.required_keys:
            isvalid = self.validate()
            if not isvalid:
                self._log_environment_info()
                self.logger.error(f"Missing required environment variables: {self.missing}")
                sys.exit(1)
        else:
            self.logger.info("ENV initialized with graceful degradation mode - no required keys")

    def _load_environment(self) -> Dict[str, str]:
        """Load environment variables from dotenv files."""
        return load_dotenv_files(self.project_root, verbose=False)

    def _load_all_variables(self):
        """Load all environment variables, both required and optional."""
        # Load all potential keys
        all_keys = [
            "ANTHROPIC_API_KEY",
            "HF_TOKEN", 
            "MEM0_API_KEY",
            "OPENAI_API_KEY",
            "SERPER_API_KEY",
            "VECTOR_STORE_PROVIDER",
            "EMBEDDING_PROVIDER", 
            "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSIONS",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "QDRANT_COLLECTION_NAME",
            "QDRANT_USE_DOCKER",
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "QDRANT_DEPLOYMENT_MODE",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN"
        ]
        
        for key in all_keys:
            value = os.getenv(key)
            if value:
                self.set(key, value)

    def _log_environment_info(self) -> None:
        """Log detailed environment information for debugging."""
        self.logger.info(f"Environment: {self.environment}")
        self.logger.info(f"Project root: {self.project_root}")
        
        dotenv_files = find_dotenv_files(self.project_root)
        if dotenv_files:
            self.logger.info(f"Loaded dotenv files: {[f.name for f in dotenv_files]}")
        else:
            self.logger.info("No dotenv files found")
        
        if self.loaded_dotenv_vars:
            self.logger.debug(f"Loaded variables from dotenv: {list(self.loaded_dotenv_vars.keys())}")

    def set(self, key: str, value: str) -> None:
        """Set environment variable value to the appropriate attribute."""
        if key == "ANTHROPIC_API_KEY":
            self.anthropic_api_key = value
        elif key == "HF_TOKEN":
            self.hf_token = value
        elif key == "MEM0_API_KEY":
            self.mem0_api_key = value
        elif key == "OPENAI_API_KEY":    
            self.openai_api_key = value
        elif key == "SERPER_API_KEY":
            self.serper_api_key = value
        # Vector store and embedding configuration
        elif key == "VECTOR_STORE_PROVIDER":
            self.vector_store_provider = value
        elif key == "EMBEDDING_PROVIDER":
            self.embedding_provider = value
        elif key == "EMBEDDING_MODEL":
            self.embedding_model = value
        elif key == "EMBEDDING_DIMENSIONS":
            self.embedding_dimensions = int(value) if value else None
        elif key == "QDRANT_HOST":
            self.qdrant_host = value
        elif key == "QDRANT_PORT":
            self.qdrant_port = int(value) if value else None
        elif key == "QDRANT_COLLECTION_NAME":
            self.qdrant_collection_name = value
        elif key == "QDRANT_USE_DOCKER":
            self.qdrant_use_docker = value.lower() in ('true', '1', 'yes', 'on')
        elif key == "QDRANT_URL":
            self.qdrant_url = value
        elif key == "QDRANT_API_KEY":
            self.qdrant_api_key = value
        elif key == "QDRANT_DEPLOYMENT_MODE":
            self.qdrant_deployment_mode = value
        elif key == "AWS_REGION":
            self.aws_region = value
        elif key == "AWS_ACCESS_KEY_ID":
            self.aws_access_key_id = value
        elif key == "AWS_SECRET_ACCESS_KEY":
            self.aws_secret_access_key = value
        elif key == "AWS_SESSION_TOKEN":
            self.aws_session_token = value

    def validate(self) -> bool:
        """Validate that all required environment variables are present."""
        self.missing = []
        
        # Check required API keys
        for key in self.required_keys:
            value = os.getenv(key)
            if not value:
                self.missing.append(key)
            else:
                self.set(key, value)
        
        # Load optional configuration variables with defaults
        optional_vars = [
            "VECTOR_STORE_PROVIDER",
            "EMBEDDING_PROVIDER", 
            "EMBEDDING_MODEL",
            "EMBEDDING_DIMENSIONS",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "QDRANT_COLLECTION_NAME",
            "QDRANT_USE_DOCKER",
            "QDRANT_URL",
            "QDRANT_API_KEY",
            "QDRANT_DEPLOYMENT_MODE",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN"
        ]
        
        for key in optional_vars:
            value = os.getenv(key)
            if value:
                self.set(key, value)
        
        return len(self.missing) == 0
    
    def get_openai_key(self) -> str:
        """Get OpenAI API key."""
        return self.openai_api_key
    
    def get_serper_key(self) -> str:
        """Get Serper API key.""" 
        return self.serper_api_key
    
    def get_anthropic_key(self) -> str:
        """Get Anthropic API key."""
        return self.anthropic_api_key
    
    def get_hf_token(self) -> str:
        """Get Hugging Face token."""
        return self.hf_token
    
    def get_mem0_key(self) -> str:
        """Get Mem0 API key."""
        return self.mem0_api_key
    
    def get_environment(self) -> str:
        """Get current environment."""
        return self.environment
    
    def get_project_root(self) -> Path:
        """Get project root directory."""
        return self.project_root
    
    def get_loaded_dotenv_files(self) -> List[Path]:
        """Get list of loaded dotenv files."""
        return find_dotenv_files(self.project_root)
    
    # === VECTOR STORE CONFIGURATION ===
    
    def get_vector_store_provider(self) -> str:
        """Get vector store provider (defaults to 'qdrant')."""
        return self.vector_store_provider or "qdrant"
    
    def get_embedding_provider(self) -> str:
        """Get embedding provider (defaults to 'openai')."""
        return self.embedding_provider or "openai"
    
    def get_embedding_model(self) -> str:
        """Get embedding model (defaults based on provider)."""
        if self.embedding_model:
            return self.embedding_model
        
        provider = self.get_embedding_provider()
        if provider == "openai":
            return "text-embedding-3-small"
        elif provider == "bedrock":
            return "amazon.titan-embed-text-v1"
        else:
            return "text-embedding-3-small"  # fallback
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions (defaults based on model)."""
        if self.embedding_dimensions:
            return self.embedding_dimensions
        
        model = self.get_embedding_model()
        if "text-embedding-3-small" in model:
            return 1536
        elif "text-embedding-3-large" in model:
            return 3072
        elif "titan-embed" in model:
            return 1536
        else:
            return 1536  # fallback
    
    def get_qdrant_host(self) -> str:
        """Get Qdrant host (defaults to 'localhost')."""
        return self.qdrant_host or "localhost"
    
    def get_qdrant_port(self) -> int:
        """Get Qdrant port (defaults to 6333)."""
        return self.qdrant_port or 6333
    
    def get_qdrant_collection_name(self) -> str:
        """Get Qdrant collection name (defaults to 'system__mem0')."""
        return self.qdrant_collection_name or "system__mem0"
    
    def get_qdrant_use_docker(self) -> bool:
        """Get whether to use Docker Qdrant (defaults to True)."""
        return self.qdrant_use_docker if self.qdrant_use_docker is not None else True
    
    def get_qdrant_url(self) -> Optional[str]:
        """Get Qdrant URL (defaults to None)."""
        return self.qdrant_url
    
    def get_qdrant_api_key(self) -> Optional[str]:
        """Get Qdrant API key (defaults to None)."""
        return self.qdrant_api_key
    
    def get_qdrant_deployment_mode(self) -> Optional[str]:
        """Get Qdrant deployment mode (defaults to None)."""
        return self.qdrant_deployment_mode
    
    def get_aws_region(self) -> str:
        """Get AWS region (defaults to 'us-east-1')."""
        return self.aws_region or "us-east-1"
    
    def get_aws_access_key_id(self) -> Optional[str]:
        """Get AWS access key ID."""
        return self.aws_access_key_id
    
    def get_aws_secret_access_key(self) -> Optional[str]:
        """Get AWS secret access key."""
        return self.aws_secret_access_key
    
    def get_aws_session_token(self) -> Optional[str]:
        """Get AWS session token."""
        return self.aws_session_token
    
    def reload(self) -> bool:
        """
        Reload environment variables from dotenv files and system environment.
        
        Returns:
            True if all required variables are present after reload
        """
        self.loaded_dotenv_vars = self._load_environment()
        return self.validate()
    
    def get_env_info(self) -> Dict[str, Any]:
        """
        Get comprehensive environment information for debugging.
        
        Returns:
            Dictionary with environment details
        """
        return {
            'environment': self.environment,
            'project_root': str(self.project_root),
            'dotenv_files': [str(f) for f in self.get_loaded_dotenv_files()],
            'loaded_dotenv_vars': list(self.loaded_dotenv_vars.keys()),
            'required_keys': self.required_keys,
            'missing_keys': self.missing,
            'all_keys_present': len(self.missing) == 0,
            'vector_config': {
                'vector_store_provider': self.get_vector_store_provider(),
                'embedding_provider': self.get_embedding_provider(),
                'embedding_model': self.get_embedding_model(),
                'embedding_dimensions': self.get_embedding_dimensions(),
                'qdrant_host': self.get_qdrant_host(),
                'qdrant_port': self.get_qdrant_port(),
                'qdrant_collection_name': self.get_qdrant_collection_name(),
                'qdrant_use_docker': self.get_qdrant_use_docker(),
                'qdrant_url': self.get_qdrant_url(),
                'qdrant_api_key': self.get_qdrant_api_key(),
                'qdrant_deployment_mode': self.get_qdrant_deployment_mode(),
                'aws_region': self.get_aws_region(),
            }
        }

    def get_available_credentials(self) -> Dict[str, bool]:
        """
        Check which credentials are actually available.
        
        Returns:
            Dictionary mapping credential types to availability
        """
        return {
            'openai_api_key': bool(self.get_openai_key()),
            'anthropic_api_key': bool(self.get_anthropic_key()),
            'aws_access_key_id': bool(self.get_aws_access_key_id()),
            'aws_secret_access_key': bool(self.get_aws_secret_access_key()),
            'aws_profile_available': self._check_aws_profile_available(),
            'aws_instance_profile': self._check_aws_instance_profile(),
            'serper_api_key': bool(self.get_serper_key()),
            'mem0_api_key': bool(self.get_mem0_key()),
            'hf_token': bool(self.get_hf_token())
        }
    
    def _check_aws_profile_available(self) -> bool:
        """Check if AWS profile is configured."""
        try:
            import boto3
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None and credentials.access_key is not None
        except Exception:
            return False
    
    def _check_aws_instance_profile(self) -> bool:
        """Check if running on AWS instance with IAM role."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/iam/security-credentials/',
                timeout=1
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_disabled_features(self) -> List[str]:
        """
        Get list of features that are disabled due to missing credentials.
        
        Returns:
            List of disabled feature names
        """
        credentials = self.get_available_credentials()
        disabled = []
        
        if not credentials['openai_api_key']:
            disabled.append('OpenAI models')
        if not credentials['anthropic_api_key']:
            disabled.append('Anthropic models')
        if not (credentials['aws_access_key_id'] or credentials['aws_profile_available'] or credentials['aws_instance_profile']):
            disabled.append('AWS Bedrock models')
        if not credentials['serper_api_key']:
            disabled.append('Web search')
        if not credentials['mem0_api_key']:
            disabled.append('Mem0 cloud memory (falls back to local)')
        if not credentials['hf_token']:
            disabled.append('HuggingFace models and LlamaServer')
        
        return disabled
    
    def get_best_available_model_config(self) -> Dict[str, Any]:
        """
        Get the best available model configuration based on available credentials.
        
        Returns:
            Dictionary with model configuration for the best available provider
        """
        credentials = self.get_available_credentials()
        
        # Priority order: OpenAI > Anthropic > Bedrock
        if credentials['openai_api_key']:
            return {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": self.get_openai_key()
            }
        elif credentials['anthropic_api_key']:
            return {
                "provider": "anthropic", 
                "model": "claude-3-5-sonnet-20241022",
                "api_key": self.get_anthropic_key()
            }
        elif (credentials['aws_access_key_id'] or credentials['aws_profile_available'] or credentials['aws_instance_profile']):
            config = {
                "provider": "bedrock",
                "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "region_name": self.get_aws_region()
            }
            if credentials['aws_access_key_id']:
                config.update({
                    "aws_access_key_id": self.get_aws_access_key_id(),
                    "aws_secret_access_key": self.get_aws_secret_access_key()
                })
                if self.get_aws_session_token():
                    config["aws_session_token"] = self.get_aws_session_token()
            return config
        else:
            # No model providers available
            self.logger.warning("No model provider credentials available")
            return None
    
    def get_provider_availability(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed availability information for each provider.
        
        Returns:
            Dictionary with provider availability details
        """
        credentials = self.get_available_credentials()
        
        return {
            "openai": {
                "available": credentials['openai_api_key'],
                "required": ["OPENAI_API_KEY"],
                "missing": [] if credentials['openai_api_key'] else ["OPENAI_API_KEY"]
            },
            "anthropic": {
                "available": credentials['anthropic_api_key'],
                "required": ["ANTHROPIC_API_KEY"],
                "missing": [] if credentials['anthropic_api_key'] else ["ANTHROPIC_API_KEY"]
            },
            "bedrock": {
                "available": (credentials['aws_access_key_id'] or credentials['aws_profile_available'] or credentials['aws_instance_profile']),
                "required": ["AWS credentials (multiple methods supported)"],
                "missing": self._get_aws_missing_credentials(credentials),
                "methods": {
                    "environment_vars": credentials['aws_access_key_id'],
                    "aws_profile": credentials['aws_profile_available'],
                    "instance_profile": credentials['aws_instance_profile']
                }
            },
            "websearch": {
                "available": credentials['serper_api_key'],
                "required": ["SERPER_API_KEY"],
                "missing": [] if credentials['serper_api_key'] else ["SERPER_API_KEY"]
            },
            "memory": {
                "available": True,  # Always available with local fallback
                "cloud_available": credentials['mem0_api_key'],
                "required": ["MEM0_API_KEY (optional, falls back to local)"],
                "missing": [] if credentials['mem0_api_key'] else ["MEM0_API_KEY"]
            },
            "huggingface": {
                "available": credentials['hf_token'],
                "required": ["HF_TOKEN"],
                "missing": [] if credentials['hf_token'] else ["HF_TOKEN"]
            }
        }
    
    def _get_aws_missing_credentials(self, credentials: Dict[str, bool]) -> List[str]:
        """Get missing AWS credentials."""
        if (credentials['aws_access_key_id'] or credentials['aws_profile_available'] or credentials['aws_instance_profile']):
            return []
        else:
            return ["AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY", "AWS_PROFILE", "IAM_ROLE"]