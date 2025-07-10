import os
import pytest
from unittest.mock import patch, Mock
from coreflow_sdk.utils.env import ENV
from coreflow_sdk.vector._default import (
    create_vector_config,
    create_qdrant_config,
    get_embedding_client_config,
    create_embedding_client,
)


class TestEmbeddingConfiguration:
    """Test the dynamic embedding configuration system."""

    def test_default_configuration(self):
        """Test default configuration values."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
            },
            clear=True,
        ):
            env = ENV()

            # Test default values
            assert env.get_embedding_provider() == "openai"
            assert env.get_embedding_model() == "text-embedding-3-small"
            assert env.get_embedding_dimensions() == 1536
            assert env.get_qdrant_host() == "localhost"
            assert env.get_qdrant_port() == 6333
            assert env.get_qdrant_use_docker() == True

    def test_openai_configuration(self):
        """Test OpenAI embedding configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-large",
                "EMBEDDING_DIMENSIONS": "3072",
            },
            clear=True,
        ):
            env = ENV()
            config = create_vector_config(env)

            assert config["embedder"]["provider"] == "openai"
            assert config["embedder"]["config"]["model"] == "text-embedding-3-large"
            assert config["vector_store"]["config"]["embedding_model_dims"] == 3072

    def test_bedrock_configuration(self):
        """Test Bedrock embedding configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "EMBEDDING_PROVIDER": "bedrock",
                "EMBEDDING_MODEL": "amazon.titan-embed-text-v1",
                "AWS_REGION": "us-west-2",
                "AWS_ACCESS_KEY_ID": "test_key_id",
                "AWS_SECRET_ACCESS_KEY": "test_secret",
            },
            clear=True,
        ):
            env = ENV()
            config = create_vector_config(env)

            assert config["embedder"]["provider"] == "bedrock"
            assert config["embedder"]["config"]["model"] == "amazon.titan-embed-text-v1"
            assert config["embedder"]["config"]["region_name"] == "us-west-2"
            assert config["embedder"]["config"]["aws_access_key_id"] == "test_key_id"
            assert (
                config["embedder"]["config"]["aws_secret_access_key"] == "test_secret"
            )

    def test_qdrant_local_configuration(self):
        """Test local Qdrant configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "QDRANT_HOST": "localhost",
                "QDRANT_PORT": "6333",
                "QDRANT_USE_DOCKER": "true",
            },
            clear=True,
        ):
            env = ENV()
            config = create_qdrant_config(env)

            assert config["host"] == "localhost"
            assert config["port"] == 6333
            assert config["use_docker"] == True
            assert config["deployment_mode"] == "local"

    def test_qdrant_cloud_configuration(self):
        """Test cloud Qdrant configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "QDRANT_URL": "https://test-cluster.qdrant.tech",
                "QDRANT_API_KEY": "test_api_key",
                "QDRANT_DEPLOYMENT_MODE": "cloud",
            },
            clear=True,
        ):
            env = ENV()
            config = create_qdrant_config(env)

            assert config["url"] == "https://test-cluster.qdrant.tech"
            assert config["api_key"] == "test_api_key"
            assert config["deployment_mode"] == "cloud"

    def test_qdrant_fargate_configuration(self):
        """Test Fargate Qdrant configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "QDRANT_URL": "http://qdrant-service.namespace.svc.cluster.local:6333",
                "QDRANT_DEPLOYMENT_MODE": "fargate",
            },
            clear=True,
        ):
            env = ENV()
            config = create_qdrant_config(env)

            assert (
                config["url"]
                == "http://qdrant-service.namespace.svc.cluster.local:6333"
            )
            assert config["deployment_mode"] == "fargate"
            assert "api_key" not in config  # No API key needed for service discovery

    def test_embedding_client_config(self):
        """Test embedding client configuration."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "EMBEDDING_PROVIDER": "bedrock",
                "EMBEDDING_MODEL": "amazon.titan-embed-text-v1",
                "EMBEDDING_DIMENSIONS": "1536",
                "AWS_REGION": "us-east-1",
            },
            clear=True,
        ):
            env = ENV()
            config = get_embedding_client_config(env)

            assert config["provider"] == "bedrock"
            assert config["model"] == "amazon.titan-embed-text-v1"
            assert config["dimensions"] == 1536
            assert config["region_name"] == "us-east-1"

    @patch("coreflow_sdk.model.api.openai.OpenAI")
    def test_create_openai_embedding_client(self, mock_openai):
        """Test creating OpenAI embedding client."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "EMBEDDING_PROVIDER": "openai",
            },
            clear=True,
        ):
            env = ENV()
            client = create_embedding_client(env)

            assert client is not None
            # Should create OpenAI client
            assert hasattr(client, "generate_embedding")

    def test_create_bedrock_embedding_client(self):
        """Test creating Bedrock embedding client."""
        # Skip test if boto3 is not available
        try:
            import boto3
        except ImportError:
            pytest.skip(
                "boto3 not available (install with 'pip install coreflow-sdk[aws]')"
            )

        with patch("boto3.Session") as mock_session:
            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "test_key",
                    "OPENAI_API_KEY": "test_key",
                    "HF_TOKEN": "test_token",
                    "MEM0_API_KEY": "test_key",
                    "SERPER_API_KEY": "test_key",
                    "EMBEDDING_PROVIDER": "bedrock",
                    "AWS_REGION": "us-east-1",
                },
                clear=True,
            ):
                # Mock boto3 session and clients
                mock_session_instance = Mock()
                mock_session.return_value = mock_session_instance
                mock_session_instance.client.return_value = Mock()

                env = ENV()
                client = create_embedding_client(env)

                assert client is not None
                # Should create Bedrock client
                assert hasattr(client, "generate_embedding")

    def test_environment_info(self):
        """Test environment info retrieval."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
                "EMBEDDING_PROVIDER": "openai",
                "QDRANT_HOST": "localhost",
            },
            clear=True,
        ):
            env = ENV()
            info = env.get_env_info()

            assert "vector_config" in info
            assert info["vector_config"]["embedding_provider"] == "openai"
            assert info["vector_config"]["qdrant_host"] == "localhost"
            assert info["all_keys_present"] == True

    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_key",
                "HF_TOKEN": "test_token",
                "MEM0_API_KEY": "test_key",
                "SERPER_API_KEY": "test_key",
            },
            clear=True,
        ):
            # Should work without any embedding-specific env vars
            env = ENV()
            config = create_vector_config(env)

            # Should use defaults
            assert config["embedder"]["provider"] == "openai"
            assert config["embedder"]["config"]["model"] == "text-embedding-3-small"
            assert config["vector_store"]["config"]["host"] == "localhost"
            assert config["vector_store"]["config"]["port"] == 6333
