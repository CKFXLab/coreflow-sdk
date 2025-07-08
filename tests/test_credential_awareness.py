"""
Test suite for credential-aware functionality in CoreFlow SDK.

This module tests the complete credential-aware system including:
- ENV credential detection and validation
- Model registry filtering based on credentials
- Workflow auto-configuration and feature disabling
- FastAPI endpoint integration
- Graceful degradation when credentials are missing
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

from sdk.utils.env import ENV
from sdk.model.registry import get_model_registry
from sdk.workflow._wabc import BaseWorkflow
from sdk.workflow._default import (
    get_credential_aware_workflow_config,
    get_best_available_model_config,
    get_available_features,
    create_workflow_with_best_model
)


@pytest.mark.credentials
@pytest.mark.env
class TestCredentialDetection:
    """Test credential detection and validation."""
    
    def test_env_graceful_initialization(self):
        """Test that ENV initializes gracefully without required credentials."""
        env = ENV()
        assert env is not None
        assert hasattr(env, 'get_available_credentials')
        assert hasattr(env, 'get_disabled_features')
        assert hasattr(env, 'get_best_available_model_config')
    
    def test_credential_detection_all_available(self):
        """Test credential detection when all credentials are available."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'AWS_ACCESS_KEY_ID': 'test-aws-key',
            'AWS_SECRET_ACCESS_KEY': 'test-aws-secret',
            'SERPER_API_KEY': 'test-serper-key',
            'MEM0_API_KEY': 'test-mem0-key',
            'HF_TOKEN': 'test-hf-token'
        }):
            env = ENV()
            creds = env.get_available_credentials()
            
            assert creds['openai_api_key'] is True
            assert creds['anthropic_api_key'] is True
            assert creds['aws_access_key_id'] is True
            assert creds['aws_secret_access_key'] is True
            assert creds['serper_api_key'] is True
            assert creds['mem0_api_key'] is True
            assert creds['hf_token'] is True
    
    def test_credential_detection_missing_credentials(self):
        """Test credential detection when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            env = ENV()
            creds = env.get_available_credentials()
            
            assert creds['openai_api_key'] is False
            assert creds['anthropic_api_key'] is False
            assert creds['aws_access_key_id'] is False
            assert creds['aws_secret_access_key'] is False
            assert creds['serper_api_key'] is False
            assert creds['mem0_api_key'] is False
            assert creds['hf_token'] is False
    
    def test_disabled_features_detection(self):
        """Test detection of disabled features based on missing credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            env = ENV()
            disabled = env.get_disabled_features()
            
            assert 'Anthropic models' in disabled
            assert 'AWS Bedrock models' in disabled
            assert 'Web search' in disabled
            assert 'HuggingFace models and LlamaServer' in disabled
            assert 'OpenAI models' not in disabled
    
    def test_best_available_model_selection(self):
        """Test automatic selection of best available model."""
        # Test OpenAI priority
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key'
        }, clear=True):
            env = ENV()
            config = env.get_best_available_model_config()
            assert config['provider'] == 'openai'
            assert config['model'] == 'gpt-4o-mini'
        
        # Test Anthropic fallback
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-anthropic-key'
        }, clear=True):
            env = ENV()
            config = env.get_best_available_model_config()
            assert config['provider'] == 'anthropic'
            assert config['model'] == 'claude-3-5-sonnet-20241022'
        
        # Test Bedrock fallback
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test-aws-key',
            'AWS_SECRET_ACCESS_KEY': 'test-aws-secret'
        }, clear=True):
            env = ENV()
            config = env.get_best_available_model_config()
            assert config['provider'] == 'bedrock'
            assert 'claude' in config['model']
    
    def test_no_credentials_available(self):
        """Test behavior when no model credentials are available."""
        with patch.dict(os.environ, {}, clear=True):
            env = ENV()
            config = env.get_best_available_model_config()
            assert config is None
    
    def test_provider_availability_details(self):
        """Test detailed provider availability information."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            env = ENV()
            availability = env.get_provider_availability()
            
            assert availability['openai']['available'] is True
            assert availability['openai']['missing'] == []
            
            assert availability['anthropic']['available'] is False
            assert 'ANTHROPIC_API_KEY' in availability['anthropic']['missing']
            
            assert availability['bedrock']['available'] is False
            assert len(availability['bedrock']['missing']) > 0


@pytest.mark.credentials
@pytest.mark.registry
class TestModelRegistryFiltering:
    """Test model registry credential-aware filtering."""
    
    def test_model_registry_with_all_credentials(self):
        """Test model registry when all credentials are available."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'AWS_ACCESS_KEY_ID': 'test-aws-key',
            'AWS_SECRET_ACCESS_KEY': 'test-aws-secret'
        }):
            registry = get_model_registry()
            available = registry.get_available_models_by_credentials()
            
            assert available['providers']['openai']['available'] is True
            assert available['providers']['anthropic']['available'] is True
            assert available['providers']['bedrock']['available'] is True
            assert available['total_models'] > 0
    
    def test_model_registry_with_limited_credentials(self):
        """Test model registry with only OpenAI credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            registry = get_model_registry()
            available = registry.get_available_models_by_credentials()
            
            assert available['providers']['openai']['available'] is True
            assert available['providers']['anthropic']['available'] is False
            assert available['providers']['bedrock']['available'] is False
            assert len(available['providers']['openai']['models']) > 0
            assert len(available['providers']['anthropic']['models']) == 0
            assert len(available['providers']['bedrock']['models']) == 0
    
    def test_model_registry_no_credentials(self):
        """Test model registry with no credentials."""
        with patch.dict(os.environ, {}, clear=True):
            registry = get_model_registry()
            available = registry.get_available_models_by_credentials()
            
            assert available['providers']['openai']['available'] is False
            assert available['providers']['anthropic']['available'] is False
            assert available['providers']['bedrock']['available'] is False
            assert available['total_models'] == 0
    
    def test_best_available_model_from_registry(self):
        """Test getting best available model from registry."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            registry = get_model_registry()
            best_model = registry.get_best_available_model()
            assert best_model == 'gpt-4o-mini'
    
    def test_available_providers_list(self):
        """Test getting list of available providers."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key'
        }, clear=True):
            registry = get_model_registry()
            providers = registry.get_available_providers_list()
            assert 'openai' in providers
            assert 'anthropic' in providers
            assert 'bedrock' not in providers


@pytest.mark.credentials
@pytest.mark.workflow
class TestWorkflowCredentialAwareness:
    """Test workflow credential-aware functionality."""
    
    def test_workflow_auto_configuration(self):
        """Test workflow auto-configuration based on credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'SERPER_API_KEY': 'test-serper-key'
        }, clear=True):
            # Mock the actual clients to avoid real API calls
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Memory'), \
                 patch('sdk.vector.rag.QdrantClient'), \
                 patch('sdk.websearch.search.Search'), \
                 patch('sdk.websearch.scrape.Scrape'):
                
                workflow = BaseWorkflow()
                assert workflow.model_config['provider'] == 'openai'
                assert workflow.enable_websearch is True
                assert 'AWS Bedrock models' in workflow.disabled_features
    
    def test_workflow_with_limited_credentials(self):
        """Test workflow with limited credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Mock the actual clients
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Memory'), \
                 patch('sdk.vector.rag.QdrantClient'):
                
                workflow = BaseWorkflow()
                assert workflow.model_config['provider'] == 'openai'
                assert workflow.enable_websearch is False
                assert 'Web search' in workflow.disabled_features
    
    def test_workflow_no_model_credentials(self):
        """Test workflow behavior with no model credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No model provider credentials available"):
                BaseWorkflow()
    
    def test_workflow_component_status(self):
        """Test workflow component status reporting."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Mock the actual clients
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Memory'), \
                 patch('sdk.vector.rag.QdrantClient'):
                
                workflow = BaseWorkflow()
                status = workflow.get_component_status()
                
                assert 'credentials' in status
                assert 'disabled_features' in status
                assert status['credentials']['openai_api_key'] is True
                assert status['credentials']['serper_api_key'] is False
    
    def test_workflow_validation(self):
        """Test workflow validation with credential warnings."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Mock the actual clients
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Memory'), \
                 patch('sdk.vector.rag.QdrantClient'):
                
                workflow = BaseWorkflow()
                validation = workflow.validate_workflow()
                
                assert validation['valid'] is True
                assert 'credential_status' in validation
                assert len(validation['warnings']) > 0
                assert any('Feature disabled' in warning for warning in validation['warnings'])


@pytest.mark.credentials
@pytest.mark.workflow
class TestWorkflowDefaultFunctions:
    """Test workflow default configuration functions."""
    
    def test_credential_aware_workflow_config(self):
        """Test credential-aware workflow configuration."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'SERPER_API_KEY': 'test-serper-key'
        }, clear=True):
            config = get_credential_aware_workflow_config()
            assert config['enable_websearch'] is True
    
    def test_credential_aware_workflow_config_no_serper(self):
        """Test workflow config without SERPER_API_KEY."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            config = get_credential_aware_workflow_config()
            assert config['enable_websearch'] is False
    
    def test_get_available_features(self):
        """Test getting available features based on credentials."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key',
            'HF_TOKEN': 'test-hf-token'
        }, clear=True):
            features = get_available_features()
            assert features['model_openai'] is True
            assert features['model_anthropic'] is False
            assert features['model_bedrock'] is False
            assert features['websearch'] is False
            assert features['huggingface'] is True
            assert features['memory_local'] is True
    
    def test_create_workflow_with_best_model(self):
        """Test creating workflow configuration with best model."""
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test-anthropic-key'
        }, clear=True):
            config = create_workflow_with_best_model()
            assert config['model_config']['provider'] == 'anthropic'
            assert config['model_config']['model'] == 'claude-3-5-sonnet-20241022'
            assert config['enable_websearch'] is False


@pytest.mark.credentials
@pytest.mark.fastapi
class TestFastAPIIntegration:
    """Test FastAPI credential-aware endpoints."""
    
    def test_models_endpoint_filtering(self):
        """Test that models endpoint returns only available models."""
        from server.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            response = client.get('/models')
            assert response.status_code == 200
            
            data = response.json()
            assert data['providers']['openai']['available'] is True
            assert data['providers']['anthropic']['available'] is False
            assert data['providers']['bedrock']['available'] is False
            assert len(data['providers']['openai']['models']) > 0
            assert len(data['providers']['anthropic']['models']) == 0
    
    def test_credentials_endpoint(self):
        """Test credentials status endpoint."""
        from server.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            response = client.get('/credentials')
            assert response.status_code == 200
            
            data = response.json()
            assert data['credentials']['openai_api_key'] is True
            assert data['credentials']['anthropic_api_key'] is False
            assert 'Anthropic models' in data['disabled_features']
    
    def test_workflow_status_endpoint(self):
        """Test workflow status endpoint."""
        from server.api.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Mock the actual clients
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Memory'), \
                 patch('sdk.vector.rag.QdrantClient'):
                
                response = client.get('/workflow/status')
                assert response.status_code == 200
                
                data = response.json()
                assert 'credentials' in data
                assert 'disabled_features' in data
                assert data['credentials']['openai_api_key'] is True


@pytest.mark.credentials
@pytest.mark.edge_cases
@pytest.mark.degradation
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_aws_profile_detection(self):
        """Test AWS profile detection functionality."""
        env = ENV()
        # This will test the actual AWS profile detection
        # The result depends on the system's AWS configuration
        profile_available = env._check_aws_profile_available()
        assert isinstance(profile_available, bool)
    
    def test_aws_instance_profile_detection(self):
        """Test AWS instance profile detection."""
        env = ENV()
        # This will test the actual AWS instance profile detection
        # Should be False in local development environment
        instance_profile = env._check_aws_instance_profile()
        assert isinstance(instance_profile, bool)
    
    def test_env_error_handling(self):
        """Test ENV error handling for malformed configurations."""
        env = ENV()
        
        # Test with None values
        with patch.dict(os.environ, {}, clear=True):
            creds = env.get_available_credentials()
            assert all(isinstance(v, bool) for v in creds.values())
    
    def test_model_registry_error_handling(self):
        """Test model registry error handling."""
        registry = get_model_registry()
        
        # Test with None env
        available = registry.get_available_models_by_credentials(env=None)
        assert 'providers' in available
        assert 'models' in available
        assert 'total_models' in available
    
    def test_workflow_graceful_degradation(self):
        """Test workflow graceful degradation with component failures."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-openai-key'
        }, clear=True):
            # Mock successful model client but failed search client
            with patch('sdk.model.api.openai.OpenAIClient'), \
                 patch('sdk.vector.memory.Mem0'), \
                 patch('sdk.vector.rag.QdrantClient'), \
                 patch('sdk.websearch.search.Search', side_effect=Exception("Search failed")):
                
                workflow = BaseWorkflow()
                status = workflow.get_component_status()
                
                assert status['model_client'] is True
                assert status['memory_client'] is True
                assert status['search_client'] is False  # This should fail due to no SERPER_API_KEY
                
                # Workflow should still be valid with just model and memory client
                validation = workflow.validate_workflow()
                assert validation['valid'] is True
                assert len(validation['warnings']) > 0  # Should have warnings about missing components


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 