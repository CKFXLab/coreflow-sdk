# Main model abstractions
from ._mabc import Model, TrainingJob, ReinforceJob

# Model factory and defaults
from ._factory import (
    ModelFactory,
    create_model,
    get_available_providers,
    create_default_model,
)
from ._default import Provider, ModelType

# Provider-specific abstractions
from .api._mabc import APIModel, APITrainingJob, APIReinforceJob

# Optional Bedrock abstractions (requires boto3)
try:
    from .bedrock._mabc import BedrockModel, BedrockTrainingJob, BedrockReinforceJob

    BEDROCK_AVAILABLE = True
except ImportError:
    BedrockModel = None
    BedrockTrainingJob = None
    BedrockReinforceJob = None
    BEDROCK_AVAILABLE = False

from .llamaserver._mabc import (
    LlamaServerModel,
    LlamaServerTrainingJob,
    LlamaServerReinforceJob,
)

# Concrete implementations
from .api.openai import OpenAIClient, OpenAITrainingJob, OpenAIReinforceJob

# Optional Anthropic API implementations
try:
    from .api.anthropic import (
        AnthropicClient,
        AnthropicTrainingJob,
        AnthropicReinforceJob,
    )

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AnthropicClient = None
    AnthropicTrainingJob = None
    AnthropicReinforceJob = None
    ANTHROPIC_AVAILABLE = False

# Optional Bedrock implementations
try:
    from .bedrock.anthropic import BedrockAnthropicClient

    BEDROCK_ANTHROPIC_AVAILABLE = True
except ImportError:
    BedrockAnthropicClient = None
    BEDROCK_ANTHROPIC_AVAILABLE = False

# Optional HuggingFace utilities
try:
    from .utils import HuggingFace, ModelInfo

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HuggingFace = None
    ModelInfo = None
    HUGGINGFACE_AVAILABLE = False

# Configuration helper functions
from .utils import (
    openai_config,
    anthropic_config,
    bedrock_config,
    gpt4_config,
    gpt4o_mini_config,
    claude35_sonnet_config,
    claude35_haiku_config,
    claude3_opus_config,
    claude3_sonnet_config,
    claude3_haiku_config,
    claude2_config,
    bedrock_claude4_sonnet_config,
    bedrock_claude37_sonnet_config,
    bedrock_claude35_sonnet_v2_config,
    bedrock_claude35_sonnet_config,
    bedrock_claude3_sonnet_config,
    bedrock_claude3_haiku_config,
)

__all__ = [
    # Main abstractions
    "Model",
    "TrainingJob",
    "ReinforceJob",
    # Model factory and utilities
    "ModelFactory",
    "create_model",
    "get_available_providers",
    "create_default_model",
    "Provider",
    "ModelType",
    # API abstractions
    "APIModel",
    "APITrainingJob",
    "APIReinforceJob",
    # LlamaServer abstractions
    "LlamaServerModel",
    "LlamaServerTrainingJob",
    "LlamaServerReinforceJob",
    # Concrete implementations
    "OpenAIClient",
    "OpenAITrainingJob",
    "OpenAIReinforceJob",
    # Configuration helpers
    "openai_config",
    "anthropic_config",
    "bedrock_config",
    "gpt4_config",
    "gpt4o_mini_config",
    "claude35_sonnet_config",
    "claude35_haiku_config",
    "claude3_opus_config",
    "claude3_sonnet_config",
    "claude3_haiku_config",
    "claude2_config",
    "bedrock_claude4_sonnet_config",
    "bedrock_claude37_sonnet_config",
    "bedrock_claude35_sonnet_v2_config",
    "bedrock_claude35_sonnet_config",
    "bedrock_claude3_sonnet_config",
    "bedrock_claude3_haiku_config",
]

# Add Bedrock exports if available
if BEDROCK_AVAILABLE:
    __all__.extend(
        [
            "BedrockModel",
            "BedrockTrainingJob",
            "BedrockReinforceJob",
        ]
    )

if BEDROCK_ANTHROPIC_AVAILABLE:
    __all__.append("BedrockAnthropicClient")

# Add Anthropic API exports if available
if ANTHROPIC_AVAILABLE:
    __all__.extend(
        [
            "AnthropicClient",
            "AnthropicTrainingJob",
            "AnthropicReinforceJob",
        ]
    )

# Add HuggingFace exports if available
if HUGGINGFACE_AVAILABLE:
    __all__.extend(
        [
            "HuggingFace",
            "ModelInfo",
        ]
    )
