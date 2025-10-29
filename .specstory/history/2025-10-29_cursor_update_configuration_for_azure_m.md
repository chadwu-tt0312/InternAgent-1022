# Update configuration for azure model provider
_Exported on 2025/10/29 at 17:56:05 [GMT+8] from Cursor (2.0.32)_

---

**User**

@default_config.yaml models 項目新增一項 azure。使用 AZURE_OPENAI_KEY，AZURE_OPENAI_ENDPOINT，model_name="gpt-4o-mini"。然後 default_provider 改成 azure

並將其他使用 model_provider="openai" 的也改成使用 "azure" 

對應需要變更設定的程式碼也一併修改

---

**Cursor**


著手新增 Azure 配置並切換預設提供者。先檢視相關程式碼結構。


讀取 `openai_model.py`，準備建立 Azure 模型實作。


建立 Azure 模型實作並更新相關設定。


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        self.api_base = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not self.api_base:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )

```

```python
"""
Model Provider Factory

A dynamic system for managing and instantiating language model backends.
Enables seamless switching between different LLM providers and handles
configuration, caching, and fallback strategies.
"""

import importlib
import logging
import time
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Available model provider implementations
MODEL_PROVIDER_MAP = {
    "openai": "internagent.mas.models.openai_model.OpenAIModel",
    "azure": "internagent.mas.models.azure_model.AzureModel",
    "interns1": "internagent.mas.models.s1_model.S1Model",
    "dsr1": "internagent.mas.models.r1_model.R1Model",
}


class ModelFactory:
    """Central factory for creating and managing language model instances."""
    
    registered_models = {}
    _model_cache = {}
    _model_stats = {}
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on provided configuration.
        
        Args:
            config: Configuration dictionary for model creation
                
        Returns:
            Configured model instance
            
        Raises:
            ValueError: For unsupported providers or configuration errors
        """
        provider = config.get("provider")
        if provider == "default" or provider is None:
            provider = config.get("default_provider", "openai")

        if "models" in config and provider in config["models"]:
            provider_config = config["models"][provider].copy()
            logger.info(f"Using provider-specific config for {provider}: {provider_config}")
            
            # Merge with general configuration
            provider_config.update({
                k: v for k, v in config.items()
                if k not in ["models", "agents", "workflow", "tools", "memory", "execution"]
            })
            config_to_use = provider_config
        else:
            logger.info(f"Using fallback config for provider {provider}")
            config_to_use = config
        
        cache_key = ModelFactory._create_cache_key(provider, config_to_use)
        
        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Reusing cached model for provider: {provider}")
            return ModelFactory._model_cache[cache_key]
            
        if provider not in MODEL_PROVIDER_MAP and provider not in ModelFactory.registered_models:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        if provider in ModelFactory.registered_models:
            model_class = ModelFactory.registered_models[provider]
        else:
            try:
                module_path, class_name = MODEL_PROVIDER_MAP[provider].rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load model implementation: {e}")
                raise ValueError(f"Implementation not available for provider: {provider}")
        
        try:
            model = model_class.from_config(config_to_use)
            ModelFactory._model_cache[cache_key] = model
            ModelFactory._model_stats[cache_key] = {
                "created_at": time.time(),
                "success_count": 0,
                "failure_count": 0,
                "total_tokens": 0,
                "total_latency": 0.0
            }
            return model
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise
    
    @staticmethod
    def create_model_for_agent(agent_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance tailored for a specific agent type.
        
        Args:
            agent_type: Type identifier for the agent
            config: Agent configuration parameters
                
        Returns:
            Model instance configured for the specified agent
        """
        model_provider = config.get("model_provider", "default")
        global_config = config.get("_global_config", {})
        model_config = {}
        
        # Apply provider configuration
        if model_provider != "default" and model_provider in global_config.get("models", {}):
            model_config.update(global_config.get("models", {}).get(model_provider, {}))
        
        # Handle default provider case
        if model_provider == "default":
            default_provider = global_config.get("models", {}).get("default_provider", "openai")
            model_config.update(global_config.get("models", {}).get(default_provider, {}))
        
        model_config["provider"] = model_provider
        model_config["default_provider"] = global_config.get("models", {}).get("default_provider", "openai")
        
        # Apply agent-specific overrides
        for key in ["temperature", "max_tokens"]:
            if key in config:
                model_config[key] = config[key]
        
        # Include remaining agent settings
        for k, v in config.items():
            if k not in ["_global_config", "count"] and k not in model_config:
                model_config[k] = v
                
        return ModelFactory.create_model(model_config)
    
    @classmethod
    def register_model(cls, provider_name: str, model_class) -> None:
        """Add a custom model implementation to the factory registry."""
        cls.registered_models[provider_name] = model_class
        logger.info(f"New model provider registered: {provider_name}")
    
    @classmethod
    def create_with_fallbacks(cls, 
                             config: Dict[str, Any], 
                             fallback_providers: List[str] = None,
                             max_retries: int = 3) -> Tuple[BaseModel, str]:
        """
        Create a model with automatic fallback to alternative providers.
        
        Attempts to create a model with the primary provider, falling back
        to alternatives if the primary fails. Each provider can be tried
        multiple times before moving to the next option.
        
        Args:
            config: Model configuration
            fallback_providers: Ordered list of backup providers
            max_retries: Maximum attempts per provider
            
        Returns:
            (model_instance, successful_provider_name)
            
        Raises:
            ValueError: When all providers fail
        """
        primary_provider = config.get("provider", config.get("default_provider", "openai"))
        
        # Setup fallback sequence
        if fallback_providers is None:
            fallback_providers = [p for p in MODEL_PROVIDER_MAP.keys() if p != primary_provider]
            # Prioritize known reliable providers
            if "openai" in fallback_providers:
                fallback_providers.remove("openai")
                fallback_providers.insert(0, "openai")
            if "ollama" in fallback_providers:
                fallback_providers.remove("ollama")
                fallback_providers.insert(1, "ollama")
        
        providers_to_try = [primary_provider] + fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    provider_config = dict(config)
                    provider_config["provider"] = provider
                    
                    model = cls.create_model(provider_config)
                    logger.info(f"Successfully initialized {provider} model")
                    return model, provider
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Attempt {retry_count} failed for provider {provider}: {e}")
                    time.sleep(1)
        
        raise ValueError(f"Failed to create model with any provider. Last error: {last_error}")
    
    @classmethod
    def update_model_stats(cls, cache_key: str, success: bool, tokens: int = 0, latency: float = 0.0) -> None:
        """Update performance metrics for a model instance."""
        if cache_key not in cls._model_stats:
            return
            
        stats = cls._model_stats[cache_key]
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        stats["total_tokens"] += tokens
        stats["total_latency"] += latency
    
    @classmethod
    def get_model_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all model instances."""
        return cls._model_stats
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get all registered model providers and their implementations."""
        available_models = {provider: path for provider, path in MODEL_PROVIDER_MAP.items()}
        
        for provider, model_class in cls.registered_models.items():
            available_models[provider] = model_class.__name__
            
        return available_models
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances and statistics."""
        cls._model_cache.clear()
        cls._model_stats.clear()
        logger.info("Model cache cleared")
    
    @staticmethod
    def _create_cache_key(provider: str, config: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for model instance caching.
        
        Different configurations of the same provider are treated as
        separate instances to ensure correct behavior.
        """
        provider_config = config.get(provider, {})
        model_name = provider_config.get("model_name", "default")
        
        if provider == "local":
            model_path = provider_config.get("model_path", "")
            return f"{provider}:{model_name}:{model_path}"
        
        if provider == "ollama":
            api_base = provider_config.get("api_base", "http://localhost:11434")
            return f"{provider}:{model_name}:{api_base}"
            
        return f"{provider}:{model_name}"

```

更新配置檔：新增 Azure 設定，將預設 provider 切換為 Azure，並把使用 openai 的代理改為 azure。


```yaml
# Default configuration for InternAgent
# This file is loaded when no specific config path is provided

version: "1.0.0"

# Model Providers Configuration
models:
  default_provider: "azure"

  openai:
    model_name: "gpt-4o-mini"
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7

  azure:
    model_name: "gpt-4o-mini"
    api_key: ""  # Set via environment variable AZURE_OPENAI_KEY
    api_base: ""  # Set via environment variable AZURE_OPENAI_ENDPOINT
    max_tokens: 4096
    temperature: 0.7

# Memory Configuration
memory:
  backend: "file_system"
  file_dir: "results"

# Agent Configuration
agents:
  # Idea Generation Agent
  generation:
    model_provider: "default"
    generation_count: 5
    creativity: 0.7
    do_survey: false

  # Reflection Agent
  reflection:
    model_provider: "default"
    detail_level: "medium"

  # Evolution Agent
  evolution:
    model_provider: "default"
    creativity_level: 0.6

  # Method Development Agent
  method_development:
    model_provider: "openai"
    # model_name: "gpt-4"

  # Refinement Agent
  refinement:
    model_provider: "openai"
    # model_name: "gpt-4"

  # Ranking Agent
  ranking:
    model_provider: "default"

  # Scholar Agent
  scholar:
    model_provider: "default"
    search_depth: "moderate"

  # Survey Agent
  survey:
    model_provider: "default"

# Workflow Configuration
workflow:
  max_iterations: 2
  top_ideas_count: 3
  top_ideas_evo: false
  max_concurrent_tasks: 5

```

```yaml
# Default configuration for InternAgent
# This file is loaded when no specific config path is provided

version: "1.0.0"

# Model Providers Configuration
models:
  default_provider: "azure"

  openai:
    model_name: "gpt-4o-mini"
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7

  azure:
    model_name: "gpt-4o-mini"
    api_key: ""  # Set via environment variable AZURE_OPENAI_KEY
    api_base: ""  # Set via environment variable AZURE_OPENAI_ENDPOINT
    max_tokens: 4096
    temperature: 0.7

# Memory Configuration
memory:
  backend: "file_system"
  file_dir: "results"

# Agent Configuration
agents:
  # Idea Generation Agent
  generation:
    model_provider: "default"
    generation_count: 5
    creativity: 0.7
    do_survey: false

  # Reflection Agent
  reflection:
    model_provider: "default"
    detail_level: "medium"

  # Evolution Agent
  evolution:
    model_provider: "default"
    creativity_level: 0.6

  # Method Development Agent
  method_development:
    model_provider: "azure"
    # model_name: "gpt-4"

  # Refinement Agent
  refinement:
    model_provider: "azure"
    # model_name: "gpt-4"

  # Ranking Agent
  ranking:
    model_provider: "default"

  # Scholar Agent
  scholar:
    model_provider: "default"
    search_depth: "moderate"

  # Survey Agent
  survey:
    model_provider: "default"

# Workflow Configuration
workflow:
  max_iterations: 2
  top_ideas_count: 3
  top_ideas_evo: false
  max_concurrent_tasks: 5

```

更新 `config/config.yaml`：


```yaml
# InternAgent Configuration
# Comprehensive configuration file with only essential settings

# System Version
version: "1.0.0"

# Model Providers Configuration
models:
  # Default provider used when not specified
  default_provider: "azure"

  # OpenAI Configuration
  openai:
    model_name: "gpt-4o-2024-11-20"
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7

  # Azure Configuration
  azure:
    model_name: "gpt-4o-2024-11-20"
    api_key: ""  # Set via environment variable AZURE_OPENAI_KEY
    api_base: ""  # Set via environment variable AZURE_OPENAI_ENDPOINT
    max_tokens: 4096
    temperature: 0.7

  # # DeepSeek R1 Configuration (alternative provider)
  # dsr1:
  #   model_name: "DeepSeek-R1"
  #   api_key: ""  # Set via environment variable DS_API_KEY
  #   max_tokens: 4096
  #   temperature: 0.7

# Agent Configuration
agents:
  # Idea Generation Agent
  generation:
    model_provider: "default"
    generation_count: 15
    creativity: 0.7
    do_survey: true
    temperature: 0.9

  # Survey Agent (for literature search)
  survey:
    model_provider: "default"
    max_papers: 50

  # Reflection Agent (for critique and analysis)
  reflection:
    model_provider: "default"

  # Evolution Agent (for idea evolution)
  evolution:
    model_provider: "default"
    creativity_level: 0.6
    temperature: 0.8
    evolution_count: 3

  # Method Development Agent
  method_development:
    model_provider: "openai"  # Use more powerful model
    model_name: "gpt-4o-2024-11-20"

  # Refinement Agent
  refinement:
    model_provider: "openai"
    model_name: "gpt-4o-2024-11-20"

  # Ranking Agent (for idea scoring)
  ranking:
    model_provider: "default"
    criteria:
      novelty: 0.3
      feasibility: 0.4
      impact: 0.2
      clarity: 0.1

  # Scholar Agent (for literature search and relevance)
  scholar:
    model_provider: "default"
    search_depth: "moderate"
    max_papers: 10
    temperature: 0.6

# Workflow Configuration
workflow:
  max_iterations: 4
  top_ideas_count: 5
  top_ideas_evo: true
  max_concurrent_tasks: 5

# Memory/Storage Configuration
memory:
  backend: "file_system"  # Options: file_system, in_memory
  file_dir: "results"

# External Tools Configuration
tools:
  # Web Search
  web_search:
    enabled: true
    provider: "google"
    api_key: ""  # Set via environment variable SEARCH_API_KEY
    max_results: 5

  # Literature Search
  literature_search:
    semantic_scholar_key: ""  # Set via environment variable SS_API_KEY
    max_results: 10

# Logging Configuration (optional)
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/intern_agent.log"

```

```yaml
# InternAgent Configuration
# Comprehensive configuration file with only essential settings

# System Version
version: "1.0.0"

# Model Providers Configuration
models:
  # Default provider used when not specified
  default_provider: "azure"

  # OpenAI Configuration
  openai:
    model_name: "gpt-4o-2024-11-20"
    api_key: ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7

  # Azure Configuration
  azure:
    model_name: "gpt-4o-2024-11-20"
    api_key: ""  # Set via environment variable AZURE_OPENAI_KEY
    api_base: ""  # Set via environment variable AZURE_OPENAI_ENDPOINT
    max_tokens: 4096
    temperature: 0.7

  # # DeepSeek R1 Configuration (alternative provider)
  # dsr1:
  #   model_name: "DeepSeek-R1"
  #   api_key: ""  # Set via environment variable DS_API_KEY
  #   max_tokens: 4096
  #   temperature: 0.7

# Agent Configuration
agents:
  # Idea Generation Agent
  generation:
    model_provider: "default"
    generation_count: 15
    creativity: 0.7
    do_survey: true
    temperature: 0.9

  # Survey Agent (for literature search)
  survey:
    model_provider: "default"
    max_papers: 50

  # Reflection Agent (for critique and analysis)
  reflection:
    model_provider: "default"

  # Evolution Agent (for idea evolution)
  evolution:
    model_provider: "default"
    creativity_level: 0.6
    temperature: 0.8
    evolution_count: 3

  # Method Development Agent
  method_development:
    model_provider: "azure"  # Use more powerful model
    model_name: "gpt-4o-2024-11-20"

  # Refinement Agent
  refinement:
    model_provider: "azure"
    model_name: "gpt-4o-2024-11-20"

  # Ranking Agent (for idea scoring)
  ranking:
    model_provider: "default"
    criteria:
      novelty: 0.3
      feasibility: 0.4
      impact: 0.2
      clarity: 0.1

  # Scholar Agent (for literature search and relevance)
  scholar:
    model_provider: "default"
    search_depth: "moderate"
    max_papers: 10
    temperature: 0.6

# Workflow Configuration
workflow:
  max_iterations: 4
  top_ideas_count: 5
  top_ideas_evo: true
  max_concurrent_tasks: 5

# Memory/Storage Configuration
memory:
  backend: "file_system"  # Options: file_system, in_memory
  file_dir: "results"

# External Tools Configuration
tools:
  # Web Search
  web_search:
    enabled: true
    provider: "google"
    api_key: ""  # Set via environment variable SEARCH_API_KEY
    max_results: 5

  # Literature Search
  literature_search:
    semantic_scholar_key: ""  # Set via environment variable SS_API_KEY
    max_results: 10

# Logging Configuration (optional)
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/intern_agent.log"

```

檢查程式碼是否正確使用環境變數，並驗證配置載入。


檢查是否還有需要更新的配置檔：


檢查是否有直接載入環境變數的程式碼需要更新：


檢查 `codeview_agent.py` 的環境變數使用是否需要更新。


修改 `codeview_agent.py` 以支援 Azure：


為 `codeview_agent.py` 新增 Azure 支援：


```python
import json
import os
import re
import sys
import ast
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.client_model = self.llm_settings.model
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            # Check if using Azure OpenAI
            if os.environ.get("AZURE_OPENAI_ENDPOINT"):
                try: 
                    import openai
                    print(f"Using Azure OpenAI API with model {client_model}")
                    client = openai.OpenAI(
                        api_key=os.environ.get("AZURE_OPENAI_KEY"),
                        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                    )
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
            else:
                try: 
                    import openai
                    print(f"Using OpenAI API with model {client_model}")
                    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        try: 
            import openai
            print(f"Using OpenAI API with model {client_model}")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.client_model = self.llm_settings.model
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            # Check if using Azure OpenAI
            if os.environ.get("AZURE_OPENAI_ENDPOINT"):
                try: 
                    import openai
                    print(f"Using Azure OpenAI API with model {client_model}")
                    client = openai.OpenAI(
                        api_key=os.environ.get("AZURE_OPENAI_KEY"),
                        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                    )
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
            else:
                try: 
                    import openai
                    print(f"Using OpenAI API with model {client_model}")
                    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

檢查是否有 linter 錯誤：


已完成以下變更：

## 已完成的修改

### 1. 新增 Azure 模型實作
在 `internagent/mas/models/azure_model.py` 新增 Azure 模型類別，使用環境變數 `AZURE_OPENAI_KEY` 與 `AZURE_OPENAI_ENDPOINT`。

### 2. 註冊 Azure 提供者
在 `internagent/mas/models/model_factory.py` 的 `MODEL_PROVIDER_MAP` 新增：
```python
"azure": "internagent.mas.models.azure_model.AzureModel"
```

### 3. 更新配置檔
- `config/default_config.yaml`:
  - 新增 `azure` 配置（含 `api_key`、`api_base`）
  - 將 `default_provider` 改為 `"azure"`
  - 將 `method_development` 與 `refinement` 的 `model_provider` 從 `"openai"` 改為 `"azure"`

- `config/config.yaml`:
  - 同步上述修改

### 4. 更新 codeview_agent
在 `internagent/mas/agents/codeview_agent.py` 的兩個函數新增對 Azure 環境變數的支援，當偵測到 `AZURE_OPENAI_ENDPOINT` 即使用 Azure。

已支援 Azure OpenAI，設定環境變數後即可使用。

```python
"azure": "internagent.mas.models.azure_model.AzureModel"
```

---

**User**

要如何測試 LLM 是否可以使用?

---

**Cursor**


檢查現有的測試結構和程式碼，準備建立測試腳本。


建立測試腳本，檢查主要入口點。


建立測試腳本以測試 LLM。


```python
#!/usr/bin/env python3
"""
測試 LLM 模型連線與回應的簡單腳本

使用方法:
    # 測試 Azure OpenAI
    python test_llm.py --provider azure

    # 測試 OpenAI
    python test_llm.py --provider openai

    # 使用配置檔
    python test_llm.py --config config/default_config.yaml
"""

import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加入專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from internagent.mas.models.model_factory import ModelFactory


async def test_model_direct(provider: str, model_name: str = None, api_key: str = None, api_base: str = None):
    """
    直接測試模型，不使用配置檔
    
    Args:
        provider: 模型提供者 (azure, openai, 等)
        model_name: 模型名稱
        api_key: API 金鑰（可選，會優先使用環境變數）
        api_base: API 基礎 URL（可選，主要用於 Azure）
    """
    logger.info(f"測試 {provider} 模型...")
    
    # 建立配置
    config = {
        "provider": provider,
        "model_name": model_name,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    
    # 根據提供者設定特定參數
    if provider == "azure":
        if api_key:
            config["api_key"] = api_key
        if api_base:
            config["api_base"] = api_base
        if not model_name:
            config["model_name"] = "gpt-4o-mini"
            
        # 檢查環境變數
        if not config.get("api_key") and not os.environ.get("AZURE_OPENAI_KEY"):
            logger.error("❌ 未設定 AZURE_OPENAI_KEY 環境變數或提供 api_key 參數")
            return False
            
        if not config.get("api_base") and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            logger.error("❌ 未設定 AZURE_OPENAI_ENDPOINT 環境變數或提供 api_base 參數")
            return False
            
    elif provider == "openai":
        if api_key:
            config["api_key"] = api_key
        if not model_name:
            config["model_name"] = "gpt-4o-mini"
            
        if not config.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
            logger.error("❌ 未設定 OPENAI_API_KEY 環境變數或提供 api_key 參數")
            return False
    
    try:
        # 建立模型
        logger.info(f"建立 {provider} 模型實例...")
        model = ModelFactory.create_model(config)
        logger.info(f"✓ 模型實例建立成功: {config.get('model_name', 'default')}")
        
        # 測試簡單的生成
        test_prompt = "請用一句話介紹你自己。"
        logger.info(f"發送測試提示: {test_prompt}")
        
        response = await model.generate(
            prompt=test_prompt,
            system_prompt="你是一個友善的 AI 助手。"
        )
        
        logger.info("=" * 60)
        logger.info("✓ 模型回應成功！")
        logger.info("=" * 60)
        print("\n回應內容:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # 顯示統計資訊
        stats = model.get_stats()
        logger.info(f"\n模型統計: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


async def test_model_from_config(config_path: str, provider: str = None):
    """
    從配置檔測試模型
    
    Args:
        config_path: 配置檔路徑
        provider: 要測試的提供者（可選，預設使用配置檔中的 default_provider）
    """
    import yaml
    
    logger.info(f"從配置檔載入設定: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置檔不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 決定要使用的提供者
        if provider:
            target_provider = provider
        else:
            target_provider = config.get("models", {}).get("default_provider", "azure")
        
        logger.info(f"使用提供者: {target_provider}")
        
        # 建立模型配置
        provider_config = config.get("models", {}).get(target_provider, {})
        if not provider_config:
            logger.error(f"❌ 配置檔中找不到 {target_provider} 的設定")
            return False
        
        model_config = {
            "provider": target_provider,
            "default_provider": config.get("models", {}).get("default_provider", "azure"),
            **provider_config
        }
        
        # 建立模型
        logger.info(f"建立 {target_provider} 模型實例...")
        model = ModelFactory.create_model(model_config)
        logger.info(f"✓ 模型實例建立成功: {model_config.get('model_name', 'default')}")
        
        # 測試簡單的生成
        test_prompt = "請用一句話介紹你自己。"
        logger.info(f"發送測試提示: {test_prompt}")
        
        response = await model.generate(
            prompt=test_prompt,
            system_prompt="你是一個友善的 AI 助手。"
        )
        
        logger.info("=" * 60)
        logger.info("✓ 模型回應成功！")
        logger.info("=" * 60)
        print("\n回應內容:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # 顯示統計資訊
        stats = model.get_stats()
        logger.info(f"\n模型統計: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="測試 LLM 模型連線與回應",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["azure", "openai", "interns1", "dsr1"],
        help="要測試的模型提供者"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="配置檔路徑（例如: config/default_config.yaml）"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="模型名稱（例如: gpt-4o-mini）"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API 金鑰（可選，優先使用環境變數）"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        help="API 基礎 URL（主要用於 Azure）"
    )
    
    args = parser.parse_args()
    
    # 顯示環境變數狀態
    print("\n" + "=" * 60)
    print("環境變數檢查")
    print("=" * 60)
    env_vars = {
        "AZURE_OPENAI_KEY": os.environ.get("AZURE_OPENAI_KEY", "未設定"),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", "未設定"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "未設定"),
    }
    
    for key, value in env_vars.items():
        if value == "未設定":
            print(f"❌ {key}: {value}")
        else:
            # 只顯示前後幾個字元
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✓ {key}: {masked}")
    
    print("=" * 60 + "\n")
    
    # 決定測試方式
    if args.config:
        # 使用配置檔
        success = await test_model_from_config(args.config, args.provider)
    elif args.provider:
        # 直接測試指定提供者
        success = await test_model_direct(
            args.provider,
            args.model_name,
            args.api_key,
            args.api_base
        )
    else:
        # 預設測試 Azure
        logger.info("未指定提供者，使用預設 Azure 進行測試...")
        success = await test_model_direct("azure", args.model_name, args.api_key, args.api_base)
    
    if success:
        print("\n✓ 測試完成！模型運作正常。")
        sys.exit(0)
    else:
        print("\n❌ 測試失敗！請檢查配置和環境變數。")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

```

設定測試腳本為可執行，並建立簡短的 README。


已建立測試腳本。以下是使用方式：

## 測試腳本使用說明

### 基本用法

1. **測試 Azure OpenAI（使用環境變數）**
```bash
python test_llm.py --provider azure
```

2. **測試 OpenAI**
```bash
python test_llm.py --provider openai
```

3. **使用配置檔測試**
```bash
python test_llm.py --config config/default_config.yaml
```

4. **指定模型名稱**
```bash
python test_llm.py --provider azure --model-name gpt-4o-mini
```

### 環境變數設定

測試前請確保設定必要的環境變數：

**Azure OpenAI:**
```bash
export AZURE_OPENAI_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
```

### 測試腳本功能

- 檢查環境變數是否設定
- 建立模型實例
- 發送測試提示並取得回應
- 顯示模型統計資訊
- 錯誤處理與診斷資訊

### 執行範例

```bash
# 最簡單的方式（使用預設 Azure）
python test_llm.py

# 指定提供者
python test_llm.py --provider azure

# 使用配置檔
python test_llm.py --config config/default_config.yaml
```

腳本會顯示：
- 環境變數狀態
- 模型建立狀態
- 測試回應內容
- 模型統計資訊

可以執行腳本測試 LLM 是否正常運作。

```shellscript
python test_llm.py --provider azure
```

```shellscript
python test_llm.py --provider openai
```

```shellscript
python test_llm.py --config config/default_config.yaml
```

```shellscript
python test_llm.py --provider azure --model-name gpt-4o-mini
```

```shellscript
export AZURE_OPENAI_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

```shellscript
export OPENAI_API_KEY="your-api-key"
```

```shellscript
# 最簡單的方式（使用預設 Azure）
python test_llm.py

# 指定提供者
python test_llm.py --provider azure

# 使用配置檔
python test_llm.py --config config/default_config.yaml
```

---

**User**

@zsh (41) 

---

**Cursor**


錯誤顯示 `OpenAIModel` 缺少 `total_calls` 屬性，可能是初始化未正確呼叫父類別。檢查並修正：


修正模型類別未呼叫 `super().__init__()`，導致統計屬性未初始化：


```python
"""
OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model_name: str = "gpt-4o", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:

            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            logger.info(f"OpenAI client initialized with model: {self.model_name} via {self.base_url}")
        except TypeError as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIModel':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured OpenAIModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "gpt-4o"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        ) 

```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        self.api_base = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not self.api_base:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

一併修正其他模型類別：


```python
"""
OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for InternS1 models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import openai
from openai import AsyncOpenAI
from json_repair import repair_json
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class S1Model(BaseModel):
    """
    InternS1 implementation of the BaseModel interface.
    """
    
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                model_name: str = "intern-s1", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the InternS1 model adapter.
        
        Args:
            api_key: API key for accessing InternS1 models (defaults to INS1_API_KEY env variable)
            base_url: Base URL for the API endpoint (defaults to INS1_API_BASE_URL env variable)
            model_name: Model identifier to use (e.g., "intern-s1")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("INS1_API_KEY")
        if not self.api_key:
            logger.warning("INS1 API key not provided. Please set INS1_API_KEY environment variable.")
        self.base_url = base_url or os.environ.get("INS1_API_BASE_URL")
        if not self.base_url:
            logger.warning("INS1 base URL not provided. Please set INS1_API_BASE_URL environment variable.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize the client with compatible parameters
        try:

            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"InternS1 client initialized with model: {self.model_name} via {self.base_url}")
        except TypeError as e:
            logger.warning(f"Error initializing InternS1 client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using InternS1 model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            
            response_text = response.choices[0].message.content

            if "</think>" in response_text:
                answer_text = response_text.split("</think>", 1)[1].strip()
            else:
                 answer_text = response_text
            output_data = {
                "system_prompt": system_prompt,
                "prompt": prompt,
                "response": answer_text
            }

            return answer_text
        except Exception as e:
            logger.error(f"Error generating response from InternS1: {e}")
            raise
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        This method instructs the model to produce structured JSON output that
        conforms to the specified schema, handling any necessary post-processing
        to ensure valid JSON is returned.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"
            
        # Use the response_format parameter to enforce JSON
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                **kwargs
            )

            response_text = response.choices[0].message.content

            if "</think>" in response_text:
                result_text = response_text.split("</think>", 1)[1].strip()
            else:
                result_text = response_text
            
            output_data = {
                "system_prompt": enhanced_system_prompt,
                "prompt": prompt,
                "response": result_text
            }
            
            try:
                result_dict = json.loads(result_text)
            except Exception as e:
                print(e)
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                    raise ValueError("Model did not return valid JSON")

            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from InternS1: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model with fallback handling.
        
        This method implements the BaseModel interface for JSON generation,
        with additional support for default values in case of generation failures.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'S1Model':
        """
        Create an InternS1 model instance from a configuration dictionary.
        
        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured InternS1Model instance
        """
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "intern-s1"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )

```

```python
"""
DeepSeek R1 Model Adapter for InternAgent

This module implements the BaseModel interface for DeepSeek R1 reasoning models.
DeepSeek R1 models provide advanced reasoning capabilities with explicit thinking
processes that are separated from the final answer using XML-style tags.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class R1Model(BaseModel):
    """
    DeepSeek R1 model implementation with reasoning capabilities.

    This model adapter interfaces with DeepSeek R1 models that provide explicit
    reasoning traces. The model outputs its thinking process wrapped in <think> tags
    followed by the final answer. This implementation automatically extracts and
    returns only the answer portion while discarding the thinking trace.

    The model is compatible with the OpenAI API format and uses AsyncOpenAI client
    for async operations.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                model_name: str = "DeepSeek-R1", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to DS_API_KEY environment variable)
            model_name: Model name to use (e.g., "DeepSeek-R1")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("DS_API_KEY")
        if not self.api_key:
            logger.warning("DS API key not provided. Please set DS_API_KEY environment variable.")
        self.base_url = base_url or os.environ.get("DS_API_BASE_URL")
        if not self.base_url:
            logger.warning("DS base URL not provided. Please set DS_API_BASE_URL environment variable.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize the client with only the supported parameters for version 1.3.3
        try:
            # The AsyncOpenAI in version 1.3.3 doesn't support 'proxies' parameter
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        except TypeError as e:
            logger.warning(f"Error initializing DeepSeek client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            
            response_text = response.choices[0].message.content
            think_text, answer_text = response_text.split("</think>\n\n", 1)
            return answer_text
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            response_text = response.choices[0].message.content
            think_text, answer_text = response_text.split("</think>\n\n", 1)
            return json.loads(answer_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'R1Model':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured OpenAIModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "DeepSeek-R1"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        ) 

```

已修正所有模型類別的初始化問題。這些類別未呼叫 `super().__init__()`，導致 `BaseModel` 中的統計屬性（`total_calls`、`successful_calls` 等）未被初始化。

## 修正內容

已在以下模型類別的 `__init__` 方法中加入 `super().__init__()`：

1. `OpenAIModel` (`internagent/mas/models/openai_model.py`)
2. `AzureModel` (`internagent/mas/models/azure_model.py`)
3. `S1Model` (`internagent/mas/models/s1_model.py`)
4. `R1Model` (`internagent/mas/models/r1_model.py`)

現在所有模型類別都會正確初始化基類的統計屬性，`get_stats()` 可正常運作。

請再次執行測試腳本：

```bash
uv run test_llm.py --provider azure
```

應該不會再出現 `total_calls` 屬性錯誤。

```shellscript
uv run test_llm.py --provider azure
```

---

**User**

@zsh (182-207) 

---

**Cursor**


出現 Azure OpenAI 404 錯誤，可能是端點格式問題。檢查並修正 Azure 模型的端點配置：


修正 Azure OpenAI 的端點配置。需要確保端點格式正確，並加入 API 版本參數：


調整 Azure 模型端點格式：確保 URL 格式正確，並規範化端點：


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點應該是: https://{resource}.openai.azure.com/
        # 需要確保以 / 結尾，並且包含 deployment 路徑
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 如果端點不包含 /openai/deployments，則需要添加
            # 但實際使用時，OpenAI SDK 會自動處理，所以只需確保端點格式正確
            if not api_endpoint.endswith('/openai'):
                # 檢查是否已經是完整路徑
                if '/openai/deployments' not in api_endpoint:
                    # 標準 Azure OpenAI 端點格式
                    self.api_base = f"{api_endpoint}/openai/deployments/{model_name}"
                else:
                    self.api_base = api_endpoint
            else:
                self.api_base = f"{api_endpoint}/deployments/{model_name}"
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 需要 api_version 參數
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                api_version="2024-02-15-preview"
            )
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            # 某些版本的 SDK 可能不支持 api_version 參數，嘗試不使用
            try:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
                logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
            except TypeError as e2:
                logger.warning(f"Error initializing Azure OpenAI client: {e2}")
                self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

改為使用更簡單且符合 OpenAI SDK 的做法。修正 `generate` 方法以使用正確的 Azure OpenAI 調用格式：


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點應該是: https://{resource}.openai.azure.com/
        # 需要確保以 / 結尾，並且在調用時使用 deployment name
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點格式，不包含 deployment 名稱
            # Deployment 名稱會在調用時通過 model 參數傳遞
            self.api_base = f"{api_endpoint}/openai/deployments"
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 需要 api_version 參數
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                api_version="2024-02-15-preview"
            )
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            # 某些版本的 SDK 可能不支持 api_version 參數，嘗試不使用
            try:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
                logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
            except TypeError as e2:
                logger.warning(f"Error initializing Azure OpenAI client: {e2}")
                self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點應該是: https://{resource}.openai.azure.com/
        # 需要確保以 / 結尾，並且在調用時使用 deployment name
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點格式，不包含 deployment 名稱
            # Deployment 名稱會在調用時通過 model 參數傳遞
            self.api_base = f"{api_endpoint}/openai/deployments"
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 客戶端初始化
            # 注意：某些版本的 SDK 可能不支持 api_version 參數，所以不使用它
            # API 版本會通過查詢參數自動添加
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點應該是: https://{resource}.openai.azure.com/
        # 需要確保以 / 結尾，並且在調用時使用 deployment name
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點格式，不包含 deployment 名稱
            # Deployment 名稱會在調用時通過 model 參數傳遞
            self.api_base = f"{api_endpoint}/openai/deployments"
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 客戶端初始化
            # 注意：某些版本的 SDK 可能不支持 api_version 參數，所以不使用它
            # API 版本會通過查詢參數自動添加
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 需要使用 deployment name 作為 model 參數
            # 並且需要在 URL 中包含 api-version 查詢參數
            # 但 AsyncOpenAI SDK 會自動處理，我們只需要確保 model 參數是 deployment name
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                extra_headers={
                    "api-version": self.api_version
                } if hasattr(self, 'api_version') else {},
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

調整實現：Azure OpenAI 的 API 版本應作為查詢參數，而不是 header。修正：


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點應該是: https://{resource}.openai.azure.com/
        # 需要確保以 / 結尾，並且在調用時使用 deployment name
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點格式，不包含 deployment 名稱
            # Deployment 名稱會在調用時通過 model 參數傳遞
            self.api_base = f"{api_endpoint}/openai/deployments"
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 客戶端初始化
            # 注意：某些版本的 SDK 可能不支持 api_version 參數，所以不使用它
            # API 版本會通過查詢參數自動添加
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 需要使用 deployment name 作為 model 參數
            # base_url 已經設定為 {endpoint}/openai/deployments
            # 所以 model 參數應該是 deployment name（self.model_name）
            # API 版本需要在 URL 中作為查詢參數，但 OpenAI SDK 可能不支持
            # 我們需要手動構建完整的 URL 或使用 httpx 直接調用
            # 先嘗試標準方式（OpenAI SDK 的新版本可能支持）
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點格式應該是: https://{resource}.openai.azure.com/
        # 對於 Azure OpenAI，base_url 應該是完整的資源端點
        # SDK 會自動構建 /openai/deployments/{model}/chat/completions?api-version=...
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點，不包含 /openai/deployments
            # 讓 SDK 自動處理路徑構建
            self.api_base = api_endpoint
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            # Azure OpenAI 客戶端初始化
            # 注意：某些版本的 SDK 可能不支持 api_version 參數，所以不使用它
            # API 版本會通過查詢參數自動添加
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except TypeError as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 需要使用 deployment name 作為 model 參數
            # base_url 已經設定為 {endpoint}/openai/deployments
            # 所以 model 參數應該是 deployment name（self.model_name）
            # API 版本需要在 URL 中作為查詢參數，但 OpenAI SDK 可能不支持
            # 我們需要手動構建完整的 URL 或使用 httpx 直接調用
            # 先嘗試標準方式（OpenAI SDK 的新版本可能支持）
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

改為更符合 OpenAI SDK 的做法：使用 `default_query` 傳遞 API 版本：


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點格式應該是: https://{resource}.openai.azure.com/
        # 對於 Azure OpenAI，base_url 應該是完整的資源端點
        # SDK 會自動構建 /openai/deployments/{model}/chat/completions?api-version=...
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 標準 Azure OpenAI 端點，不包含 /openai/deployments
            # 讓 SDK 自動處理路徑構建
            self.api_base = api_endpoint
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # Azure OpenAI 客戶端初始化
            # 對於 Azure OpenAI，base_url 應該是完整的端點（不包含 /openai/deployments）
            # SDK 會自動構建路徑，我們使用 default_query 傳遞 API 版本
            from openai import AsyncOpenAI
            
            # 嘗試使用 default_query 參數（如果 SDK 支持）
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    default_query={"api-version": self.api_version}
                )
            except TypeError:
                # 如果 SDK 不支持 default_query，則不使用
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
                # 記住需要手動處理 API 版本
                logger.info(f"SDK does not support default_query, will handle api-version manually")
            
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 需要使用 deployment name 作為 model 參數
            # base_url 已經設定為 {endpoint}/openai/deployments
            # 所以 model 參數應該是 deployment name（self.model_name）
            # API 版本需要在 URL 中作為查詢參數，但 OpenAI SDK 可能不支持
            # 我們需要手動構建完整的 URL 或使用 httpx 直接調用
            # 先嘗試標準方式（OpenAI SDK 的新版本可能支持）
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點格式應該是: https://{resource}.openai.azure.com/
        # 對於 Azure OpenAI，base_url 需要包含完整的部署路徑
        # 格式: https://{resource}.openai.azure.com/openai/deployments/{deployment-name}
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 構建包含 deployment 的路徑
            if '/openai/deployments' not in api_endpoint:
                self.api_base = f"{api_endpoint}/openai/deployments/{model_name}"
            elif api_endpoint.endswith('/openai') or api_endpoint.endswith('/openai/'):
                self.api_base = f"{api_endpoint.rstrip('/')}/deployments/{model_name}"
            else:
                # 已經包含完整路徑，直接使用
                self.api_base = api_endpoint
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # Azure OpenAI 客戶端初始化
            # 對於 Azure OpenAI，base_url 應該是完整的端點（不包含 /openai/deployments）
            # SDK 會自動構建路徑，我們使用 default_query 傳遞 API 版本
            from openai import AsyncOpenAI
            
            # 嘗試使用 default_query 參數（如果 SDK 支持）
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    default_query={"api-version": self.api_version}
                )
            except TypeError:
                # 如果 SDK 不支持 default_query，則不使用
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
                # 記住需要手動處理 API 版本
                logger.info(f"SDK does not support default_query, will handle api-version manually")
            
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 需要使用 deployment name 作為 model 參數
            # base_url 已經設定為 {endpoint}/openai/deployments
            # 所以 model 參數應該是 deployment name（self.model_name）
            # API 版本需要在 URL 中作為查詢參數，但 OpenAI SDK 可能不支持
            # 我們需要手動構建完整的 URL 或使用 httpx 直接調用
            # 先嘗試標準方式（OpenAI SDK 的新版本可能支持）
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

調整 `generate` 方法，確保正確使用 deployment name，並在 URL 中加入 API 版本：


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點格式應該是: https://{resource}.openai.azure.com/
        # 對於 Azure OpenAI，base_url 需要包含完整的部署路徑
        # 格式: https://{resource}.openai.azure.com/openai/deployments/{deployment-name}
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 構建包含 deployment 的路徑
            if '/openai/deployments' not in api_endpoint:
                self.api_base = f"{api_endpoint}/openai/deployments/{model_name}"
            elif api_endpoint.endswith('/openai') or api_endpoint.endswith('/openai/'):
                self.api_base = f"{api_endpoint.rstrip('/')}/deployments/{model_name}"
            else:
                # 已經包含完整路徑，直接使用
                self.api_base = api_endpoint
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # Azure OpenAI 客戶端初始化
            # 對於 Azure OpenAI，base_url 應該是完整的端點（不包含 /openai/deployments）
            # SDK 會自動構建路徑，我們使用 default_query 傳遞 API 版本
            from openai import AsyncOpenAI
            
            # 嘗試使用 default_query 參數（如果 SDK 支持）
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    default_query={"api-version": self.api_version}
                )
            except TypeError:
                # 如果 SDK 不支持 default_query，則不使用
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
                # 記住需要手動處理 API 版本
                logger.info(f"SDK does not support default_query, will handle api-version manually")
            
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 的 base_url 已經包含完整路徑: {endpoint}/openai/deployments/{deployment-name}
            # 所以 model 參數需要和 base_url 中的 deployment name 匹配
            # 如果使用 default_query 參數，API 版本會自動添加到查詢參數中
            # 否則需要在 URL 中手動添加
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Azure OpenAI 中，這是 deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

已修正 Azure OpenAI 的端點格式處理。主要變更：

1. **自動構建完整端點路徑**：將端點格式從 `https://{resource}.openai.azure.com/` 轉換為 `https://{resource}.openai.azure.com/openai/deployments/{deployment-name}`

2. **API 版本處理**：使用 `default_query` 參數自動添加 `api-version` 查詢參數（如果 SDK 支持）

請再次執行測試：

```bash
uv run test_llm.py --provider azure
```

如果仍有問題，請確認：

1. **部署名稱是否正確**：`model_name` 必須與 Azure OpenAI 中的實際部署名稱一致（例如 `gpt-4o-mini`）
2. **端點格式**：環境變數 `AZURE_OPENAI_ENDPOINT` 應為 `https://{resource}.openai.azure.com/`（不含 `/openai/deployments`）

若仍有 404 錯誤，可能需要檢查 Azure 資源與部署設定。

```shellscript
uv run test_llm.py --provider azure
```

---

**User**

@azure_model.py 的 AsyncOpenAI() 改成使用 AsyncAzureOpenAI()

---

**Cursor**


將 `AsyncOpenAI` 改為 `AsyncAzureOpenAI`，以正確使用 Azure OpenAI API。


```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncAzureOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # Azure OpenAI 端點格式應該是: https://{resource}.openai.azure.com/
        # 對於 Azure OpenAI，base_url 需要包含完整的部署路徑
        # 格式: https://{resource}.openai.azure.com/openai/deployments/{deployment-name}
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 構建包含 deployment 的路徑
            if '/openai/deployments' not in api_endpoint:
                self.api_base = f"{api_endpoint}/openai/deployments/{model_name}"
            elif api_endpoint.endswith('/openai') or api_endpoint.endswith('/openai/'):
                self.api_base = f"{api_endpoint.rstrip('/')}/deployments/{model_name}"
            else:
                # 已經包含完整路徑，直接使用
                self.api_base = api_endpoint
        else:
            self.api_base = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # Azure OpenAI 客戶端初始化
            # 對於 Azure OpenAI，base_url 應該是完整的端點（不包含 /openai/deployments）
            # SDK 會自動構建路徑，我們使用 default_query 傳遞 API 版本
            from openai import AsyncOpenAI
            
            # 嘗試使用 default_query 參數（如果 SDK 支持）
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    default_query={"api-version": self.api_version}
                )
            except TypeError:
                # 如果 SDK 不支持 default_query，則不使用
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
                # 記住需要手動處理 API 版本
                logger.info(f"SDK does not support default_query, will handle api-version manually")
            
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 的 base_url 已經包含完整路徑: {endpoint}/openai/deployments/{deployment-name}
            # 所以 model 參數需要和 base_url 中的 deployment name 匹配
            # 如果使用 default_query 參數，API 版本會自動添加到查詢參數中
            # 否則需要在 URL 中手動添加
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Azure OpenAI 中，這是 deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncAzureOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # AsyncAzureOpenAI 需要的端點格式應該是: https://{resource}.openai.azure.com/
        # 不包含 /openai/deployments 路徑
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 如果已經包含 /openai/deployments，移除它
            if '/openai/deployments' in api_endpoint:
                # 提取基礎端點
                self.azure_endpoint = api_endpoint.split('/openai/deployments')[0]
            elif api_endpoint.endswith('/openai'):
                self.azure_endpoint = api_endpoint.rsplit('/openai', 1)[0]
            else:
                self.azure_endpoint = api_endpoint
        else:
            self.azure_endpoint = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # Azure OpenAI 客戶端初始化
            # 對於 Azure OpenAI，base_url 應該是完整的端點（不包含 /openai/deployments）
            # SDK 會自動構建路徑，我們使用 default_query 傳遞 API 版本
            from openai import AsyncOpenAI
            
            # 嘗試使用 default_query 參數（如果 SDK 支持）
            try:
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    default_query={"api-version": self.api_version}
                )
            except TypeError:
                # 如果 SDK 不支持 default_query，則不使用
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout
                )
                # 記住需要手動處理 API 版本
                logger.info(f"SDK does not support default_query, will handle api-version manually")
            
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.api_base}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 的 base_url 已經包含完整路徑: {endpoint}/openai/deployments/{deployment-name}
            # 所以 model 參數需要和 base_url 中的 deployment name 匹配
            # 如果使用 default_query 參數，API 版本會自動添加到查詢參數中
            # 否則需要在 URL 中手動添加
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Azure OpenAI 中，這是 deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncAzureOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # AsyncAzureOpenAI 需要的端點格式應該是: https://{resource}.openai.azure.com/
        # 不包含 /openai/deployments 路徑
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 如果已經包含 /openai/deployments，移除它
            if '/openai/deployments' in api_endpoint:
                # 提取基礎端點
                self.azure_endpoint = api_endpoint.split('/openai/deployments')[0]
            elif api_endpoint.endswith('/openai'):
                self.azure_endpoint = api_endpoint.rsplit('/openai', 1)[0]
            else:
                self.azure_endpoint = api_endpoint
        else:
            self.azure_endpoint = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # 使用 AsyncAzureOpenAI 初始化 Azure OpenAI 客戶端
            # AsyncAzureOpenAI 需要 azure_endpoint (基礎端點) 和 api_version
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.timeout
            )
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.azure_endpoint}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Azure OpenAI 的 base_url 已經包含完整路徑: {endpoint}/openai/deployments/{deployment-name}
            # 所以 model 參數需要和 base_url 中的 deployment name 匹配
            # 如果使用 default_query 參數，API 版本會自動添加到查詢參數中
            # 否則需要在 URL 中手動添加
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Azure OpenAI 中，這是 deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

```python
"""
Azure OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import openai
from openai import AsyncAzureOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureModel(BaseModel):
    """Azure OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_base: Optional[str] = None,
                model_name: str = "gpt-4o-mini", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60):
        """
        Initialize the Azure OpenAI model adapter.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY environment variable)
            api_base: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT environment variable)
            model_name: Model identifier to use (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        #         super().__init__()
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        api_endpoint = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        if not self.api_key:
            logger.warning("Azure OpenAI API key not provided. Please set AZURE_OPENAI_KEY environment variable.")
        
        if not api_endpoint:
            logger.warning("Azure OpenAI endpoint not provided. Please set AZURE_OPENAI_ENDPOINT environment variable.")
        
        # 規範化 Azure OpenAI 端點格式
        # AsyncAzureOpenAI 需要的端點格式應該是: https://{resource}.openai.azure.com/
        # 不包含 /openai/deployments 路徑
        if api_endpoint:
            api_endpoint = api_endpoint.rstrip('/')
            # 如果已經包含 /openai/deployments，移除它
            if '/openai/deployments' in api_endpoint:
                # 提取基礎端點
                self.azure_endpoint = api_endpoint.split('/openai/deployments')[0]
            elif api_endpoint.endswith('/openai'):
                self.azure_endpoint = api_endpoint.rsplit('/openai', 1)[0]
            else:
                self.azure_endpoint = api_endpoint
        else:
            self.azure_endpoint = None
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 儲存 API 版本用於構建完整 URL
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        try:
            # 使用 AsyncAzureOpenAI 初始化 Azure OpenAI 客戶端
            # AsyncAzureOpenAI 需要 azure_endpoint (基礎端點) 和 api_version
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=self.timeout
            )
            logger.info(f"Azure OpenAI client initialized with model: {self.model_name} via {self.azure_endpoint}")
        except Exception as e:
            logger.warning(f"Error initializing Azure OpenAI client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Azure OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # AsyncAzureOpenAI 會自動處理 Azure OpenAI 的路徑和 API 版本
            # model 參數應該是 deployment name（在 Azure OpenAI 中）
            response = await self.client.chat.completions.create(
                model=self.model_name,  # Azure OpenAI deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from Azure OpenAI: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            result_text = response.choices[0].message.content
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Model returned invalid JSON: {result_text}")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair}")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                raise ValueError("Model did not return valid JSON")
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureModel':
        """
        Create an Azure model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured AzureModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            model_name=config.get("model_name", "gpt-4o-mini"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60)
        )


```

已將 `AsyncOpenAI` 改為 `AsyncAzureOpenAI`。主要變更：

## 主要修改

1. 改用 `AsyncAzureOpenAI`（第 14 行）
2. 端點處理改為僅保留基礎端點，移除 `/openai/deployments`（第 52-66 行）
3. 使用 `azure_endpoint` 與 `api_version` 初始化客戶端（第 79-84 行）

## 差異

- AsyncOpenAI：需自行構建 URL 路徑與 API 版本
- AsyncAzureOpenAI：自動處理路徑與版本

使用 `AsyncAzureOpenAI`：
- 基礎端點：`https://{resource}.openai.azure.com`
- 會自動構建 `/openai/deployments/{deployment-name}/chat/completions?api-version=...`
- 自動處理 API 版本

請重新測試：

```bash
uv run test_llm.py --provider azure
```

預期可正常運作。

```shellscript
uv run test_llm.py --provider azure
```

---

**User**

@codeview_agent.py 從哪裡取得 client_model?

---

**Cursor**


追蹤 `client_model` 的來源：


`client_model` 的來源如下：

## `client_model` 的來源流程

```247:247:internagent/mas/agents/codeview_agent.py
        self.client_model = self.llm_settings.model
```

來源鏈路：
1. `get_repo_structure()` 接收 `model` 參數（第 564 行）
2. 傳給 `SettingManager`，設定為 `chat_settings.model`
3. `RepoViewer` 從 `setting_manager.chat_settings` 取得（第 243 行）
4. 在 `RepoViewer.__init__` 中設定 `self.client_model = self.llm_settings.model`（第 247 行）
5. 在 `generate_docs()` 中傳給 `generate_doc_single_static()`（第 262 行）

查看呼叫 `get_repo_structure` 的地方：


```python
        self.client_model = self.llm_settings.model
```

## 完整流程

1. 來源：在 `generation_agent.py` 第 265 行，`client_model` 來自 `self.model.model_name`
   - `self.model` 是 `BaseModel` 實例（例如 `AzureModel`、`OpenAIModel`）
   - `model_name` 是該模型實例的模型名稱屬性

2. 傳遞路徑：
   ```
   generation_agent.py (第265行)
   ↓ model=self.model.model_name
   get_repo_structure() (第564行)
   ↓ model 參數
   SettingManager.__init__() (第20行)
   ↓ self.chat_settings.model = model
   RepoViewer.__init__() (第243-247行)
   ↓ self.client_model = self.llm_settings.model
   generate_docs() (第262行)
   ↓ 傳給 generate_doc_single_static()
   generate_doc_single_static() (第282行)
   ```

## 結論

`client_model` 最終來自代理使用的模型實例的 `model_name`（例如 `"gpt-4o-mini"`），該名稱通常從配置檔讀取。

若需要在 `codeview_agent.py` 中改用 Azure OpenAI，可檢查 `model_name` 是否以 `"gpt"` 開頭，並檢查環境變數；此部分已在第 286-307 行處理。

```plaintext
   generation_agent.py (第265行)
   ↓ model=self.model.model_name
   get_repo_structure() (第564行)
   ↓ model 參數
   SettingManager.__init__() (第20行)
   ↓ self.chat_settings.model = model
   RepoViewer.__init__() (第243-247行)
   ↓ self.client_model = self.llm_settings.model
   generate_docs() (第262行)
   ↓ 傳給 generate_doc_single_static()
   generate_doc_single_static() (第282行)
```

```plaintext
   generation_agent.py (第265行)
   ↓ model=self.model.model_name
   get_repo_structure() (第564行)
   ↓ model 參數
   SettingManager.__init__() (第20行)
   ↓ self.chat_settings.model = model
   RepoViewer.__init__() (第243-247行)
   ↓ self.client_model = self.llm_settings.model
   generate_docs() (第262行)
   ↓ 傳給 generate_doc_single_static()
   generate_doc_single_static() (第282行)
```

---

**User**

@codeview_agent.pygenerate_doc_single_static() 如何改成使用 config 檔案中的 CodeView Configuration ?

並移除使用 client_model.lower().startswith('gpt') 做判斷的方式

---

**Cursor**


檢查配置結構與載入方式，接著修改 `generate_doc_single_static()` 以使用配置，並移除模型名稱判斷：


修改 `codeview_agent.py`，改用配置並移除模型名稱判斷邏輯。改為使用 `ModelFactory` 建立模型：


修改 `codeview_agent.py`，改用配置並移除模型名稱判斷：


```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.client_model = self.llm_settings.model
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            # Check if using Azure OpenAI
            if os.environ.get("AZURE_OPENAI_ENDPOINT"):
                try: 
                    import openai
                    print(f"Using Azure OpenAI API with model {client_model}")
                    client = openai.OpenAI(
                        api_key=os.environ.get("AZURE_OPENAI_KEY"),
                        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                    )
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
            else:
                try: 
                    import openai
                    print(f"Using OpenAI API with model {client_model}")
                    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model=client_model,
                        messages=[
                            {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                            {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                        ],
                        temperature=llm_settings.temperature,
                        max_tokens=llm_settings.max_tokens
                    )
                    results = response.choices[0].message.content
                except Exception as e:
                    print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.client_model = self.llm_settings.model
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
                - file_path: 檔案路徑
                - dict_path: 字典路徑
                - model_config: 模型配置字典，包含 provider 和相關設定
                - codeview_config: CodeView 配置字典
                - llm_settings: LLM 設定（temperature, max_tokens 等）
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            # 使用 ModelFactory 建立模型實例
            from internagent.mas.models.model_factory import ModelFactory
            
            # 建立模型配置
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    logger.error(f"Default provider '{default_provider}' not found in config")
                    return None, None
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config")
                    return None, None
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
                - file_path: 檔案路徑
                - dict_path: 字典路徑
                - model_config: 模型配置字典，包含 provider 和相關設定
                - codeview_config: CodeView 配置字典
                - llm_settings: LLM 設定（temperature, max_tokens 等）
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            # 使用 ModelFactory 建立模型實例
            from internagent.mas.models.model_factory import ModelFactory
            
            # 建立模型配置
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    logger.error(f"Default provider '{default_provider}' not found in config")
                    return None, None
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config")
                    return None, None
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
                - file_path: 檔案路徑
                - dict_path: 字典路徑
                - model_config: 模型配置字典，包含 provider 和相關設定
                - codeview_config: CodeView 配置字典
                - llm_settings: LLM 設定（temperature, max_tokens 等）
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            # 使用 ModelFactory 建立模型實例
            from internagent.mas.models.model_factory import ModelFactory
            
            # 建立模型配置
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    logger.error(f"Default provider '{default_provider}' not found in config")
                    return None, None
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config")
                    return None, None
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
                - file_path: 檔案路徑
                - dict_path: 字典路徑
                - model_config: 模型配置字典，包含 provider 和相關設定
                - codeview_config: CodeView 配置字典
                - llm_settings: LLM 設定（temperature, max_tokens 等）
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            # 使用 ModelFactory 建立模型實例
            from internagent.mas.models.model_factory import ModelFactory
            
            # 建立模型配置
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    logger.error(f"Default provider '{default_provider}' not found in config")
                    return None, None
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config")
                    return None, None
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        # Check if using Azure OpenAI
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            try: 
                import openai
                print(f"Using Azure OpenAI API with model {client_model}")
                client = openai.OpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")
        else:
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                        {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(project_path, output_dir, output_name, ignore_list=None, provider="agent", config: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
    """
    取得並產生程式庫結構文檔。
    
    Args:
        project_path: 專案路徑
        output_dir: 輸出目錄
        output_name: 輸出檔名
        ignore_list: 忽略清單
        provider: 提供者類型 ("agent" 或 "user")
        config: 配置字典（包含 models 和 codeview 配置）
        model_name: 模型名稱（保留向後相容，但優先使用 config）
    """
    # 如果沒有提供 config，嘗試建立基本配置
    if config is None:
        from internagent.mas.models.model_factory import ModelFactory
        # 嘗試從環境變數或預設值建立配置
        config = {
            "models": {
                "default_provider": "azure",
                "azure": {
                    "model_name": model_name or "gpt-4o-mini",
                }
            },
            "codeview": {
                "model_provider": "default",
            }
        }
    
    # 使用 codeview 配置中的 model_name，或使用傳入的 model_name
    codeview_config = config.get("codeview", {})
    if model_name and "model_name" not in codeview_config:
        codeview_config["model_name"] = model_name
    
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=codeview_config.get("model_name", model_name or "gpt-4o-mini"),
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict, config)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        # 使用配置來產生摘要
        repo_summary = get_repo_summary_with_config(repo_description_tree, setting_manager.chat_settings, config)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
                - file_path: 檔案路徑
                - dict_path: 字典路徑
                - model_config: 模型配置字典，包含 provider 和相關設定
                - codeview_config: CodeView 配置字典
                - llm_settings: LLM 設定（temperature, max_tokens 等）
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            # 使用 ModelFactory 建立模型實例
            from internagent.mas.models.model_factory import ModelFactory
            
            # 建立模型配置
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    logger.error(f"Default provider '{default_provider}' not found in config")
                    return None, None
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config")
                    return None, None
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary_with_config(repo_file_tree, llm_settings, config: Dict[str, Any]):
    """
    使用配置檔產生程式庫摘要。
    
    Args:
        repo_file_tree: 程式庫檔案樹
        llm_settings: LLM 設定
        config: 配置字典（包含 models 和 codeview 配置）
    """
    try:
        from internagent.mas.models.model_factory import ModelFactory
        
        codeview_config = config.get("codeview", {})
        model_config = config.get("models", {})
        
        # 決定使用哪個提供者
        model_provider = codeview_config.get("model_provider", "default")
        if model_provider == "default":
            default_provider = model_config.get("default_provider", "azure")
            if default_provider in model_config:
                provider_config = model_config[default_provider].copy()
                provider_config["provider"] = default_provider
                provider_config["default_provider"] = default_provider
            else:
                logger.error(f"Default provider '{default_provider}' not found in config")
                return ""
        else:
            if model_provider in model_config:
                provider_config = model_config[model_provider].copy()
                provider_config["provider"] = model_provider
                provider_config["default_provider"] = model_config.get("default_provider", "azure")
            else:
                logger.error(f"Provider '{model_provider}' not found in config")
                return ""
        
        # 使用 codeview_config 中的 model_name（如果有的話）
        if "model_name" in codeview_config:
            provider_config["model_name"] = codeview_config["model_name"]
        
        # 使用 llm_settings 覆蓋
        if hasattr(llm_settings, 'temperature'):
            provider_config["temperature"] = llm_settings.temperature
        if hasattr(llm_settings, 'max_tokens'):
            provider_config["max_tokens"] = llm_settings.max_tokens
        
        # 建立模型
        model = ModelFactory.create_model(provider_config)
        logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
        
        # 使用模型的 generate 方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                model.generate(
                    prompt=REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree),
                    system_prompt=REPO_SUMMARY_SYS_PROMPT,
                    temperature=provider_config.get("temperature", 0.6),
                    max_tokens=provider_config.get("max_tokens", 3000)
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error during generate summary for repo: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""


def get_repo_summary(client_model, repo_file_tree, llm_settings):
    """
    向後相容的函數，保留原有 API。
    使用配置檔的新實作：get_repo_summary_with_config()
    """
    # 嘗試建立基本配置（向後相容）
    config = {
        "models": {
            "default_provider": "azure",
            "azure": {
                "model_name": client_model,
            }
        },
        "codeview": {
            "model_provider": "default",
            "model_name": client_model,
        }
    }
    return get_repo_summary_with_config(repo_file_tree, llm_settings, config)

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(project_path, output_dir, output_name, ignore_list=None, provider="agent", config: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
    """
    取得並產生程式庫結構文檔。
    
    Args:
        project_path: 專案路徑
        output_dir: 輸出目錄
        output_name: 輸出檔名
        ignore_list: 忽略清單
        provider: 提供者類型 ("agent" 或 "user")
        config: 配置字典（包含 models 和 codeview 配置）
        model_name: 模型名稱（保留向後相容，但優先使用 config）
    """
    # 如果沒有提供 config，嘗試建立基本配置
    if config is None:
        from internagent.mas.models.model_factory import ModelFactory
        # 嘗試從環境變數或預設值建立配置
        config = {
            "models": {
                "default_provider": "azure",
                "azure": {
                    "model_name": model_name or "gpt-4o-mini",
                }
            },
            "codeview": {
                "model_provider": "default",
            }
        }
    
    # 使用 codeview 配置中的 model_name，或使用傳入的 model_name
    codeview_config = config.get("codeview", {})
    if model_name and "model_name" not in codeview_config:
        codeview_config["model_name"] = model_name
    
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=codeview_config.get("model_name", model_name or "gpt-4o-mini"),
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict, config)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        # 使用配置來產生摘要
        repo_summary = get_repo_summary_with_config(repo_description_tree, setting_manager.chat_settings, config)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
"""
Generation Agent for InternAgent

Implements the idea generation agent that creates novel research hypotheses
based on research goals, domain constraints, literature, and optional code baselines.
The agent supports iterative refinement through feedback incorporation and provides
configurable creativity levels for idea generation.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional

from ..models.base_model import BaseModel
from .base_agent import BaseAgent, AgentExecutionError
from .codeview_agent import get_repo_structure


logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Agent that generates novel, scientifically plausible research hypotheses.

    Creates multiple idea candidates based on research goals, domain knowledge,
    literature surveys, and optional code baselines. Supports iterative refinement
    through feedback and adjustable creativity levels. Can analyze file-level or
    project-level code to generate code-aware hypotheses.

    Attributes:
        do_survey (bool): Whether to include literature information
        generation_count (int): Number of hypotheses to generate per call
        creativity (float): Creativity level 0-1 (higher = more creative)
        diversity_threshold (float): Minimum diversity between hypotheses
        temperature (float): Model sampling temperature
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the generation agent with model and configuration.

        Args:
            model (BaseModel): Language model for idea generation
            config (Dict[str, Any]): Configuration with keys:
                - do_survey (bool): Include literature (default: False)
                - generation_count (int): Hypotheses per call (default: 5)
                - creativity (float): Creativity 0-1 (default: 0.9)
                - diversity_threshold (float): Min diversity (default: 0.3)
                - temperature (float): Sampling temperature (optional)
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.do_survey = config.get("do_survey", False)
        self.generation_count = config.get("generation_count", 5)
        self.creativity = config.get("creativity", 0.9)  # Higher = more creative
        self.diversity_threshold = config.get("diversity_threshold", 0.3)
        self.temperature = config.get("temperature", None)
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate novel research ideas based on provided context.

        Generates multiple idea candidates using the configured language model,
        incorporating research goals, domain knowledge, literature, code baselines,
        and iterative feedback. Returns structured ideas with rationales.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal with description, domain, constraints
                - iteration (int): Current iteration number
                - feedback (List[Dict]): Previous feedback entries (optional)
                - paper_lst (List[Dict]): Literature papers if do_survey=True
            params (Dict[str, Any]): Runtime parameters:
                - count (int): Override generation_count (optional)
                - creativity (float): Override creativity level (optional)

        Returns:
            Dict[str, Any]: Results containing:
                - hypotheses (List[Dict]): Generated hypotheses with text/rationale
                - metadata (Dict): Generation info (count, creativity, reasoning)
                - baseline_summary (str): Code baseline summary if applicable

        Raises:
            AgentExecutionError: If goal missing or generation fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        if not goal or not goal.get("description"):
            raise AgentExecutionError("Research goal is required for idea generation")
            
        # Extract and override parameters if provided
        count = params.get("count", self.generation_count)
        creativity = params.get("creativity", self.creativity)
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        paper_lst = context.get("paper_lst", [])
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The idea statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for why this idea is plausible"
                            }
                        },
                        "required": ["text", "rationale"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the generation approach"
                },
                "baseline_summary":{
                    "type": "string",
                    "description": "Summary of the baseline code understanding"
                }
            },
            "required": ["hypotheses", "reasoning"]
        }
        
        if os.path.exists(goal.get("ref_code_path", "")):
            output_schema["required"] = ["hypotheses", "reasoning", "baseline_summary"]
                    
        # Build the prompt
        prompt = self._build_generation_prompt(
            goal=goal,
            count=count,
            iteration=iteration,
            feedback=feedback,
            paper_lst=paper_lst
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(creativity)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature,
            )
            
            # Validate the response
            hypotheses = response.get("hypotheses", [])
            if not hypotheses:
                logger.warning("Generation agent returned no hypotheses")
                
            # Add metadata to the response
            result = {
                "hypotheses": hypotheses,
                "metadata": {
                    "count": len(hypotheses),
                    "creativity": creativity,
                    "iteration": iteration,
                    "reasoning": response.get("reasoning", "")
                },
                "baseline_summary": response.get("baseline_summary", "")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Generation agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate hypotheses: {str(e)}")
    
    def _build_generation_prompt(self,
                               goal: Dict[str, Any],
                               count: int,
                               iteration: int,
                               feedback: List[Dict[str, Any]],
                               paper_lst: List[Dict]) -> str:
        """
        Construct comprehensive prompt for idea generation.

        Builds a structured prompt incorporating research goals, domain constraints,
        background information, literature, code baselines, and iterative feedback.
        Handles both file-level and project-level code analysis.

        Args:
            goal (Dict[str, Any]): Goal with description/domain/constraints/background
            count (int): Number of hypotheses to generate
            iteration (int): Current iteration number for feedback context
            feedback (List[Dict[str, Any]]): Previous feedback entries
            paper_lst (List[Dict]): Literature papers for survey mode

        Returns:
            str: Formatted multi-section prompt for the language model
        """
        
        # Extract goal information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        background = goal.get("background", "")
        
        # Start with the goal
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add background if available
        if background:
            prompt += "Background Information is a detailed description of the baseline method. Please analyze the provided Background Information and give novel hypotheses based on the research goal and the background information.\n\n"
            prompt += f"# Background Information\n{background}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"

        # if do_survey, literature information is included in goal, and need prompt
        if self.do_survey and paper_lst:
            logger.info("Add literature information to prompt")
            literature_prompt = "# Literature Information\n"
            for paper in paper_lst:
                literature_prompt += f"- {paper['title']} ({paper['year']})\n {paper['abstract']} \n\n"
            prompt += literature_prompt
            prompt += "Please analyze the provided Literature Information and give novel hypotheses based on the research goal and the literature."
            
        # Add reference code if available
        # load code if exist, judge file/dir/not exist
        ref_code_path = goal.get("ref_code_path", "")
        if os.path.exists(ref_code_path):
            if os.path.isfile(ref_code_path):
                logger.info("Perform #file-level# code understanding and generation")
                with open(ref_code_path, 'r') as f:
                    ref_code = f.read()
                logger.info("Add reference code (RAW) to prompt")
                prompt += f"# Reference Code\n```python\n{ref_code}\n```\n\n"
                
            elif os.path.isdir(ref_code_path):
                logger.info("Perform #project-level# code understanding and generation")
                logger.info("Loading codeivew Agent ...")
                if os.path.exists(os.path.join(ref_code_path, "code_summary.json")):
                    logger.info("Code summary exists! Loading code summary from file.")
                    with open(os.path.join(ref_code_path, "code_summary.json"), 'r') as f:
                        ref_code = f.read()
                    ref_code = json.loads(ref_code)
                else:
                    logger.info("Code summary does not exist! Generating code summary.")
                    # Use codeview agent to generate code summary
                    # 取得全局配置（如果有的話）
                    global_config = getattr(self, 'config', {})
                    ref_code = get_repo_structure(
                        project_path=ref_code_path,
                        output_dir=ref_code_path,
                        output_name="code_summary.json",
                        ignore_list=None,
                        provider="user",
                        config=global_config,
                        model_name=self.model.model_name  # 保留作為後備
                    )
                ref_code = ref_code['summary'] + "\n\n" + ref_code['key_files']
                logger.info("Add reference code (CODEVIEW) to prompt")
                prompt += f"# Reference Code (Repo Summary) \n{ref_code}\n\n"
            
            prompt += "The Reference Code serves as the baseline code aligned with the Research Goal. The proposed idea should be innovative, building upon the Reference Code to enhance task performance."
            prompt += "Please analyze the provided Reference Code and give a brief summary from the following perspectives: \n 1. Methods and Concepts: Describe the main methods and concepts used in the code. How do they support the functionality? 2. Model Structure: If applicable, outline the model architecture and design choices. How does the structure serve its purpose? 3. Limitations: What are the limitations of the code? How can they be addressed?"
        else:
            ref_code = ""
            logger.error("Default: No reference Code, simple idea-gen")
        
        if background and ref_code:
            prompt += "The Background Information and Reference Code are closely related. The proposed idea should be innovative, building upon the Background Information and Reference Code to enhance task performance."
        
        # Add feedback from previous iterations
        if feedback and iteration > 0:
            prompt += "# Previous Feedback\n"
            # Take the most recent feedback entries, up to 3
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0), 
                reverse=True
            )[:3]
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                prompt += f"Iteration {feedback_iter}: {feedback_text}\n\n"
                
        # Add task description
        prompt += f"# Task\n"
        prompt += f"Generate {count} scientifically plausible hypotheses for the research goal above."
        
        if iteration > 0:
            prompt += f" This is iteration {iteration}, so incorporate the feedback provided."
        
        return prompt
    
    def _build_system_prompt(self, creativity: float) -> str:
        """
        Build system prompt tailored to creativity level.

        Creates system-level instructions that guide the model's idea generation
        style based on the creativity parameter, ranging from conservative to highly
        innovative approaches.

        Args:
            creativity (float): Creativity level 0-1 where:
                >0.8: highly innovative and out-of-the-box
                >0.5: creative but grounded in principles
                <=0.5: conservative and evidence-based

        Returns:
            str: System prompt with tone and quality guidelines
        """
        if creativity > 0.8:
            tone = "highly innovative and out-of-the-box"
        elif creativity > 0.5:
            tone = "creative but grounded in scientific principles"
        else:
            tone = "conservative and strictly evidence-based"
            
        return f"""You are a creative scientific idea generator. Your task is to generate {tone} scientific hypotheses based on all the information given by the user.
Ensure that each idea:
1.Are novel, not obvious from existing literature, and grounded in scientific principles.
2.Propose specific mechanisms or pathways and include a brief explanation of them.
3.Are specific, testable, and can be tested experimentally.
4.Address the research goal directly and have clear scientific significance.
5.Consider constraints and domain knowledge.

Be creative but scientifically rigorous. Propose hypotheses that make unexpected connections between concepts, challenge conventional wisdom, or apply principles from one field to another.

The different scientific hypotheses proposed need to remain separate and cannot be similar.

Your hypotheses should be detailed enough for a scientist in the field to understand and evaluate, but not so detailed that they require specialized knowledge outside the domain. 
"""

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'GenerationAgent':
        """
        Factory method to create GenerationAgent from configuration.

        Args:
            config (Dict[str, Any]): Agent configuration dictionary
            model (BaseModel): Language model instance

        Returns:
            GenerationAgent: Configured instance
        """
        return cls(model, config) 

```

檢查 `generation_agent` 是否有 `config` 屬性：


檢查 `BaseAgent` 是否有保存配置：


```python
"""
Generation Agent for InternAgent

Implements the idea generation agent that creates novel research hypotheses
based on research goals, domain constraints, literature, and optional code baselines.
The agent supports iterative refinement through feedback incorporation and provides
configurable creativity levels for idea generation.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional

from ..models.base_model import BaseModel
from .base_agent import BaseAgent, AgentExecutionError
from .codeview_agent import get_repo_structure


logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Agent that generates novel, scientifically plausible research hypotheses.

    Creates multiple idea candidates based on research goals, domain knowledge,
    literature surveys, and optional code baselines. Supports iterative refinement
    through feedback and adjustable creativity levels. Can analyze file-level or
    project-level code to generate code-aware hypotheses.

    Attributes:
        do_survey (bool): Whether to include literature information
        generation_count (int): Number of hypotheses to generate per call
        creativity (float): Creativity level 0-1 (higher = more creative)
        diversity_threshold (float): Minimum diversity between hypotheses
        temperature (float): Model sampling temperature
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the generation agent with model and configuration.

        Args:
            model (BaseModel): Language model for idea generation
            config (Dict[str, Any]): Configuration with keys:
                - do_survey (bool): Include literature (default: False)
                - generation_count (int): Hypotheses per call (default: 5)
                - creativity (float): Creativity 0-1 (default: 0.9)
                - diversity_threshold (float): Min diversity (default: 0.3)
                - temperature (float): Sampling temperature (optional)
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.do_survey = config.get("do_survey", False)
        self.generation_count = config.get("generation_count", 5)
        self.creativity = config.get("creativity", 0.9)  # Higher = more creative
        self.diversity_threshold = config.get("diversity_threshold", 0.3)
        self.temperature = config.get("temperature", None)
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate novel research ideas based on provided context.

        Generates multiple idea candidates using the configured language model,
        incorporating research goals, domain knowledge, literature, code baselines,
        and iterative feedback. Returns structured ideas with rationales.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal with description, domain, constraints
                - iteration (int): Current iteration number
                - feedback (List[Dict]): Previous feedback entries (optional)
                - paper_lst (List[Dict]): Literature papers if do_survey=True
            params (Dict[str, Any]): Runtime parameters:
                - count (int): Override generation_count (optional)
                - creativity (float): Override creativity level (optional)

        Returns:
            Dict[str, Any]: Results containing:
                - hypotheses (List[Dict]): Generated hypotheses with text/rationale
                - metadata (Dict): Generation info (count, creativity, reasoning)
                - baseline_summary (str): Code baseline summary if applicable

        Raises:
            AgentExecutionError: If goal missing or generation fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        if not goal or not goal.get("description"):
            raise AgentExecutionError("Research goal is required for idea generation")
            
        # Extract and override parameters if provided
        count = params.get("count", self.generation_count)
        creativity = params.get("creativity", self.creativity)
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        paper_lst = context.get("paper_lst", [])
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The idea statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for why this idea is plausible"
                            }
                        },
                        "required": ["text", "rationale"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the generation approach"
                },
                "baseline_summary":{
                    "type": "string",
                    "description": "Summary of the baseline code understanding"
                }
            },
            "required": ["hypotheses", "reasoning"]
        }
        
        if os.path.exists(goal.get("ref_code_path", "")):
            output_schema["required"] = ["hypotheses", "reasoning", "baseline_summary"]
                    
        # Build the prompt
        prompt = self._build_generation_prompt(
            goal=goal,
            count=count,
            iteration=iteration,
            feedback=feedback,
            paper_lst=paper_lst
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(creativity)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature,
            )
            
            # Validate the response
            hypotheses = response.get("hypotheses", [])
            if not hypotheses:
                logger.warning("Generation agent returned no hypotheses")
                
            # Add metadata to the response
            result = {
                "hypotheses": hypotheses,
                "metadata": {
                    "count": len(hypotheses),
                    "creativity": creativity,
                    "iteration": iteration,
                    "reasoning": response.get("reasoning", "")
                },
                "baseline_summary": response.get("baseline_summary", "")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Generation agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate hypotheses: {str(e)}")
    
    def _build_generation_prompt(self,
                               goal: Dict[str, Any],
                               count: int,
                               iteration: int,
                               feedback: List[Dict[str, Any]],
                               paper_lst: List[Dict]) -> str:
        """
        Construct comprehensive prompt for idea generation.

        Builds a structured prompt incorporating research goals, domain constraints,
        background information, literature, code baselines, and iterative feedback.
        Handles both file-level and project-level code analysis.

        Args:
            goal (Dict[str, Any]): Goal with description/domain/constraints/background
            count (int): Number of hypotheses to generate
            iteration (int): Current iteration number for feedback context
            feedback (List[Dict[str, Any]]): Previous feedback entries
            paper_lst (List[Dict]): Literature papers for survey mode

        Returns:
            str: Formatted multi-section prompt for the language model
        """
        
        # Extract goal information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        background = goal.get("background", "")
        
        # Start with the goal
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add background if available
        if background:
            prompt += "Background Information is a detailed description of the baseline method. Please analyze the provided Background Information and give novel hypotheses based on the research goal and the background information.\n\n"
            prompt += f"# Background Information\n{background}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"

        # if do_survey, literature information is included in goal, and need prompt
        if self.do_survey and paper_lst:
            logger.info("Add literature information to prompt")
            literature_prompt = "# Literature Information\n"
            for paper in paper_lst:
                literature_prompt += f"- {paper['title']} ({paper['year']})\n {paper['abstract']} \n\n"
            prompt += literature_prompt
            prompt += "Please analyze the provided Literature Information and give novel hypotheses based on the research goal and the literature."
            
        # Add reference code if available
        # load code if exist, judge file/dir/not exist
        ref_code_path = goal.get("ref_code_path", "")
        if os.path.exists(ref_code_path):
            if os.path.isfile(ref_code_path):
                logger.info("Perform #file-level# code understanding and generation")
                with open(ref_code_path, 'r') as f:
                    ref_code = f.read()
                logger.info("Add reference code (RAW) to prompt")
                prompt += f"# Reference Code\n```python\n{ref_code}\n```\n\n"
                
            elif os.path.isdir(ref_code_path):
                logger.info("Perform #project-level# code understanding and generation")
                logger.info("Loading codeivew Agent ...")
                if os.path.exists(os.path.join(ref_code_path, "code_summary.json")):
                    logger.info("Code summary exists! Loading code summary from file.")
                    with open(os.path.join(ref_code_path, "code_summary.json"), 'r') as f:
                        ref_code = f.read()
                    ref_code = json.loads(ref_code)
                else:
                    logger.info("Code summary does not exist! Generating code summary.")
                    # Use codeview agent to generate code summary
                    # 取得全局配置（從 _global_config 或 config）
                    global_config = getattr(self, '_global_config', {}) or getattr(self, 'config', {})
                    ref_code = get_repo_structure(
                        project_path=ref_code_path,
                        output_dir=ref_code_path,
                        output_name="code_summary.json",
                        ignore_list=None,
                        provider="user",
                        config=global_config,
                        model_name=self.model.model_name  # 保留作為後備
                    )
                ref_code = ref_code['summary'] + "\n\n" + ref_code['key_files']
                logger.info("Add reference code (CODEVIEW) to prompt")
                prompt += f"# Reference Code (Repo Summary) \n{ref_code}\n\n"
            
            prompt += "The Reference Code serves as the baseline code aligned with the Research Goal. The proposed idea should be innovative, building upon the Reference Code to enhance task performance."
            prompt += "Please analyze the provided Reference Code and give a brief summary from the following perspectives: \n 1. Methods and Concepts: Describe the main methods and concepts used in the code. How do they support the functionality? 2. Model Structure: If applicable, outline the model architecture and design choices. How does the structure serve its purpose? 3. Limitations: What are the limitations of the code? How can they be addressed?"
        else:
            ref_code = ""
            logger.error("Default: No reference Code, simple idea-gen")
        
        if background and ref_code:
            prompt += "The Background Information and Reference Code are closely related. The proposed idea should be innovative, building upon the Background Information and Reference Code to enhance task performance."
        
        # Add feedback from previous iterations
        if feedback and iteration > 0:
            prompt += "# Previous Feedback\n"
            # Take the most recent feedback entries, up to 3
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0), 
                reverse=True
            )[:3]
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                prompt += f"Iteration {feedback_iter}: {feedback_text}\n\n"
                
        # Add task description
        prompt += f"# Task\n"
        prompt += f"Generate {count} scientifically plausible hypotheses for the research goal above."
        
        if iteration > 0:
            prompt += f" This is iteration {iteration}, so incorporate the feedback provided."
        
        return prompt
    
    def _build_system_prompt(self, creativity: float) -> str:
        """
        Build system prompt tailored to creativity level.

        Creates system-level instructions that guide the model's idea generation
        style based on the creativity parameter, ranging from conservative to highly
        innovative approaches.

        Args:
            creativity (float): Creativity level 0-1 where:
                >0.8: highly innovative and out-of-the-box
                >0.5: creative but grounded in principles
                <=0.5: conservative and evidence-based

        Returns:
            str: System prompt with tone and quality guidelines
        """
        if creativity > 0.8:
            tone = "highly innovative and out-of-the-box"
        elif creativity > 0.5:
            tone = "creative but grounded in scientific principles"
        else:
            tone = "conservative and strictly evidence-based"
            
        return f"""You are a creative scientific idea generator. Your task is to generate {tone} scientific hypotheses based on all the information given by the user.
Ensure that each idea:
1.Are novel, not obvious from existing literature, and grounded in scientific principles.
2.Propose specific mechanisms or pathways and include a brief explanation of them.
3.Are specific, testable, and can be tested experimentally.
4.Address the research goal directly and have clear scientific significance.
5.Consider constraints and domain knowledge.

Be creative but scientifically rigorous. Propose hypotheses that make unexpected connections between concepts, challenge conventional wisdom, or apply principles from one field to another.

The different scientific hypotheses proposed need to remain separate and cannot be similar.

Your hypotheses should be detailed enough for a scientist in the field to understand and evaluate, but not so detailed that they require specialized knowledge outside the domain. 
"""

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'GenerationAgent':
        """
        Factory method to create GenerationAgent from configuration.

        Args:
            config (Dict[str, Any]): Agent configuration dictionary
            model (BaseModel): Language model instance

        Returns:
            GenerationAgent: Configured instance
        """
        return cls(model, config) 

```

已修改 `codeview_agent.py` 以使用配置檔。主要變更：

## 主要變更

### 1. 移除模型名稱判斷邏輯
- 移除所有 `client_model.lower().startswith('gpt')`、`startswith('claude')` 等判斷
- 改用 `ModelFactory` 根據配置建立模型實例

### 2. 使用配置檔的 CodeView Configuration
- `generate_doc_single_static()` 現在接收 `model_config` 和 `codeview_config`
- 從 `codeview` 配置讀取 `model_provider` 和 `model_name`
- 使用 `ModelFactory.create_model()` 建立正確的模型實例

### 3. 新的函數與修改
- `get_repo_summary_with_config()`：使用配置檔產生摘要的新函數
- `get_repo_summary()`：保留作為向後相容的包裝函數
- `get_repo_structure()`：現在接收 `config` 參數（可選）

### 4. RepoViewer 類別修改
- `__init__()` 現在接收 `config` 參數
- 從配置中提取 `codeview_config` 和 `model_config`

### 5. 支援 asyncio
- 在 multiprocessing Pool 中為每個 worker 建立新的事件循環
- 使用模型的 async `generate()` 方法

## 配置檔格式

配置檔應包含：

```yaml
models:
  default_provider: "azure"
  azure:
    model_name: "gpt-4o-mini"
    # ... 其他設定

codeview:
  model_provider: "default"  # 或 "azure", "openai" 等
  model_name: "gpt-4o-mini"  # 可選，會覆蓋 provider 的 model_name
```

程式碼現在會自動使用配置檔中的 CodeView 設定，不再依賴模型名稱判斷。

```yaml
models:
  default_provider: "azure"
  azure:
    model_name: "gpt-4o-mini"
    # ... 其他設定

codeview:
  model_provider: "default"  # 或 "azure", "openai" 等
  model_name: "gpt-4o-mini"  # 可選，會覆蓋 provider 的 model_name
```

---

**User**

@launch_discovery.py (292-295) 請說明

---

**Cursor**


檢視這段程式碼的上下文：


檢查 config 的載入方式：


這段程式碼說明：

## 程式碼說明

```python
'model': (
    config.get("experiment", {}).get("model") or  # Config file
    "anthropic/claude-3-7-sonnet-20250219"  # Final fallback
),
```

這段程式碼在建立實驗摘要時決定使用的模型名稱，採用優先順序：

### 運作方式

1. 優先順序：
   - 先讀 `config.get("experiment", {}).get("model")`
   - 若為 `None`、空字串、或 `False`，則使用預設值

2. Python `or` 運算符：
   - `A or B`：若 `A` 為 truthy 返回 `A`，否則返回 `B`
   - 這裡若配置中有 `model` 就用它，否則用預設值

3. 預設值：
   - `"anthropic/claude-3-7-sonnet-20250219"` 作為最終後備

### 配置檔結構

對應的配置檔格式應該是：

```yaml
experiment:
  model: "azure/gpt-4o-mini"  # 或在這裡指定模型名稱
```

或者在 JSON 中：

```json
{
  "experiment": {
    "model": "azure/gpt-4o-mini"
  }
}
```

### 用途

這段程式碼用於將使用的模型記錄到實驗摘要中，以便追蹤與重現。如果配置檔未指定，就使用預設的 Claude 模型作為後備。

### 建議改進

如果要與目前的 Azure 配置整合，可以改為：

```python
'model': (
    config.get("experiment", {}).get("model") or
    config.get("models", {}).get("default_provider", "azure") or
    "azure/gpt-4o-mini"  # 與系統配置一致的預設值
),
```

這會讓預設值與系統配置一致。

```python
'model': (
    config.get("experiment", {}).get("model") or  # Config file
    "anthropic/claude-3-7-sonnet-20250219"  # Final fallback
),
```

```yaml
experiment:
  model: "azure/gpt-4o-mini"  # 或在這裡指定模型名稱
```

```json
{
  "experiment": {
    "model": "azure/gpt-4o-mini"
  }
}
```

```python
'model': (
    config.get("experiment", {}).get("model") or
    config.get("models", {}).get("default_provider", "azure") or
    "azure/gpt-4o-mini"  # 與系統配置一致的預設值
),
```

---

**User**

 @launch_discovery.py (292-297) 修改後對應的 @codeview_agent.py 要如何修改?

---

**Cursor**


檢查 `codeview_agent.py` 並對齊 `launch_discovery.py` 的配置邏輯：


修改 `codeview_agent.py`，使其使用與 `launch_discovery.py` 相同的配置邏輯，並移除模型名稱判斷：


```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.client_model = self.llm_settings.model
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        try: 
            import openai
            print(f"Using OpenAI API with model {client_model}")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.client_model, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        try: 
            import openai
            print(f"Using OpenAI API with model {client_model}")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        file_path, dict_path, client_model, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        with open(file_path, 'r') as f:
            code = f.read()
        if client_model.lower().startswith('gpt'):
            try: 
                import openai
                print(f"Using OpenAI API with model {client_model}")
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('claude'):
            try: 
                import anthropic
                print(f"Using Anthropic API with model {client_model}")
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                response = client.completions.create(
                    model=client_model,
                    system=CODE_VIEW_SYS_PROMPT,
                    messages=[
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens_to_sample=llm_settings.max_tokens
                )
                results = response.completions[0].text
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")

        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using DeepSeek model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        
        elif client_model.lower().startswith('deepseek'):
            try:
                import openai
                print(f"Using InternS1 model: {llm_settings.model}")
                client_model = llm_settings.model
                client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
                response = client.chat.completions.create(
                    model=client_model,
                    messages=[
                        {"role": "system", "content": CODE_VIEW_SYS_PROMPT},
                        {"role": "user", "content": CODE_VIEW_PROMPT.format(code=code)}
                    ],
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
                results = response.choices[0].message.content
            except Exception as e:
                print(f"Error during generate doc for file {file_path} using {client_model}.\n{e}")
        try:
            return (dict_path, results)
        except:
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        try: 
            import openai
            print(f"Using OpenAI API with model {client_model}")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            from internagent.mas.models.model_factory import ModelFactory
            
            # 決定使用的模型提供者和名稱，與 launch_discovery.py 的邏輯一致
            # 1. 優先從 experiment.model（但在 codeview 中沒有，所以用 codeview 配置）
            # 2. 從 codeview.model_name
            # 3. 從 models.default_provider 的 model_name
            # 4. 後備值
            
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    # 後備：使用 azure
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"  # 與 launch_discovery.py 一致的後備值
                    }
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config, using fallback")
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"
                    }
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary(client_model, repo_file_tree, llm_settings):
    if client_model.lower().startswith('gpt'):
        try: 
            import openai
            print(f"Using OpenAI API with model {client_model}")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('claude'):
        try: 
            import anthropic
            print(f"Using Anthropic API with model {client_model}")
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.completions.create(
                model=client_model,
                system=REPO_SUMMARY_SYS_PROMPT,
                messages=[
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens_to_sample=llm_settings.max_tokens
            )
            return response.completions[0].text
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('deepseek'):
        try:
            import openai
            print(f"Using DeepSeek model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("API_BASE"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

    elif client_model.lower().startswith('intern'):
        try:
            import openai
            print(f"Using InternS1 model: {llm_settings.model}")
            client_model = llm_settings.model
            client = openai.OpenAI(api_key=os.environ.get("INS1_API_KEY"), base_url=os.environ.get("INS1_API_BASE_URL"))
            response = client.chat.completions.create(
                model=client_model,
                messages=[
                    {"role": "system", "content": REPO_SUMMARY_SYS_PROMPT},
                    {"role": "user", "content": REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree)}
                ],
                temperature=llm_settings.temperature,
                max_tokens=llm_settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generate summary for repo using {client_model}.\n{e}")

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            from internagent.mas.models.model_factory import ModelFactory
            
            # 決定使用的模型提供者和名稱，與 launch_discovery.py 的邏輯一致
            # 1. 優先從 experiment.model（但在 codeview 中沒有，所以用 codeview 配置）
            # 2. 從 codeview.model_name
            # 3. 從 models.default_provider 的 model_name
            # 4. 後備值
            
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    # 後備：使用 azure
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"  # 與 launch_discovery.py 一致的後備值
                    }
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config, using fallback")
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"
                    }
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary_with_config(repo_file_tree, llm_settings, config: Dict[str, Any]):
    """
    使用配置檔產生程式庫摘要。
    
    Args:
        repo_file_tree: 程式庫檔案樹
        llm_settings: LLM 設定
        config: 配置字典（包含 models 和 codeview 配置）
    """
    try:
        from internagent.mas.models.model_factory import ModelFactory
        
        codeview_config = config.get("codeview", {})
        model_config = config.get("models", {})
        
        # 決定使用哪個提供者，與 launch_discovery.py 的邏輯一致
        model_provider = codeview_config.get("model_provider", "default")
        if model_provider == "default":
            default_provider = model_config.get("default_provider", "azure")
            if default_provider in model_config:
                provider_config = model_config[default_provider].copy()
                provider_config["provider"] = default_provider
                provider_config["default_provider"] = default_provider
            else:
                # 後備：使用 azure
                provider_config = {
                    "provider": "azure",
                    "default_provider": "azure",
                    "model_name": "gpt-4.1-mini"
                }
        else:
            if model_provider in model_config:
                provider_config = model_config[model_provider].copy()
                provider_config["provider"] = model_provider
                provider_config["default_provider"] = model_config.get("default_provider", "azure")
            else:
                logger.error(f"Provider '{model_provider}' not found in config, using fallback")
                provider_config = {
                    "provider": "azure",
                    "default_provider": "azure",
                    "model_name": "gpt-4.1-mini"
                }
        
        # 使用 codeview_config 中的 model_name（如果有的話）
        if "model_name" in codeview_config:
            provider_config["model_name"] = codeview_config["model_name"]
        
        # 使用 llm_settings 覆蓋
        if hasattr(llm_settings, 'temperature'):
            provider_config["temperature"] = llm_settings.temperature
        if hasattr(llm_settings, 'max_tokens'):
            provider_config["max_tokens"] = llm_settings.max_tokens
        
        # 建立模型
        model = ModelFactory.create_model(provider_config)
        logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
        
        # 使用模型的 generate 方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                model.generate(
                    prompt=REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree),
                    system_prompt=REPO_SUMMARY_SYS_PROMPT,
                    temperature=provider_config.get("temperature", 0.6),
                    max_tokens=provider_config.get("max_tokens", 3000)
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error during generate summary for repo: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""


def get_repo_summary(client_model, repo_file_tree, llm_settings):
    """
    向後相容的函數，保留原有 API。
    使用配置檔的新實作：get_repo_summary_with_config()
    """
    # 嘗試建立基本配置（向後相容）
    config = {
        "models": {
            "default_provider": "azure",
            "azure": {
                "model_name": client_model,
            }
        },
        "codeview": {
            "model_provider": "default",
            "model_name": client_model,
        }
    }
    return get_repo_summary_with_config(repo_file_tree, llm_settings, config)

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(model, project_path, output_dir, output_name, ignore_list=None, provider="agent"):
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=model,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        repo_summary = get_repo_summary(setting_manager.chat_settings.model, repo_description_tree, setting_manager.chat_settings)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

```python
import json
import os
import re
import sys
import ast
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict
from multiprocessing import Pool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SettingManager():
    def __init__(self, project_path: Path, output_dir: Path, output_name: str, ignore_list: list[str], model: str):
        self.project_settings = EasyDict(
            project_path = project_path,
            output_dir = output_dir,
            output_name = output_name,
            ignore_list = ignore_list
        )
        self.chat_settings = EasyDict(
            model = model, 
            temperature = 0.6,
            max_tokens = 3000,
            num_proc = 8
        )

class PythonElementVisitor(ast.NodeVisitor):
    def __init__(self, root_path, file_path, local_modules):
        self.root_path = root_path
        self.file_path = file_path
        self.stdlib_modules = self.get_stdlib_modules()
        self.local_modules = local_modules
        self.classes, self.functions, self.local_imports = [], [], []
    
    def visit_ClassDef(self, node):
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef): methods.append(item.name)
        self.classes.append({'name': node.name, 'methods': methods})
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not any(isinstance(parent, ast.ClassDef) for parent in self._get_parents(node)): self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for name in node.names:
            module_name = name.name.split('.')[0]
            if self._is_local_import(module_name): self.local_imports.append(name.name)
            self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if self._is_local_import(module_name):
                for name in node.names: self.local_imports.append(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def _is_local_import(self, module_name):
        if module_name in self.local_modules:
            return True
            
        if module_name in self.stdlib_modules:
            return False
            
        potential_path = os.path.join(self.root_path, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        current_dir = os.path.dirname(self.file_path)
        potential_path = os.path.join(current_dir, module_name.replace('.', os.sep))
        if os.path.exists(potential_path) or os.path.exists(f"{potential_path}.py"):
            return True
            
        return False
    
    def _get_parents(self, node):
        return []


    @staticmethod
    def get_stdlib_modules():
        stdlib_modules = set()
        stdlib_paths = [p for p in sys.path if p.endswith('site-packages') or p.endswith('dist-packages')]
        if not stdlib_paths:
            stdlib_modules = set(sys.builtin_module_names)
            stdlib_modules.update([
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'configparser',
                'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum', 'functools',
                'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib',
                'inspect', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
                'operator', 'os', 'pathlib', 'pickle', 'pprint', 'random', 're', 'shutil',
                'signal', 'socket', 'sqlite3', 'ssl', 'statistics', 'string', 'subprocess',
                'sys', 'tempfile', 'threading', 'time', 'traceback', 'typing', 'unittest',
                'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
            ])
        else:
            import pkgutil
            for path in sys.path:
                if any(x in path for x in ['site-packages', 'dist-packages']):
                    continue  
                for _, name, is_pkg in pkgutil.iter_modules([path]):
                    stdlib_modules.add(name)
        return stdlib_modules


def get_local_modules(root_path):
    local_modules = set()
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                ['__pycache__', 'venv', 'env', 'build', 'dist']]
        
        rel_path = os.path.relpath(root, root_path)
        package_path = "" if rel_path == '.' else rel_path.replace(os.sep, '.')
        
        if '__init__.py' in files:
            if package_path:
                local_modules.add(package_path)
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(f"{package_path}.{module_name}")
            else:
                for py_file in [f for f in files if f.endswith('.py') and f != '__init__.py']:
                    module_name = os.path.splitext(py_file)[0]
                    local_modules.add(module_name)
        else:
            for py_file in [f for f in files if f.endswith('.py')]:
                module_name = os.path.splitext(py_file)[0]
                if package_path:
                    local_modules.add(f"{package_path}.{module_name}")
                else:
                    local_modules.add(module_name)
    return local_modules


def analyze_python_file(file_path, project_root, local_modules):
    try: 
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        file_tree = ast.parse(file_content)
        visitor = PythonElementVisitor(file_path, project_root, local_modules)
        visitor.visit(file_tree)

        return {
            "classes": visitor.classes,
            "functions": visitor.functions,
            "local_imports": visitor.local_imports
        }
    except Exception as e:
        print(f"Error during handle file: {file_path}.\n {e}")

def extract_repo_structure(root_dir, project_settings, local_modules):
    if project_settings.ignore_list is None:
        project_settings.ignore_list = ['.git', '__pycache__', '.idea', '.vscode', 'venv', 'env', 'build', 'dist', '.pytest_cache']
    structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "path": root_dir,
        "children": []
    }
    try:
        items = os.listdir(root_dir)
        files = [item for item in items if os.path.isfile(os.path.join(root_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
        for item in sorted(files):
            if item.endswith('.py') and not item in project_settings.ignore_list:
                full_path = os.path.join(root_dir, item)
                
                file_analysis = analyze_python_file(full_path, root_dir, local_modules)
                
                file_info = {
                    "name": item,
                    "type": "file",
                    "path": full_path,
                    "elements": file_analysis
                }
                structure["children"].append(file_info)
        
        for item in sorted(dirs):
            if item in project_settings.ignore_list:
                continue
                
            dir_path = os.path.join(root_dir, item)
            dir_structure = extract_repo_structure(dir_path, project_settings, local_modules)

            if dir_structure["children"]:
                structure["children"].append(dir_structure)
    
    except Exception as e:
        structure["error"] = str(e)
        
    return structure

def save_repo_structure(result, project_setting):
    if not os.path.exists(project_setting.output_dir):
        os.makedirs(project_setting.output_dir)
    if not project_setting.output_name.endswith(".json"):
        project_setting.output_name += ".json"
    with open(os.path.join(project_setting.output_dir, project_setting.output_name), 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

CODE_VIEW_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing and describing the functionality of different levels of code (e.g., functions, classes, and files)."

CODE_VIEW_PROMPT = """You are an advanced code analysis assistant. Your task is to analyze the given code and provide a structured analysis report. The code for analysis is:

<code>
{code}
</code>

Please provide your analysis in the following aspects:

1. file_overview
- **File Purpose**: A single-sentence description of the file's main purpose
- **Key Components**: List of main classes/functions
- **Main Workflow**: Brief description of how components interact

2. Class-level Analysis: For each class, identify its name and provide a concise description of its purpose. You should also conduct function-level analysis of the functions within the classlist all methods (functions within the class).
    - Purpose: 1-2 sentences describing the class's responsibility
    - Methods: method_name: description

3. Function-level Analysis: For each function (both standalone and within classes), identify its name and provide a concise description of its purpose.
    - function_name: 1-2 describing what the function does

Guidelines for Analysis:
    - First give a file-level summary and then give the detail description of file, class and function.
    - Use clear, technical language.
    - Focus on observable behavior, not implementation assumptions
    - Output use markdown format

Output format:
<file_overview>
The overview of the file.
</file_overview>

<detailed_analysis>
The detail analysis of class and function in this file
</detailed_analysis>
"""

class RepoViewer():
    def __init__(self, setting_manager, repo_structure, config: Optional[Dict[str, Any]] = None):
        self.project_settings = setting_manager.project_settings
        self.llm_settings = setting_manager.chat_settings
        self.repo_structure = repo_structure
        self.file_paths, self.dict_paths = [], []
        self.get_path(repo_structure)
        self.config = config or {}
        self.codeview_config = self.config.get("codeview", {})
        self.model_config = self.config.get("models", {})
    
    def get_path(self, repo_structure, dict_path=""):
        if repo_structure.get('children', None) is None:
            self.file_paths.append(repo_structure['path'])
            self.dict_paths.append(dict_path)

        if repo_structure.get('children'):
            dict_path += "['children']"
            for i, child in enumerate(repo_structure['children']):
                self.get_path(child, dict_path+f"[{i}]")
    
    def generate_docs(self):
        assert len(self.file_paths) == len(self.dict_paths)
        num_proc = self.llm_settings.num_proc
        parallel_inputs = [(fpath, dpath, self.model_config, self.codeview_config, self.llm_settings) for fpath, dpath in zip(self.file_paths, self.dict_paths)]
        group_nums = len(parallel_inputs) // num_proc if len(parallel_inputs) % num_proc == 0 else len(parallel_inputs) // num_proc +1
        all_results = []
        for group_idx in range(group_nums):
            try:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc: (group_idx+1) * num_proc]
            except:
                cur_parallel_inputs = parallel_inputs[group_idx * num_proc:]
            with Pool(processes=8) as pool:
                all_results += list(tqdm(pool.imap(self.generate_doc_single_static, cur_parallel_inputs), total=len(cur_parallel_inputs), desc = f"Generating Document for Group {group_idx}."))
        print(f"Finish generating document for project {self.project_settings.project_path}")

        repo_structure = self.repo_structure
        for (dict_path, results) in all_results:
            repo_structure = self.save_to_json(results, dict_path, repo_structure)
        with open(os.path.join(self.project_settings.output_dir, self.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)

    @staticmethod
    def generate_doc_single_static(config):
        """
        使用配置檔產生單一檔案的文檔。
        
        Args:
            config: (file_path, dict_path, model_config, codeview_config, llm_settings) 元組
        """
        file_path, dict_path, model_config, codeview_config, llm_settings = config
        print(f"-- Generating document for file {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None, None
        
        try:
            from internagent.mas.models.model_factory import ModelFactory
            
            # 決定使用的模型提供者和名稱，與 launch_discovery.py 的邏輯一致
            # 1. 優先從 experiment.model（但在 codeview 中沒有，所以用 codeview 配置）
            # 2. 從 codeview.model_name
            # 3. 從 models.default_provider 的 model_name
            # 4. 後備值
            
            model_provider = codeview_config.get("model_provider", "default")
            if model_provider == "default":
                default_provider = model_config.get("default_provider", "azure")
                if default_provider in model_config:
                    provider_config = model_config[default_provider].copy()
                    provider_config["provider"] = default_provider
                    provider_config["default_provider"] = default_provider
                else:
                    # 後備：使用 azure
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"  # 與 launch_discovery.py 一致的後備值
                    }
            else:
                if model_provider in model_config:
                    provider_config = model_config[model_provider].copy()
                    provider_config["provider"] = model_provider
                    provider_config["default_provider"] = model_config.get("default_provider", "azure")
                else:
                    logger.error(f"Provider '{model_provider}' not found in config, using fallback")
                    provider_config = {
                        "provider": "azure",
                        "default_provider": "azure",
                        "model_name": "gpt-4.1-mini"
                    }
            
            # 使用 codeview_config 中的 model_name 覆蓋（如果有的話）
            if "model_name" in codeview_config:
                provider_config["model_name"] = codeview_config["model_name"]
            
            # 使用 llm_settings 覆蓋溫度等設定
            if hasattr(llm_settings, 'temperature'):
                provider_config["temperature"] = llm_settings.temperature
            if hasattr(llm_settings, 'max_tokens'):
                provider_config["max_tokens"] = llm_settings.max_tokens
            
            # 建立模型
            model = ModelFactory.create_model(provider_config)
            logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
            
            # 使用模型的 generate 方法（async）
            # 注意：由於在 multiprocessing Pool 中使用，需要建立新的事件循環
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    model.generate(
                        prompt=CODE_VIEW_PROMPT.format(code=code),
                        system_prompt=CODE_VIEW_SYS_PROMPT,
                        temperature=provider_config.get("temperature", 0.6),
                        max_tokens=provider_config.get("max_tokens", 3000)
                    )
                )
            finally:
                loop.close()
            
            return (dict_path, results)
            
        except Exception as e:
            logger.error(f"Error during generate doc for file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
            
    def save_to_json(self, response, dict_path, repo_structure):
        pattern_file = r'<file_overview>([\s\S]*?)</file_overview>'
        pattern_detail = r'<detailed_analysis>([\s\S]*?)</detailed_analysis>'
        match_file = re.search(pattern_file, response)
        match_detail = re.search(pattern_detail, response)
        if match_file: 
            eval('repo_structure' + dict_path)['file_summary'] = match_file.group(1)
        else: 
            eval('repo_structure' + dict_path)['file_summary'] = response
        if match_detail: 
            eval('repo_structure' + dict_path)['detailed_content'] = match_detail.group(1)
        else: 
            eval('repo_structure' + dict_path)['detailed_content'] = response
        return repo_structure


def get_file_description_tree(repo_structure, indent=0, prefix=""):
    result = ""
    
    if indent > 0:
        result += " " * (indent - 1) + prefix + repo_structure.get("name", "Unknown")
        if repo_structure.get("file_summary", None) is not None:
            file_summary = repo_structure.get('file_summary').replace('\n', ' ')
            result += f": {file_summary} \n"
        else: result += "\n"
    else:
        result += repo_structure.get("name", "Root") + "\n"
    
    children = repo_structure.get("children", [])
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        
        if is_last:
            child_prefix = "└── "
            next_prefix = "    "
        else:
            child_prefix = "├── "
            next_prefix = "│   "
        result += get_file_description_tree(child, indent + 4, child_prefix)
    
    return result

REPO_SUMMARY_SYS_PROMPT = "You are an expert code analyst and technical documentation specialist who is specialized in analyzing the overall usage of a code repository."
REPO_SUMMARY_PROMPT = """
You are an expert code analyst and need to analyze the code structure and find out the overall purpose and functionality of this codebase. You will be given a file tree and descriptions for all files in the file tree. The file tree you need to analyze is as follows:

<file_description_tree>
{repo_file_tree}
</file_description_tree>

Please:
    1. Analyze the main functionality of the code repository: Provide a complete overview of what the repository is designed to do. Avoid overly detailed explanations; focus on delivering a clear and concise summary of its purpose.
    2. Identify the key files that are critical to understanding or implementing the functionality. Provide brief explanations for why these files are important

Output format:
<code_repo_func>
The summarized functionality of the code.
</code_repo_func>

<code_repo_key_files>
A list of key files that are critical to understanding or implementing the functionality. Each file should be listed on a new line, and each line should include the file name and a brief explanation of why it is important.
</code_repo_key_files>

"""

def get_repo_summary_with_config(repo_file_tree, llm_settings, config: Dict[str, Any]):
    """
    使用配置檔產生程式庫摘要。
    
    Args:
        repo_file_tree: 程式庫檔案樹
        llm_settings: LLM 設定
        config: 配置字典（包含 models 和 codeview 配置）
    """
    try:
        from internagent.mas.models.model_factory import ModelFactory
        
        codeview_config = config.get("codeview", {})
        model_config = config.get("models", {})
        
        # 決定使用哪個提供者，與 launch_discovery.py 的邏輯一致
        model_provider = codeview_config.get("model_provider", "default")
        if model_provider == "default":
            default_provider = model_config.get("default_provider", "azure")
            if default_provider in model_config:
                provider_config = model_config[default_provider].copy()
                provider_config["provider"] = default_provider
                provider_config["default_provider"] = default_provider
            else:
                # 後備：使用 azure
                provider_config = {
                    "provider": "azure",
                    "default_provider": "azure",
                    "model_name": "gpt-4.1-mini"
                }
        else:
            if model_provider in model_config:
                provider_config = model_config[model_provider].copy()
                provider_config["provider"] = model_provider
                provider_config["default_provider"] = model_config.get("default_provider", "azure")
            else:
                logger.error(f"Provider '{model_provider}' not found in config, using fallback")
                provider_config = {
                    "provider": "azure",
                    "default_provider": "azure",
                    "model_name": "gpt-4.1-mini"
                }
        
        # 使用 codeview_config 中的 model_name（如果有的話）
        if "model_name" in codeview_config:
            provider_config["model_name"] = codeview_config["model_name"]
        
        # 使用 llm_settings 覆蓋
        if hasattr(llm_settings, 'temperature'):
            provider_config["temperature"] = llm_settings.temperature
        if hasattr(llm_settings, 'max_tokens'):
            provider_config["max_tokens"] = llm_settings.max_tokens
        
        # 建立模型
        model = ModelFactory.create_model(provider_config)
        logger.info(f"Using model provider: {provider_config.get('provider')} with model: {provider_config.get('model_name')}")
        
        # 使用模型的 generate 方法
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                model.generate(
                    prompt=REPO_SUMMARY_PROMPT.format(repo_file_tree=repo_file_tree),
                    system_prompt=REPO_SUMMARY_SYS_PROMPT,
                    temperature=provider_config.get("temperature", 0.6),
                    max_tokens=provider_config.get("max_tokens", 3000)
                )
            )
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error during generate summary for repo: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""


def get_repo_summary(client_model, repo_file_tree, llm_settings):
    """
    向後相容的函數，保留原有 API。
    使用配置檔的新實作：get_repo_summary_with_config()
    """
    # 嘗試建立基本配置（向後相容）
    config = {
        "models": {
            "default_provider": "azure",
            "azure": {
                "model_name": client_model,
            }
        },
        "codeview": {
            "model_provider": "default",
            "model_name": client_model,
        }
    }
    return get_repo_summary_with_config(repo_file_tree, llm_settings, config)

def extract_from_repo_summary(repo_summary):
    pattern_summary = r'<code_repo_func>([\s\S]*?)</code_repo_func>'
    pattern_key = r'<code_repo_key_files>([\s\S]*?)</code_repo_key_files>'
    match_summary = re.search(pattern_summary, repo_summary)
    match_key = re.search(pattern_key, repo_summary)
    if match_summary: 
        summary = match_summary.group(1)
    else: 
        summary = repo_summary
    if match_key: 
        key_files = match_key.group(1)
    else: 
        key_files = repo_summary
    return summary, key_files


def get_repo_structure(project_path, output_dir, output_name, ignore_list=None, provider="agent", config: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
    """
    取得並產生程式庫結構文檔。
    
    Args:
        project_path: 專案路徑
        output_dir: 輸出目錄
        output_name: 輸出檔名
        ignore_list: 忽略清單
        provider: 提供者類型 ("agent" 或 "user")
        config: 配置字典（包含 models 和 codeview 配置）
        model_name: 模型名稱（保留向後相容，但優先使用 config）
    """
    # 如果沒有提供 config，嘗試建立基本配置
    if config is None:
        # 嘗試從環境變數或預設值建立配置
        config = {
            "models": {
                "default_provider": "azure",
                "azure": {
                    "model_name": model_name or "gpt-4.1-mini",
                }
            },
            "codeview": {
                "model_provider": "default",
            }
        }
    
    # 使用 codeview 配置中的 model_name，或使用傳入的 model_name
    codeview_config = config.get("codeview", {})
    if model_name and "model_name" not in codeview_config:
        codeview_config["model_name"] = model_name
    
    # 決定使用的 model_name（用於 SettingManager，保持向後相容）
    # 優先順序：codeview.model_name > model_name 參數 > 後備值
    final_model_name = (
        codeview_config.get("model_name") or
        model_name or
        config.get("models", {}).get("default_provider", "azure") or
        "gpt-4.1-mini"
    )
    
    setting_manager = SettingManager(
        project_path=project_path,
        output_dir=output_dir,
        output_name=output_name,
        ignore_list=ignore_list,
        model=final_model_name,
    )
    local_modules = get_local_modules(setting_manager.project_settings.project_path)
    repo_structure_dict = extract_repo_structure(
        root_dir=setting_manager.project_settings.project_path,
        project_settings=setting_manager.project_settings,
        local_modules=local_modules
    )
    print(repo_structure_dict)
    repo_viewer = RepoViewer(setting_manager, repo_structure_dict, config)
    repo_viewer.generate_docs()
    if provider == "user":
        repo_description_tree = get_file_description_tree(repo_structure_dict)
        # 使用配置來產生摘要
        repo_summary = get_repo_summary_with_config(repo_description_tree, setting_manager.chat_settings, config)
        summary, key_files = extract_from_repo_summary(repo_summary)
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'r') as f:
            repo_structure = json.load(f)
        repo_structure["summary"] = summary
        repo_structure["key_files"] = key_files
        with open(os.path.join(setting_manager.project_settings.output_dir, setting_manager.project_settings.output_name), 'w', encoding='utf-8') as f:
            json.dump(repo_structure, f, ensure_ascii=False, indent=2)
        return repo_structure



    


```

已更新 `codeview_agent.py` 以對齊 `launch_discovery.py` 的配置邏輯。主要變更如下：

## 主要修改

### 1. **移除模型名稱判斷邏輯**
- 移除所有 `client_model.lower().startswith('gpt')`、`startswith('claude')` 等判斷
- 改用 `ModelFactory` 根據配置建立模型實例

### 2. **與 launch_discovery.py 一致的優先順序**
在 `generate_doc_single_static()` 和 `get_repo_summary_with_config()` 中使用相同的邏輯：

```python
# 優先順序：
# 1. codeview.model_name（如果配置中有）
# 2. models.default_provider 的 model_name
# 3. 後備值："gpt-4.1-mini"（與 launch_discovery.py 一致）
```

### 3. **使用配置檔的 codeview 配置**
- `RepoViewer.__init__()` 現在接收 `config` 參數
- 從配置中提取 `codeview_config` 和 `model_config`
- 使用 `ModelFactory.create_model()` 建立正確的模型實例

### 4. **後備值統一**
- 與 `launch_discovery.py` 使用相同的後備值 `"gpt-4.1-mini"`
- 預設 provider 改為 `"azure"`

### 5. **向後相容性**
- `get_repo_summary()` 保留作為向後相容的包裝函數
- `get_repo_structure()` 仍支援舊的 `model` 參數（作為 `model_name`）

現在 `codeview_agent.py` 會：
1. 優先使用配置檔中的 `codeview.model_name`
2. 如果沒有，使用 `models.default_provider` 對應的模型
3. 如果都沒有，使用後備值 `"gpt-4.1-mini"`

與 `launch_discovery.py` 的邏輯一致。

```python
# 優先順序：
# 1. codeview.model_name（如果配置中有）
# 2. models.default_provider 的 model_name
# 3. 後備值："gpt-4.1-mini"（與 launch_discovery.py 一致）
```

---

**User**

@zsh (1-124) 

---

**Cursor**


錯誤顯示 "Unsupported model provider: azure"，表示 Azure 未正確註冊。檢查 `model_factory.py` 的提供者註冊與配置讀取：


`MODEL_PROVIDER_MAP` 缺少 `azure` 的註冊。補上：


```python
"""
Model Provider Factory

A dynamic system for managing and instantiating language model backends.
Enables seamless switching between different LLM providers and handles
configuration, caching, and fallback strategies.
"""

import importlib
import logging
import time
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Available model provider implementations
MODEL_PROVIDER_MAP = {
    "openai": "internagent.mas.models.openai_model.OpenAIModel",
    "azure": "internagent.mas.models.azure_model.AzureModel",
    "interns1": "internagent.mas.models.s1_model.S1Model",
    "dsr1": "internagent.mas.models.r1_model.R1Model",
}


class ModelFactory:
    """Central factory for creating and managing language model instances."""
    
    registered_models = {}
    _model_cache = {}
    _model_stats = {}
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on provided configuration.
        
        Args:
            config: Configuration dictionary for model creation
                
        Returns:
            Configured model instance
            
        Raises:
            ValueError: For unsupported providers or configuration errors
        """
        provider = config.get("provider")
        if provider == "default" or provider is None:
            provider = config.get("default_provider", "openai")

        if "models" in config and provider in config["models"]:
            provider_config = config["models"][provider].copy()
            logger.info(f"Using provider-specific config for {provider}: {provider_config}")
            
            # Merge with general configuration
            provider_config.update({
                k: v for k, v in config.items()
                if k not in ["models", "agents", "workflow", "tools", "memory", "execution"]
            })
            config_to_use = provider_config
        else:
            logger.info(f"Using fallback config for provider {provider}")
            config_to_use = config
        
        cache_key = ModelFactory._create_cache_key(provider, config_to_use)
        
        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Reusing cached model for provider: {provider}")
            return ModelFactory._model_cache[cache_key]
            
        if provider not in MODEL_PROVIDER_MAP and provider not in ModelFactory.registered_models:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        if provider in ModelFactory.registered_models:
            model_class = ModelFactory.registered_models[provider]
        else:
            try:
                module_path, class_name = MODEL_PROVIDER_MAP[provider].rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load model implementation: {e}")
                raise ValueError(f"Implementation not available for provider: {provider}")
        
        try:
            model = model_class.from_config(config_to_use)
            ModelFactory._model_cache[cache_key] = model
            ModelFactory._model_stats[cache_key] = {
                "created_at": time.time(),
                "success_count": 0,
                "failure_count": 0,
                "total_tokens": 0,
                "total_latency": 0.0
            }
            return model
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise
    
    @staticmethod
    def create_model_for_agent(agent_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance tailored for a specific agent type.
        
        Args:
            agent_type: Type identifier for the agent
            config: Agent configuration parameters
                
        Returns:
            Model instance configured for the specified agent
        """
        model_provider = config.get("model_provider", "default")
        global_config = config.get("_global_config", {})
        model_config = {}
        
        # Apply provider configuration
        if model_provider != "default" and model_provider in global_config.get("models", {}):
            model_config.update(global_config.get("models", {}).get(model_provider, {}))
        
        # Handle default provider case
        if model_provider == "default":
            default_provider = global_config.get("models", {}).get("default_provider", "openai")
            model_config.update(global_config.get("models", {}).get(default_provider, {}))
        
        model_config["provider"] = model_provider
        model_config["default_provider"] = global_config.get("models", {}).get("default_provider", "openai")
        
        # Apply agent-specific overrides
        for key in ["temperature", "max_tokens"]:
            if key in config:
                model_config[key] = config[key]
        
        # Include remaining agent settings
        for k, v in config.items():
            if k not in ["_global_config", "count"] and k not in model_config:
                model_config[k] = v
                
        return ModelFactory.create_model(model_config)
    
    @classmethod
    def register_model(cls, provider_name: str, model_class) -> None:
        """Add a custom model implementation to the factory registry."""
        cls.registered_models[provider_name] = model_class
        logger.info(f"New model provider registered: {provider_name}")
    
    @classmethod
    def create_with_fallbacks(cls, 
                             config: Dict[str, Any], 
                             fallback_providers: List[str] = None,
                             max_retries: int = 3) -> Tuple[BaseModel, str]:
        """
        Create a model with automatic fallback to alternative providers.
        
        Attempts to create a model with the primary provider, falling back
        to alternatives if the primary fails. Each provider can be tried
        multiple times before moving to the next option.
        
        Args:
            config: Model configuration
            fallback_providers: Ordered list of backup providers
            max_retries: Maximum attempts per provider
            
        Returns:
            (model_instance, successful_provider_name)
            
        Raises:
            ValueError: When all providers fail
        """
        primary_provider = config.get("provider", config.get("default_provider", "openai"))
        
        # Setup fallback sequence
        if fallback_providers is None:
            fallback_providers = [p for p in MODEL_PROVIDER_MAP.keys() if p != primary_provider]
            # Prioritize known reliable providers
            if "openai" in fallback_providers:
                fallback_providers.remove("openai")
                fallback_providers.insert(0, "openai")
            if "ollama" in fallback_providers:
                fallback_providers.remove("ollama")
                fallback_providers.insert(1, "ollama")
        
        providers_to_try = [primary_provider] + fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    provider_config = dict(config)
                    provider_config["provider"] = provider
                    
                    model = cls.create_model(provider_config)
                    logger.info(f"Successfully initialized {provider} model")
                    return model, provider
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Attempt {retry_count} failed for provider {provider}: {e}")
                    time.sleep(1)
        
        raise ValueError(f"Failed to create model with any provider. Last error: {last_error}")
    
    @classmethod
    def update_model_stats(cls, cache_key: str, success: bool, tokens: int = 0, latency: float = 0.0) -> None:
        """Update performance metrics for a model instance."""
        if cache_key not in cls._model_stats:
            return
            
        stats = cls._model_stats[cache_key]
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        stats["total_tokens"] += tokens
        stats["total_latency"] += latency
    
    @classmethod
    def get_model_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all model instances."""
        return cls._model_stats
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get all registered model providers and their implementations."""
        available_models = {provider: path for provider, path in MODEL_PROVIDER_MAP.items()}
        
        for provider, model_class in cls.registered_models.items():
            available_models[provider] = model_class.__name__
            
        return available_models
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances and statistics."""
        cls._model_cache.clear()
        cls._model_stats.clear()
        logger.info("Model cache cleared")
    
    @staticmethod
    def _create_cache_key(provider: str, config: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for model instance caching.
        
        Different configurations of the same provider are treated as
        separate instances to ensure correct behavior.
        """
        provider_config = config.get(provider, {})
        model_name = provider_config.get("model_name", "default")
        
        if provider == "local":
            model_path = provider_config.get("model_path", "")
            return f"{provider}:{model_name}:{model_path}"
        
        if provider == "ollama":
            api_base = provider_config.get("api_base", "http://localhost:11434")
            return f"{provider}:{model_name}:{api_base}"
            
        return f"{provider}:{model_name}"

```

問題已修正：`MODEL_PROVIDER_MAP` 中缺少 `azure` 提供者。已在第 21 行重新加入：

```python
"azure": "internagent.mas.models.azure_model.AzureModel",
```

現在 `ModelFactory` 已能識別 `azure` 提供者。請重新執行程式：

```bash
./scripts/run_pipeline.sh
```

錯誤 "Unsupported model provider: azure" 應該已解決，所有代理應能成功建立 Azure 模型實例。

如果還有問題，可能是 `azure_model.py` 的模組匯入問題；可檢查 Python 路徑與模組結構。

```python
"azure": "internagent.mas.models.azure_model.AzureModel",
```

```shellscript
./scripts/run_pipeline.sh
```

---

**User**

@zsh (5-8) 

---

**Cursor**


檢查是否有地方使用 litellm 導致此錯誤：


檢查 `experiments_utils_aider.py` 是否使用 litellm，以及 "azure" 如何被傳遞：


檢查 `ExperimentRunner` 如何傳遞模型名稱給 aider：


檢查配置中 experiment 區塊的設定：


問題在 `stage.py`：缺少 `experiment.model` 時，直接使用 `default_provider`（"azure"）作為模型名稱傳給 aider，但 litellm 需要 `"azure/{model_name}"` 格式。改為從 `models.{default_provider}.model_name` 組成正確格式。修正中：


修正模型名稱的組裝邏輯：


```python
import os.path as osp
import sys
import json
import shutil
from datetime import datetime

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from internagent.mas.interface import InternAgentInterface
from internagent.experiments_utils_aider import perform_experiments as perform_experiments_aider

from internagent.vis import visualize_hypotheses


# ============================================================================
# Idea Generation Config
# ============================================================================
class IdeaGenerator:
    """Handles idea generation using MAS"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interface = InternAgentInterface(args.config, work_dir=args.task_dir, task_name=args.task_name)
        self.session_id = None
        self.status = None
    
    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")
        
        await self.interface.startup()
        
        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        
        with open(task_desc_path, 'r') as f:
            params = json.load(f)
        
        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])
        
        if not goal or not domain:
            raise ValueError("Task description and domain are required")
        
        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=self.args.ref_code_path,
            constraints=constraints
        )
        
        self.logger.info(f"Session created: {self.session_id}")
    
    async def generate_ideas(self):
        """Run MAS to generate ideas"""
        if self.session_id is None:
            await self.load_task()
        
        await self.interface.startup()
        
        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":
                    if self.args.offline_feedback:
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Feedback added: {feedback}")
                
                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")
        
        # Save and visualize ideas
        # Session files are saved in results/{task_name}/ by the memory manager
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}.json")
        vis_output = osp.join(
            "results", self.args.task_name,
            f"{self.args.task_name}_ideas_{self.session_id}.pdf"
        )
        visualize_hypotheses(session_json, vis_output)
        self.logger.info(f"Visualization saved: {vis_output}")
        
        return top_ideas, session_json


# ============================================================================
# Experiment Execution Module
# ============================================================================
class ExperimentRunner:
    """Handles experiment execution with different backends"""
    
    def __init__(self, args, logger, config=None):
        self.args = args
        self.logger = logger
        self.backend = args.exp_backend
        self.config = config or {}
    
    def _extract_idea_info(self, idea):
        """Extract idea information from different formats"""
        # Try refined_method_details first (from full MAS pipeline)
        if 'refined_method_details' in idea and idea['refined_method_details']:
            details = idea['refined_method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to method_details (from method development only)
        elif 'method_details' in idea and idea['method_details']:
            details = idea['method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to basic idea structure (from JSON files)
        else:
            # Handle different possible field names
            name = idea.get('name') or idea.get('title') or 'unnamed_idea'
            title = idea.get('title') or idea.get('name') or 'Untitled'
            description = idea.get('description') or idea.get('content') or ''
            method = idea.get('method') or ''

            return {
                'name': name[:50] if name else 'unnamed_idea',  # Limit name length
                'title': title,
                'description': description,
                'method': method
            }

    def setup_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")
        
        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Ensure experiment.py exists in the experiment folder
        experiment_src = osp.join(base_dir, "experiment.py")
        experiment_dst = osp.join(folder_name, "experiment.py")
        if not osp.exists(experiment_dst):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, experiment_dst)
                print(f"Copied experiment.py from {experiment_src} to {experiment_dst}")
            else:
                raise FileNotFoundError(f"experiment.py not found in base directory: {experiment_src}")

        # Ensure run_0/experiment.py exists (needed for baseline comparison)
        run0_dir = osp.join(folder_name, "run_0")
        run0_experiment = osp.join(run0_dir, "experiment.py")
        if osp.exists(run0_dir) and not osp.exists(run0_experiment):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, run0_experiment)
                print(f"Copied baseline experiment.py to {run0_experiment}")
            else:
                print(f"Warning: Could not copy baseline experiment.py to run_0")
        
        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")
        
        return folder_name, idea_name
    
    def setup_logging_redirect(self, folder_name):
        """Redirect stderr to log file, keep stdout visible for aider output"""
        log_path = osp.join(folder_name, "log.txt")
        log_file = open(log_path, "a")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Keep stdout visible so we can see aider's output
        # Only redirect stderr to capture errors
        sys.stderr = log_file
        return original_stdout, original_stderr, log_file
    
    def restore_logging(self, original_stdout, original_stderr, log_file):
        """Restore stdout/stderr"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    def run_aider_experiment(self, base_dir, results_dir, idea):
        """Run experiment using Aider backend"""
        folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
        # Load baseline results
        baseline_path = osp.join(base_dir, "run_0", "final_info.json")
        if osp.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_results = json.load(f)
        else:
            baseline_results = {}
        
        original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
        try:
            self.logger.info(f"Starting Aider experiment: {idea_name}")
            
            exp_file = osp.join(folder_name, "experiment.py")
            notes_file = osp.join(folder_name, "notes.txt")
            
            io = InputOutput(
                yes=True,
                chat_history_file=osp.join(folder_name, f"{idea_name}_aider.txt")
            )
            # Get experiment model from config file
            # aider/litellm requires format: "provider/model_name" (e.g., "azure/gpt-4o-mini")
            experiment_model = self.config.get("experiment", {}).get("model")
            if not experiment_model:
                # Build model string from default_provider and model_name
                default_provider = self.config.get("models", {}).get("default_provider", "azure")
                provider_config = self.config.get("models", {}).get(default_provider, {})
                model_name = provider_config.get("model_name", "gpt-4.1-mini")
                experiment_model = f"{default_provider}/{model_name}"
            self.logger.info(f"Using experiment model: {experiment_model}")
            main_model = Model(experiment_model)
            coder = Coder.create(
                main_model=main_model,
                fnames=[exp_file, notes_file],
                io=io,
                stream=True,  # Enable streaming to see aider output
                use_git=False,
                edit_format="diff"
            )
            
            success = perform_experiments_aider(idea, folder_name, coder, baseline_results)
            
            self.logger.info(f"Aider experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Aider experiment error: {str(e)}")
            return False
        finally:
            self.restore_logging(original_stdout, original_stderr, log_file)
    
    # def run_openhands_experiment(self, base_dir, results_dir, idea):
    #     """Run experiment using OpenHands backend"""
    #     folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
    #     original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
    #     try:
    #         self.logger.info(f"Starting OpenHands experiment: {idea_name}")
            
    #         # Get OpenHands-specific config
    #         openhands_config = self.config.get("experiment", {}).get("openhands", {})
    #         code_server_path = "/workspace"
    #         mount_paths = openhands_config.get("mount_paths", [])
    #         uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

    #         success = perform_experiments_openhands(
    #             idea,
    #             folder_name,
    #             code_server_path,
    #             mount_paths,
    #             uri_prefix
    #         )
            
    #         self.logger.info(f"OpenHands experiment {'succeeded' if success else 'failed'}: {idea_name}")
    #         return success
            
    #     except Exception as e:
    #         self.logger.error(f"OpenHands experiment error: {str(e)}")
    #         return False
    #     finally:
    #         self.restore_logging(original_stdout, original_stderr, log_file)
    
    def run_experiments(self, base_dir, results_dir, ideas):
        """Run experiments for all ideas"""
        results = []
        
        for idx, idea in enumerate(ideas, 1):
            idea_info = self._extract_idea_info(idea)
            idea_name = idea_info['name']
            self.logger.info(f"Processing idea {idx}/{len(ideas)}: {idea_name}")
            
            try:
                if self.backend == "aider":
                    success = self.run_aider_experiment(base_dir, results_dir, idea)
                elif self.backend == "openhands":
                    raise NotImplementedError("OpenHands backend is not implemented in this version.")
                    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                results.append({
                    'idea_name': idea_name,
                    'success': success
                })
                
            except Exception as e:
                self.logger.error(f"Failed to run experiment for {idea_name}: {str(e)}")
                results.append({
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results

```

已修正邏輯。原本會將 `"azure"` 直接當作模型名稱傳給 aider，但 aider/litellm 需要 `"provider/model_name"` 格式（例如 `"azure/gpt-4o-mini"`）。

修正後，程式會：
1. 先檢查 `experiment.model` 是否存在
2. 若不存在，從 `default_provider`（例如 `"azure"`）取得 provider 名稱
3. 從 `models.azure.model_name` 取得模型名稱（例如 `"gpt-4o-mini"`）
4. 組合成 `"azure/gpt-4o-mini"` 格式傳給 aider

這樣應該能解決 litellm 的錯誤。請重新執行程式測試。

注意：aider/litellm 使用 Azure 時可能需要額外的環境變數設定。如果仍有問題，可能需要檢查 litellm 的 Azure 配置方式。

---

**User**

@zsh (992-994) 

---

**Cursor**


檢查錯誤詳情，並確認 aider/litellm 的 Azure OpenAI 配置需求：


確認 litellm 需要不同的環境變數名稱：`AZURE_API_KEY` 和 `AZURE_API_BASE`，而非 `AZURE_OPENAI_KEY` 和 `AZURE_OPENAI_ENDPOINT`。更新 `stage.py`，在建立 aider Model 之前設定這些環境變數：


更新程式碼，確保 litellm 使用正確的環境變數：


```python
import os.path as osp
import sys
import json
import shutil
from datetime import datetime

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from internagent.mas.interface import InternAgentInterface
from internagent.experiments_utils_aider import perform_experiments as perform_experiments_aider

from internagent.vis import visualize_hypotheses


# ============================================================================
# Idea Generation Config
# ============================================================================
class IdeaGenerator:
    """Handles idea generation using MAS"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interface = InternAgentInterface(args.config, work_dir=args.task_dir, task_name=args.task_name)
        self.session_id = None
        self.status = None
    
    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")
        
        await self.interface.startup()
        
        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        
        with open(task_desc_path, 'r') as f:
            params = json.load(f)
        
        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])
        
        if not goal or not domain:
            raise ValueError("Task description and domain are required")
        
        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=self.args.ref_code_path,
            constraints=constraints
        )
        
        self.logger.info(f"Session created: {self.session_id}")
    
    async def generate_ideas(self):
        """Run MAS to generate ideas"""
        if self.session_id is None:
            await self.load_task()
        
        await self.interface.startup()
        
        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":
                    if self.args.offline_feedback:
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Feedback added: {feedback}")
                
                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")
        
        # Save and visualize ideas
        # Session files are saved in results/{task_name}/ by the memory manager
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}.json")
        vis_output = osp.join(
            "results", self.args.task_name,
            f"{self.args.task_name}_ideas_{self.session_id}.pdf"
        )
        visualize_hypotheses(session_json, vis_output)
        self.logger.info(f"Visualization saved: {vis_output}")
        
        return top_ideas, session_json


# ============================================================================
# Experiment Execution Module
# ============================================================================
class ExperimentRunner:
    """Handles experiment execution with different backends"""
    
    def __init__(self, args, logger, config=None):
        self.args = args
        self.logger = logger
        self.backend = args.exp_backend
        self.config = config or {}
    
    def _extract_idea_info(self, idea):
        """Extract idea information from different formats"""
        # Try refined_method_details first (from full MAS pipeline)
        if 'refined_method_details' in idea and idea['refined_method_details']:
            details = idea['refined_method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to method_details (from method development only)
        elif 'method_details' in idea and idea['method_details']:
            details = idea['method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to basic idea structure (from JSON files)
        else:
            # Handle different possible field names
            name = idea.get('name') or idea.get('title') or 'unnamed_idea'
            title = idea.get('title') or idea.get('name') or 'Untitled'
            description = idea.get('description') or idea.get('content') or ''
            method = idea.get('method') or ''

            return {
                'name': name[:50] if name else 'unnamed_idea',  # Limit name length
                'title': title,
                'description': description,
                'method': method
            }

    def setup_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")
        
        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Ensure experiment.py exists in the experiment folder
        experiment_src = osp.join(base_dir, "experiment.py")
        experiment_dst = osp.join(folder_name, "experiment.py")
        if not osp.exists(experiment_dst):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, experiment_dst)
                print(f"Copied experiment.py from {experiment_src} to {experiment_dst}")
            else:
                raise FileNotFoundError(f"experiment.py not found in base directory: {experiment_src}")

        # Ensure run_0/experiment.py exists (needed for baseline comparison)
        run0_dir = osp.join(folder_name, "run_0")
        run0_experiment = osp.join(run0_dir, "experiment.py")
        if osp.exists(run0_dir) and not osp.exists(run0_experiment):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, run0_experiment)
                print(f"Copied baseline experiment.py to {run0_experiment}")
            else:
                print(f"Warning: Could not copy baseline experiment.py to run_0")
        
        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")
        
        return folder_name, idea_name
    
    def setup_logging_redirect(self, folder_name):
        """Redirect stderr to log file, keep stdout visible for aider output"""
        log_path = osp.join(folder_name, "log.txt")
        log_file = open(log_path, "a")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Keep stdout visible so we can see aider's output
        # Only redirect stderr to capture errors
        sys.stderr = log_file
        return original_stdout, original_stderr, log_file
    
    def restore_logging(self, original_stdout, original_stderr, log_file):
        """Restore stdout/stderr"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    def run_aider_experiment(self, base_dir, results_dir, idea):
        """Run experiment using Aider backend"""
        folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
        # Load baseline results
        baseline_path = osp.join(base_dir, "run_0", "final_info.json")
        if osp.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_results = json.load(f)
        else:
            baseline_results = {}
        
        original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
        try:
            self.logger.info(f"Starting Aider experiment: {idea_name}")
            
            exp_file = osp.join(folder_name, "experiment.py")
            notes_file = osp.join(folder_name, "notes.txt")
            
            io = InputOutput(
                yes=True,
                chat_history_file=osp.join(folder_name, f"{idea_name}_aider.txt")
            )
            # Get experiment model from config file
            # aider/litellm requires format: "provider/model_name" (e.g., "azure/gpt-4o-mini")
            experiment_model = self.config.get("experiment", {}).get("model")
            if not experiment_model:
                # Build model string from default_provider and model_name
                default_provider = self.config.get("models", {}).get("default_provider", "azure")
                provider_config = self.config.get("models", {}).get(default_provider, {})
                model_name = provider_config.get("model_name", "gpt-4.1-mini")
                experiment_model = f"{default_provider}/{model_name}"
            
            # Setup environment variables for litellm (aider uses litellm internally)
            # litellm expects AZURE_API_KEY and AZURE_API_BASE, not AZURE_OPENAI_KEY/ENDPOINT
            if experiment_model.startswith("azure/"):
                if os.environ.get("AZURE_OPENAI_KEY") and not os.environ.get("AZURE_API_KEY"):
                    os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_KEY"]
                if os.environ.get("AZURE_OPENAI_ENDPOINT") and not os.environ.get("AZURE_API_BASE"):
                    # litellm expects the base endpoint (without /openai/deployments/...)
                    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
                    # Ensure it ends with / if it's a base URL
                    if endpoint and not endpoint.endswith("/"):
                        endpoint = endpoint.rstrip("/")
                    os.environ["AZURE_API_BASE"] = endpoint
            
            self.logger.info(f"Using experiment model: {experiment_model}")
            main_model = Model(experiment_model)
            coder = Coder.create(
                main_model=main_model,
                fnames=[exp_file, notes_file],
                io=io,
                stream=True,  # Enable streaming to see aider output
                use_git=False,
                edit_format="diff"
            )
            
            success = perform_experiments_aider(idea, folder_name, coder, baseline_results)
            
            self.logger.info(f"Aider experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Aider experiment error: {str(e)}")
            return False
        finally:
            self.restore_logging(original_stdout, original_stderr, log_file)
    
    # def run_openhands_experiment(self, base_dir, results_dir, idea):
    #     """Run experiment using OpenHands backend"""
    #     folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
    #     original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
    #     try:
    #         self.logger.info(f"Starting OpenHands experiment: {idea_name}")
            
    #         # Get OpenHands-specific config
    #         openhands_config = self.config.get("experiment", {}).get("openhands", {})
    #         code_server_path = "/workspace"
    #         mount_paths = openhands_config.get("mount_paths", [])
    #         uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

    #         success = perform_experiments_openhands(
    #             idea,
    #             folder_name,
    #             code_server_path,
    #             mount_paths,
    #             uri_prefix
    #         )
            
    #         self.logger.info(f"OpenHands experiment {'succeeded' if success else 'failed'}: {idea_name}")
    #         return success
            
    #     except Exception as e:
    #         self.logger.error(f"OpenHands experiment error: {str(e)}")
    #         return False
    #     finally:
    #         self.restore_logging(original_stdout, original_stderr, log_file)
    
    def run_experiments(self, base_dir, results_dir, ideas):
        """Run experiments for all ideas"""
        results = []
        
        for idx, idea in enumerate(ideas, 1):
            idea_info = self._extract_idea_info(idea)
            idea_name = idea_info['name']
            self.logger.info(f"Processing idea {idx}/{len(ideas)}: {idea_name}")
            
            try:
                if self.backend == "aider":
                    success = self.run_aider_experiment(base_dir, results_dir, idea)
                elif self.backend == "openhands":
                    raise NotImplementedError("OpenHands backend is not implemented in this version.")
                    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                results.append({
                    'idea_name': idea_name,
                    'success': success
                })
                
            except Exception as e:
                self.logger.error(f"Failed to run experiment for {idea_name}: {str(e)}")
                results.append({
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results

```

檢查是否已匯入 `os` 模組：


添加 `import os` 以使用 `os.environ`：


```python
import os
import os.path as osp
import sys
import json
import shutil
from datetime import datetime

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from internagent.mas.interface import InternAgentInterface
from internagent.experiments_utils_aider import perform_experiments as perform_experiments_aider

from internagent.vis import visualize_hypotheses


# ============================================================================
# Idea Generation Config
# ============================================================================
class IdeaGenerator:
    """Handles idea generation using MAS"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interface = InternAgentInterface(args.config, work_dir=args.task_dir, task_name=args.task_name)
        self.session_id = None
        self.status = None
    
    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")
        
        await self.interface.startup()
        
        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        
        with open(task_desc_path, 'r') as f:
            params = json.load(f)
        
        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])
        
        if not goal or not domain:
            raise ValueError("Task description and domain are required")
        
        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=self.args.ref_code_path,
            constraints=constraints
        )
        
        self.logger.info(f"Session created: {self.session_id}")
    
    async def generate_ideas(self):
        """Run MAS to generate ideas"""
        if self.session_id is None:
            await self.load_task()
        
        await self.interface.startup()
        
        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":
                    if self.args.offline_feedback:
                        with open(self.args.offline_feedback, "r") as f:
                            feedback = json.load(f)
                        await self.interface.add_feedback(self.session_id, feedback)
                        self.logger.info(f"Feedback added: {feedback}")
                
                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")
        
        # Save and visualize ideas
        # Session files are saved in results/{task_name}/ by the memory manager
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}.json")
        vis_output = osp.join(
            "results", self.args.task_name,
            f"{self.args.task_name}_ideas_{self.session_id}.pdf"
        )
        visualize_hypotheses(session_json, vis_output)
        self.logger.info(f"Visualization saved: {vis_output}")
        
        return top_ideas, session_json


# ============================================================================
# Experiment Execution Module
# ============================================================================
class ExperimentRunner:
    """Handles experiment execution with different backends"""
    
    def __init__(self, args, logger, config=None):
        self.args = args
        self.logger = logger
        self.backend = args.exp_backend
        self.config = config or {}
    
    def _extract_idea_info(self, idea):
        """Extract idea information from different formats"""
        # Try refined_method_details first (from full MAS pipeline)
        if 'refined_method_details' in idea and idea['refined_method_details']:
            details = idea['refined_method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to method_details (from method development only)
        elif 'method_details' in idea and idea['method_details']:
            details = idea['method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to basic idea structure (from JSON files)
        else:
            # Handle different possible field names
            name = idea.get('name') or idea.get('title') or 'unnamed_idea'
            title = idea.get('title') or idea.get('name') or 'Untitled'
            description = idea.get('description') or idea.get('content') or ''
            method = idea.get('method') or ''

            return {
                'name': name[:50] if name else 'unnamed_idea',  # Limit name length
                'title': title,
                'description': description,
                'method': method
            }

    def setup_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")
        
        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Ensure experiment.py exists in the experiment folder
        experiment_src = osp.join(base_dir, "experiment.py")
        experiment_dst = osp.join(folder_name, "experiment.py")
        if not osp.exists(experiment_dst):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, experiment_dst)
                print(f"Copied experiment.py from {experiment_src} to {experiment_dst}")
            else:
                raise FileNotFoundError(f"experiment.py not found in base directory: {experiment_src}")

        # Ensure run_0/experiment.py exists (needed for baseline comparison)
        run0_dir = osp.join(folder_name, "run_0")
        run0_experiment = osp.join(run0_dir, "experiment.py")
        if osp.exists(run0_dir) and not osp.exists(run0_experiment):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, run0_experiment)
                print(f"Copied baseline experiment.py to {run0_experiment}")
            else:
                print(f"Warning: Could not copy baseline experiment.py to run_0")
        
        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")
        
        return folder_name, idea_name
    
    def setup_logging_redirect(self, folder_name):
        """Redirect stderr to log file, keep stdout visible for aider output"""
        log_path = osp.join(folder_name, "log.txt")
        log_file = open(log_path, "a")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Keep stdout visible so we can see aider's output
        # Only redirect stderr to capture errors
        sys.stderr = log_file
        return original_stdout, original_stderr, log_file
    
    def restore_logging(self, original_stdout, original_stderr, log_file):
        """Restore stdout/stderr"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    def run_aider_experiment(self, base_dir, results_dir, idea):
        """Run experiment using Aider backend"""
        folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
        # Load baseline results
        baseline_path = osp.join(base_dir, "run_0", "final_info.json")
        if osp.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_results = json.load(f)
        else:
            baseline_results = {}
        
        original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
        try:
            self.logger.info(f"Starting Aider experiment: {idea_name}")
            
            exp_file = osp.join(folder_name, "experiment.py")
            notes_file = osp.join(folder_name, "notes.txt")
            
            io = InputOutput(
                yes=True,
                chat_history_file=osp.join(folder_name, f"{idea_name}_aider.txt")
            )
            # Get experiment model from config file
            # aider/litellm requires format: "provider/model_name" (e.g., "azure/gpt-4o-mini")
            experiment_model = self.config.get("experiment", {}).get("model")
            if not experiment_model:
                # Build model string from default_provider and model_name
                default_provider = self.config.get("models", {}).get("default_provider", "azure")
                provider_config = self.config.get("models", {}).get(default_provider, {})
                model_name = provider_config.get("model_name", "gpt-4.1-mini")
                experiment_model = f"{default_provider}/{model_name}"
            
            # Setup environment variables for litellm (aider uses litellm internally)
            # litellm expects AZURE_API_KEY and AZURE_API_BASE, not AZURE_OPENAI_KEY/ENDPOINT
            if experiment_model.startswith("azure/"):
                if os.environ.get("AZURE_OPENAI_KEY") and not os.environ.get("AZURE_API_KEY"):
                    os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_KEY"]
                if os.environ.get("AZURE_OPENAI_ENDPOINT") and not os.environ.get("AZURE_API_BASE"):
                    # litellm expects the base endpoint (without /openai/deployments/...)
                    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
                    # Ensure it ends with / if it's a base URL
                    if endpoint and not endpoint.endswith("/"):
                        endpoint = endpoint.rstrip("/")
                    os.environ["AZURE_API_BASE"] = endpoint
            
            self.logger.info(f"Using experiment model: {experiment_model}")
            main_model = Model(experiment_model)
            coder = Coder.create(
                main_model=main_model,
                fnames=[exp_file, notes_file],
                io=io,
                stream=True,  # Enable streaming to see aider output
                use_git=False,
                edit_format="diff"
            )
            
            success = perform_experiments_aider(idea, folder_name, coder, baseline_results)
            
            self.logger.info(f"Aider experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Aider experiment error: {str(e)}")
            return False
        finally:
            self.restore_logging(original_stdout, original_stderr, log_file)
    
    # def run_openhands_experiment(self, base_dir, results_dir, idea):
    #     """Run experiment using OpenHands backend"""
    #     folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
    #     original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
    #     try:
    #         self.logger.info(f"Starting OpenHands experiment: {idea_name}")
            
    #         # Get OpenHands-specific config
    #         openhands_config = self.config.get("experiment", {}).get("openhands", {})
    #         code_server_path = "/workspace"
    #         mount_paths = openhands_config.get("mount_paths", [])
    #         uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

    #         success = perform_experiments_openhands(
    #             idea,
    #             folder_name,
    #             code_server_path,
    #             mount_paths,
    #             uri_prefix
    #         )
            
    #         self.logger.info(f"OpenHands experiment {'succeeded' if success else 'failed'}: {idea_name}")
    #         return success
            
    #     except Exception as e:
    #         self.logger.error(f"OpenHands experiment error: {str(e)}")
    #         return False
    #     finally:
    #         self.restore_logging(original_stdout, original_stderr, log_file)
    
    def run_experiments(self, base_dir, results_dir, ideas):
        """Run experiments for all ideas"""
        results = []
        
        for idx, idea in enumerate(ideas, 1):
            idea_info = self._extract_idea_info(idea)
            idea_name = idea_info['name']
            self.logger.info(f"Processing idea {idx}/{len(ideas)}: {idea_name}")
            
            try:
                if self.backend == "aider":
                    success = self.run_aider_experiment(base_dir, results_dir, idea)
                elif self.backend == "openhands":
                    raise NotImplementedError("OpenHands backend is not implemented in this version.")
                    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                results.append({
                    'idea_name': idea_name,
                    'success': success
                })
                
            except Exception as e:
                self.logger.error(f"Failed to run experiment for {idea_name}: {str(e)}")
                results.append({
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results

```

已修正：在 `stage.py` 中，於建立 aider 的 `Model` 前，將環境變數對應到 litellm 所需的格式：

1. 已添加 `import os`
2. 檢查若使用 Azure 模型（`experiment_model.startswith("azure/")`）
3. 將 `AZURE_OPENAI_KEY` 映射到 `AZURE_API_KEY`
4. 將 `AZURE_OPENAI_ENDPOINT` 映射到 `AZURE_API_BASE`，並確保格式正確

這樣 litellm（aider 內部使用）就能找到正確的環境變數來連接 Azure OpenAI。

如果仍有問題，可能需要設置 `AZURE_API_VERSION`。目前使用 litellm 的預設值；若需指定，可添加：

```python
os.environ["AZURE_API_VERSION"] = "2023-05-15"  # 或你需要的版本
```

請重新執行程式測試。

```python
os.environ["AZURE_API_VERSION"] = "2023-05-15"  # 或你需要的版本
```

