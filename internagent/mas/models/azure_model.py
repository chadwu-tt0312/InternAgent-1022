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

