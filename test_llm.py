#!/usr/bin/env python3
"""
測試 LLM 模型連線與回應的簡單腳本

使用方法:
    # 測試 Azure OpenAI
    uv run test_llm.py --provider azure

    # 測試 OpenAI
    uv run test_llm.py --provider openai

    # 指定模型名稱
    uv run test_llm.py --provider azure --model-name gpt-4o-mini

    # 使用配置檔
    uv run test_llm.py --config config/default_config.yaml
"""

import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path

import dotenv
dotenv.load_dotenv()

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
        test_prompt = "你是誰?請用一句話介紹你自己。"
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
        # stats = model.get_stats()
        # logger.info(f"\n模型統計: {stats}")
        
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
        test_prompt = "你是誰？請用一句話介紹你自己。"
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
        # stats = model.get_stats()
        # logger.info(f"\n模型統計: {stats}")
        
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
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "未設定"),
        "AZURE_OPENAI_KEY": os.environ.get("AZURE_OPENAI_KEY", "未設定"),
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT", "未設定"),
    }
    
    for key, value in env_vars.items():
        if value == "未設定":
            print(f"❌ {key}: {value}")
        else:
            # 只顯示前後幾個字元
            masked = value[:12] + "..." + value[-4:] if len(value) > 16 else "***"
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

