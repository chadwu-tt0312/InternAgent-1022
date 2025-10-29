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



    

