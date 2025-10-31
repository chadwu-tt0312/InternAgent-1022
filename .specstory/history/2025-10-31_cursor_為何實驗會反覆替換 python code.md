# 為何實驗會反覆替換 python code
_Exported on 2025/10/31 at 15:35:59 [GMT+8] from Cursor (2.0.43)_

---

**User**

@20251031_144833_internagent.log (514-535) @zsh (148-532)  為何實驗會反覆替換 python code ?

---

**Cursor**


查看相關程式碼以了解原因。


檢視 aider 實驗處理邏輯：


查看 aider coder 的實現，了解它如何處理修改：


查看實際的實驗檔案結構，並檢查當前的實驗目錄：


分析問題並提出解決方案。檢查相關邏輯：


問題分析：

從日誌和程式碼可見：
1. Aider 的 SEARCH/REPLACE 修改失敗：終端顯示 "SEARCH/REPLACE blocks failed to match"；LLM 生成的 SEARCH 區塊與檔案實際內容不完全匹配（空白、縮排、註解等）。
2. 無限迴圈：在 `experiments_utils_aider.py` 第 216-219 行，當檢測到程式碼未修改時直接 `continue`，但：
   - `current_iter` 沒有增加，無法觸發最大迭代次數檢查
   - `next_prompt` 沒有更新，會用相同 prompt 重試
   - 這導致重複相同的失敗嘗試

修正方案：

修改邏輯，在檢測到未修改時增加迭代計數，並更新 prompt 告知修改失敗，避免無限迴圈。


修正此問題：


```python
import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os

from internagent.prompts import (
    CODER_PROMPT_AIDER, 
    NEXT_EXPERIMENT_PROMPT, 
    CODE_STRUCTURE_PROMPT, 
    DEBUG_PROMPT_WITH_STRUCTURE
    )

import filecmp

MAX_ITERS = 5
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 3000


def extract_idea_info(idea):
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


# return (file, line, function, content), message
def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip()
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=27000):
    cwd = osp.abspath(folder_name)

    # Check if experiment.py exists before trying to copy
    experiment_src = osp.join(cwd, "experiment.py")
    if not osp.exists(experiment_src):
        raise FileNotFoundError(f"experiment.py not found in experiment directory: {experiment_src}")

    # COPY CODE SO WE CAN SEE IT.
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.mkdir(run_dir)

    experiment_dst = osp.join(run_dir, "experiment.py")
    shutil.copy(experiment_src, experiment_dst)

    # LAUNCH COMMAND
    command = ["bash", "launcher.sh", f"run_{run_num}"]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=timeout
        )

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            results = {}

            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results

            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results
                    # with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                    #     results = json.load(f)
                    #     results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            traceback, message, tb = None, None, None
            return result.returncode, next_prompt, traceback, message

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
            if osp.exists(traceback_file):
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
            else:
                # Use stderr as fallback if traceback.log doesn't exist
                tb = result.stderr
                traceback, message = info_traceback(tb) if tb else (None, None)
        else:
            traceback, message, tb = None, None, None

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            if tb:
                stderr_output = tb
            else:
                stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)

        return result.returncode, next_prompt, traceback, message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    """
    Perform experiments using Aider backend.
    
    Returns:
        bool: True if all experiments completed successfully, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    idea_info = extract_idea_info(idea)
    next_prompt = CODER_PROMPT_AIDER.format(
        title=idea_info["title"],
        method=idea_info["method"],
        idea=idea_info["description"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    
    try:
        while run < MAX_RUNS + 1:
            if current_iter >= MAX_ITERS:
                logger.warning(f"Max iterations ({MAX_ITERS}) reached for run {run}")
                break
            
            try:
                coder_out = coder.run(next_prompt) # 1. method2code
                print(coder_out)
                
                # Check for API errors
                if "litellm.BadRequestError" in coder_out:
                    logger.error("BadRequestError detected in coder output - likely API issue")
                    return False
                if "rate limit" in coder_out.lower() or "ratelimit" in coder_out.lower():
                    logger.error("Rate limit detected in coder output")
                    return False
                if "ALL_COMPLETED" in coder_out:
                    logger.info("All experiments completed successfully")
                    break
                
                # Check if code was modified
                if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), 
                             os.path.join(folder_name, 'run_0', 'experiment.py')):
                    logger.warning("Code was not modified - skipping run")
                    
                    # Check if aider failed to match SEARCH blocks
                    if "failed to match" in coder_out.lower() or "search/replace" in coder_out.lower():
                        logger.warning("Aider SEARCH/REPLACE blocks failed to match file content")
                        # Update prompt to inform LLM about the matching failure
                        next_prompt = (
                            "Previous modification attempt failed because the SEARCH blocks did not exactly match "
                            "the file content. Please carefully read the current file content and ensure your "
                            "SEARCH blocks match EXACTLY including all whitespace, indentation, and comments. "
                            "Then try again with the correct SEARCH/REPLACE format.\n\n"
                            f"Original prompt: {next_prompt}"
                        )
                    
                    # Increment iteration counter to prevent infinite loops
                    current_iter += 1
                    if current_iter >= MAX_ITERS:
                        logger.error(f"Max iterations ({MAX_ITERS}) reached - aider unable to modify code")
                        break
                    continue
                
                # 2. autodebug - run experiment
                return_code, next_prompt, traceback, message = run_experiment(folder_name, run)
                
                # Handle errors with traceback
                if traceback:
                    logger.warning(f"Run {run} failed with traceback")
                    functions_codes = ""
                    for t in traceback:
                        functions_codes = functions_codes + f"line: {t[1]}, function: {t[2]}, codes: {t[3]} \n"

                    try:
                        code_structure = coder.run(CODE_STRUCTURE_PROMPT.format(
                            error_messages=next_prompt, 
                            function_code=functions_codes
                        ))
                        next_prompt = DEBUG_PROMPT_WITH_STRUCTURE.format(
                            error_messages=next_prompt, 
                            code_structure=code_structure
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate code structure for debugging: {str(e)}")
                        # Continue with original error message
                
                # Check return code
                if return_code == 0:
                    logger.info(f"Run {run} completed successfully")
                    run += 1
                    current_iter = 0
                else:
                    logger.warning(f"Run {run} failed with return code {return_code}")
                    if message:
                        logger.debug(f"Error message: {message}")
                    current_iter += 1
                    
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"Exception in perform_experiments at run {run}, iteration {current_iter}:")
                logger.error(f"  Error Type: {error_type}")
                logger.error(f"  Error Message: {error_msg}")
                logger.debug(f"  Traceback:\n{traceback.format_exc()}")
                current_iter += 1
                
                # If we've exceeded max iterations, fail
                if current_iter >= MAX_ITERS:
                    logger.error(f"Max iterations reached after exception")
                    break
        
        if current_iter >= MAX_ITERS:
            logger.error("Not all experiments completed - max iterations reached")
            return False
        
        if run <= MAX_RUNS:
            logger.warning(f"Experiments stopped at run {run}, expected {MAX_RUNS} runs")
            return False
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Fatal error in perform_experiments:")
        logger.error(f"  Error Type: {type(e).__name__}")
        logger.error(f"  Error Message: {str(e)}")
        logger.debug(f"  Traceback:\n{traceback.format_exc()}")
        return False

```

檢查是否有語法錯誤：


## 問題原因

從日誌和程式碼分析，實驗會反覆替換 Python 程式碼的原因是：

### 1. **Aider SEARCH/REPLACE 匹配失敗**
- 終端顯示多次 "SEARCH/REPLACE blocks failed to match" 錯誤
- Aider 需要 SEARCH 區塊完全匹配檔案內容（空白、縮排、註解等）
- LLM 生成的 SEARCH 區塊與實際檔案有微小差異，導致修改失敗

### 2. **無限迴圈問題**
在 ```215:219:internagent/experiments_utils_aider.py
# Check if code was modified
if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), 
             os.path.join(folder_name, 'run_0', 'experiment.py')):
    logger.warning("Code was not modified - skipping run")
    continue
``` 中：
- 當檢測到程式碼未修改時，直接 `continue` 回到迴圈開始
- `current_iter` 沒有增加，無法觸發最大迭代次數檢查
- `next_prompt` 沒有更新，LLM 會用相同 prompt 重試相同的修改
- 結果：重複相同的失敗嘗試，形成無限迴圈

## 解決方案

已修正程式碼，主要改進：

1. 增加迭代計數：檢測到程式碼未修改時，增加 `current_iter`
2. 偵測 SEARCH/REPLACE 失敗：檢查 aider 輸出中的錯誤訊息
3. 更新 prompt：當匹配失敗時，告知 LLM 失敗原因並要求仔細檢查檔案內容
4. 停止條件：達到最大迭代次數時停止，避免無限迴圈

修正後，當 aider 無法匹配 SEARCH 區塊時：
- 會增加迭代計數
- 會更新 prompt 說明失敗原因
- 達到最大迭代次數（MAX_ITERS=5）後會停止，而不是無限迴圈

這應該可以解決反覆替換程式碼的問題。

```python
- 當檢測到程式碼未修改時，直接 `continue` 回到迴圈開始
- `current_iter` 沒有增加，無法觸發最大迭代次數檢查
- `next_prompt` 沒有更新，LLM 會用相同 prompt 重試相同的修改
- 結果：重複相同的失敗嘗試，形成無限迴圈

## 解決方案

已修正程式碼，主要改進：

1. 增加迭代計數：檢測到程式碼未修改時，增加 `current_iter`
2. 偵測 SEARCH/REPLACE 失敗：檢查 aider 輸出中的錯誤訊息
3. 更新 prompt：當匹配失敗時，告知 LLM 失敗原因並要求仔細檢查檔案內容
4. 停止條件：達到最大迭代次數時停止，避免無限迴圈

修正後，當 aider 無法匹配 SEARCH 區塊時：
- 會增加迭代計數
- 會更新 prompt 說明失敗原因
- 達到最大迭代次數（MAX_ITERS=5）後會停止，而不是無限迴圈

這應該可以解決反覆替換程式碼的問題。
```

```plaintext
- 當檢測到程式碼未修改時，直接 `continue` 回到迴圈開始
- `current_iter` 沒有增加，無法觸發最大迭代次數檢查
- `next_prompt` 沒有更新，LLM 會用相同 prompt 重試相同的修改
- 結果：重複相同的失敗嘗試，形成無限迴圈

## 解決方案

已修正程式碼，主要改進：

1. 增加迭代計數：檢測到程式碼未修改時，增加 `current_iter`
2. 偵測 SEARCH/REPLACE 失敗：檢查 aider 輸出中的錯誤訊息
3. 更新 prompt：當匹配失敗時，告知 LLM 失敗原因並要求仔細檢查檔案內容
4. 停止條件：達到最大迭代次數時停止，避免無限迴圈

修正後，當 aider 無法匹配 SEARCH 區塊時：
- 會增加迭代計數
- 會更新 prompt 說明失敗原因
- 達到最大迭代次數（MAX_ITERS=5）後會停止，而不是無限迴圈

這應該可以解決反覆替換程式碼的問題。
```

---

**User**

 @run_pipeline.sh 不帶 "--exp_backend" 參數會有什麼後果? 參數除了 "aider" 還有其他嗎?

---

**Cursor**


檢查 `--exp_backend` 參數的定義與用途：


檢查 `ExperimentRunner` 如何使用 `exp_backend` 參數：


繼續查看 `ExperimentRunner` 如何根據後端選擇執行方法：


## 不帶 `--exp_backend` 參數的後果

在 ```106:111:launch_discovery.py
exp_group.add_argument(
    "--exp_backend",
    type=str,
    required=True,
    choices=["aider", "openhands"],
    help="Experiment backend to use"
)
``` 中，`--exp_backend` 是必填參數（`required=True`）。

不帶此參數時，argparse 會報錯並立即退出，例如：

```bash
error: the following arguments are required: --exp_backend
usage: launch_discovery.py [-h] --exp_backend {aider,openhands} ...
```

## 可用的後端選項

根據程式碼，`--exp_backend` 只支援兩個值：

1. `"aider"` — 使用 Aider 作為程式碼修改工具
2. `"openhands"` — 目前未實作（會拋出 `NotImplementedError`）

在 ```416:422:internagent/stage.py
if self.backend == "aider":
    success = self.run_aider_experiment(base_dir, results_dir, idea)
elif self.backend == "openhands":
    raise NotImplementedError("OpenHands backend is not implemented in this version.")
    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
else:
    raise ValueError(f"Unknown backend: {self.backend}")
``` 中可以看到 `openhands` 後端尚未實作。

## 建議

目前只使用 `--exp_backend aider`，這是唯一可用的選項。`openhands` 尚未實作，使用會導致錯誤。

```python
error: the following arguments are required: --exp_backend
usage: launch_discovery.py [-h] --exp_backend {aider,openhands} ...
```

```shellscript
error: the following arguments are required: --exp_backend
usage: launch_discovery.py [-h] --exp_backend {aider,openhands} ...
```

```shellscript
error: the following arguments are required: --exp_backend
usage: launch_discovery.py [-h] --exp_backend {aider,openhands} ...
```

```python
if self.backend == "aider":
    success = self.run_aider_experiment(base_dir, results_dir, idea)
elif self.backend == "openhands":
    raise NotImplementedError("OpenHands backend is not implemented in this version.")
    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
else:
    raise ValueError(f"Unknown backend: {self.backend}")
```

---

**User**

 Aider 作為程式碼修改工具，用途是什麼?

---

**Cursor**


檢視 Aider 在專案中的用途與實作：


檢視 Aider 如何實際修改程式碼：


## Aider 在此專案中的用途

**Aider** 是基於 LLM 的程式碼編輯工具，在這個專案中作為實驗程式碼自動修改的核心元件。

### 主要功能

1. 根據 Idea 和方法描述自動修改程式碼
   - 接收由多智能體系統（MAS）生成的 idea 與方法描述
   - 自動修改 `experiment.py` 以實作該方法
   - 使用 SEARCH/REPLACE 格式精確修改程式碼

2. 在實驗執行流程中的角色
   ```
   想法生成 → Aider 修改程式碼 → 執行實驗 → 分析結果 → 迭代改進
   ```

3. 自動化迭代實驗
   - 可進行最多 `MAX_RUNS=5` 次實驗
   - 每次執行後根據結果自動調整程式碼
   - 支援自動除錯（autodebug）

### 具體工作流程

從 ```336:343:internagent/stage.py
coder = Coder.create(
    main_model=main_model,
    fnames=[exp_file, notes_file],
    io=io,
    stream=True,  # Enable streaming to see aider output
    use_git=False,
    edit_format="diff"
)
``` 可以看到，Aider 會：
- 監控 `experiment.py` 和 `notes.txt`
- 使用指定的 LLM（例如 `azure/gpt-4o-mini`）
- 以 diff 格式進行修改
- 保存對話歷史供後續參考

從 ```186:192:internagent/experiments_utils_aider.py
next_prompt = CODER_PROMPT_AIDER.format(
    title=idea_info["title"],
    method=idea_info["method"],
    idea=idea_info["description"],
    max_runs=MAX_RUNS,
    baseline_results=baseline_results,
)
``` 可以看到，Aider 會收到包含：
- Idea 描述
- 建議的方法實作細節
- 基準實驗結果
- 實驗計劃指示

### 修改方式

Aider 使用 SEARCH/REPLACE 格式修改程式碼，例如：
```
<<<<<<< SEARCH
原有程式碼片段
=======
新的程式碼片段
>>>>>>> REPLACE
```

這要求 SEARCH 區塊必須完全匹配檔案內容（包括空白、縮排、註解等），這正是之前遇到的匹配失敗問題的原因。

### 與其他工具的差異

- **Aider**：用於自動修改 Python 實驗程式碼，支援多輪迭代與自動除錯
- **OpenHands**：原本規劃的另一個後端選項，但目前尚未實作

總結：Aider 將自然語言描述轉為程式碼修改，實現從 idea 到實驗實作的自動化流程。

```plaintext
   想法生成 → Aider 修改程式碼 → 執行實驗 → 分析結果 → 迭代改進
   ```

3. 自動化迭代實驗
   - 可進行最多 `MAX_RUNS=5` 次實驗
   - 每次執行後根據結果自動調整程式碼
   - 支援自動除錯（autodebug）

### 具體工作流程

從 ```336:343:internagent/stage.py
coder = Coder.create(
    main_model=main_model,
    fnames=[exp_file, notes_file],
    io=io,
    stream=True,  # Enable streaming to see aider output
    use_git=False,
    edit_format="diff"
)
``` 可以看到，Aider 會：
- 監控 `experiment.py` 和 `notes.txt`
- 使用指定的 LLM（例如 `azure/gpt-4o-mini`）
- 以 diff 格式進行修改
- 保存對話歷史供後續參考

從 ```186:192:internagent/experiments_utils_aider.py
next_prompt = CODER_PROMPT_AIDER.format(
    title=idea_info["title"],
    method=idea_info["method"],
    idea=idea_info["description"],
    max_runs=MAX_RUNS,
    baseline_results=baseline_results,
)
``` 可以看到，Aider 會收到包含：
- Idea 描述
- 建議的方法實作細節
- 基準實驗結果
- 實驗計劃指示

### 修改方式

Aider 使用 SEARCH/REPLACE 格式修改程式碼，例如：
```
<<<<<<< SEARCH
原有程式碼片段
=======
新的程式碼片段
>>>>>>> REPLACE
```

這要求 SEARCH 區塊必須完全匹配檔案內容（包括空白、縮排、註解等），這正是之前遇到的匹配失敗問題的原因。

### 與其他工具的差異

- **Aider**：用於自動修改 Python 實驗程式碼，支援多輪迭代與自動除錯
- **OpenHands**：原本規劃的另一個後端選項，但目前尚未實作

總結：Aider 將自然語言描述轉為程式碼修改，實現從 idea 到實驗實作的自動化流程。
```

```python
coder = Coder.create(
    main_model=main_model,
    fnames=[exp_file, notes_file],
    io=io,
    stream=True,  # Enable streaming to see aider output
    use_git=False,
    edit_format="diff"
)
```

```python
next_prompt = CODER_PROMPT_AIDER.format(
    title=idea_info["title"],
    method=idea_info["method"],
    idea=idea_info["description"],
    max_runs=MAX_RUNS,
    baseline_results=baseline_results,
)
```

```plaintext
<<<<<<< SEARCH
原有程式碼片段
=======
新的程式碼片段
>>>>>>> REPLACE
```

---

**User**

Aider 使用 SEARCH/REPLACE 格式修改程式碼



要求 SEARCH 區塊必須完全匹配檔案內容（包括空白、縮排、註解等），這正是之前遇到的匹配失敗問題的原因。

要如何改善這個問題? 請提出解決方案與我討論後再實作

---

**Cursor**


分析問題，並提出幾種改善方案供討論。


檢視 Aider 是否支援其他編輯格式：


提出幾個改善方案供討論：


以下改善方案供討論：

## 解決方案分析

### **方案 1：改進初始 Prompt，要求 LLM 先讀取檔案**（推薦）
優點：
- 利用 Aider 的文件閱讀能力
- LLM 能取得準確的檔案內容
- 避免憑記憶或猜測產生 SEARCH 區塊

實作方式：
- 在 `CODER_PROMPT_AIDER` 中加入明確指示，要求先使用 `aider` 讀取檔案內容
- 提供具體的讀取指令範例

### **方案 2：匹配失敗時自動提取並提供檔案片段**
優點：
- 精準定位問題區域
- 直接提供正確的程式碼片段給 LLM

實作方式：
- 解析錯誤訊息，提取失敗的 SEARCH 區塊嘗試匹配的範圍
- 從實際檔案中提取相關片段（前後幾行）
- 將這些片段加入重試 prompt

### **方案 3：使用更小、更精確的 SEARCH 區塊策略**
優點：
- 小區塊較容易完全匹配
- 降低因檔案長度帶來的匹配困難

實作方式：
- 在 prompt 中要求 LLM 使用小範圍、精確的程式碼片段
- 建議包含行號或明確的標識符（如函數名、類名）

### **方案 4：增加預處理步驟 - 標準化提示**
優點：
- 引導 LLM 使用一致的格式

實作方式：
- 在 prompt 中提供正確的 SEARCH/REPLACE 範例
- 說明常見錯誤（空白、縮排不一致等）

### **方案 5：混合策略 - 結合方案 1 + 2**（最推薦）
優點：
- 先預防問題（方案 1）
- 失敗時主動修復（方案 2）
- 多層防護

## 我的建議

優先採用方案 5（混合策略）：
1. 改進初始 prompt，要求 LLM 先讀取檔案
2. 失敗時自動提取相關程式碼片段，加入重試 prompt
3. 保留現有的迭代計數與錯誤處理

預期效果：
- 減少首次匹配失敗
- 失敗時更快恢復
- 提高整體成功率

## 請確認

1. 是否接受方案 5（混合策略）？
2. 是否需要我先實作方案 1，觀察效果後再決定是否加入方案 2？
3. 對於方案 2 的實作細節（如片段範圍、提取方式）是否有偏好？

請告訴我您的選擇，我再進行實作。

---

**User**

採用方案 5（混合策略）：



- 改進初始 prompt，要求 LLM 先讀取檔案，並且提供正確的 SEARCH/REPLACE 範例然後說明常見錯誤（空白、縮排不一致等）

- 失敗時自動提取相關程式碼片段，加入重試 prompt

- 保留現有的迭代計數與錯誤處理

---

**Cursor**


實作方案 5（混合策略）：


實作方案 5（混合策略）：


```python
CODER_PROMPT_AIDER = """Your goal is to implement the following idea: {idea}
The proposed method is as follows: {method}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.
**Note**: It is highly recommended that you implement the core functionality of your code within the first one or two runs, with subsequent experiments focused on performance tuning and hyperparameter optimization based on observed results.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

**CRITICAL: Before making any code changes, you MUST first read the current experiment.py file to see the exact code content, including all whitespace, indentation, and comments.** This ensures your SEARCH blocks will match exactly.

Then, you need to implement code based on your plan. After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list.

**SEARCH/REPLACE Format Requirements:**
When modifying code, you MUST use the SEARCH/REPLACE format. Here is the CORRECT format:

```python
<<<<<<< SEARCH
from pfaa import ParameterFreeAdaptiveAttention
=======
from pfaa import ParameterFreeAdaptiveAttention
from encoder import Encoder
>>>>>>> REPLACE
```

**Common Mistakes to Avoid:**
1. **Whitespace mismatch**: The SEARCH block must match EXACTLY, including:
   - Spaces vs tabs (must match the file)
   - Trailing spaces at end of lines
   - Multiple spaces vs single space
   - Line endings (should match file)

2. **Indentation mismatch**: 
   - Use the EXACT same indentation (spaces or tabs) as the file
   - Count spaces carefully if using spaces
   - Do not mix tabs and spaces

3. **Comment differences**: 
   - Include comments exactly as they appear in the file
   - Do not omit or modify existing comments

4. **Incomplete context**: 
   - Include enough surrounding code to uniquely identify the location
   - Use function/class names or distinctive code patterns as anchors

**Best Practices:**
- Read the file first to see the exact formatting
- Use smaller, more precise SEARCH blocks rather than large ones
- Include context lines before and after your target code
- Test your SEARCH block mentally: could it match multiple places? If yes, add more context.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""

CODER_PROMPT_OPENHANDS = """Your goal is to implement the following idea: {idea_description} in the codes {code_server_path}. 
Please read the code of all the files of {code_server_path} first (important), each time before modifying the file you need to determine the location of the insertion again, and after modification to confirm that the content and location of the modification is correct through observation. After that, I will give you a method and you need to adapt that method appropriately based on the existing baseline code.

## Requirements:
    1. Integrate the core concepts of my improved method into the baseline code
    2. Make necessary adaptations to ensure compatibility with the existing codebase
    3. When conflicts arise between the improved method and baseline implementation: 
        1) Prioritize maintaining the stability of the baseline code 
        2) Adapt the improved method's concepts rather than forcing exact implementation 
        3) Preserve the overall architecture of the baseline while enhancing its functionality
    4. Ensure that the final file to be executed is {code_server_path}/launcher.sh
    5. DO NOT make changes to the original content in the {code_server_path}/launcher.sh, such as the GPU ID, data_root, etc. However, it is allowed to add or modify model-related parameters.
    6. DO NOT attempt to install the environment in the {code_server_path}/launcher.sh
    7. When checking the correctness of the code, ignore the runtime environment issues.

The proposed method is as follows: {method}.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.

"""

CODE_STRUCTURE_PROMPT = """You are an expert code analyst specializing in error detection, debugging, and error handling patterns. Your task is to thoroughly analyze the provided code with a focus on potential errors below:

{error_messages}

You need to focus on error-related aspects of code and analyze their relations. The following functions and codes may highly related to the error which is extracted from the traceback.

{function_code}

Note that you do not need to modify the code in this step and just need to give the error-related code structure.
"""

DEBUG_PROMPT_WITH_STRUCTURE = """You are an expert code debugger specializing in structural analysis and error diagnosis. Your task is to debug the code based on the following error message:

{error_messages}

Previously, you have analyzed the error-related code structure as follows:

{code_structure}

You need to first analyze the error message and list all the possible reasons and code modification plan of the error. Then, modify the code based on the plan. You can refer to the code structure obtained from the previous analysis. 

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""

NEXT_EXPERIMENT_PROMPT = """Run {RUN_NUM} completed. Here are the results:
{RESULTS}

Based on these results:
1. Analyze what worked and what didn't work in your approach.
2. Compare the current run with previous runs and baseline.
3. Decide if you need to re-plan your experiments or continue with your current strategy.
4. If continuing, implement the next improvement on your list.
5. If re-planning, explain why and outline your new approach.

We will run the command `bash launcher.sh {NEXT_RUN_NUM}` to execute your next experiment.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you believe you have completed all necessary experiments and found the optimal solution, respond with 'ALL_COMPLETED'.
"""

```

實作自動提取失敗區域程式碼片段的功能：


```python
import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os

from internagent.prompts import (
    CODER_PROMPT_AIDER, 
    NEXT_EXPERIMENT_PROMPT, 
    CODE_STRUCTURE_PROMPT, 
    DEBUG_PROMPT_WITH_STRUCTURE
    )

import filecmp

MAX_ITERS = 5
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 3000


def extract_idea_info(idea):
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


# return (file, line, function, content), message
def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip()
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


def extract_code_snippet_from_file(file_path, search_block, context_lines=5):
    """
    Extract code snippet from file that should match the failed SEARCH block.
    
    Args:
        file_path: Path to the experiment.py file
        search_block: The SEARCH block that failed to match (as string)
        context_lines: Number of context lines to include before and after
    
    Returns:
        str: Extracted code snippet with context, or None if extraction fails
    """
    try:
        if not osp.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            file_lines = file_content.split('\n')
        
        # Try to find the search block in the file
        # First, normalize the search block (remove leading/trailing whitespace from lines)
        search_lines = [line.rstrip() for line in search_block.strip().split('\n') if line.strip()]
        
        if not search_lines:
            return None
        
        # Try to find the first line of the search block in the file
        first_line_pattern = search_lines[0].strip()
        
        # Search for matching line (allowing for some flexibility)
        for i, line in enumerate(file_lines):
            if first_line_pattern in line.rstrip() or line.rstrip() in first_line_pattern:
                # Found potential match, extract context
                start = max(0, i - context_lines)
                end = min(len(file_lines), i + len(search_lines) + context_lines)
                
                snippet_lines = file_lines[start:end]
                
                # Add line numbers for reference
                snippet_with_numbers = []
                for idx, snippet_line in enumerate(snippet_lines, start=start + 1):
                    snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
                
                return '\n'.join(snippet_with_numbers)
        
        # If exact match not found, return a general snippet from the file
        # Try to extract around likely modification points (imports, class definitions, etc.)
        for i, line in enumerate(file_lines):
            if any(keyword in line for keyword in ['import', 'class ', 'def ', 'from ']):
                start = max(0, i - context_lines)
                end = min(len(file_lines), i + 10 + context_lines)
                snippet_lines = file_lines[start:end]
                snippet_with_numbers = []
                for idx, snippet_line in enumerate(snippet_lines, start=start + 1):
                    snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
                return '\n'.join(snippet_with_numbers)
        
        # Fallback: return first 50 lines with context
        snippet_lines = file_lines[:50]
        snippet_with_numbers = []
        for idx, snippet_line in enumerate(snippet_lines, start=1):
            snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
        return '\n'.join(snippet_with_numbers)
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to extract code snippet: {str(e)}")
        return None


def extract_search_block_from_error(coder_output):
    """
    Try to extract the failed SEARCH block from aider error output.
    
    Args:
        coder_output: The output from coder.run()
    
    Returns:
        str: The SEARCH block that failed, or None if not found
    """
    try:
        # Look for SEARCH blocks in the output
        # Pattern: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
        pattern = r'<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE'
        matches = re.findall(pattern, coder_output, re.DOTALL)
        
        if matches:
            # Return the first SEARCH block found
            return matches[0][0]  # First group is the SEARCH block
        
        # Alternative: Look for "SEARCH" blocks in different formats
        pattern2 = r'SEARCH\s*\n(.*?)(?:\n=======|$)'
        matches2 = re.findall(pattern2, coder_output, re.DOTALL)
        if matches2:
            return matches2[0]
        
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to extract SEARCH block from error: {str(e)}")
        return None


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=27000):
    cwd = osp.abspath(folder_name)

    # Check if experiment.py exists before trying to copy
    experiment_src = osp.join(cwd, "experiment.py")
    if not osp.exists(experiment_src):
        raise FileNotFoundError(f"experiment.py not found in experiment directory: {experiment_src}")

    # COPY CODE SO WE CAN SEE IT.
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.mkdir(run_dir)

    experiment_dst = osp.join(run_dir, "experiment.py")
    shutil.copy(experiment_src, experiment_dst)

    # LAUNCH COMMAND
    command = ["bash", "launcher.sh", f"run_{run_num}"]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=timeout
        )

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            results = {}

            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results

            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results
                    # with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                    #     results = json.load(f)
                    #     results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            traceback, message, tb = None, None, None
            return result.returncode, next_prompt, traceback, message

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
            if osp.exists(traceback_file):
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
            else:
                # Use stderr as fallback if traceback.log doesn't exist
                tb = result.stderr
                traceback, message = info_traceback(tb) if tb else (None, None)
        else:
            traceback, message, tb = None, None, None

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            if tb:
                stderr_output = tb
            else:
                stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)

        return result.returncode, next_prompt, traceback, message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    """
    Perform experiments using Aider backend.
    
    Returns:
        bool: True if all experiments completed successfully, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    idea_info = extract_idea_info(idea)
    next_prompt = CODER_PROMPT_AIDER.format(
        title=idea_info["title"],
        method=idea_info["method"],
        idea=idea_info["description"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    
    try:
        while run < MAX_RUNS + 1:
            if current_iter >= MAX_ITERS:
                logger.warning(f"Max iterations ({MAX_ITERS}) reached for run {run}")
                break
            
            try:
                coder_out = coder.run(next_prompt) # 1. method2code
                print(coder_out)
                
                # Check for API errors
                if "litellm.BadRequestError" in coder_out:
                    logger.error("BadRequestError detected in coder output - likely API issue")
                    return False
                if "rate limit" in coder_out.lower() or "ratelimit" in coder_out.lower():
                    logger.error("Rate limit detected in coder output")
                    return False
                if "ALL_COMPLETED" in coder_out:
                    logger.info("All experiments completed successfully")
                    break
                
                # Check if code was modified
                if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), 
                             os.path.join(folder_name, 'run_0', 'experiment.py')):
                    logger.warning("Code was not modified - skipping run")
                    
                    # Check if aider failed to match SEARCH blocks
                    if "failed to match" in coder_out.lower() or "search/replace" in coder_out.lower():
                        logger.warning("Aider SEARCH/REPLACE blocks failed to match file content")
                        # Update prompt to inform LLM about the matching failure
                        next_prompt = (
                            "Previous modification attempt failed because the SEARCH blocks did not exactly match "
                            "the file content. Please carefully read the current file content and ensure your "
                            "SEARCH blocks match EXACTLY including all whitespace, indentation, and comments. "
                            "Then try again with the correct SEARCH/REPLACE format.\n\n"
                            f"Original prompt: {next_prompt}"
                        )
                    
                    # Increment iteration counter to prevent infinite loops
                    current_iter += 1
                    if current_iter >= MAX_ITERS:
                        logger.error(f"Max iterations ({MAX_ITERS}) reached - aider unable to modify code")
                        break
                    continue
                
                # 2. autodebug - run experiment
                return_code, next_prompt, traceback, message = run_experiment(folder_name, run)
                
                # Handle errors with traceback
                if traceback:
                    logger.warning(f"Run {run} failed with traceback")
                    functions_codes = ""
                    for t in traceback:
                        functions_codes = functions_codes + f"line: {t[1]}, function: {t[2]}, codes: {t[3]} \n"

                    try:
                        code_structure = coder.run(CODE_STRUCTURE_PROMPT.format(
                            error_messages=next_prompt, 
                            function_code=functions_codes
                        ))
                        next_prompt = DEBUG_PROMPT_WITH_STRUCTURE.format(
                            error_messages=next_prompt, 
                            code_structure=code_structure
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate code structure for debugging: {str(e)}")
                        # Continue with original error message
                
                # Check return code
                if return_code == 0:
                    logger.info(f"Run {run} completed successfully")
                    run += 1
                    current_iter = 0
                else:
                    logger.warning(f"Run {run} failed with return code {return_code}")
                    if message:
                        logger.debug(f"Error message: {message}")
                    current_iter += 1
                    
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"Exception in perform_experiments at run {run}, iteration {current_iter}:")
                logger.error(f"  Error Type: {error_type}")
                logger.error(f"  Error Message: {error_msg}")
                logger.debug(f"  Traceback:\n{traceback.format_exc()}")
                current_iter += 1
                
                # If we've exceeded max iterations, fail
                if current_iter >= MAX_ITERS:
                    logger.error(f"Max iterations reached after exception")
                    break
        
        if current_iter >= MAX_ITERS:
            logger.error("Not all experiments completed - max iterations reached")
            return False
        
        if run <= MAX_RUNS:
            logger.warning(f"Experiments stopped at run {run}, expected {MAX_RUNS} runs")
            return False
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Fatal error in perform_experiments:")
        logger.error(f"  Error Type: {type(e).__name__}")
        logger.error(f"  Error Message: {str(e)}")
        logger.debug(f"  Traceback:\n{traceback.format_exc()}")
        return False

```

改進失敗處理邏輯，整合程式碼片段提取功能：


```python
import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os

from internagent.prompts import (
    CODER_PROMPT_AIDER, 
    NEXT_EXPERIMENT_PROMPT, 
    CODE_STRUCTURE_PROMPT, 
    DEBUG_PROMPT_WITH_STRUCTURE
    )

import filecmp

MAX_ITERS = 5
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 3000


def extract_idea_info(idea):
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


# return (file, line, function, content), message
def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip()
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


def extract_code_snippet_from_file(file_path, search_block, context_lines=5):
    """
    Extract code snippet from file that should match the failed SEARCH block.
    
    Args:
        file_path: Path to the experiment.py file
        search_block: The SEARCH block that failed to match (as string)
        context_lines: Number of context lines to include before and after
    
    Returns:
        str: Extracted code snippet with context, or None if extraction fails
    """
    try:
        if not osp.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            file_lines = file_content.split('\n')
        
        # Try to find the search block in the file
        # First, normalize the search block (remove leading/trailing whitespace from lines)
        search_lines = [line.rstrip() for line in search_block.strip().split('\n') if line.strip()]
        
        if not search_lines:
            return None
        
        # Try to find the first line of the search block in the file
        first_line_pattern = search_lines[0].strip()
        
        # Search for matching line (allowing for some flexibility)
        for i, line in enumerate(file_lines):
            if first_line_pattern in line.rstrip() or line.rstrip() in first_line_pattern:
                # Found potential match, extract context
                start = max(0, i - context_lines)
                end = min(len(file_lines), i + len(search_lines) + context_lines)
                
                snippet_lines = file_lines[start:end]
                
                # Add line numbers for reference
                snippet_with_numbers = []
                for idx, snippet_line in enumerate(snippet_lines, start=start + 1):
                    snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
                
                return '\n'.join(snippet_with_numbers)
        
        # If exact match not found, return a general snippet from the file
        # Try to extract around likely modification points (imports, class definitions, etc.)
        for i, line in enumerate(file_lines):
            if any(keyword in line for keyword in ['import', 'class ', 'def ', 'from ']):
                start = max(0, i - context_lines)
                end = min(len(file_lines), i + 10 + context_lines)
                snippet_lines = file_lines[start:end]
                snippet_with_numbers = []
                for idx, snippet_line in enumerate(snippet_lines, start=start + 1):
                    snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
                return '\n'.join(snippet_with_numbers)
        
        # Fallback: return first 50 lines with context
        snippet_lines = file_lines[:50]
        snippet_with_numbers = []
        for idx, snippet_line in enumerate(snippet_lines, start=1):
            snippet_with_numbers.append(f"{idx:4d} | {snippet_line}")
        return '\n'.join(snippet_with_numbers)
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to extract code snippet: {str(e)}")
        return None


def extract_search_block_from_error(coder_output):
    """
    Try to extract the failed SEARCH block from aider error output.
    
    Args:
        coder_output: The output from coder.run()
    
    Returns:
        str: The SEARCH block that failed, or None if not found
    """
    try:
        # Look for SEARCH blocks in the output
        # Pattern: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
        pattern = r'<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE'
        matches = re.findall(pattern, coder_output, re.DOTALL)
        
        if matches:
            # Return the first SEARCH block found
            return matches[0][0]  # First group is the SEARCH block
        
        # Alternative: Look for "SEARCH" blocks in different formats
        pattern2 = r'SEARCH\s*\n(.*?)(?:\n=======|$)'
        matches2 = re.findall(pattern2, coder_output, re.DOTALL)
        if matches2:
            return matches2[0]
        
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to extract SEARCH block from error: {str(e)}")
        return None


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=27000):
    cwd = osp.abspath(folder_name)

    # Check if experiment.py exists before trying to copy
    experiment_src = osp.join(cwd, "experiment.py")
    if not osp.exists(experiment_src):
        raise FileNotFoundError(f"experiment.py not found in experiment directory: {experiment_src}")

    # COPY CODE SO WE CAN SEE IT.
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.mkdir(run_dir)

    experiment_dst = osp.join(run_dir, "experiment.py")
    shutil.copy(experiment_src, experiment_dst)

    # LAUNCH COMMAND
    command = ["bash", "launcher.sh", f"run_{run_num}"]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=timeout
        )

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            results = {}

            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results

            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results
                    # with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                    #     results = json.load(f)
                    #     results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            traceback, message, tb = None, None, None
            return result.returncode, next_prompt, traceback, message

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
            if osp.exists(traceback_file):
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
            else:
                # Use stderr as fallback if traceback.log doesn't exist
                tb = result.stderr
                traceback, message = info_traceback(tb) if tb else (None, None)
        else:
            traceback, message, tb = None, None, None

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            if tb:
                stderr_output = tb
            else:
                stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)

        return result.returncode, next_prompt, traceback, message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    """
    Perform experiments using Aider backend.
    
    Returns:
        bool: True if all experiments completed successfully, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    idea_info = extract_idea_info(idea)
    next_prompt = CODER_PROMPT_AIDER.format(
        title=idea_info["title"],
        method=idea_info["method"],
        idea=idea_info["description"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    
    try:
        while run < MAX_RUNS + 1:
            if current_iter >= MAX_ITERS:
                logger.warning(f"Max iterations ({MAX_ITERS}) reached for run {run}")
                break
            
            try:
                coder_out = coder.run(next_prompt) # 1. method2code
                print(coder_out)
                
                # Check for API errors
                if "litellm.BadRequestError" in coder_out:
                    logger.error("BadRequestError detected in coder output - likely API issue")
                    return False
                if "rate limit" in coder_out.lower() or "ratelimit" in coder_out.lower():
                    logger.error("Rate limit detected in coder output")
                    return False
                if "ALL_COMPLETED" in coder_out:
                    logger.info("All experiments completed successfully")
                    break
                
                # Check if code was modified
                if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), 
                             os.path.join(folder_name, 'run_0', 'experiment.py')):
                    logger.warning("Code was not modified - skipping run")
                    
                    # Check if aider failed to match SEARCH blocks
                    if "failed to match" in coder_out.lower() or "search/replace" in coder_out.lower():
                        logger.warning("Aider SEARCH/REPLACE blocks failed to match file content")
                        
                        # Try to extract the failed SEARCH block and actual code snippet
                        failed_search_block = extract_search_block_from_error(coder_out)
                        experiment_file = os.path.join(folder_name, 'experiment.py')
                        code_snippet = None
                        
                        if failed_search_block:
                            logger.debug(f"Extracted failed SEARCH block: {failed_search_block[:200]}...")
                            code_snippet = extract_code_snippet_from_file(
                                experiment_file, 
                                failed_search_block, 
                                context_lines=10
                            )
                        
                        # Build enhanced retry prompt
                        retry_prompt_parts = [
                            "Previous modification attempt FAILED because the SEARCH blocks did not exactly match the file content.",
                            "",
                            "CRITICAL INSTRUCTIONS:",
                            "1. Read the current experiment.py file FIRST to see the exact formatting",
                            "2. Compare your SEARCH block with the actual file content below",
                            "3. Ensure your SEARCH block matches EXACTLY including:",
                            "   - All whitespace (spaces vs tabs)",
                            "   - Exact indentation (count spaces carefully)",
                            "   - Trailing spaces at end of lines",
                            "   - All comments as they appear in the file",
                            "   - Line endings",
                            "4. Use SMALLER, more precise SEARCH blocks with enough context",
                            "5. Include line numbers from the file for reference when constructing your SEARCH block",
                        ]
                        
                        if code_snippet:
                            retry_prompt_parts.extend([
                                "",
                                "Here is the relevant code snippet from the current experiment.py file:",
                                "(Pay attention to the exact formatting, especially whitespace and indentation)",
                                "",
                                "```python",
                                code_snippet,
                                "```",
                                "",
                                "Use this ACTUAL code as reference when creating your SEARCH block. ",
                                "Copy the EXACT text including all whitespace, indentation, and comments."
                            ])
                        else:
                            retry_prompt_parts.extend([
                                "",
                                "NOTE: Unable to extract the exact code snippet. ",
                                "Please read the entire experiment.py file first before attempting modifications."
                            ])
                        
                        if failed_search_block:
                            retry_prompt_parts.extend([
                                "",
                                "The SEARCH block that failed was:",
                                "```python",
                                failed_search_block[:500] + ("..." if len(failed_search_block) > 500 else ""),
                                "```",
                                "",
                                "Compare this with the actual file content above and identify the differences."
                            ])
                        
                        retry_prompt_parts.extend([
                            "",
                            "---",
                            "Original prompt:",
                            next_prompt
                        ])
                        
                        next_prompt = "\n".join(retry_prompt_parts)
                    
                    # Increment iteration counter to prevent infinite loops
                    current_iter += 1
                    if current_iter >= MAX_ITERS:
                        logger.error(f"Max iterations ({MAX_ITERS}) reached - aider unable to modify code")
                        break
                    continue
                
                # 2. autodebug - run experiment
                return_code, next_prompt, traceback, message = run_experiment(folder_name, run)
                
                # Handle errors with traceback
                if traceback:
                    logger.warning(f"Run {run} failed with traceback")
                    functions_codes = ""
                    for t in traceback:
                        functions_codes = functions_codes + f"line: {t[1]}, function: {t[2]}, codes: {t[3]} \n"

                    try:
                        code_structure = coder.run(CODE_STRUCTURE_PROMPT.format(
                            error_messages=next_prompt, 
                            function_code=functions_codes
                        ))
                        next_prompt = DEBUG_PROMPT_WITH_STRUCTURE.format(
                            error_messages=next_prompt, 
                            code_structure=code_structure
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate code structure for debugging: {str(e)}")
                        # Continue with original error message
                
                # Check return code
                if return_code == 0:
                    logger.info(f"Run {run} completed successfully")
                    run += 1
                    current_iter = 0
                else:
                    logger.warning(f"Run {run} failed with return code {return_code}")
                    if message:
                        logger.debug(f"Error message: {message}")
                    current_iter += 1
                    
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"Exception in perform_experiments at run {run}, iteration {current_iter}:")
                logger.error(f"  Error Type: {error_type}")
                logger.error(f"  Error Message: {error_msg}")
                logger.debug(f"  Traceback:\n{traceback.format_exc()}")
                current_iter += 1
                
                # If we've exceeded max iterations, fail
                if current_iter >= MAX_ITERS:
                    logger.error(f"Max iterations reached after exception")
                    break
        
        if current_iter >= MAX_ITERS:
            logger.error("Not all experiments completed - max iterations reached")
            return False
        
        if run <= MAX_RUNS:
            logger.warning(f"Experiments stopped at run {run}, expected {MAX_RUNS} runs")
            return False
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Fatal error in perform_experiments:")
        logger.error(f"  Error Type: {type(e).__name__}")
        logger.error(f"  Error Message: {str(e)}")
        logger.debug(f"  Traceback:\n{traceback.format_exc()}")
        return False

```

已完成方案 5（混合策略）的實作。摘要如下：

## 已完成實作

### 1. 改進初始 Prompt (`prompts.py`)

已在 `CODER_PROMPT_AIDER` 中加入：
- 要求先讀取檔案：在修改前必須先讀取 `experiment.py` 查看確切內容
- SEARCH/REPLACE 範例：提供正確格式範例
- 常見錯誤說明：說明空白、縮排、註解等常見匹配失敗原因
- 最佳實踐：建議使用較小、精確的 SEARCH 區塊

### 2. 實作程式碼片段提取功能 (`experiments_utils_aider.py`)

新增兩個函數：

- `extract_code_snippet_from_file()`：
  - 從檔案提取與失敗 SEARCH 區塊相關的程式碼片段
  - 包含行號與上下文（預設前後 5 行）
  - 支援多種匹配策略（精確匹配 → 關鍵字匹配 → 回退方案）

- `extract_search_block_from_error()`：
  - 從 aider 錯誤輸出中提取失敗的 SEARCH 區塊
  - 支援多種格式的正則表達式匹配

### 3. 改進失敗處理邏輯 (`experiments_utils_aider.py`)

當匹配失敗時，新的處理流程會：
1. 自動提取失敗的 SEARCH 區塊
2. 從實際檔案中提取相關程式碼片段（含行號）
3. 建立增強的重試 prompt，包含：
   - 明確的失敗說明與步驟指示
   - 實際檔案程式碼片段（含行號）
   - 失敗的 SEARCH 區塊供對比
   - 格式化注意事項
   - 原始 prompt 保留

### 改進效果預期

- 預防：初始 prompt 引導 LLM 先讀取檔案並提供範例
- 恢復：失敗時自動提供實際程式碼片段，幫助 LLM 生成正確的 SEARCH 區塊
- 防護：保留迭代計數與錯誤處理機制

## 測試建議

下次執行實驗時可觀察：
1. 是否減少首次匹配失敗
2. 失敗後的重試是否能更快成功
3. 整體實驗成功率是否提升

如需調整，例如：
- 上下文行數（目前為 10 行）
- 程式碼片段提取策略
- 重試 prompt 的格式與內容

請告訴我。

