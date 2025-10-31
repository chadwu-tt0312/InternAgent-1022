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
