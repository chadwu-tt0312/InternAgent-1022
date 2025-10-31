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
    
    def _log_experiment_failure_details(self, folder_name: str, idea_name: str):
        """
        Log detailed failure information from experiment folder.
        
        Args:
            folder_name: Path to experiment folder
            idea_name: Name of the idea/experiment
        """
        try:
            # Check for log file
            log_path = osp.join(folder_name, "log.txt")
            if osp.exists(log_path):
                with open(log_path, "r") as f:
                    log_content = f.read()
                    if log_content.strip():
                        # Show last 1000 characters of log
                        log_snippet = log_content[-1000:] if len(log_content) > 1000 else log_content
                        self.logger.debug(f"  Last log entries:\n{log_snippet}")
            
            # Check for traceback files
            run_dirs = [d for d in os.listdir(folder_name) if d.startswith("run_") and osp.isdir(osp.join(folder_name, d))]
            for run_dir in sorted(run_dirs, reverse=True):  # Check most recent runs first
                traceback_path = osp.join(folder_name, run_dir, "traceback.log")
                if osp.exists(traceback_path):
                    with open(traceback_path, "r") as f:
                        traceback_content = f.read()
                        if traceback_content.strip():
                            self.logger.error(f"  Traceback from {run_dir}:")
                            # Show first 2000 characters of traceback
                            tb_snippet = traceback_content[:2000] if len(traceback_content) > 2000 else traceback_content
                            self.logger.error(f"    {tb_snippet}")
                            if len(traceback_content) > 2000:
                                self.logger.error(f"    ... (truncated, see {traceback_path} for full traceback)")
                            break
            
            # Check for aider chat history
            aider_chat_path = osp.join(folder_name, f"{idea_name}_aider.txt")
            if osp.exists(aider_chat_path):
                with open(aider_chat_path, "r") as f:
                    chat_content = f.read()
                    if chat_content.strip():
                        # Look for error messages in chat
                        lines = chat_content.split('\n')
                        error_lines = [line for line in lines if any(keyword in line.lower() 
                                                                    for keyword in ['error', 'failed', 'exception', 'traceback'])]
                        if error_lines:
                            self.logger.debug(f"  Error mentions in aider chat ({len(error_lines)} lines found)")
                            for line in error_lines[-5:]:  # Show last 5 error lines
                                self.logger.debug(f"    {line[:200]}")
            
            # Check experiment.py for syntax errors
            exp_path = osp.join(folder_name, "experiment.py")
            if osp.exists(exp_path):
                # Could add syntax check here if needed
                pass
                
        except Exception as e:
            self.logger.debug(f"  Could not read failure details: {str(e)}")
    
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
            
            if success:
                self.logger.info(f"Aider experiment succeeded: {idea_name}")
            else:
                self.logger.error(f"Aider experiment failed: {idea_name}")
                # Try to read error details from experiment folder
                self._log_experiment_failure_details(folder_name, idea_name)
            
            return success
            
        except Exception as e:
            import traceback
            error_type = type(e).__name__
            error_msg = str(e)
            error_tb = traceback.format_exc()
            
            self.logger.error(f"Aider experiment error for {idea_name}:")
            self.logger.error(f"  Error Type: {error_type}")
            self.logger.error(f"  Error Message: {error_msg}")
            self.logger.debug(f"  Traceback:\n{error_tb}")
            
            # Try to log additional context if available
            self._log_experiment_failure_details(folder_name, idea_name)
            
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
