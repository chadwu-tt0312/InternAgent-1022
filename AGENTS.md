# Repository Guidelines

## Project Structure & Module Organization
- `internagent/` hosts orchestration logic, MAS interfaces, and visualization helpers (`vis_tree.py`).
- `tasks/` contains reference research tasks; each subfolder carries configs/datasets and is the template for new studies.
- `config/` stores global YAML presets (model routing, agent budgets) consumed by `launch_discovery.py` and shell scripts.
- `scripts/` bundles runnable pipelines such as `run_pipeline.sh` and `run_skip-idea.sh`; automation outputs land in `results/` and logs in `logs/`.
- Documentation assets (`assets/`, `eval/`, `materials_*.md`) hold figures, benchmark tables, and narrative reports.

## Build, Test, and Development Commands
```bash
uv sync                               # install Python 3.11 deps from pyproject/uv.lock
bash scripts/run_pipeline.sh          # full closed-loop experiment (idea → execution)
bash scripts/run_skip-idea.sh         # reuse an existing idea plan, start at coding
uv run python test_llm.py --provider openai  # sanity-check API keys + model factory
python launch_discovery.py --config config/default_config.yaml  # launch a custom project
```

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, and type hints on public functions; favor dataclasses for structured agent state.
- Module names are lowercase with underscores (`internagent/mas/models`); classes use PascalCase; functions/vars use snake_case (`plan_iteration`, `idea_buffer`).
- Before pushing, run `uv run flake8 internagent tasks scripts` to enforce linting and keep imports sorted manually (no autoformatter is bundled).

## Testing Guidelines
- Lightweight connectivity tests live in `test_llm.py`; extend it or add `tests/` modules prefixed with `test_` to mirror the agent or task being validated.
- Prefer asyncio-based tests when probing model backends, mock external calls to stay deterministic, and document dataset/hardware checks inside each `tasks/<domain>/README.md` with matching artifacts in `results/`.

## Commit & Pull Request Guidelines
- Follow the existing log style: short imperative subjects (`Add flow search eval`, `Clean pretrain ckpt`), optionally prefixed with the task ID (e.g., `1027 update`). Keep body lines wrapped at 72 chars.
- Each PR should include: scope summary, linked issue/benchmark, configuration deltas, dependency changes, and validation evidence (command logs, key metrics, or screenshots of `assets/` additions).
- Call out breaking changes and newly required secrets or cloud resources so reviewers can mirror the setup.

## Environment & Secrets
- Copy `.env.example` to `.env` and supply provider keys (OpenAI, Azure, Intern-S1); never commit secrets—scripts load them via `dotenv`.
- Update `config/default_config.yaml` instead of hard-coding credentials; reference environment tokens so configs remain shareable.
