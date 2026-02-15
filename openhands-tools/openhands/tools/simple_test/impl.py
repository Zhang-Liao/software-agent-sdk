"""Executor for SimpleTest tool."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import maybe_truncate
from openhands.tools.simple_test.definition import (
    SimpleTestAction,
    SimpleTestObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.llm import LLM


class SimpleTestExecutor(ToolExecutor[SimpleTestAction, SimpleTestObservation]):
    def __init__(self, *, llm: "LLM", working_dir: str):
        self.llm = llm
        self.working_dir = working_dir

    def __call__(
        self, action: SimpleTestAction, conversation: "LocalConversation | None" = None
    ) -> SimpleTestObservation:
        try:
            repo_path = (action.repo_path or self.working_dir).strip()
            if not repo_path:
                repo_path = self.working_dir

            python_code = action.python_code
            if not python_code:
                python_code = self._draft_minimal_repro(
                    issue_description=action.issue_description,
                    context=action.context,
                    repo_path=repo_path,
                )

            # draft_only mode: do not execute
            if action.mode == "draft_only" or not action.run:
                return SimpleTestObservation.from_text(
                    text="Drafted minimal reproduction script.",
                    status="ok",
                    repo_path=repo_path,
                    python_code=python_code,
                    command=None,
                    exit_code=None,
                    output=None,
                )

            if conversation is None:
                raise RuntimeError("SimpleTest requires a conversation context to run commands.")

            cmd = self._build_bash_command(
                repo_path=repo_path,
                python_code=python_code,
                conda_env=action.conda_env,
                install_editable=action.install_editable,
            )
            result = conversation.workspace.execute_command(
                cmd, cwd=self.working_dir, timeout=float(action.timeout_s)
            )
            combined = (result.stdout or "") + (result.stderr or "")
            combined = maybe_truncate(
                combined, truncate_after=int(action.output_limit_chars), save_dir=None, tool_prefix="simple_test"
            )

            exit_code = result.exit_code
            status = "ok" if exit_code == 0 else "fail"
            return SimpleTestObservation.from_text(
                text=f"SimpleTest finished with exit_code={exit_code}.",
                status=status,  # type: ignore[arg-type]
                repo_path=repo_path,
                python_code=python_code,
                command=cmd,
                exit_code=exit_code,
                output=combined,
                is_error=False,
            )
        except Exception as e:
            return SimpleTestObservation.from_text(
                text=f"{type(e).__name__}: {e}",
                status="error",
                repo_path=(action.repo_path or self.working_dir).strip() or self.working_dir,
                python_code=action.python_code or "",
                command=None,
                exit_code=None,
                output=None,
                is_error=True,
            )

    def _draft_minimal_repro(
        self, *, issue_description: str, context: str | None, repo_path: str
    ) -> str:
        from openhands.sdk.llm.message import Message, TextContent, content_to_str

        system_prompt = """You are a senior Python test engineer.
Draft a minimal Python reproduction script for the described bug.

Requirements:
- Output ONLY valid Python code (no markdown fences, no extra text).
- Keep it minimal: 1-5 cases max.
- Prefer using any reproduction/test code included in the issue description as-is (minimal adaptation for imports).
- Use assertions or explicit exception checks so the script exits non-zero on failure.
- Do NOT write any files. Do NOT access network.
- Assume the repository is available locally; use imports that work after `pip install -e . --no-deps`.

Structure suggestion:
- Put logic in a main() and call it under if __name__ == "__main__".
- Print a short message before/after, but rely on asserts for pass/fail.
"""

        user_parts = [
            "Repository path (for reference):",
            repo_path,
            "",
            "Issue description:",
            (issue_description or "").strip(),
        ]
        if context:
            user_parts += ["", "Additional context:", context.strip()]
        user_prompt = "\n".join(user_parts).strip() + "\n"

        resp = self.llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ],
            temperature=0.1,
        )
        raw = "".join(content_to_str(resp.message.content)).strip()

        # Strip common fence patterns defensively
        raw = re.sub(r"^```(?:python)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def _build_bash_command(
        self,
        *,
        repo_path: str,
        python_code: str,
        conda_env: str,
        install_editable: bool,
    ) -> str:
        # Use single-quoted heredoc delimiter to avoid variable expansion.
        # Note: we intentionally do NOT use `set -e` (runtime may not support it reliably).
        install = ""
        if install_editable:
            install = "pip install -e . --no-deps && "

        # Ensure repo_path is shell-safe enough (best effort)
        repo_path_escaped = repo_path.replace('"', '\\"')
        conda_env_escaped = conda_env.replace('"', '\\"')

        return (
            'bash -lc "'
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            f'conda activate \\"{conda_env_escaped}\\" && '
            f'cd \\"{repo_path_escaped}\\" && '
            + install
            + "python - <<'PY'\n"
            + python_code
            + "\nPY"
            + '"'
        )

