"""Executor for AssertionTest tool."""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import maybe_truncate
from openhands.tools.assertion_test.definition import (
    AssertionTestAction,
    AssertionTestObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.llm import LLM


class AssertionTestExecutor(ToolExecutor[AssertionTestAction, AssertionTestObservation]):
    def __init__(self, *, llm: "LLM", working_dir: str):
        self.llm = llm
        self.working_dir = working_dir

    def __call__(
        self, action: AssertionTestAction, conversation: "LocalConversation | None" = None
    ) -> AssertionTestObservation:
        try:
            repo_path = (action.repo_path or self.working_dir).strip() or self.working_dir
            drafted = self._draft_pytest_code(
                issue_description=action.issue_description,
                context=action.context,
                repo_path=repo_path,
            )

            draft_path: str | None = None
            if action.write_mode == "ignored":
                if conversation is None:
                    raise RuntimeError("AssertionTest requires a conversation context to write files.")
                draft_path = self._write_draft_file(
                    conversation=conversation,
                    repo_path=repo_path,
                    draft_dir=action.draft_dir,
                    draft_basename=action.draft_basename,
                    content=drafted,
                )

            run_command: str | None = None
            pytest_exit_code: int | None = None
            pytest_output: str | None = None

            if action.run_draft:
                if conversation is None:
                    raise RuntimeError("AssertionTest requires a conversation context to run commands.")
                if not draft_path:
                    raise ValueError("run_draft=true requires write_mode=ignored (so there is a file to run).")
                run_command = self._build_pytest_command(
                    repo_path=repo_path,
                    test_file_path=draft_path,
                    conda_env=action.conda_env,
                    install_editable=action.install_editable,
                )
                res = conversation.workspace.execute_command(
                    run_command, cwd=self.working_dir, timeout=float(action.timeout_s)
                )
                pytest_exit_code = res.exit_code
                combined = (res.stdout or "") + (res.stderr or "")
                pytest_output = maybe_truncate(
                    combined, truncate_after=int(action.output_limit_chars), save_dir=None, tool_prefix="assertion_test"
                )

            return AssertionTestObservation.from_text(
                text="Drafted assertion-based pytest regression test.",
                status="ok",
                repo_path=repo_path,
                draft_path=draft_path,
                drafted_code=drafted,
                run_command=run_command,
                pytest_exit_code=pytest_exit_code,
                pytest_output=pytest_output,
                is_error=False,
            )
        except Exception as e:
            return AssertionTestObservation.from_text(
                text=f"{type(e).__name__}: {e}",
                status="error",
                repo_path=(action.repo_path or self.working_dir).strip() or self.working_dir,
                draft_path=None,
                drafted_code="",
                run_command=None,
                pytest_exit_code=None,
                pytest_output=None,
                is_error=True,
            )

    def _draft_pytest_code(
        self, *, issue_description: str, context: str | None, repo_path: str
    ) -> str:
        from openhands.sdk.llm.message import Message, TextContent, content_to_str

        system_prompt = """You are a senior Python test engineer.
Draft a deterministic pytest regression test for the described bug.

Requirements:
- Output ONLY valid Python code (no markdown fences, no extra text).
- Use pytest.
- Keep the test narrow and stable; avoid brittle string matching unless required by the issue.
- Prefer any test/repro code included in the issue description as-is (minimal adaptation for imports/fixtures).
- Do NOT write files, do NOT access network.
- The test should fail before the fix and pass after.

Conventions:
- Use `def test_...():` naming.
- Use `pytest.raises(...)` / `pytest.warns(...)` where appropriate.
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
        raw = re.sub(r"^```(?:python)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def _write_draft_file(
        self,
        *,
        conversation: "LocalConversation",
        repo_path: str,
        draft_dir: str,
        draft_basename: str | None,
        content: str,
    ) -> str:
        # Resolve path inside repo
        base = Path(repo_path)
        rel_dir = Path(draft_dir)
        out_dir = base / rel_dir
        # Create directory
        mkdir_cmd = f'bash -lc "mkdir -p \\"{str(out_dir).replace(chr(34), r"\\\"")}\\""'
        _ = conversation.workspace.execute_command(mkdir_cmd, cwd=self.working_dir, timeout=30.0)

        if draft_basename:
            fname = draft_basename
        else:
            fname = f"test_regression_draft_{int(time.time())}.py"
        out_path = out_dir / fname

        # Write file content via base64 to avoid shell escaping issues
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        out_path_escaped = str(out_path).replace('"', '\\"')
        write_cmd = (
            'bash -lc "python - <<\'PY\'\n'
            "import base64, pathlib\n"
            f"data = base64.b64decode({b64!r})\n"
            f"path = pathlib.Path({out_path_escaped!r})\n"
            "path.write_bytes(data)\n"
            "print(str(path))\n"
            "PY\""
        )
        res = conversation.workspace.execute_command(write_cmd, cwd=self.working_dir, timeout=30.0)
        if res.exit_code != 0:
            raise RuntimeError(f"Failed to write draft file: {res.stderr}")
        return str(out_path)

    def _build_pytest_command(
        self,
        *,
        repo_path: str,
        test_file_path: str,
        conda_env: str,
        install_editable: bool,
    ) -> str:
        install = ""
        if install_editable:
            install = "pip install -e . --no-deps && "
        repo_path_escaped = repo_path.replace('"', '\\"')
        test_path_escaped = test_file_path.replace('"', '\\"')
        conda_env_escaped = conda_env.replace('"', '\\"')
        return (
            'bash -lc "'
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            f'conda activate \\"{conda_env_escaped}\\" && '
            f'cd \\"{repo_path_escaped}\\" && '
            + install
            + f'python -m pytest -q \\"{test_path_escaped}\\""'
        )

