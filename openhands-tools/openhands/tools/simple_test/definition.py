"""SimpleTest tool: generate and run a minimal Python reproduction (1-5 cases).

This tool is designed to support the `simple_test` path chosen by `test_strategy_decider`.
It runs Python via a bash heredoc so the reproduction does not need to be committed to git.
"""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class SimpleTestAction(Action):
    """Schema for minimal reproduction generation/execution."""

    issue_description: str = Field(
        description="The full issue/problem statement to reproduce."
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional additional context: reproduction steps, stack traces, suspected files, "
            "and any constraints you discovered."
        ),
    )
    repo_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the target repository in the workspace. "
            "If omitted, uses the workspace working directory."
        ),
    )
    python_code: str | None = Field(
        default=None,
        description=(
            "Optional Python script to run. If omitted, the tool will draft a minimal "
            "repro script based on issue_description/context."
        ),
    )
    run: bool = Field(
        default=True,
        description="If true, execute the script. If false, only draft python_code.",
    )
    conda_env: str = Field(
        default="testbed",
        description=(
            "Conda environment name to use when running inside SWE-bench style containers."
        ),
    )
    install_editable: bool = Field(
        default=False,
        description=(
            "If true, run `pip install -e . --no-deps` before executing the script "
            "(from repo_path)."
        ),
    )
    timeout_s: float = Field(
        default=120.0,
        ge=0,
        description="Command timeout in seconds for running the reproduction.",
    )
    output_limit_chars: int = Field(
        default=12000,
        ge=1000,
        description="Truncate combined stdout/stderr to this many characters.",
    )
    mode: Literal["draft_only", "run"] = Field(
        default="run",
        description="draft_only: only draft a minimal script; run: draft (if needed) then execute.",
    )


class SimpleTestObservation(Observation):
    """Observation from SimpleTest tool."""

    status: Literal["ok", "fail", "error"] = Field(
        description="ok: ran successfully; fail: script failed/asserted; error: tool error."
    )
    repo_path: str = Field(description="Repository path used for execution/drafting.")
    command: str | None = Field(
        default=None, description="The executed shell command (if run)."
    )
    exit_code: int | None = Field(
        default=None, description="Exit code of the executed command (if run)."
    )
    python_code: str = Field(description="The drafted/executed Python code.")
    output: str | None = Field(
        default=None, description="Truncated combined stdout/stderr from execution."
    )


TOOL_DESCRIPTION = """Generate and run a minimal Python reproduction test (1-5 cases).

Use this when `test_strategy_decider` chooses `simple_test`:
- Oracle/spec is unclear, or bug is integration-heavy.
- You want a tiny script that fails before the fix and passes after.

Safety:
- The script is executed via a bash heredoc (no tracked test files required).
- If you provide `python_code`, it will be executed as-is.
"""


class SimpleTestTool(ToolDefinition[SimpleTestAction, SimpleTestObservation]):
    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["SimpleTestTool"]:
        from openhands.tools.simple_test.impl import SimpleTestExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        executor = SimpleTestExecutor(llm=conv_state.agent.llm, working_dir=working_dir)
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=SimpleTestAction,
                observation_type=SimpleTestObservation,
                annotations=ToolAnnotations(
                    title="SimpleTest",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool(SimpleTestTool.name, SimpleTestTool)

