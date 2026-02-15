"""AssertionTest tool: draft a deterministic pytest regression test.

This tool supports the `assertion` path chosen by `test_strategy_decider`.
By default it writes drafts under `.openhands/` (gitignored) to avoid polluting patches.
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


class AssertionTestAction(Action):
    """Schema for drafting/running deterministic regression tests."""

    issue_description: str = Field(description="The full issue/problem statement.")
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
    draft_dir: str = Field(
        default=".openhands/test_drafts",
        description="Directory to write the draft test file to (relative to repo_path).",
    )
    draft_basename: str | None = Field(
        default=None,
        description=(
            "Optional filename (e.g. test_regression_issue123.py). "
            "If omitted, a unique name is generated."
        ),
    )
    write_mode: Literal["ignored", "return_only"] = Field(
        default="ignored",
        description=(
            "ignored: write under draft_dir (recommended, typically gitignored). "
            "return_only: do not write any file, only return drafted code."
        ),
    )
    run_draft: bool = Field(
        default=False,
        description="If true, run `python -m pytest -q <draft_file>` after drafting.",
    )
    conda_env: str = Field(
        default="testbed",
        description="Conda environment name used when executing pytest in SWE-bench containers.",
    )
    install_editable: bool = Field(
        default=False,
        description=(
            "If true, run `pip install -e . --no-deps` before running pytest (from repo_path)."
        ),
    )
    timeout_s: float = Field(
        default=180.0,
        ge=0,
        description="Command timeout in seconds for running pytest.",
    )
    output_limit_chars: int = Field(
        default=20000,
        ge=1000,
        description="Truncate pytest output to this many characters.",
    )


class AssertionTestObservation(Observation):
    status: Literal["ok", "error"] = Field(description="ok or error.")
    repo_path: str = Field(description="Repository path used.")
    draft_path: str | None = Field(
        default=None, description="Path to the written draft file (if write_mode=ignored)."
    )
    drafted_code: str = Field(description="The drafted pytest code.")
    run_command: str | None = Field(default=None, description="Pytest command executed.")
    pytest_exit_code: int | None = Field(default=None, description="Exit code from pytest run.")
    pytest_output: str | None = Field(default=None, description="Truncated pytest output.")


TOOL_DESCRIPTION = """Draft a deterministic pytest regression test (assertion-based).

Use this when `test_strategy_decider` chooses `assertion`:
- A specific failing input exists and the expected behavior is stable.
- You can assert return values / exceptions / warnings deterministically.

By default, drafts are written under `.openhands/` (gitignored) so they won't be included
in a final patch (important for SWE-bench workflows).
"""


class AssertionTestTool(ToolDefinition[AssertionTestAction, AssertionTestObservation]):
    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["AssertionTestTool"]:
        from openhands.tools.assertion_test.impl import AssertionTestExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        executor = AssertionTestExecutor(llm=conv_state.agent.llm, working_dir=working_dir)
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=AssertionTestAction,
                observation_type=AssertionTestObservation,
                annotations=ToolAnnotations(
                    title="AssertionTest",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool(AssertionTestTool.name, AssertionTestTool)

