"""TestStrategyDecider tool - decide fuzz vs assertion vs simple test.

This tool wraps a rubric prompt (derived from benchmarks/swebench/prompts/hypothesis_default_lz.j2)
to decide which kind of test is appropriate for the current bug-fix task:
 - fuzz / PBT (Hypothesis)
 - deterministic assertion-based regression test
 - simple reproduction script / minimal test case
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, override

from pydantic import Field

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


Decision = Literal["fuzz", "assertion", "simple_test"]


class TestStrategyDeciderAction(Action):
    """Schema for deciding what kind of test to write."""

    issue_description: str = Field(
        description="The issue/bug description to base the decision on."
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional additional context: suspected files/functions, stack traces, "
            "constraints, partial reproduction steps, etc."
        ),
    )
    preferred_language: Literal["python"] = Field(
        default="python",
        description="Programming language context (currently only python is supported).",
    )
    python_target: str | None = Field(
        default=None,
        description=(
            "Optional Python symbol target to inspect internals, e.g. "
            "'path/to/file.py:ClassName' or 'module.py:ClassName'."
        ),
    )
    include_python_internal_signals: bool = Field(
        default=True,
        description=(
            "Whether to inspect Python-internal signals (e.g., subclass counts) "
            "and use them in strategy decision."
        ),
    )


class TestStrategyDeciderObservation(Observation):
    """Observation containing a structured decision and next steps."""

    decision: Decision = Field(
        description="One of: 'fuzz', 'assertion', 'simple_test'."
    )
    rationale: str = Field(
        default="",
        description="Concise rationale for the decision based on the rubric.",
    )
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete next steps to follow for the chosen decision.",
    )
    suggested_oracle: str | None = Field(
        default=None,
        description=(
            "If decision is 'fuzz', propose a general oracle/property to check. "
            "If not applicable, omit."
        ),
    )
    should_use_python_internal_info: bool = Field(
        default=False,
        description=(
            "Whether the generated tests should incorporate Python internal structure "
            "(for example, subclass hierarchy coverage)."
        ),
    )
    python_internal_signals: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Collected Python-internal signals used for decision support "
            "(e.g., subclass_count, direct_subclasses, method_count)."
        ),
    )

    @property
    @override
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        parts: list[str] = []
        if self.is_error:
            parts.append(self.ERROR_MESSAGE_HEADER)

        parts.append(f"Decision: {self.decision}")
        if self.rationale:
            parts.append(f"Rationale:\n{self.rationale}")

        if self.suggested_oracle:
            parts.append(f"Suggested oracle/property:\n{self.suggested_oracle}")

        parts.append(
            "Use Python internal info: "
            + ("yes" if self.should_use_python_internal_info else "no")
        )
        if self.python_internal_signals:
            parts.append(
                "Python internal signals:\n"
                + "\n".join(
                    f"- {k}: {v}" for k, v in self.python_internal_signals.items()
                )
            )

        if self.recommended_next_steps:
            steps = "\n".join(f"- {s}" for s in self.recommended_next_steps[:10])
            parts.append(f"Recommended next steps:\n{steps}")

        return [TextContent(text="\n\n".join(parts).strip())]


TOOL_DESCRIPTION = """Choose the best test strategy for a bugfix: fuzz/PBT vs assertion vs simple repro.

Call this when you are about to write tests (or a repro script) and you need to decide:
- **fuzz**: Hypothesis/property-based testing, ONLY if you can state a general oracle/invariant
- **assertion**: deterministic regression test(s) with explicit expected outputs/exceptions/warnings
- **simple_test**: minimal reproduction (1-5 cases) when a strong oracle is unclear/expensive/ambiguous

High-recall triggers:
- parser/IO robustness, weird inputs, large/combinatorial input space
- round-trip/idempotence/format constraints or other invariants
- you can reproduce a failure but cannot yet assert a full spec (start minimal)

Uses a rubric derived from `benchmarks/swebench/prompts/hypothesis_default_lz.j2`.
Returns structured output: decision + rationale + concrete next steps (+ oracle when fuzz)."""


class TestStrategyDeciderTool(
    ToolDefinition[TestStrategyDeciderAction, TestStrategyDeciderObservation]
):
    """Tool for deciding fuzz vs assertion vs simple test."""

    @classmethod
    def create(
        cls, conv_state: "ConversationState"
    ) -> Sequence["TestStrategyDeciderTool"]:
        from openhands.tools.test_strategy_decider.impl import TestStrategyDeciderExecutor

        executor = TestStrategyDeciderExecutor(llm=conv_state.agent.llm)

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=TestStrategyDeciderAction,
                observation_type=TestStrategyDeciderObservation,
                annotations=ToolAnnotations(
                    title="TestStrategyDecider",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# Automatically register the tool when this module is imported
register_tool(TestStrategyDeciderTool.name, TestStrategyDeciderTool)

