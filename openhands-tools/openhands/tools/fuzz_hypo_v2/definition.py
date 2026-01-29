"""FuzzHypo V2 tool implementation with root cause analysis and iterative fuzzing."""

from collections.abc import Sequence
import sys
from typing import TYPE_CHECKING, Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

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


class RootCauseCandidate(Action):
    """Schema for a potential root cause location."""
    function_path: str = Field(description="Path to the candidate function (module:func)")
    confidence: float = Field(description="Confidence score 0-1 that this is the root cause")
    reasoning: str = Field(description="Why this might be the root cause")


class FuzzHypoV2Action(Action):
    """Schema for enhanced LLM-assisted Python fuzzing operations with root cause analysis."""

    target: str = Field(
        description="Python function to test, e.g., 'module:func' or 'path/to/file.py:func'"
    )
    spec: str | None = Field(
        default=None,
        description="Optional: Structured YAML/JSON string defining input/pre/post conditions. "
                    "If not provided, spec will be auto-generated from code analysis."
    )
    mode: Literal["quick", "deep", "iterative", "analyze"] = Field(
        default="quick",
        description="'quick': 200 iterations, 'deep': 2000 iterations, "
                    "'iterative': multi-round fuzz-fix cycle, 'analyze': root cause analysis only"
    )
    issue_description: str | None = Field(
        default=None,
        description="Optional: Issue description to help auto-generate better specs"
    )
    previous_findings: list[str] | None = Field(
        default=None,
        description="Optional: Previous failing inputs to verify fix"
    )


class FuzzHypoV2Observation(Observation):
    """Enhanced observation with root cause analysis and fix suggestions."""

    status: str = Field(description="'ok', 'fail', 'error', 'needs_upstream_analysis', or 'timeout'")
    bug_found: bool = Field(default=False, description="True if a falsifying example was found")
    failing_input: str | None = Field(default=None, description="The input that caused the crash")
    failure_kind: str | None = Field(default=None, description="Type of failure (assertion or exception)")
    harness_path: str | None = Field(default=None, description="Path to the generated test file")
    message: str = Field(default="", description="Detailed summary or error message")
    
    # Original fields
    strategy: str | None = Field(default=None, description="Hypothesis strategy used")
    post_condition: str | None = Field(default=None, description="Post-condition assertion code")
    examples_tested: int | None = Field(default=None, description="Number of examples tested")
    sample_inputs: list[str] | None = Field(default=None, description="Sample inputs tested")
    harness_code: str | None = Field(default=None, description="Generated test harness code")
    
    # NEW: Root cause analysis fields (Plan 1)
    is_root_cause: bool | None = Field(
        default=None, 
        description="Whether this function is likely the root cause vs just showing symptoms"
    )
    root_cause_analysis: str | None = Field(
        default=None,
        description="Detailed analysis of why this might or might not be the root cause"
    )
    call_chain: list[str] | None = Field(
        default=None,
        description="Call chain leading to the failure (for tracing upstream)"
    )
    upstream_candidates: list[str] | None = Field(
        default=None,
        description="Functions that might be the actual root cause (to fuzz next)"
    )
    
    # NEW: Auto-generated spec info (Plan 2)
    auto_generated_spec: bool = Field(
        default=False,
        description="Whether the spec was auto-generated vs user-provided"
    )
    spec_source: str | None = Field(
        default=None,
        description="How the spec was generated (type_hints, docstring, issue_analysis, etc.)"
    )
    
    # NEW: Iterative testing fields (Plan 3)
    iteration: int | None = Field(
        default=None,
        description="Current iteration number in iterative mode"
    )
    related_functions_discovered: list[str] | None = Field(
        default=None,
        description="Related functions discovered during iterative testing"
    )
    fix_verified: bool | None = Field(
        default=None,
        description="Whether a previous fix was verified to work"
    )
    regression_detected: bool | None = Field(
        default=None,
        description="Whether a regression was detected after a fix"
    )
    
    # NEW: Fix suggestions (Plan 1 extension)
    suggested_fix_locations: list[str] | None = Field(
        default=None,
        description="Suggested locations to apply fixes, ordered by priority"
    )
    fix_strategy: str | None = Field(
        default=None,
        description="Recommended fix strategy based on analysis"
    )

    @property
    @override
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Convert observation to LLM-readable content with enhanced analysis."""
        llm_content: list[TextContent | ImageContent] = []

        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        content_parts: list[str] = []
        
        # Status header
        examples_info = f" ({self.examples_tested} examples)" if self.examples_tested else ""
        iteration_info = f" [Iteration {self.iteration}]" if self.iteration else ""
        
        if self.status == "ok":
            content_parts.append(f"✅ FuzzHypo V2 completed successfully{examples_info}{iteration_info} - no bugs found.")
        elif self.status == "fail":
            content_parts.append(f"🐛 FuzzHypo V2 found a bug{examples_info}{iteration_info}!")
        elif self.status == "needs_upstream_analysis":
            content_parts.append(f"⚠️ FuzzHypo V2: Bug found but likely NOT the root cause{iteration_info}")
        elif self.status == "error":
            content_parts.append("❌ FuzzHypo V2 encountered an error.")
        elif self.status == "timeout":
            content_parts.append(f"⏱️ FuzzHypo V2 timed out{examples_info}.")
        else:
            content_parts.append(f"FuzzHypo V2 status: {self.status}")

        # Bug details
        if self.bug_found and self.failing_input:
            content_parts.append(f"\n**Failing input:** `{self.failing_input}`")
        
        if self.failure_kind:
            content_parts.append(f"**Failure type:** {self.failure_kind}")

        # NEW: Root Cause Analysis Section (Plan 1)
        if self.bug_found and self.root_cause_analysis:
            content_parts.append("\n--- 🔍 ROOT CAUSE ANALYSIS ---")
            
            if self.is_root_cause is True:
                content_parts.append("✅ **This IS likely the root cause**")
            elif self.is_root_cause is False:
                content_parts.append("⚠️ **This is likely a SYMPTOM, not the root cause**")
            
            content_parts.append(f"\n{self.root_cause_analysis}")
            
            if self.call_chain:
                chain_str = " → ".join(self.call_chain[:5])
                content_parts.append(f"\n**Call chain:** {chain_str}")
            
            if self.upstream_candidates:
                content_parts.append("\n**📍 Recommended: Fuzz these upstream functions next:**")
                for i, candidate in enumerate(self.upstream_candidates[:3], 1):
                    content_parts.append(f"  {i}. `{candidate}`")
                content_parts.append("\nUse: `fuzz_hypo_v2(target=\"<function>\", mode=\"analyze\")` to trace upstream")
        
        # NEW: Fix suggestions (Plan 1 extension)
        if self.suggested_fix_locations:
            content_parts.append("\n--- 🔧 SUGGESTED FIX LOCATIONS ---")
            for i, loc in enumerate(self.suggested_fix_locations[:3], 1):
                content_parts.append(f"  {i}. `{loc}`")
            
            if self.fix_strategy:
                content_parts.append(f"\n**Fix strategy:** {self.fix_strategy}")

        # Auto-generated spec info (Plan 2)
        if self.auto_generated_spec:
            content_parts.append(f"\n📝 Spec was auto-generated from: {self.spec_source or 'code analysis'}")

        # Iterative testing info (Plan 3)
        if self.fix_verified is True:
            content_parts.append("\n✅ **Previous fix verified: All originally failing inputs now pass**")
        elif self.fix_verified is False:
            content_parts.append("\n❌ **Fix verification failed: Some inputs still fail**")
        
        if self.regression_detected:
            content_parts.append("⚠️ **Regression detected: New failures introduced**")
        
        if self.related_functions_discovered:
            content_parts.append(f"\n**Related functions discovered:** {', '.join(self.related_functions_discovered[:5])}")

        # Test configuration
        if self.strategy or self.post_condition:
            content_parts.append("\n--- Test Configuration ---")
            
        if self.strategy:
            content_parts.append(f"**Strategy:** `{self.strategy}`")
        
        if self.post_condition:
            pc = self.post_condition
            if len(pc) > 300:
                pc = pc[:300] + "..."
            content_parts.append(f"**Post-condition:**\n```python\n{pc}\n```")
        
        if self.sample_inputs:
            samples = self.sample_inputs[:5]
            content_parts.append(f"**Sample inputs tested:** {samples}")

        if self.message:
            content_parts.append(f"\n{self.message}")

        if self.harness_path and (self.status == "error" or self.bug_found):
            content_parts.append(f"\n**Harness file:** {self.harness_path}")

        full_text = "\n".join(content_parts) if content_parts else "[FuzzHypo V2 completed]"
        llm_content.append(TextContent(text=full_text))

        return llm_content


TOOL_DESCRIPTION = """Run enhanced LLM-assisted Python fuzzing with ROOT CAUSE ANALYSIS.

This is FuzzHypo V2 with major improvements:

1. **ROOT CAUSE ANALYSIS**: When a bug is found, analyzes whether this is the actual 
   root cause or just a symptom. Suggests upstream functions to investigate.

2. **AUTO SPEC GENERATION**: If no spec is provided, automatically generates one from:
   - Type hints and docstrings
   - Issue description analysis
   - Code pattern recognition

3. **ITERATIVE FUZZING**: In 'iterative' mode, performs multi-round fuzz-analyze-fix cycles
   to discover all related issues and verify fixes.

4. **FIX VERIFICATION**: Validates that fixes actually resolve the issues without 
   introducing regressions.

USAGE:
- `fuzz_hypo_v2(target="module:func")` - Auto-generate spec and run quick fuzz
- `fuzz_hypo_v2(target="module:func", mode="analyze")` - Focus on root cause analysis
- `fuzz_hypo_v2(target="module:func", mode="iterative")` - Full iterative cycle
- `fuzz_hypo_v2(target="...", previous_findings=["input1", "input2"])` - Verify fix

IMPORTANT: Always check `is_root_cause` field before fixing. If False, fuzz the 
`upstream_candidates` first to find the true root cause.
"""


class FuzzHypoV2Tool(ToolDefinition[FuzzHypoV2Action, FuzzHypoV2Observation]):
    """Enhanced tool for automated bug discovery with root cause analysis."""

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["FuzzHypoV2Tool"]:
        from openhands.tools.fuzz_hypo_v2.impl import FuzzHypoV2Executor

        working_dir = conv_state.workspace.working_dir
        llm = conv_state.agent.llm
        executor = FuzzHypoV2Executor(working_dir=working_dir, llm=llm)

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=FuzzHypoV2Action,
                observation_type=FuzzHypoV2Observation,
                annotations=ToolAnnotations(
                    title="FuzzHypo V2",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool(FuzzHypoV2Tool.name, FuzzHypoV2Tool)
