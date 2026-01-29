"""FuzzHypoAgent tool - LLM-assisted property-based testing using sub-agents.

This tool uses sub-agents to perform more robust fuzzing:
1. Analyzer agent: Analyzes target function and extracts key information
2. Generator agent: Generates Hypothesis strategies and post-conditions
3. Validator agent: Validates and fixes generated code before execution
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, override

from pydantic import Field
from rich.text import Text

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


class FuzzHypoAgentAction(Action):
    """Schema for LLM-assisted Python fuzzing operations with sub-agents."""

    target: str = Field(
        description="Python function to test, e.g., 'module:func' or 'path/to/file.py:func'"
    )
    spec: str = Field(
        description="Structured YAML/JSON string or file path defining input/pre/post conditions"
    )
    mode: Literal["quick", "deep", "debug"] = Field(
        default="quick",
        description="'quick': 200 iterations, 'deep': 2000 iterations, 'debug': test harness only"
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum number of fix attempts if validation fails (default: 3)"
    )


class FuzzHypoAgentObservation(Observation):
    """Observation from fuzzing operations with sub-agents."""

    status: str = Field(description="'ok', 'fail', 'error', or 'timeout'")
    bug_found: bool = Field(default=False, description="True if a falsifying example was found")
    failing_input: str | None = Field(default=None, description="The input that caused the crash")
    failure_kind: str | None = Field(default=None, description="Type of failure (assertion or exception)")
    harness_path: str | None = Field(default=None, description="Path to the generated test file")
    message: str = Field(default="", description="Detailed summary or error message")
    
    # Additional fields for sub-agent workflow
    analysis_summary: str | None = Field(default=None, description="Summary from analysis agent")
    generation_attempts: int = Field(default=0, description="Number of generation attempts made")
    validation_fixes: list[str] = Field(default_factory=list, description="List of fixes applied during validation")

    @property
    @override
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Convert observation to LLM-readable content."""
        llm_content: list[TextContent | ImageContent] = []

        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        content_parts: list[str] = []
        
        # Status header
        if self.status == "ok":
            content_parts.append("✅ FuzzHypoAgent completed successfully - no bugs found.")
        elif self.status == "fail":
            content_parts.append("🐛 FuzzHypoAgent found a bug!")
        elif self.status == "error":
            content_parts.append("❌ FuzzHypoAgent encountered an error.")
        elif self.status == "timeout":
            content_parts.append("⏱️ FuzzHypoAgent timed out.")
        else:
            content_parts.append(f"FuzzHypoAgent status: {self.status}")

        # Analysis summary
        if self.analysis_summary:
            content_parts.append(f"\n📊 Analysis: {self.analysis_summary}")

        # Generation info
        if self.generation_attempts > 0:
            content_parts.append(f"🔄 Generation attempts: {self.generation_attempts}")
        
        # Validation fixes
        if self.validation_fixes:
            content_parts.append(f"🔧 Validation fixes applied: {len(self.validation_fixes)}")
            for fix in self.validation_fixes[:3]:  # Show first 3 fixes
                content_parts.append(f"   - {fix}")

        # Bug details
        if self.bug_found and self.failing_input:
            content_parts.append(f"\n🔴 Failing input: {self.failing_input}")
        
        if self.failure_kind:
            content_parts.append(f"Failure type: {self.failure_kind}")

        # Message (contains detailed output)
        if self.message:
            content_parts.append(f"\n{self.message}")

        # Harness path for reference
        if self.harness_path:
            content_parts.append(f"\n📁 Test harness: {self.harness_path}")

        full_text = "\n".join(content_parts) if content_parts else "[FuzzHypoAgent completed]"
        llm_content.append(TextContent(text=full_text))

        return llm_content


TOOL_DESCRIPTION = """Run LLM-assisted Python fuzzing using intelligent sub-agents.

**Enhanced Features over FuzzHypo:**
- Uses specialized sub-agents for analysis, generation, and validation
- Multi-turn dialogue for better understanding of target function
- Automatic fix attempts when generated code fails validation
- More robust error handling and recovery

**Workflow:**
1. **Analysis Agent**: Examines target function source code and extracts:
   - Function signature and parameter types
   - Return type and possible exceptions
   - Edge cases and boundary conditions

2. **Generator Agent**: Creates Hypothesis testing components:
   - Smart strategy selection based on analysis
   - Context-aware post-conditions
   - Meaningful seed values for edge cases

3. **Validator Agent**: Ensures generated code is correct:
   - Syntax validation
   - Import checking
   - Test execution dry-run
   - Automatic fix attempts if issues found

**Usage:**
Provide a target function and specification, and the tool will orchestrate
sub-agents to generate and execute robust property-based tests.
"""


class FuzzHypoAgentTool(ToolDefinition[FuzzHypoAgentAction, FuzzHypoAgentObservation]):
    """Tool for automated discovery of software vulnerabilities using sub-agents."""

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["FuzzHypoAgentTool"]:
        from openhands.tools.fuzz_hypo_agent.impl import FuzzHypoAgentExecutor

        working_dir = conv_state.workspace.working_dir
        llm = conv_state.agent.llm
        executor = FuzzHypoAgentExecutor(working_dir=working_dir, llm=llm)

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=FuzzHypoAgentAction,
                observation_type=FuzzHypoAgentObservation,
                annotations=ToolAnnotations(
                    title="FuzzHypoAgent",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# Automatically register the tool when this module is imported
register_tool(FuzzHypoAgentTool.name, FuzzHypoAgentTool)
