"""FuzzHypo tool implementation for LLM-assisted property-based testing."""

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


class FuzzHypoAction(Action):
    """Schema for LLM-assisted Python fuzzing operations."""

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
    extra_imports: list[str] | None = Field(
        default=None,
        description="List of additional packages to import in the test harness. "
                    "E.g., ['numpy', 'pandas', 'datetime']. These will be installed if missing."
    )
    import_statements: list[str] | None = Field(
        default=None,
        description="Custom import statements to add to the test harness. "
                    "E.g., ['import numpy as np', 'from datetime import datetime']. "
                    "Use this for more control over imports (aliases, specific imports)."
    )


class FuzzHypoObservation(Observation):
    """Observation from fuzzing operations, containing structured crash reports."""

    status: str = Field(description="'ok', 'fail', 'error', or 'timeout'")
    bug_found: bool = Field(default=False, description="True if a falsifying example was found")
    failing_input: str | None = Field(default=None, description="The first input that caused the crash (for backwards compatibility)")
    failing_inputs: list[dict] | None = Field(
        default=None, 
        description="List of all failing inputs (up to 10), each with 'input', 'error', 'repr' fields"
    )
    num_failures: int = Field(default=0, description="Total number of unique failures found")
    failure_kind: str | None = Field(default=None, description="Type of failure (assertion or exception)")
    harness_path: str | None = Field(default=None, description="Path to the generated test file")
    message: str = Field(default="", description="Detailed summary or error message")
    
    # Test configuration details
    strategy: str | None = Field(default=None, description="Hypothesis strategy used for input generation")
    post_condition: str | None = Field(default=None, description="Post-condition assertion code")
    examples_tested: int | None = Field(default=None, description="Number of examples tested")
    sample_inputs: list[str] | None = Field(default=None, description="Sample inputs that were tested")
    harness_code: str | None = Field(default=None, description="Generated test harness code (truncated)")

    @property
    @override
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Convert observation to LLM-readable content.
        
        This method is required to provide content for the LLM. Without it,
        the base class returns an empty list which causes IndexError when
        prompt caching tries to access content[-1].
        """
        llm_content: list[TextContent | ImageContent] = []

        # If is_error is true, prepend error message header
        if self.is_error:
            llm_content.append(TextContent(text=self.ERROR_MESSAGE_HEADER))

        # Build the content text from our custom fields
        content_parts: list[str] = []
        
        # Status header with examples count
        if self.examples_tested is not None and self.examples_tested > 0:
            examples_info = f" ({self.examples_tested} examples)"
        else:
            examples_info = ""
        
        if self.status == "ok":
            if self.examples_tested is None or self.examples_tested == 0:
                content_parts.append("⚠️ FuzzHypo completed but no examples were tested. Check harness for errors.")
            else:
                content_parts.append(f"✅ FuzzHypo completed successfully{examples_info} - no issues found.")
        elif self.status == "fail":
            content_parts.append(f"🔍 FuzzHypo detected issues{examples_info}!")
        elif self.status == "error":
            content_parts.append("❌ FuzzHypo encountered an error.")
        elif self.status == "timeout":
            content_parts.append(f"⏱️ FuzzHypo timed out{examples_info}.")
        else:
            content_parts.append(f"FuzzHypo status: {self.status}")

        # Bug details - group by error type to avoid repetition
        if self.bug_found:
            if self.failing_inputs and len(self.failing_inputs) > 0:
                # Group failures by error message
                error_groups: dict[str, list[str]] = {}
                for failure in self.failing_inputs:
                    error_msg = failure.get("error", "unknown error")
                    input_repr = failure.get("repr", failure.get("input", "unknown"))
                    if error_msg not in error_groups:
                        error_groups[error_msg] = []
                    error_groups[error_msg].append(input_repr)
                
                num_error_types = len(error_groups)
                total_examples = sum(len(inputs) for inputs in error_groups.values())
                content_parts.append(f"\n**Found {num_error_types} error type(s) across {total_examples} example(s):**")
                
                for i, (error_msg, inputs) in enumerate(error_groups.items(), 1):
                    # Truncate error message
                    display_error = error_msg[:150] + "..." if len(error_msg) > 150 else error_msg
                    content_parts.append(f"\n  {i}. **Error:** {display_error}")
                    content_parts.append(f"     **Triggered by {len(inputs)} input(s):**")
                    # Show each input on its own line for readability
                    for j, inp in enumerate(inputs[:5]):  # Show up to 5 inputs
                        truncated_inp = inp[:100] + "..." if len(inp) > 100 else inp
                        content_parts.append(f"       - `{truncated_inp}`")
                    if len(inputs) > 5:
                        content_parts.append(f"       - ... and {len(inputs) - 5} more")
            elif self.failing_input:
                # Backwards compatibility
                content_parts.append(f"\n**Failing input:** `{self.failing_input}`")

        # Test configuration details
        if self.strategy or self.post_condition:
            content_parts.append("\n--- Test Configuration ---")
            
        if self.strategy:
            content_parts.append(f"**Strategy:** `{self.strategy}`")
        
        if self.post_condition:
            # 截断过长的后验条件
            pc = self.post_condition
            if len(pc) > 300:
                pc = pc[:300] + "..."
            content_parts.append(f"**Post-condition:**\n```python\n{pc}\n```")
        
        # Sample inputs
        if self.sample_inputs:
            samples = self.sample_inputs[:5]  # 最多显示5个
            content_parts.append(f"**Sample inputs tested:** {samples}")

        # Message (contains detailed output)
        if self.message:
            content_parts.append(f"\n{self.message}")

        # Note: harness_code is intentionally NOT included to reduce output size
        # The harness is saved to harness_path for debugging if needed
        
        # Always show harness path for debugging
        if self.harness_path:
            content_parts.append(f"\n**Harness file:** {self.harness_path}")

        # Join all parts into a single text content
        full_text = "\n".join(content_parts) if content_parts else "[FuzzHypo completed]"
        llm_content.append(TextContent(text=full_text))

        return llm_content


TOOL_DESCRIPTION = """Run LLM-assisted Python fuzzing on a target function.
* Automatically generates Hypothesis strategies and post-conditions based on your provided spec.
* Detects edge cases, crashes, and semantic violations that regular tests might miss.
* Collects up to 10 unique failures with detailed input representations.
* Use this to REPRODUCE bugs or VERIFY complex fixes in logic.

IMPORTANT: Create a Wrapper Function First!
-------------------------------------------
Before calling this tool, you MUST create a simple wrapper function in a test file. 
This is because the original function may have complex signatures (class methods, 
special methods like __new__, multiple parameters, etc.) that are hard to fuzz directly.

SWE-bench note: keep scratch files untracked.
-------------------------------------------
If you are running inside SWE-bench-style evaluation, avoid adding temporary wrapper files
to the final git patch. Put wrappers under an ignored directory such as `.fuzz_hypo/` or
`.openhands/` (recommended) so `git add -A` in the repo won't pick them up.

Steps:
1. Create a test file (e.g., `test_fuzz_wrapper.py`) in the workspace
2. In the file, import the target function/class and create a simple wrapper:
   - The wrapper should have simple inputs (e.g., int, float, str, list)
   - The wrapper calls the actual target function with appropriate setup
   - The wrapper returns a result that can be checked
3. Call fuzz_hypo with target pointing to your wrapper function

Example workflow:
-----------------
# Step 1: Create test_fuzz_wrapper.py
```python
from mymodule import Quantity
import numpy as np

def fuzz_quantity_creation(value: float, dtype_name: str):
    '''Wrapper to test Quantity creation with different dtypes.'''
    dtype_map = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}
    dtype = dtype_map.get(dtype_name, np.float64)
    arr = np.array([value], dtype=dtype)
    q = Quantity(arr, unit='m')
    # Return something checkable
    return q.dtype == dtype

def fuzz_class_method(x: int, y: int):
    '''Wrapper to test a class method.'''
    obj = MyClass(x)
    result = obj.some_method(y)
    return result
```

# Step 2: Call fuzz_hypo on the wrapper
# IMPORTANT: spec MUST clearly describe the function signature!
fuzz_hypo(
    target="test_fuzz_wrapper:fuzz_quantity_creation",
    spec='''
Function signature: fuzz_quantity_creation(value: float, dtype_name: str)
Parameters:
  - value: a float number, can be any float including edge cases like 0.0, -inf, nan
  - dtype_name: one of "float16", "float32", "float64"
Post-condition: function should return True if dtype is preserved correctly
''',
    mode="quick"
)

Parameters:
- target: Function to test (e.g., 'test_fuzz_wrapper:fuzz_quantity_creation')
- spec: MUST include:
  1. Function signature with parameter names and types
  2. Description of each parameter's valid values/ranges
  3. Post-condition to check
- mode: 'quick' (200 examples), 'deep' (2000 examples), or 'debug'
- extra_imports: List of packages to import (e.g., ['numpy', 'pandas'])
- import_statements: Custom import lines (e.g., ['import numpy as np'])

The wrapper approach solves these problems:
- Class methods (__init__, __new__, instance methods)
- Functions with complex parameter types
- Functions requiring setup/teardown
- Functions with side effects that need isolation
"""


class FuzzHypoTool(ToolDefinition[FuzzHypoAction, FuzzHypoObservation]):
    """Tool for automated discovery of software vulnerabilities."""

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["FuzzHypoTool"]:
        from openhands.tools.fuzz_hypo.impl import FuzzHypoExecutor

        working_dir = conv_state.workspace.working_dir
        # 传递 LLM 配置给 executor，使其使用与 openhands 相同的 API
        llm = conv_state.agent.llm
        executor = FuzzHypoExecutor(working_dir=working_dir, llm=llm)

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=FuzzHypoAction,
                observation_type=FuzzHypoObservation,
                annotations=ToolAnnotations(
                    title="FuzzHypo",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]

register_tool(FuzzHypoTool.name, FuzzHypoTool)