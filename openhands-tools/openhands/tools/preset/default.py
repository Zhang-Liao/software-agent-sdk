"""Default preset configuration for OpenHands agents."""

from openhands.sdk import Agent
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import Tool


logger = get_logger(__name__)

# Define which tools are "extra" (not included by default)
# These tools must be explicitly enabled via --extra-tools
EXTRA_TOOL_NAMES = {
    # Fuzzing tools
    "fuzz_hypo",
    "fuzz_hypo_agent",
    "fuzz_hypo_v2",
    # Delegation tool (CLI name: subagent; internal tool name: delegate)
    "subagent",
    # Test tooling (now opt-in)
    "test_strategy_decider",
    "assertion_test",
    "simple_test",
}


def register_default_tools(enable_browser: bool = True) -> None:
    """Register the default set of tools (always available)."""
    # Tools are now automatically registered when imported
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    logger.debug(f"Tool: {TerminalTool.name} registered.")
    logger.debug(f"Tool: {FileEditorTool.name} registered.")
    logger.debug(f"Tool: {TaskTrackerTool.name} registered.")

    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet

        logger.debug(f"Tool: {BrowserToolSet.name} registered.")


def register_extra_tools(tool_names: list[str] | None = None) -> None:
    """Register extra tools that are not included by default.
    
    Args:
        tool_names: List of extra tool names to register. If None, no extra tools
                   are registered.
    """
    if not tool_names:
        return
    
    tool_names_lower = {name.lower() for name in tool_names}

    if "subagent" in tool_names_lower:
        from openhands.tools.delegate import DelegateTool
        logger.debug(
            f"Extra tool: {DelegateTool.name} registered (enabled via --extra-tools subagent)."
        )
    
    if "fuzz_hypo" in tool_names_lower:
        from openhands.tools.fuzz_hypo import FuzzHypoTool
        logger.debug(f"Extra tool: {FuzzHypoTool.name} registered.")
    
    if "fuzz_hypo_agent" in tool_names_lower:
        from openhands.tools.fuzz_hypo_agent import FuzzHypoAgentTool
        logger.debug(f"Extra tool: {FuzzHypoAgentTool.name} registered.")
    
    if "fuzz_hypo_v2" in tool_names_lower:
        from openhands.tools.fuzz_hypo_v2 import FuzzHypoV2Tool
        logger.debug(f"Extra tool: {FuzzHypoV2Tool.name} registered.")

    if "test_strategy_decider" in tool_names_lower:
        from openhands.tools.test_strategy_decider import TestStrategyDeciderTool
        logger.debug(f"Extra tool: {TestStrategyDeciderTool.name} registered.")

    if "assertion_test" in tool_names_lower:
        from openhands.tools.assertion_test import AssertionTestTool
        logger.debug(f"Extra tool: {AssertionTestTool.name} registered.")

    if "simple_test" in tool_names_lower:
        from openhands.tools.simple_test import SimpleTestTool
        logger.debug(f"Extra tool: {SimpleTestTool.name} registered.")


def get_available_extra_tools() -> list[str]:
    """Get list of available extra tool names."""
    return list(EXTRA_TOOL_NAMES)


def get_default_tools(
    enable_browser: bool = True,
    extra_tools: list[str] | None = None,
) -> list[Tool]:
    """Get the default set of tool specifications for the standard experience.

    Args:
        enable_browser: Whether to include browser tools.
        extra_tools: List of extra tool names to include. These are tools not
                    enabled by default. Examples: fuzz_hypo, fuzz_hypo_agent, fuzz_hypo_v2,
                    subagent, test_strategy_decider, assertion_test, simple_test.
    """
    register_default_tools(enable_browser=enable_browser)
    register_extra_tools(extra_tools)

    # Import tools to access their name attributes
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    # Default tools (always included)
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
    
    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet
        tools.append(Tool(name=BrowserToolSet.name))
    
    # Add extra tools if requested
    if extra_tools:
        extra_tools_lower = {name.lower() for name in extra_tools}

        if "subagent" in extra_tools_lower:
            from openhands.tools.delegate import DelegateTool
            tools.append(Tool(name=DelegateTool.name))
            logger.info(f"Added extra tool: {DelegateTool.name} (via --extra-tools subagent)")
        
        if "fuzz_hypo" in extra_tools_lower:
            from openhands.tools.fuzz_hypo import FuzzHypoTool
            tools.append(Tool(name=FuzzHypoTool.name))
            logger.info(f"Added extra tool: {FuzzHypoTool.name}")
        
        if "fuzz_hypo_agent" in extra_tools_lower:
            from openhands.tools.fuzz_hypo_agent import FuzzHypoAgentTool
            tools.append(Tool(name=FuzzHypoAgentTool.name))
            logger.info(f"Added extra tool: {FuzzHypoAgentTool.name}")
        
        if "fuzz_hypo_v2" in extra_tools_lower:
            from openhands.tools.fuzz_hypo_v2 import FuzzHypoV2Tool
            tools.append(Tool(name=FuzzHypoV2Tool.name))
            logger.info(f"Added extra tool: {FuzzHypoV2Tool.name}")

        if "test_strategy_decider" in extra_tools_lower:
            from openhands.tools.test_strategy_decider import TestStrategyDeciderTool
            tools.append(Tool(name=TestStrategyDeciderTool.name))
            logger.info(f"Added extra tool: {TestStrategyDeciderTool.name}")

        if "assertion_test" in extra_tools_lower:
            from openhands.tools.assertion_test import AssertionTestTool
            tools.append(Tool(name=AssertionTestTool.name))
            logger.info(f"Added extra tool: {AssertionTestTool.name}")

        if "simple_test" in extra_tools_lower:
            from openhands.tools.simple_test import SimpleTestTool
            tools.append(Tool(name=SimpleTestTool.name))
            logger.info(f"Added extra tool: {SimpleTestTool.name}")
    
    return tools


def get_default_condenser(llm: LLM) -> CondenserBase:
    # Create a condenser to manage the context. The condenser will automatically
    # truncate conversation history when it exceeds max_size, and replaces the dropped
    # events with an LLM-generated summary.
    condenser = LLMSummarizingCondenser(llm=llm, max_size=80, keep_first=4)

    return condenser


def get_default_agent(
    llm: LLM,
    cli_mode: bool = False,
    extra_tools: list[str] | None = None,
) -> Agent:
    tools = get_default_tools(
        # Disable browser tools in CLI mode
        enable_browser=not cli_mode,
        extra_tools=extra_tools,
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": cli_mode},
        condenser=get_default_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
    )
    return agent
