# Core tool interface for LLM-assisted Fuzzing with Sub-Agents
"""FuzzHypoAgent: Enhanced property-based testing using specialized sub-agents.

This module provides a more robust fuzzing tool that uses multiple sub-agents
for analysis, generation, and validation of test components.

Key Components:
- FuzzHypoAgentTool: Main tool definition
- FuzzHypoAgentAction: Input schema (target, spec, mode, max_iterations)
- FuzzHypoAgentObservation: Output schema with detailed results
- FuzzHypoAgentExecutor: Executor that orchestrates sub-agents

Usage:
    from openhands.tools.fuzz_hypo_agent import (
        FuzzHypoAgentTool,
        FuzzHypoAgentAction,
        FuzzHypoAgentObservation,
    )
    
    # The tool is automatically registered when imported
    # Use it like any other OpenHands tool

Sub-Agent Workflow:
    1. Analyzer Agent: Deep analysis of target function
    2. Generator Agent: Hypothesis strategy generation
    3. Validator Agent: Code validation and auto-fix
"""

from openhands.tools.fuzz_hypo_agent.definition import (
    FuzzHypoAgentAction,
    FuzzHypoAgentObservation,
    FuzzHypoAgentTool,
)
from openhands.tools.fuzz_hypo_agent.impl import FuzzHypoAgentExecutor

__all__ = [
    "FuzzHypoAgentTool",
    "FuzzHypoAgentAction",
    "FuzzHypoAgentObservation",
    "FuzzHypoAgentExecutor",
]
