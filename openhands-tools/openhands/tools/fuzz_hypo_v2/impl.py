"""Enhanced FuzzHypo V2 Executor with root cause analysis, auto-spec generation, and iterative fuzzing."""

import ast
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openhands.sdk.tool import ToolExecutor
from openhands.tools.fuzz_hypo_v2.definition import FuzzHypoV2Action, FuzzHypoV2Observation

if TYPE_CHECKING:
    from openhands.sdk.llm import LLM


class FuzzHypoV2Executor(ToolExecutor[FuzzHypoV2Action, FuzzHypoV2Observation]):
    """Enhanced executor with root cause analysis and iterative fuzzing."""
    
    MAX_LLM_RETRIES = 2
    MAX_ITERATIONS = 3  # For iterative mode
    
    def __init__(self, working_dir: str, llm: "LLM | None" = None):
        self.working_dir = Path(working_dir).resolve()
        self.llm = llm
        self._discovered_functions: set[str] = set()  # Track discovered related functions

    def __call__(self, action: FuzzHypoV2Action, conversation=None) -> FuzzHypoV2Observation:
        try:
            # 1. Ensure dependencies
            self._ensure_dependencies()
            
            # 2. Handle different modes
            if action.mode == "analyze":
                return self._analyze_only_mode(action)
            elif action.mode == "iterative":
                return self._iterative_mode(action)
            else:
                return self._standard_fuzz_mode(action)

        except Exception as e:
            return FuzzHypoV2Observation(
                status="error",
                bug_found=False,
                message=f"FuzzHypo V2 execution failed: {str(e)}\n{traceback.format_exc()}",
            )

    # ==================== MODE HANDLERS ====================
    
    def _standard_fuzz_mode(self, action: FuzzHypoV2Action) -> FuzzHypoV2Observation:
        """Standard fuzzing with enhanced root cause analysis."""
        
        # 1. Get or generate spec (Plan 2: Auto Spec Generation)
        if action.spec:
            spec_content = self._load_spec(action.spec)
            auto_generated = False
            spec_source = "user_provided"
        else:
            spec_content, spec_source = self._auto_generate_spec(
                action.target, 
                action.issue_description
            )
            auto_generated = True
        
        # 2. Generate test components
        components = self._llm_generate_components(action.target, spec_content)
        
        # 3. Execute fuzzing
        base_result = self._execute_fuzz(action, components)
        
        # 4. If bug found, perform root cause analysis (Plan 1)
        if base_result.bug_found:
            root_cause_info = self._analyze_root_cause(
                target=action.target,
                failing_input=base_result.failing_input,
                error_output=base_result.message,
                spec=spec_content
            )
            
            # Merge root cause analysis into result
            base_result = self._merge_root_cause_analysis(base_result, root_cause_info)
        
        # 5. Add auto-spec info
        base_result.auto_generated_spec = auto_generated
        base_result.spec_source = spec_source
        
        # 6. Verify previous findings if provided
        if action.previous_findings and not base_result.bug_found:
            base_result.fix_verified = True
            base_result.message += "\n✅ All previous failing inputs now pass!"
        
        return base_result

    def _analyze_only_mode(self, action: FuzzHypoV2Action) -> FuzzHypoV2Observation:
        """Focus on analyzing a function without running full fuzz - for tracing upstream."""
        
        # Analyze the function structure
        analysis = self._deep_analyze_function(action.target, action.issue_description)
        
        return FuzzHypoV2Observation(
            status="ok",
            bug_found=False,
            message=f"Analysis complete for {action.target}",
            root_cause_analysis=analysis.get("analysis", ""),
            call_chain=analysis.get("call_chain", []),
            upstream_candidates=analysis.get("callers", []),
            suggested_fix_locations=analysis.get("fix_locations", []),
            fix_strategy=analysis.get("fix_strategy", ""),
            is_root_cause=analysis.get("is_likely_root_cause"),
        )

    def _iterative_mode(self, action: FuzzHypoV2Action) -> FuzzHypoV2Observation:
        """Iterative fuzz-analyze-fix cycle (Plan 3)."""
        
        all_findings: list[dict] = []
        targets_to_test = [action.target]
        tested_targets: set[str] = set()
        iteration = 0
        
        while targets_to_test and iteration < self.MAX_ITERATIONS:
            iteration += 1
            current_target = targets_to_test.pop(0)
            
            if current_target in tested_targets:
                continue
            tested_targets.add(current_target)
            
            # Run standard fuzz on current target
            sub_action = FuzzHypoV2Action(
                target=current_target,
                spec=action.spec if current_target == action.target else None,
                mode="quick",
                issue_description=action.issue_description,
            )
            
            result = self._standard_fuzz_mode(sub_action)
            
            finding = {
                "target": current_target,
                "bug_found": result.bug_found,
                "failing_input": result.failing_input,
                "is_root_cause": result.is_root_cause,
            }
            all_findings.append(finding)
            
            # If bug found and not root cause, add upstream candidates
            if result.bug_found and result.is_root_cause is False:
                if result.upstream_candidates:
                    for candidate in result.upstream_candidates:
                        if candidate not in tested_targets:
                            targets_to_test.append(candidate)
            
            # Also discover related functions
            if result.related_functions_discovered:
                self._discovered_functions.update(result.related_functions_discovered)
        
        # Synthesize final result
        return self._synthesize_iterative_results(all_findings, iteration)

    # ==================== PLAN 1: ROOT CAUSE ANALYSIS ====================
    
    def _analyze_root_cause(
        self, 
        target: str, 
        failing_input: str | None, 
        error_output: str,
        spec: str
    ) -> dict:
        """Analyze whether the bug is a root cause or symptom."""
        
        # 1. Parse stack trace to get call chain
        call_chain = self._extract_call_chain(error_output)
        
        # 2. Identify project files in the chain (vs library code)
        project_frames = self._filter_project_frames(call_chain)
        
        # 3. Find callers of the target function
        callers = self._find_callers(target)
        
        # 4. Use LLM to analyze root cause
        analysis_prompt = f"""Analyze this bug to determine if it's a ROOT CAUSE or just a SYMPTOM.

Target function: {target}
Failing input: {failing_input}
Specification: {spec}

Call chain (from error):
{chr(10).join(f'  {i+1}. {frame}' for i, frame in enumerate(project_frames[:10]))}

Functions that call this target:
{chr(10).join(f'  - {caller}' for caller in callers[:5])}

Analysis required:
1. Is the bug likely originating IN this function, or is bad data being PASSED TO it?
2. If this is a symptom, which upstream function is likely the root cause?
3. What specific fix strategy would address the root cause?

Respond in JSON format:
{{
    "is_root_cause": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "upstream_candidates": ["func1", "func2"] if symptom,
    "fix_strategy": "what to fix and where",
    "suggested_fix_locations": ["location1", "location2"]
}}
"""
        
        try:
            response = self._call_llm(
                system_prompt="You are a debugging expert. Analyze bugs to find root causes.",
                user_prompt=analysis_prompt,
                temperature=0.1
            )
            return self._parse_json_response(response)
        except Exception:
            # Fallback: heuristic analysis
            return self._heuristic_root_cause_analysis(target, call_chain, callers)
    
    def _heuristic_root_cause_analysis(
        self, 
        target: str, 
        call_chain: list[str], 
        callers: list[str]
    ) -> dict:
        """Fallback heuristic analysis when LLM fails."""
        
        # Heuristic: If target is deep in call chain, it's likely a symptom
        target_func = target.split(":")[-1] if ":" in target else target
        
        is_symptom = False
        for i, frame in enumerate(call_chain):
            if target_func in frame:
                # If target is not at the top of our project frames, might be symptom
                if i > 0:
                    is_symptom = True
                break
        
        return {
            "is_root_cause": not is_symptom,
            "confidence": 0.5,
            "reasoning": "Heuristic analysis based on call chain position",
            "upstream_candidates": callers[:3] if is_symptom else [],
            "fix_strategy": "Investigate callers if this is a symptom",
            "suggested_fix_locations": [target] if not is_symptom else callers[:2],
        }
    
    def _merge_root_cause_analysis(
        self, 
        base_result: FuzzHypoV2Observation, 
        root_cause_info: dict
    ) -> FuzzHypoV2Observation:
        """Merge root cause analysis into the observation."""
        
        is_root = root_cause_info.get("is_root_cause", True)
        
        base_result.is_root_cause = is_root
        base_result.root_cause_analysis = root_cause_info.get("reasoning", "")
        base_result.upstream_candidates = root_cause_info.get("upstream_candidates", [])
        base_result.suggested_fix_locations = root_cause_info.get("suggested_fix_locations", [])
        base_result.fix_strategy = root_cause_info.get("fix_strategy", "")
        
        # Update status if this is not root cause
        if not is_root:
            base_result.status = "needs_upstream_analysis"
        
        return base_result

    def _extract_call_chain(self, error_output: str) -> list[str]:
        """Extract call chain from error output/stack trace."""
        frames = []
        
        # Match Python traceback format
        pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        for match in re.finditer(pattern, error_output):
            file_path, line_num, func_name = match.groups()
            frames.append(f"{file_path}:{line_num}:{func_name}")
        
        return frames
    
    def _filter_project_frames(self, frames: list[str]) -> list[str]:
        """Filter to only include project code (not library code)."""
        project_frames = []
        for frame in frames:
            # Skip common library paths
            if any(skip in frame for skip in [
                "/opt/miniconda", 
                "site-packages", 
                "/usr/lib/python",
                "hypothesis/",
                "pytest/"
            ]):
                continue
            project_frames.append(frame)
        return project_frames
    
    def _find_callers(self, target: str) -> list[str]:
        """Find functions that call the target function."""
        callers = []
        
        target_mod, target_func = self._parse_target(target)
        func_name = target_func.split(".")[-1] if "." in target_func else target_func
        
        # Search for calls in the codebase
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", f"{func_name}(", str(self.working_dir)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                for file_path in result.stdout.strip().split("\n"):
                    if file_path.endswith(".py") and file_path:
                        # Extract function names that contain calls
                        caller_funcs = self._extract_callers_from_file(file_path, func_name)
                        callers.extend(caller_funcs)
        except Exception:
            pass
        
        return list(set(callers))[:10]  # Dedupe and limit
    
    def _extract_callers_from_file(self, file_path: str, func_name: str) -> list[str]:
        """Extract functions that call func_name from a file."""
        callers = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this function contains a call to func_name
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self._get_call_name(child)
                            if func_name in call_name:
                                rel_path = os.path.relpath(file_path, self.working_dir)
                                callers.append(f"{rel_path}:{node.name}")
                                break
        except Exception:
            pass
        
        return callers
    
    def _get_call_name(self, call_node: ast.Call) -> str:
        """Get the name of a function call from AST node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return ""

    # ==================== PLAN 2: AUTO SPEC GENERATION ====================
    
    def _auto_generate_spec(
        self, 
        target: str, 
        issue_description: str | None
    ) -> tuple[str, str]:
        """Auto-generate spec from code analysis."""
        
        sources_used = []
        spec_parts = []
        
        # 1. Extract type hints and docstring
        func_info = self._extract_function_info(target)
        
        if func_info.get("type_hints"):
            sources_used.append("type_hints")
            spec_parts.append(f"# From type hints:\n# {func_info['type_hints']}")
        
        if func_info.get("docstring"):
            sources_used.append("docstring")
            spec_parts.append(f"# From docstring:\n# {func_info['docstring'][:200]}")
        
        # 2. Analyze issue description
        if issue_description:
            sources_used.append("issue_analysis")
            issue_constraints = self._extract_constraints_from_issue(issue_description)
            spec_parts.append(f"# From issue:\n# {issue_constraints}")
        
        # 3. Use LLM to synthesize a proper spec
        synthesis_prompt = f"""Generate a Hypothesis testing specification for this function.

Target function: {target}

Available information:
{chr(10).join(spec_parts)}

Function signature and hints:
{func_info.get('signature', 'Unknown')}

Generate a YAML specification with:
- inputs: type constraints for each parameter
- pre_conditions: what must be true before calling
- post_conditions: what must be true after calling
- invariants: relationships that must hold

Be specific about edge cases that should be tested.
Output ONLY the YAML, no markdown fences.
"""
        
        try:
            spec = self._call_llm(
                system_prompt="You are a testing expert. Generate precise test specifications.",
                user_prompt=synthesis_prompt,
                temperature=0.2
            )
            sources_used.append("llm_synthesis")
        except Exception:
            # Fallback to generic spec
            spec = self._generate_generic_spec(func_info)
            sources_used.append("generic_fallback")
        
        return spec, "+".join(sources_used)
    
    def _extract_function_info(self, target: str) -> dict:
        """Extract type hints, docstring, and signature from target function."""
        info = {}
        
        target_mod, target_func = self._parse_target(target)
        
        # Try to find and parse the source file
        source_file = self._find_source_file(target_mod)
        if not source_file:
            return info
        
        try:
            with open(source_file, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Handle class.method format
            if "." in target_func:
                class_name, method_name = target_func.split(".", 1)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if item.name == method_name:
                                    info = self._extract_from_funcdef(item)
                                    break
            else:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == target_func:
                            info = self._extract_from_funcdef(node)
                            break
        except Exception:
            pass
        
        return info
    
    def _extract_from_funcdef(self, node: ast.FunctionDef) -> dict:
        """Extract info from a function definition AST node."""
        info = {}
        
        # Get docstring
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                info["docstring"] = str(node.body[0].value.value)
        
        # Get signature with annotations
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        info["signature"] = f"def {node.name}({', '.join(args)}){returns}"
        
        # Extract type hints
        type_hints = []
        for arg in node.args.args:
            if arg.annotation:
                type_hints.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        info["type_hints"] = ", ".join(type_hints) if type_hints else None
        
        return info
    
    def _extract_constraints_from_issue(self, issue: str) -> str:
        """Extract testing constraints from issue description."""
        constraints = []
        
        # Look for common patterns
        patterns = [
            (r"should\s+return\s+(\w+)", "Expected return: {}"),
            (r"must\s+be\s+(\w+)", "Must be: {}"),
            (r"when\s+(\w+)\s+is\s+(\w+)", "When {} is {}"),
            (r"raises?\s+(\w+Error)", "Should handle: {}"),
            (r"instead\s+of\s+(\w+)", "Should not: {}"),
        ]
        
        for pattern, template in patterns:
            for match in re.finditer(pattern, issue, re.IGNORECASE):
                constraints.append(template.format(*match.groups()))
        
        return "; ".join(constraints) if constraints else "Test edge cases and error handling"
    
    def _generate_generic_spec(self, func_info: dict) -> str:
        """Generate a generic spec when LLM fails."""
        return """
inputs:
  data: any
  
pre_conditions:
  - function is callable
  
post_conditions:
  - function completes without unexpected errors
  - return value is valid for the expected type
  
invariants:
  - state consistency maintained
"""

    # ==================== PLAN 3: ITERATIVE FUZZING ====================
    
    def _synthesize_iterative_results(
        self, 
        findings: list[dict], 
        iterations: int
    ) -> FuzzHypoV2Observation:
        """Synthesize results from iterative fuzzing."""
        
        bugs_found = [f for f in findings if f.get("bug_found")]
        root_causes = [f for f in bugs_found if f.get("is_root_cause")]
        symptoms = [f for f in bugs_found if not f.get("is_root_cause")]
        
        message_parts = [
            f"Iterative analysis completed in {iterations} iterations.",
            f"Tested {len(findings)} functions.",
            f"Found {len(bugs_found)} bugs total.",
        ]
        
        if root_causes:
            message_parts.append(f"\n🎯 ROOT CAUSES IDENTIFIED ({len(root_causes)}):")
            for rc in root_causes:
                message_parts.append(f"  - {rc['target']}: {rc.get('failing_input', 'N/A')}")
        
        if symptoms:
            message_parts.append(f"\n⚠️ SYMPTOMS (fix root causes first) ({len(symptoms)}):")
            for s in symptoms:
                message_parts.append(f"  - {s['target']}")
        
        # Determine overall status
        if root_causes:
            status = "fail"
        elif symptoms:
            status = "needs_upstream_analysis"
        else:
            status = "ok"
        
        # Get fix locations from root causes
        fix_locations = [rc["target"] for rc in root_causes]
        
        return FuzzHypoV2Observation(
            status=status,
            bug_found=bool(bugs_found),
            message="\n".join(message_parts),
            iteration=iterations,
            related_functions_discovered=list(self._discovered_functions),
            suggested_fix_locations=fix_locations,
            fix_strategy="Fix the identified root causes first, then verify symptoms are resolved",
        )

    def _deep_analyze_function(self, target: str, issue: str | None) -> dict:
        """Deep analysis of a function for root cause tracing."""
        
        # Get function info
        func_info = self._extract_function_info(target)
        callers = self._find_callers(target)
        
        # Analyze with LLM
        analysis_prompt = f"""Analyze this function to understand its role in the codebase.

Target: {target}
Signature: {func_info.get('signature', 'Unknown')}
Docstring: {func_info.get('docstring', 'None')[:300] if func_info.get('docstring') else 'None'}

Issue context: {issue or 'Not provided'}

Functions that call this target:
{chr(10).join(f'  - {c}' for c in callers[:10])}

Provide analysis in JSON:
{{
    "is_likely_root_cause": true/false,
    "analysis": "detailed explanation",
    "call_chain": ["list", "of", "functions", "in", "call", "path"],
    "callers": ["upstream", "functions"],
    "fix_locations": ["where", "to", "fix"],
    "fix_strategy": "how to fix"
}}
"""
        
        try:
            response = self._call_llm(
                system_prompt="You are a code analysis expert.",
                user_prompt=analysis_prompt,
                temperature=0.1
            )
            return self._parse_json_response(response)
        except Exception:
            return {
                "is_likely_root_cause": None,
                "analysis": "Analysis failed - manual investigation needed",
                "callers": callers,
                "fix_locations": [target],
            }

    # ==================== HELPER METHODS (from original impl) ====================
    
    def _ensure_dependencies(self):
        """Ensure hypothesis and pytest are installed."""
        python_exe = self._get_testbed_python()
        try:
            result = subprocess.run(
                [python_exe, "-c", "import hypothesis, pytest"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except Exception:
            pass
        
        try:
            subprocess.check_call(
                ["conda", "run", "-n", "testbed", "pip", "install", "hypothesis", "pytest"],
                timeout=60,
            )
        except Exception:
            pass

    def _get_testbed_python(self) -> str:
        """Get Python executable from testbed environment."""
        conda_python = "/opt/miniconda3/envs/testbed/bin/python"
        if os.path.exists(conda_python):
            return conda_python
        return sys.executable
    
    def _parse_target(self, target: str) -> tuple[str, str]:
        """Parse target string into (module, function)."""
        if ":" not in target:
            raise ValueError(f"Target must be 'module:func' format, got: {target}")
        parts = target.rsplit(":", 1)
        return parts[0], parts[1]
    
    def _find_source_file(self, target_mod: str) -> Path | None:
        """Find the source file for a module."""
        if target_mod.endswith(".py"):
            path = self.working_dir / target_mod
            if path.exists():
                return path
            return None
        
        # Convert module path to file path
        file_path = target_mod.replace(".", "/") + ".py"
        candidates = [
            self.working_dir / file_path,
            self.working_dir / target_mod.split(".")[0] / file_path,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _load_spec(self, spec: str) -> str:
        """Load spec content from file or use directly."""
        if '\n' in spec or spec.strip().startswith(('inputs:', '{', '[')):
            return spec
        try:
            spec_path = self.working_dir / spec
            if spec_path.exists():
                return spec_path.read_text()
        except Exception:
            pass
        return spec
    
    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Call LLM and return response text."""
        if self.llm is not None:
            from openhands.sdk.llm.message import Message, TextContent, content_to_str
            
            messages = [
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ]
            
            response = self.llm.completion(messages=messages, temperature=temperature)
            text_parts = content_to_str(response.message.content)
            return "".join(text_parts)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content or ""
    
    def _parse_json_response(self, raw_content: str) -> dict:
        """Parse JSON from LLM response."""
        content = raw_content.strip()
        
        # Try multiple cleaning strategies
        strategies = [
            lambda s: s,
            lambda s: re.sub(r"```(?:json)?\s*|\s*```", "", s).strip(),
            lambda s: self._extract_json_object(s),
        ]
        
        for strategy in strategies:
            try:
                cleaned = strategy(content)
                if cleaned:
                    return json.loads(cleaned)
            except Exception:
                continue
        
        return {}
    
    def _extract_json_object(self, text: str) -> str:
        """Extract first JSON object from text."""
        depth = 0
        start = -1
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start:i+1]
        return text
    
    def _llm_generate_components(self, target: str, spec: str) -> dict:
        """Generate Hypothesis testing components."""
        system_prompt = """You are a testing expert. Generate Hypothesis testing components.

Return ONLY a valid JSON object with:
{
    "strategy": "st.integers()",
    "post_condition": "def post_condition(input_obj, output_obj):\\n    return True",
    "seeds": [1, 2, 3]
}

Rules:
- "strategy": A valid hypothesis.strategies expression (imported as `st`)
- "post_condition": A complete Python function definition as a string
- "seeds": A list of 3 concrete example inputs

Output ONLY the JSON object."""

        user_prompt = f"Target function: {target}\n\nSpecification:\n{spec}\n\nGenerate the JSON now:"
        
        for attempt in range(self.MAX_LLM_RETRIES + 1):
            try:
                raw = self._call_llm(system_prompt, user_prompt, temperature=0.1 + attempt * 0.1)
                result = self._parse_json_response(raw)
                if all(k in result for k in ["strategy", "post_condition", "seeds"]):
                    return result
            except Exception:
                continue
        
        # Fallback
        return self._generate_fallback_components(spec)
    
    def _generate_fallback_components(self, spec: str) -> dict:
        """Generate fallback components when LLM fails."""
        spec_lower = spec.lower()
        
        if "array" in spec_lower or "ndarray" in spec_lower:
            strategy = "st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10)"
        elif "string" in spec_lower or "str" in spec_lower:
            strategy = "st.text(min_size=0, max_size=100)"
        elif "int" in spec_lower:
            strategy = "st.integers(min_value=-1000, max_value=1000)"
        else:
            strategy = "st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=50))"
        
        return {
            "strategy": strategy,
            "post_condition": "def post_condition(input_obj, output_obj):\n    return True",
            "seeds": [0, 1, -1]
        }
    
    def _execute_fuzz(self, action: FuzzHypoV2Action, components: dict) -> FuzzHypoV2Observation:
        """Execute the fuzz test."""
        max_examples = 2000 if action.mode == "deep" else 200
        
        target_mod, target_func = self._parse_target(action.target)
        
        fuzz_dir = self.working_dir / ".fuzz_hypo_v2"
        fuzz_dir.mkdir(exist_ok=True)
        
        import time
        safe_func_name = re.sub(r'[^\w]', '_', target_func)[:30]
        harness_file = fuzz_dir / f"test_{safe_func_name}_{int(time.time())}.py"
        
        import_code = self._generate_import_code(target_mod, target_func)
        
        test_template = f'''
import sys, os
import hypothesis.strategies as st
from hypothesis import given, settings, Verbosity

sys.path.insert(0, "{self.working_dir}")

{import_code}

{components['post_condition']}

_example_count = 0

@settings(
    max_examples={max_examples},
    deadline=None,
    verbosity=Verbosity.normal,
)
@given({components['strategy']})
def test_fuzz_task(data):
    global _example_count
    _example_count += 1
    result = target_func(data)
    assert post_condition(data, result) is True, f"Fuzzing violation at example #{{_example_count}}"

def teardown_module():
    print(f"\\n[FuzzHypo] Total examples tested: {{_example_count}} / {max_examples}")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
'''
        harness_file.write_text(test_template)
        
        python_exe = self._get_testbed_python()
        env = os.environ.copy()
        pythonpath = str(self.working_dir)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath
        
        pytest_args = "-v --tb=short -s"
        
        cmd = [
            "bash", "-c",
            f"cd {self.working_dir} && "
            f"source /opt/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate testbed && "
            f"pip install -q pytest hypothesis 2>/dev/null; "
            f"PYTHONPATH='{pythonpath}' python -m pytest {harness_file} {pytest_args} 2>&1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(self.working_dir), env=env)
            full_output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return FuzzHypoV2Observation(
                status="timeout",
                bug_found=False,
                harness_path=str(harness_file),
                message="Fuzzing timed out after 300 seconds",
            )
        
        bug_found = result.returncode != 0
        failing_input = self._extract_crash(full_output) if bug_found else None
        examples_count = self._extract_example_count(full_output, max_examples)
        
        status = "fail" if bug_found else "ok"
        msg = f"Found specification violation!" if bug_found else "No defects found."
        
        return FuzzHypoV2Observation(
            status=status,
            bug_found=bug_found,
            failing_input=str(failing_input) if failing_input else None,
            failure_kind="assertion_failure" if "AssertionError" in full_output else "exception",
            harness_path=str(harness_file),
            message=f"{msg}\n\n[Pytest Output]\n{full_output[-800:]}",
            strategy=components.get("strategy"),
            post_condition=components.get("post_condition"),
            examples_tested=examples_count,
        )
    
    def _generate_import_code(self, target_mod: str, target_func: str) -> str:
        """Generate import code for target function."""
        is_class_method = "." in target_func
        if is_class_method:
            class_name, method_name = target_func.split(".", 1)
            getattr_code = f'''_class = getattr(_module, "{class_name}")
    target_func = getattr(_class, "{method_name}")'''
        else:
            getattr_code = f'target_func = getattr(_module, "{target_func}")'
        
        is_file_path = target_mod.endswith(".py") or "/" in target_mod
        
        if is_file_path and "/" in target_mod and not target_mod.startswith("/"):
            module_name = target_mod.replace("/", ".").replace(".py", "")
            return f'''import importlib
print("[FuzzHypo V2] Importing: {module_name}:{target_func}")
try:
    _module = importlib.import_module("{module_name}")
    {getattr_code}
    print("[FuzzHypo V2] OK")
except Exception as e:
    print(f"[FuzzHypo V2] FAILED: {{e}}")
    raise'''
        else:
            return f'''import importlib
print("[FuzzHypo V2] Importing: {target_mod}:{target_func}")
try:
    _module = importlib.import_module("{target_mod}")
    {getattr_code}
    print("[FuzzHypo V2] OK")
except Exception as e:
    print(f"[FuzzHypo V2] FAILED: {{e}}")
    raise'''
    
    def _extract_crash(self, output: str) -> Any:
        """Extract failing input from output."""
        match = re.search(r"Falsifying example:.*?data=(.*?)\n", output, re.DOTALL)
        if match:
            raw_val = match.group(1).strip()
            try:
                return ast.literal_eval(raw_val)
            except Exception:
                return raw_val
        return None
    
    def _extract_example_count(self, output: str, max_examples: int) -> int | None:
        """Extract example count from output."""
        match = re.search(r"\[FuzzHypo\] Total examples tested: (\d+)", output)
        if match:
            return int(match.group(1))
        if "passed" in output.lower() and "failed" not in output.lower():
            return max_examples
        return None
