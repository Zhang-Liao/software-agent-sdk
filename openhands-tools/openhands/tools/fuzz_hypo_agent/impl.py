"""Executor for FuzzHypoAgent operations using sub-agents."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openhands.sdk.tool import ToolExecutor
from openhands.tools.fuzz_hypo_agent.definition import (
    FuzzHypoAgentAction,
    FuzzHypoAgentObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.llm import LLM


class FuzzHypoAgentExecutor(ToolExecutor[FuzzHypoAgentAction, FuzzHypoAgentObservation]):
    """Executor that uses sub-agents for more robust fuzzing.
    
    This executor spawns specialized sub-agents for:
    1. Analysis: Understanding the target function
    2. Generation: Creating test strategies and assertions
    3. Validation: Ensuring generated code is correct
    """

    def __init__(self, working_dir: str, llm: "LLM | None" = None):
        self.working_dir = Path(working_dir).resolve()
        self.llm = llm
        self._parent_conversation: "LocalConversation | None" = None

    def __call__(
        self, action: FuzzHypoAgentAction, conversation: "LocalConversation | None" = None
    ) -> FuzzHypoAgentObservation:
        """Execute the fuzzing workflow using sub-agents."""
        if conversation is not None:
            self._parent_conversation = conversation

        validation_fixes: list[str] = []
        generation_attempts = 0
        analysis_summary: str | None = None

        try:
            # 1. 环境准备
            self._ensure_dependencies()

            # 2. 加载规格说明
            spec_content = self._load_spec(action.spec)

            # 3. 读取目标函数源码（用于分析）
            target_source = self._read_target_source(action.target)

            # 4. 阶段一：分析 Agent - 深入理解目标函数
            analysis_result = self._run_analyzer_agent(
                action.target, target_source, spec_content
            )
            analysis_summary = analysis_result.get("summary", "Analysis completed")

            # 5. 阶段二：生成 Agent - 生成测试组件（带重试）
            components = None
            last_error: Exception | None = None
            
            for attempt in range(action.max_iterations):
                generation_attempts += 1
                try:
                    components = self._run_generator_agent(
                        action.target,
                        spec_content,
                        analysis_result,
                        previous_error=str(last_error) if last_error else None,
                    )
                    
                    # 6. 阶段三：验证 Agent - 验证生成的代码
                    validation_result = self._run_validator_agent(components)
                    
                    if validation_result["is_valid"]:
                        break
                    else:
                        # 记录修复
                        fix_msg = validation_result.get("fix_suggestion", "Validation failed")
                        validation_fixes.append(fix_msg)
                        
                        # 应用验证器的修复建议
                        if "fixed_components" in validation_result:
                            components = validation_result["fixed_components"]
                            break
                        
                        last_error = ValueError(fix_msg)
                        
                except Exception as e:
                    last_error = e
                    validation_fixes.append(f"Attempt {attempt + 1} failed: {str(e)}")
                    continue

            if components is None:
                return FuzzHypoAgentObservation(
                    status="error",
                    bug_found=False,
                    message=f"Failed to generate valid components after {action.max_iterations} attempts. Last error: {last_error}",
                    analysis_summary=analysis_summary,
                    generation_attempts=generation_attempts,
                    validation_fixes=validation_fixes,
                )

            # 7. 执行 Fuzzing
            result = self._execute_fuzz(action, components)
            
            # 添加子代理信息到结果
            return FuzzHypoAgentObservation(
                status=result.status,
                bug_found=result.bug_found,
                failing_input=result.failing_input,
                failure_kind=result.failure_kind,
                harness_path=result.harness_path,
                message=result.message,
                analysis_summary=analysis_summary,
                generation_attempts=generation_attempts,
                validation_fixes=validation_fixes,
            )

        except Exception as e:
            return FuzzHypoAgentObservation(
                status="error",
                bug_found=False,
                message=f"FuzzHypoAgent 执行失败: {str(e)}\n{traceback.format_exc()}",
                analysis_summary=analysis_summary,
                generation_attempts=generation_attempts,
                validation_fixes=validation_fixes,
            )

    # ============ Sub-Agent Methods ============

    def _run_analyzer_agent(
        self, target: str, source: str, spec: str
    ) -> dict[str, Any]:
        """Run the analyzer sub-agent to understand the target function.
        
        This agent performs multi-turn dialogue to:
        1. Understand function signature and types
        2. Identify edge cases and boundary conditions
        3. Extract semantic meaning from docstrings
        """
        system_prompt = """You are a code analysis expert. Your job is to deeply understand 
            a Python function and provide insights for testing.

            Analyze the function and return a JSON object with:
            {
                "summary": "Brief description of what the function does",
                "input_types": {"param1": "type description", ...},
                "return_type": "description of return type",
                "edge_cases": ["list of potential edge cases to test"],
                "invariants": ["properties that should always hold"],
                "exceptions": ["list of exceptions that can be raised"],
                "constraints": ["input constraints mentioned in docs or code"]
            }

            Focus on:
            - Type hints and docstrings
            - Validation logic and error handling
            - Boundary conditions (empty, zero, negative, large values)
            - Special cases mentioned in comments

            Return ONLY the JSON object, no markdown or explanation."""

                    # 第一轮：初始分析
        user_prompt_1 = f"""Analyze this Python function for property-based testing:

            Target: {target}

            Source code:
            ```python
            {source[:3000] if source else "# Source not available - analyze based on specification"}
            ```

            Specification:
            {spec}

            Provide your analysis as JSON:"""

        analysis_1 = self._call_llm_agent(system_prompt, user_prompt_1, temperature=0.1)
        
        try:
            result = self._parse_json_response(analysis_1)
        except (json.JSONDecodeError, ValueError):
            # 如果解析失败，进行第二轮澄清
            user_prompt_2 = f"""The previous response wasn't valid JSON. 
                Please provide ONLY a valid JSON object with these exact keys:
                - summary (string)
                - input_types (object)
                - return_type (string)
                - edge_cases (array of strings)
                - invariants (array of strings)
                - exceptions (array of strings)
                - constraints (array of strings)

                Based on target: {target}
                Spec: {spec[:500]}

                JSON:"""
            analysis_2 = self._call_llm_agent(system_prompt, user_prompt_2, temperature=0.0)
            try:
                result = self._parse_json_response(analysis_2)
            except (json.JSONDecodeError, ValueError):
                # 回退到基本分析
                result = {
                    "summary": f"Function {target.split(':')[-1]}",
                    "input_types": {},
                    "return_type": "unknown",
                    "edge_cases": ["empty input", "boundary values"],
                    "invariants": [],
                    "exceptions": [],
                    "constraints": [],
                }
        
        return result

    def _run_generator_agent(
        self,
        target: str,
        spec: str,
        analysis: dict[str, Any],
        previous_error: str | None = None,
    ) -> dict[str, Any]:
        """Run the generator sub-agent to create Hypothesis components.
        
        This agent uses the analysis to generate appropriate:
        - Hypothesis strategies
        - Post-conditions
        - Seed values
        """
        # 构建增强的上下文
        edge_cases = analysis.get("edge_cases", [])
        invariants = analysis.get("invariants", [])
        input_types = analysis.get("input_types", {})
        
        system_prompt = """You are a Hypothesis testing expert. Generate test components based on analysis.

            CRITICAL: Return ONLY a valid JSON object with exactly these keys:
            {
                "strategy": "hypothesis.strategies expression (use st. prefix)",
                "post_condition": "def post_condition(input_obj, output_obj):\\n    return True/False",
                "seeds": [example1, example2, example3]
            }

            Rules:
            1. strategy: Use st.* functions (integers, floats, text, lists, etc.)
            2. post_condition: MUST be a complete function definition as a STRING
            3. seeds: 3-5 concrete values matching the strategy type

            Strategy selection guide:
            - For numbers: st.integers(), st.floats(allow_nan=False)
            - For strings: st.text(min_size=0, max_size=100)
            - For lists: st.lists(st.integers(), min_size=0, max_size=10)
            - For complex objects: st.fixed_dictionaries({...})

            Output ONLY the JSON. No markdown, no explanation."""

        error_context = ""
        if previous_error:
            error_context = f"""
            IMPORTANT: Previous attempt failed with error:
            {previous_error}

            Please fix the issue and try again."""

        user_prompt = f"""Generate Hypothesis testing components for:

            Target: {target}

            Specification:
            {spec}

            Analysis insights:
            - Summary: {analysis.get('summary', 'N/A')}
            - Input types: {json.dumps(input_types)}
            - Edge cases to cover: {json.dumps(edge_cases[:5])}
            - Invariants to check: {json.dumps(invariants[:3])}
            {error_context}

            Generate the JSON now:"""

        response = self._call_llm_agent(system_prompt, user_prompt, temperature=0.2)
        return self._parse_json_response(response)

    def _run_validator_agent(self, components: dict[str, Any]) -> dict[str, Any]:
        """Run the validator sub-agent to check generated components.
        
        This agent verifies:
        1. Syntax correctness
        2. Import availability
        3. Strategy validity
        """
        strategy = components.get("strategy", "")
        post_condition = components.get("post_condition", "")
        seeds = components.get("seeds", [])

        # 先进行基本语法检查
        validation_errors = []
        
        # 检查 post_condition 语法
        try:
            ast.parse(post_condition)
        except SyntaxError as e:
            validation_errors.append(f"post_condition syntax error: {e}")
        
        # 检查 strategy 是否引用了有效的 st.* 函数
        if not strategy.strip():
            validation_errors.append("strategy is empty")
        elif not re.search(r'st\.\w+', strategy):
            validation_errors.append("strategy should use st.* functions (e.g., st.integers())")
        
        # 检查 seeds 是否为列表
        if not isinstance(seeds, list) or len(seeds) == 0:
            validation_errors.append("seeds must be a non-empty list")

        if validation_errors:
            # 使用 LLM 尝试修复
            return self._attempt_fix(components, validation_errors)
        
        return {
            "is_valid": True,
            "components": components,
        }

    def _attempt_fix(
        self, components: dict[str, Any], errors: list[str]
    ) -> dict[str, Any]:
        """Attempt to fix validation errors using LLM."""
        system_prompt = """You are a code fixer. Fix the validation errors in the Hypothesis test components.

            Return ONLY a valid JSON object with fixed components:
            {
                "strategy": "fixed strategy expression",
                "post_condition": "def post_condition(input_obj, output_obj):\\n    return True",
                "seeds": [1, 2, 3]
            }

            Rules:
            1. Fix all syntax errors
            2. Ensure strategy uses st.* functions correctly
            3. Ensure post_condition is a valid function definition string
            4. Ensure seeds is a non-empty list

            Output ONLY JSON, no markdown."""

        user_prompt = f"""Fix these Hypothesis test components:

            Current components:
            {json.dumps(components, indent=2)}

            Validation errors:
            {json.dumps(errors)}

            Fixed JSON:"""

        try:
            response = self._call_llm_agent(system_prompt, user_prompt, temperature=0.0)
            fixed = self._parse_json_response(response)
            
            # 再次验证修复后的组件
            if self._validate_components(fixed):
                return {
                    "is_valid": True,
                    "fixed_components": fixed,
                    "fix_suggestion": f"Applied fixes for: {', '.join(errors)}",
                }
        except Exception as e:
            pass
        
        return {
            "is_valid": False,
            "fix_suggestion": f"Could not fix errors: {', '.join(errors)}",
        }

    # ============ Helper Methods ============

    def _call_llm_agent(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.1
    ) -> str:
        """Call LLM for sub-agent dialogue."""
        if self.llm is not None:
            from openhands.sdk.llm.message import Message, TextContent, content_to_str

            messages = [
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ]

            response = self.llm.completion(
                messages=messages,
                temperature=temperature,
            )

            text_parts = content_to_str(response.message.content)
            return "".join(text_parts)
        else:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )

            return response.choices[0].message.content or ""

    def _parse_json_response(self, raw_content: str) -> dict[str, Any]:
        """Robustly parse JSON from LLM response."""
        content = raw_content.strip()

        strategies = [
            lambda s: s,
            lambda s: re.sub(r"```(?:json)?\s*|\s*```", "", s).strip(),
            lambda s: self._extract_json_object(s),
            lambda s: self._extract_json_object(
                "\n".join(line.strip() for line in s.split("\n"))
            ),
        ]

        last_error: Exception | None = None

        for strategy in strategies:
            try:
                cleaned = strategy(content)
                if cleaned:
                    result = json.loads(cleaned)
                    if isinstance(result, dict):
                        return result
            except (json.JSONDecodeError, TypeError) as e:
                last_error = e
                continue

        raise json.JSONDecodeError(
            f"Could not parse JSON from LLM response. Last error: {last_error}",
            content,
            0,
        )

    def _extract_json_object(self, text: str) -> str:
        """Extract first complete JSON object from text."""
        depth = 0
        start = -1

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]

        return text

    def _validate_components(self, components: dict[str, Any]) -> bool:
        """Validate that components have required keys and valid syntax."""
        required_keys = {"strategy", "post_condition", "seeds"}
        if not required_keys.issubset(components.keys()):
            return False
        
        try:
            ast.parse(components["post_condition"])
        except SyntaxError:
            return False
        
        if not isinstance(components["seeds"], list):
            return False
        
        return True

    def _read_target_source(self, target: str) -> str:
        """Read the source code of the target function."""
        try:
            target_mod, target_func = self._parse_target(target)
            
            # Check if it's a file path
            if target_mod.endswith(".py") or "/" in target_mod:
                path = self._normalize_path(target_mod)
                if path.exists():
                    content = path.read_text()
                    # Try to extract just the function
                    return self._extract_function_source(content, target_func)
            
            # Try to import and get source
            import importlib
            import inspect
            
            try:
                module = importlib.import_module(target_mod)
                func = getattr(module, target_func, None)
                if func:
                    return inspect.getsource(func)
            except (ImportError, OSError):
                pass
            
        except Exception:
            pass
        
        return ""

    def _extract_function_source(self, content: str, func_name: str) -> str:
        """Extract function source from file content."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    lines = content.split("\n")
                    return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        except SyntaxError:
            pass
        
        # Fallback: simple regex extraction
        pattern = rf"def {func_name}\s*\([^)]*\):.*?(?=\ndef |\nclass |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(0)
        
        return content[:2000]  # Return first part of file as context

    def _parse_target(self, target: str) -> tuple[str, str]:
        """Parse target string into (module/path, function_name)."""
        if ":" not in target:
            raise ValueError(
                f"Target must be in format 'module:func' or 'path:func', got: {target}"
            )
        parts = target.rsplit(":", 1)
        return parts[0], parts[1]

    def _normalize_path(self, path_str: str) -> Path:
        """Normalize path, handling workspace prefixes."""
        path = Path(path_str)

        if path.is_absolute():
            return path

        working_dir_str = str(self.working_dir)
        path_str_clean = path_str.lstrip("/")

        if working_dir_str.endswith("/workspace") or "/workspace" in working_dir_str:
            if path_str_clean.startswith("workspace/"):
                path_str_clean = path_str_clean[len("workspace/") :]

        full_path = self.working_dir / path_str_clean

        if full_path.exists():
            return full_path.resolve()

        original_path = self.working_dir / path_str
        if original_path.exists():
            return original_path.resolve()

        return full_path

    def _execute_fuzz(
        self, action: FuzzHypoAgentAction, components: dict[str, Any]
    ) -> FuzzHypoAgentObservation:
        """Execute the fuzzing with generated components."""
        max_examples = 2000 if action.mode == "deep" else 200

        target_mod, target_func = self._parse_target(action.target)

        with tempfile.TemporaryDirectory(prefix="oh_fuzz_agent_") as tmpdir:
            tmp_path = Path(tmpdir)
            harness_file = tmp_path / "test_harness.py"

            import_code = self._generate_import_code(target_mod, target_func)

            test_template = f"""
import sys, os, json
import hypothesis.strategies as st
from hypothesis import given, settings, example

# Dynamic import path
sys.path.insert(0, "{self.working_dir}")

{import_code}

# Post-condition definition
{components['post_condition']}

@settings(max_examples={max_examples}, deadline=None)
@given({components['strategy']})
def test_fuzz_task(data):
    result = target_func(data)
    assert post_condition(data, result) is True, "Fuzzing violation found"

if __name__ == "__main__":
    pass
"""
            harness_file.write_text(test_template)

            python_exe = self._get_testbed_python()
            
            # Prepare environment with correct PYTHONPATH
            env = os.environ.copy()
            pythonpath = str(self.working_dir)
            if "PYTHONPATH" in env:
                pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
            env["PYTHONPATH"] = pythonpath

            if (
                "/opt/miniconda3/envs/testbed" in python_exe
                or python_exe != sys.executable
            ):
                # Run from working_dir to ensure correct package initialization
                # Install pytest and hypothesis if not already installed
                cmd = [
                    "bash",
                    "-c",
                    f"cd {self.working_dir} && "
                    f"source /opt/miniconda3/etc/profile.d/conda.sh && "
                    f"conda activate testbed && "
                    f"pip install -q pytest hypothesis 2>/dev/null; "
                    f"PYTHONPATH='{pythonpath}' python -m pytest {harness_file} -q --tb=short 2>&1",
                ]
            else:
                cmd = [
                    python_exe,
                    "-m",
                    "pytest",
                    str(harness_file),
                    "-q",
                    "--tb=short",
                ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.working_dir),
                env=env,
            )

            bug_found = result.returncode != 0
            failing_input = (
                self._extract_crash(result.stdout + result.stderr) if bug_found else None
            )

            status = "fail" if bug_found else "ok"
            msg = "Found specification violation!" if bug_found else "No defects found."

            return FuzzHypoAgentObservation(
                status=status,
                bug_found=bug_found,
                failing_input=str(failing_input) if failing_input else None,
                failure_kind=(
                    "assertion_failure"
                    if "AssertionError" in result.stdout
                    else "exception"
                ),
                harness_path=str(harness_file),
                message=f"{msg}\n\n[Pytest Output]\n{result.stdout[-500:]}",
            )

    def _generate_import_code(self, target_mod: str, target_func: str) -> str:
        """Generate import code for the target function.
        
        Handles complex packages (like astropy) that require proper initialization.
        Supports class methods (e.g., "Quantity.__array_ufunc__").
        """
        # Check if target_func contains a class (e.g., "Quantity.__array_ufunc__")
        is_class_method = "." in target_func
        if is_class_method:
            parts = target_func.split(".", 1)
            class_name = parts[0]
            method_name = parts[1]
            # Generate getattr code for class methods
            getattr_stmt = f'''_class = getattr(_module, "{class_name}")
    target_func = getattr(_class, "{method_name}")'''
        else:
            getattr_stmt = f'target_func = getattr(_module, "{target_func}")'
        
        is_file_path = target_mod.endswith(".py") or "/" in target_mod

        if is_file_path:
            normalized_path = self._normalize_path(target_mod)
            module_name = self._path_to_module_name(target_mod)
            pkg_root = self._find_package_root(normalized_path)
            top_package = module_name.split(".")[0] if module_name else None
            
            if module_name:
                return f'''# Import from file path (converted to module for relative imports)
import sys
import os
import importlib

# Ensure package root is in sys.path
_pkg_root = "{pkg_root}"
if _pkg_root:
    sys.path = [p for p in sys.path if p != _pkg_root]
    sys.path.insert(0, _pkg_root)

# Change to package root for proper resolution
_original_cwd = os.getcwd()
if _pkg_root and os.path.isdir(_pkg_root):
    os.chdir(_pkg_root)

try:
    # Initialize top-level package first for complex packages
    _top_package = "{top_package}"
    if _top_package:
        try:
            importlib.import_module(_top_package)
        except ImportError:
            pass
    
    _module = importlib.import_module("{module_name}")
    {getattr_stmt}
except ImportError as e:
    try:
        _parts = "{module_name}".rsplit(".", 1)
        if len(_parts) == 2:
            _parent = importlib.import_module(_parts[0])
            _child = getattr(_parent, _parts[1], None)
            if _child and hasattr(_child, "{target_func}"):
                target_func = getattr(_child, "{target_func}")
            else:
                raise ImportError(f"Cannot find {target_func}: {{e}}")
        else:
            raise
    except ImportError:
        raise ImportError(f"Cannot import {target_func} from {module_name}: {{e}}")
finally:
    os.chdir(_original_cwd)'''
            else:
                return f'''# Direct file load (no relative imports)
import sys
_file_dir = "{normalized_path.parent}"
if _file_dir not in sys.path:
    sys.path.insert(0, _file_dir)
_globals = {{"__name__": "__main__", "__file__": "{normalized_path}"}}
with open("{normalized_path}", "r") as _f:
    exec(compile(_f.read(), "{normalized_path}", "exec"), _globals)
target_func = _globals.get("{target_func}")
if target_func is None:
    raise ImportError(f"Cannot find {target_func} in {normalized_path}")'''
        else:
            top_package = target_mod.split(".")[0]
            return f'''# Import from module (with complex package support)
import importlib

# Initialize top-level package first
_top_package = "{top_package}"
if _top_package:
    try:
        importlib.import_module(_top_package)
    except ImportError:
        pass

try:
    _module = importlib.import_module("{target_mod}")
    {getattr_stmt}
except (ImportError, AttributeError) as e:
    _parts = "{target_mod}".rsplit(".", 1)
    if len(_parts) == 2:
        try:
            _parent = importlib.import_module(_parts[0])
            _child = getattr(_parent, _parts[1], None)
            if _child and hasattr(_child, "{target_func}"):
                target_func = getattr(_child, "{target_func}")
            else:
                raise ImportError(f"Cannot find {target_func} in {target_mod}")
        except ImportError:
            raise ImportError(f"Cannot import {target_func} from {target_mod}: {{e}}")
    else:
        raise ImportError(f"Cannot import {target_func} from {target_mod}: {{e}}")'''
    
    def _path_to_module_name(self, path_str: str) -> str | None:
        """Convert file path to Python module name by detecting package structure.
        
        Algorithm:
        1. Start from the file's parent directory
        2. Traverse upward, collecting directories that have __init__.py
        3. Stop when reaching a directory without __init__.py (package root's parent)
        4. Combine collected directory names + file name to form module name
        
        Examples:
        - "/workspace/django/django/urls/resolvers.py" -> "django.urls.resolvers"
        - "/workspace/astropy/astropy/units/core.py" -> "astropy.units.core"
        """
        # Normalize path
        path = Path(path_str)
        if not path.is_absolute():
            path = self.working_dir / path
        path = path.resolve()
        
        # Get module name (file name without .py)
        module_file_name = path.stem
        
        # Traverse upward from file's parent, collecting package directories
        current_dir = path.parent
        package_parts: list[str] = []
        
        while current_dir != current_dir.parent:
            init_file = current_dir / "__init__.py"
            if init_file.exists():
                package_parts.insert(0, current_dir.name)
                current_dir = current_dir.parent
            else:
                # Not a package, stop traversal
                break
        
        # Build full module name
        if package_parts:
            return ".".join(package_parts + [module_file_name])
        
        # No package structure found, return just the file name
        return module_file_name
    
    def _find_package_root(self, file_path: Path) -> str:
        """Find the root directory of the package containing the file."""
        current = file_path.parent
        package_root = None
        
        while current != current.parent:
            if (current / "__init__.py").exists():
                package_root = current.parent
            elif package_root is not None:
                break
            current = current.parent
        
        return str(package_root) if package_root else str(self.working_dir)

    def _extract_crash(self, output: str) -> Any:
        """Extract failing input from Hypothesis output."""
        match = re.search(r"Falsifying example:.*?data=(.*?)\n", output, re.DOTALL)
        if match:
            raw_val = match.group(1).strip()
            try:
                return ast.literal_eval(raw_val)
            except:
                return raw_val
        return None

    def _get_testbed_python(self) -> str:
        """Get Python executable from testbed conda environment."""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", "testbed", "which", "python"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        conda_python = "/opt/miniconda3/envs/testbed/bin/python"
        if os.path.exists(conda_python):
            return conda_python

        return sys.executable

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
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            subprocess.check_call(
                ["conda", "run", "-n", "testbed", "pip", "install", "hypothesis", "pytest"],
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", "hypothesis", "pytest"],
                    timeout=60,
                )
            except Exception:
                pass

    def _load_spec(self, spec: str) -> str:
        """Load spec content from file or use directly."""
        if "\n" in spec or spec.strip().startswith(("inputs:", "{", "[")):
            return spec

        try:
            spec_path = self.working_dir / spec
            if spec_path.exists() and spec_path.is_file():
                return spec_path.read_text()
        except (OSError, ValueError):
            pass

        return spec
