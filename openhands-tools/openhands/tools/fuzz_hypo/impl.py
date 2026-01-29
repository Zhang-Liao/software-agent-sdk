"""Executor for FuzzHypo operations."""

import ast
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openhands.sdk.tool import ToolExecutor
from openhands.tools.fuzz_hypo.definition import FuzzHypoAction, FuzzHypoObservation

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.llm import LLM


class FuzzHypoExecutor(ToolExecutor[FuzzHypoAction, FuzzHypoObservation]):
    # 最大LLM重试次数
    MAX_LLM_RETRIES = 2
    
    def __init__(self, working_dir: str, llm: "LLM | None" = None):
        self.working_dir = Path(working_dir).resolve()
        self.llm = llm  # 存储 LLM 配置，用于与 openhands 使用相同的 API

    def __call__(self, action: FuzzHypoAction, conversation=None) -> FuzzHypoObservation:
        try:
            # 1. 环境准备
            self._ensure_dependencies()

            # 2. 加载规格说明
            spec_content = self._load_spec(action.spec)
            
            # 3. 自动检测目标函数的签名（预检）
            param_info = self._detect_function_signature(action.target)

            # 4. LLM 生成测试组件 (Strategy & Assertion)
            components = self._llm_generate_components(action.target, spec_content, param_info)

            # 5. 执行 Fuzzing 并返回结果
            return self._execute_fuzz(action, components)

        except Exception as e:
            return FuzzHypoObservation(
                status="error",
                bug_found=False,
                failing_input=None,
                failure_kind="execution_error",
                harness_path=None,
                message=f"FuzzHypo execution failed: {str(e)}\n{traceback.format_exc()}",
            )

    # --- 核心方法 1: LLM 生成 ---
    def _llm_generate_components(self, target: str, spec: str, param_info: str | None = None) -> dict:
        """生成 Hypothesis 测试组件，带重试机制和健壮的JSON解析。
        
        Args:
            target: 目标函数的路径
            spec: 用户提供的规格说明
            param_info: 目标函数的参数信息（可选，用于生成多参数策略）
        """
        # 从 param_info 或 spec 中解析参数数量（用于 fallback）
        param_count = self._extract_param_count(param_info, spec)
        system_prompt = """You are a testing expert. Generate Hypothesis testing components.

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON object with NO markdown fences, NO code blocks, NO explanations.
2. The JSON must have exactly these keys: "strategies", "post_condition", "seeds"
3. CAREFULLY read the specification to determine the function's parameters!

Required JSON structure:
{
    "strategies": ["st.integers()", "st.text()"],
    "post_condition": "def post_condition(inputs, output_obj):\\n    return True",
    "seeds": [[1, "a"], [2, "b"]]
}

Rules:
- "strategies": A LIST of hypothesis.strategies expressions (imported as `st`), ONE for each function parameter
  - CRITICAL: Count the parameters from the function signature in the spec!
  - Example: "func(x: int, y: str)" -> TWO strategies: ["st.integers()", "st.text()"]
  - Example: "func(value: float)" -> ONE strategy: ["st.floats()"]
  - Example: "func()" -> ZERO strategies: []
- "post_condition": A complete Python function definition. The first argument `inputs` is a tuple of all inputs.
- "seeds": A list of example inputs. Each seed should match the number of strategies.

Strategy type mapping:
- int -> st.integers()
- float -> st.floats(allow_nan=False, allow_infinity=False)
- str -> st.text(max_size=50) or st.sampled_from(["option1", "option2"]) for enum-like
- bool -> st.booleans()
- list -> st.lists(st.integers())

IMPORTANT: Output ONLY the JSON object. Nothing else."""
        
        # Build user prompt with parameter info if available
        user_prompt = f"Target function: {target}\n\n"
        if param_info:
            user_prompt += f"Function signature info:\n{param_info}\n\n"
        user_prompt += f"Specification:\n{spec}\n\nGenerate the JSON now:"
        
        last_error: Exception | None = None
        
        for attempt in range(self.MAX_LLM_RETRIES + 1):
            try:
                raw_content = self._call_llm(system_prompt, user_prompt, temperature=0.1 + attempt * 0.1)
                return self._parse_json_response(raw_content)
            except json.JSONDecodeError as e:
                last_error = e
                # 继续重试
                continue
        
        # 所有重试都失败了，使用默认组件（传递检测到的参数数量）
        return self._generate_fallback_components(target, spec, param_count)
    
    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """调用 LLM 并返回原始响应文本。"""
        if self.llm is not None:
            # 使用 openhands 的 LLM 类（推荐方案）
            from openhands.sdk.llm.message import Message, TextContent, content_to_str
            
            messages = [
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ]
            
            response = self.llm.completion(
                messages=messages,
                temperature=temperature,
            )
            
            # 使用官方工具函数转换内容
            text_parts = content_to_str(response.message.content)
            return "".join(text_parts)
        else:
            # 回退到环境变量（向后兼容）
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
        """健壮地解析 LLM 返回的 JSON 响应。"""
        content = raw_content.strip()
        
        # 尝试多种清洗策略
        strategies = [
            # 策略1: 直接解析
            lambda s: s,
            # 策略2: 移除 markdown 代码块
            lambda s: re.sub(r"```(?:json)?\s*|\s*```", "", s).strip(),
            # 策略3: 提取第一个 JSON 对象
            lambda s: self._extract_json_object(s),
            # 策略4: 移除行首/行尾空白后提取
            lambda s: self._extract_json_object("\n".join(line.strip() for line in s.split("\n"))),
        ]
        
        last_error: Exception | None = None
        
        for strategy in strategies:
            try:
                cleaned = strategy(content)
                if cleaned:
                    result = json.loads(cleaned)
                    # 验证必需的键
                    if self._validate_components(result):
                        return result
            except (json.JSONDecodeError, TypeError) as e:
                last_error = e
                continue
        
        raise json.JSONDecodeError(
            f"Could not parse JSON from LLM response. Last error: {last_error}",
            content,
            0
        )
    
    def _extract_json_object(self, text: str) -> str:
        """从文本中提取第一个完整的 JSON 对象。"""
        # 找到第一个 { 和对应的 }
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
    
    def _validate_components(self, components: dict) -> bool:
        """验证组件是否包含必需的键。支持新旧两种格式。"""
        # New format: strategies (list)
        new_format_keys = {"strategies", "post_condition", "seeds"}
        # Old format: strategy (single) - for backwards compatibility
        old_format_keys = {"strategy", "post_condition", "seeds"}
        
        if new_format_keys.issubset(components.keys()):
            return True
        if old_format_keys.issubset(components.keys()):
            # Convert old format to new format
            strategy = components.get("strategy", "")
            components["strategies"] = [strategy] if strategy else []
            return True
        return False
    
    def _generate_fallback_components(self, target: str, spec: str, param_count: int = 1) -> dict:
        """当 LLM 无法生成有效 JSON 时，使用通用的测试组件。
        
        这个回退方案使用非常通用的策略，可能不如 LLM 生成的精确，
        但至少可以进行基本的 fuzzing。
        
        Args:
            target: 目标函数
            spec: 规格说明
            param_count: 目标函数的参数数量
        """
        # 分析 spec 来推断输入类型
        spec_lower = spec.lower()
        
        # 根据规格中的关键词选择单个策略
        if any(word in spec_lower for word in ["array", "ndarray", "numpy", "matrix"]):
            strategy = "st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10)"
        elif any(word in spec_lower for word in ["string", "str", "text"]):
            strategy = "st.text(min_size=0, max_size=100)"
        elif any(word in spec_lower for word in ["int", "integer", "number"]):
            strategy = "st.integers(min_value=-1000, max_value=1000)"
        elif any(word in spec_lower for word in ["float", "decimal", "real"]):
            strategy = "st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000)"
        elif any(word in spec_lower for word in ["bool", "boolean", "flag"]):
            strategy = "st.booleans()"
        elif any(word in spec_lower for word in ["mixed", "any", "object"]):
            strategy = "st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=20), st.none())"
        else:
            # 默认：混合类型
            strategy = "st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50))"
        
        # 根据参数数量生成策略列表
        if param_count == 0:
            strategies = []
            seeds = [[]]
        else:
            strategies = [strategy] * param_count
            # 简单的种子
            single_seeds = [0, 1, -1]
            seeds = [[s] * param_count for s in single_seeds]
        
        # 通用的后验条件：只检查函数不会崩溃
        post_condition = """def post_condition(inputs, output_obj):
    # Fallback post-condition: just check the function doesn't crash
    # The function executed successfully, so we consider it valid
    return True"""
        
        return {
            "strategies": strategies,
            "post_condition": post_condition,
            "seeds": seeds
        }

    # --- 核心方法 2: 执行驱动 ---
    def _execute_fuzz(self, action: FuzzHypoAction, components: dict) -> FuzzHypoObservation:
        max_examples = 2000 if action.mode == "deep" else 500
        
        # 解析目标：支持多种格式
        # 1. "module.path:func" (Python模块格式)
        # 2. "/path/to/file.py:func" (绝对路径)
        # 3. "path/to/file.py:func" (相对路径)
        target_mod, target_func = self._parse_target(action.target)
        
        # 使用 working_dir 下的持久化目录存储 harness 文件
        # 这样 LLM 可以在测试结束后查看文件进行调试
        fuzz_dir = self.working_dir / ".fuzz_hypo"
        fuzz_dir.mkdir(exist_ok=True)
        
        # 创建唯一的 harness 文件名（基于 target 函数名和时间戳）
        import time
        safe_func_name = re.sub(r'[^\w]', '_', target_func)[:30]
        harness_file = fuzz_dir / f"test_{safe_func_name}_{int(time.time())}.py"
        
        # 生成导入代码
        import_code = self._generate_import_code(target_mod, target_func)
        
        # Install extra packages if specified
        extra_packages = action.extra_imports or []
        if extra_packages:
            self._install_extra_packages(extra_packages)
        
        # Generate extra import statements
        extra_import_code = self._generate_extra_imports(
            action.extra_imports,
            action.import_statements
        )
        
        # Build test script that collects up to MAX_FAILURES unique failures
        max_failures = 10
        
        # Generate strategy expression for @given decorator
        strategies_list = components.get('strategies', [])
        if not strategies_list:
            # Empty strategies = no arguments needed
            given_strategies = ""
        else:
            given_strategies = ", ".join(strategies_list)
        
        # Generate parameter names for test function
        num_params = len(strategies_list)
        if num_params == 0:
            param_names = ""
            args_tuple = "()"
        elif num_params == 1:
            param_names = "arg0"
            args_tuple = "(arg0,)"
        else:
            param_names = ", ".join(f"arg{i}" for i in range(num_params))
            args_tuple = f"({param_names})"
        
        test_template = f'''
import sys, os
import json
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, Verbosity, Phase

# Extra imports for strategy/post-condition
{extra_import_code}

# Dynamic import path
sys.path.insert(0, "{self.working_dir}")

{import_code}

# Number of expected parameters
_num_params = {num_params}
print(f"[FuzzHypo] Configured for {{_num_params}} parameter(s)")

# Post-condition definition
{components['post_condition']}

# Failure collection - grouped by error type
_example_count = 0
_failures_by_type = {{}}  # Dict[error_sig, List[failure_info]]
_seen_inputs_by_type = {{}}  # Dict[error_sig, Set[input_repr]] - to deduplicate inputs
MAX_ERROR_TYPES = {max_failures}  # Max different error types to collect
MAX_EXAMPLES_PER_TYPE = 5  # Max UNIQUE examples to collect for each error type

def _serialize_input(data):
    """Serialize input to a human-readable string, handling complex objects."""
    try:
        # Try JSON first for simple types
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        pass
    
    # For complex objects, use detailed repr
    try:
        if hasattr(data, '__dict__'):
            # Object with attributes - show its type and key attributes
            attrs = {{k: v for k, v in data.__dict__.items() if not k.startswith('_')}}
            type_name = type(data).__name__
            if attrs:
                attrs_str = ', '.join(f"{{k}}={{repr(v)[:50]}}" for k, v in list(attrs.items())[:5])
                return f"<{{type_name}}: {{attrs_str}}>"
            return f"<{{type_name}} object>"
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # Iterable - show first few elements
            items = list(data)[:5]
            items_str = ', '.join(_serialize_input(item) for item in items)
            return f"[{{items_str}}{{', ...' if len(list(data)) > 5 else ''}}]"
        else:
            return repr(data)[:200]
    except Exception:
        return f"<{{type(data).__name__}} (unserializable)>"

def _get_error_signature(error_msg):
    """Get a signature for grouping similar errors."""
    # Extract the error type and first line of message
    lines = error_msg.strip().split("\\n")
    if lines:
        return lines[-1][:100]  # Last line usually has the actual error
    return error_msg[:100]

def _record_failure(data, error_msg, traceback_str=None):
    """Record a failure, grouped by error type, with input deduplication."""
    global _failures_by_type, _seen_inputs_by_type
    error_sig = _get_error_signature(error_msg)
    
    # Check if we already have too many error types
    if error_sig not in _failures_by_type:
        if len(_failures_by_type) >= MAX_ERROR_TYPES:
            return False  # Skip, already have enough error types
        _failures_by_type[error_sig] = []
        _seen_inputs_by_type[error_sig] = set()
    
    # Check if we already have enough examples for this error type
    if len(_failures_by_type[error_sig]) >= MAX_EXAMPLES_PER_TYPE:
        return False  # Skip, already have enough examples for this type
    
    input_repr = _serialize_input(data)
    
    # Deduplicate: skip if we've seen this exact input for this error type
    if input_repr in _seen_inputs_by_type[error_sig]:
        return False  # Skip duplicate input
    _seen_inputs_by_type[error_sig].add(input_repr)
    failure_info = {{
        "input": str(data)[:500],
        "repr": input_repr,
        "error": error_msg,
        "example_num": _example_count
    }}
    if traceback_str:
        failure_info["traceback"] = traceback_str[-500:]
    
    _failures_by_type[error_sig].append(failure_info)
    total_failures = sum(len(v) for v in _failures_by_type.values())
    print(f"[FuzzHypo] Failure #{{total_failures}} at example #{{_example_count}}: {{input_repr[:100]}}")
    print(f"           Error: {{error_msg[:100]}}")
    return True

# Try to define the test function with decorators
# If this fails, we'll catch it and report the error
_test_func_error = None
try:
    @settings(
        max_examples={max_examples},
        deadline=None,
        verbosity=Verbosity.normal,
        phases=[Phase.generate, Phase.target],  # Skip shrinking to find more failures
        database=None,  # Don't cache examples
    )
    @given({given_strategies if given_strategies else 'st.just(None)'})
    def test_fuzz_task({param_names if param_names else '_unused'}):
        global _example_count
        _example_count += 1
        
        # Pack arguments into a tuple for recording and post-condition
        args = {args_tuple}
        
        # Check if we've collected enough data
        total_failures = sum(len(v) for v in _failures_by_type.values())
        if len(_failures_by_type) >= MAX_ERROR_TYPES and total_failures >= MAX_ERROR_TYPES * MAX_EXAMPLES_PER_TYPE:
            return
        
        try:
            # Call target function with the correct number of arguments
            if _num_params == 0:
                result = target_func()
            else:
                result = target_func(*args)
            
            if not post_condition(args, result):
                error_msg = f"Post-condition violation: post_condition(inputs, output) returned False"
                _record_failure(args, error_msg)
        except Exception as e:
            error_msg = f"{{type(e).__name__}}: {{str(e)}}"
            _record_failure(args, error_msg, traceback.format_exc())

except Exception as e:
    _test_func_error = f"{{type(e).__name__}}: {{e}}"
    print(f"[FuzzHypo] ERROR defining test function: {{_test_func_error}}")
    import traceback as tb
    print(f"[FuzzHypo] Traceback:\\n{{tb.format_exc()}}")
    # Define a dummy function so the module doesn't crash
    def test_fuzz_task():
        pass

def teardown_module():
    """Report statistics and all failures after tests"""
    total_failures = sum(len(v) for v in _failures_by_type.values())
    print(f"\\n[FuzzHypo] Total examples tested: {{_example_count}} / {max_examples}")
    print(f"[FuzzHypo] Unique error types found: {{len(_failures_by_type)}}")
    print(f"[FuzzHypo] Total failure examples: {{total_failures}}")
    
    if _failures_by_type:
        # Flatten for JSON output (backwards compatible)
        all_failures = []
        for error_sig, failures in _failures_by_type.items():
            for f in failures:
                all_failures.append(f)
        
        print("\\n[FuzzHypo:FAILURES_JSON_START]")
        print(json.dumps(all_failures, default=str, ensure_ascii=False))
        print("[FuzzHypo:FAILURES_JSON_END]")
        
        # Human-readable summary grouped by error type
        print("\\n=== FAILURE SUMMARY (by error type) ===")
        for error_sig, failures in _failures_by_type.items():
            print(f"\\n--- Error Type: {{error_sig[:80]}} ---")
            print(f"    ({{len(failures)}} example(s) found)")
            for i, f in enumerate(failures, 1):
                print(f"    Example {{i}} (at #{{f.get('example_num', '?')}}): {{f.get('repr', f.get('input', 'N/A'))[:150]}}")

if __name__ == "__main__":
    import sys
    sys.stdout.flush()  # Ensure previous output is flushed
    
    # Check if test function was defined correctly
    if _test_func_error:
        print(f"[FuzzHypo] FATAL: Could not define test function: {{_test_func_error}}")
        print("[FuzzHypo] This usually means the @given strategy is invalid.")
        print(f"[FuzzHypo] Strategy used: {given_strategies if given_strategies else 'st.just(None)'}")
        sys.stdout.flush()
        sys.exit(1)
    
    # Run the test directly without pytest
    print("[FuzzHypo] Starting fuzz test...")
    sys.stdout.flush()
    
    try:
        test_fuzz_task()
    except Exception as e:
        import traceback
        print(f"[FuzzHypo] Test execution error: {{type(e).__name__}}: {{e}}")
        print(f"[FuzzHypo] Traceback:\\n{{traceback.format_exc()}}")
        sys.stdout.flush()
    finally:
        # Always run teardown to report results
        teardown_module()
        sys.stdout.flush()
'''
        harness_file.write_text(test_template)

        # 运行命令 - 使用 testbed conda 环境的 Python
        python_exe = self._get_testbed_python()
        
        # 准备环境变量 - 确保项目根目录在 PYTHONPATH 中
        env = os.environ.copy()
        pythonpath = str(self.working_dir)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath
        
        # Run the harness directly (no pytest needed since we catch all exceptions)
        if "/opt/miniconda3/envs/testbed" in python_exe or python_exe != sys.executable:
            # Use conda run to ensure we're in the testbed environment
            cmd = [
                "bash", "-c",
                f"cd {self.working_dir} && "
                f"source /opt/miniconda3/etc/profile.d/conda.sh && "
                f"conda activate testbed && "
                f"pip install -q hypothesis 2>/dev/null; "
                f"PYTHONPATH='{pythonpath}' python {harness_file} 2>&1"
            ]
        else:
            # Fallback to direct execution if conda is not available
            cmd = [python_exe, str(harness_file)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(self.working_dir), env=env)

        # Merge output
        full_output = result.stdout + result.stderr
        
        # Extract all failures from JSON output
        failures = self._extract_failures_json(full_output)
        num_failures = len(failures)
        bug_found = num_failures > 0
        
        # For backwards compatibility, set failing_input to the first failure
        failing_input = None
        if failures:
            first_failure = failures[0]
            failing_input = first_failure.get("repr", first_failure.get("input"))
        
        # Extract example count
        examples_count = self._extract_example_count_int(full_output, max_examples)
        
        # Extract sample inputs
        sample_inputs = self._extract_sample_inputs(full_output)
        
        status = "fail" if bug_found else "ok"
        examples_str = str(examples_count) if examples_count is not None else "unknown"
        if bug_found:
            msg = f"Found {num_failures} unique failure(s)! (after testing {examples_str} examples)"
        else:
            if examples_count is None or examples_count == 0:
                msg = "Test may not have run correctly. Check the harness output below."
            else:
                msg = f"No defects found. (tested {examples_str} examples)"
        
        # Truncate harness code for display
        harness_code_truncated = self._truncate_harness(test_template, max_lines=50)
        
        # Extract only FuzzHypo's own output, filtering out Hypothesis/pytest noise
        # This avoids confusion since Hypothesis always shows "passing" (we catch exceptions)
        fuzz_output = self._extract_fuzz_output(full_output)

        # Format strategies for display
        strategies = components.get("strategies", [])
        strategy_display = ", ".join(strategies) if strategies else "(no parameters)"
        
        return FuzzHypoObservation(
            status=status,
            bug_found=bug_found,
            failing_input=str(failing_input) if failing_input else None,
            failing_inputs=failures if failures else None,
            num_failures=num_failures,
            failure_kind="assertion_failure" if "AssertionError" in full_output else "exception",
            harness_path=str(harness_file),
            message=f"{msg}\n\n{fuzz_output}",
            strategy=strategy_display,
            post_condition=components.get("post_condition"),
            examples_tested=examples_count,
            sample_inputs=sample_inputs,
            harness_code=harness_code_truncated,
        )

    # --- 辅助方法 ---
    def _detect_function_signature(self, target: str) -> str | None:
        """自动检测目标函数的签名，返回参数信息字符串。
        
        通过运行一个小脚本来导入目标函数并检查其签名。
        这个信息会传递给 LLM，帮助它生成正确数量的策略。
        
        Returns:
            参数信息字符串，如 "Function has 2 parameters: value (float), dtype_name (str)"
            如果检测失败则返回 None
        """
        try:
            target_mod, target_func = self._parse_target(target)
        except ValueError:
            return None
        
        # 检测是否是类方法
        is_class_method = "." in target_func
        if is_class_method:
            parts = target_func.split(".", 1)
            class_name = parts[0]
            method_name = parts[1]
            func_access = f"getattr({class_name}, '{method_name}')"
        else:
            func_access = target_func
        
        # 生成导入代码
        if target_mod.endswith(".py") or "/" in target_mod:
            # 文件路径格式
            if "/" in target_mod and not target_mod.startswith("/"):
                module_name = target_mod.replace("/", ".").replace(".py", "")
            else:
                module_name = target_mod.replace(".py", "").replace("/", ".")
            import_stmt = f"from {module_name} import *"
        else:
            # 模块格式
            import_stmt = f"from {target_mod} import *"
        
        # 创建检测脚本
        script = f'''
import sys
import json
import inspect
sys.path.insert(0, "{self.working_dir}")

try:
    {import_stmt}
    target = {func_access}
    
    sig = inspect.signature(target)
    params = []
    for name, p in sig.parameters.items():
        if name in ('self', 'cls'):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = str(p.annotation) if p.annotation != inspect.Parameter.empty else "Any"
        ann = ann.replace("<class '", "").replace("'>", "")
        has_default = p.default != inspect.Parameter.empty
        params.append({{"name": name, "type": ann, "optional": has_default}})
    
    print("SIGNATURE_JSON:" + json.dumps(params))
except Exception as e:
    print("SIGNATURE_ERROR:" + str(e))
'''
        
        # 运行检测脚本
        python_exe = self._get_testbed_python()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.working_dir)
        
        try:
            if "/opt/miniconda3/envs/testbed" in python_exe:
                cmd = [
                    "bash", "-c",
                    f"source /opt/miniconda3/etc/profile.d/conda.sh && "
                    f"conda activate testbed && "
                    f"python -c {repr(script)} 2>&1"
                ]
            else:
                cmd = [python_exe, "-c", script]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10,
                cwd=str(self.working_dir),
                env=env
            )
            
            output = result.stdout + result.stderr
            
            # 解析结果
            if "SIGNATURE_JSON:" in output:
                json_str = output.split("SIGNATURE_JSON:")[1].split("\n")[0].strip()
                params = json.loads(json_str)
                
                if not params:
                    return "Function takes 0 parameters."
                
                param_strs = []
                required_count = 0
                for p in params:
                    type_hint = p.get("type", "Any")
                    if p.get("optional"):
                        param_strs.append(f"{p['name']}: {type_hint} (optional)")
                    else:
                        param_strs.append(f"{p['name']}: {type_hint}")
                        required_count += 1
                
                return f"Function has {len(params)} parameter(s) ({required_count} required): {', '.join(param_strs)}"
            
            elif "SIGNATURE_ERROR:" in output:
                error_msg = output.split("SIGNATURE_ERROR:")[1].split("\n")[0].strip()
                return f"(Could not detect signature: {error_msg[:100]})"
            
        except subprocess.TimeoutExpired:
            return "(Signature detection timed out)"
        except Exception as e:
            return f"(Signature detection failed: {str(e)[:100]})"
        
        return None
    
    def _extract_param_count(self, param_info: str | None, spec: str) -> int:
        """从 param_info 或 spec 中提取参数数量。
        
        优先使用 param_info（自动检测的结果），如果不可用则尝试从 spec 解析。
        
        Returns:
            检测到的参数数量，如果无法确定则返回 1
        """
        import re
        
        # 1. 首先尝试从 param_info 解析
        if param_info:
            # 格式: "Function has 3 parameter(s) (2 required): ..."
            match = re.search(r'Function has (\d+) parameter', param_info)
            if match:
                return int(match.group(1))
            # 格式: "Function takes 0 parameters."
            if "takes 0 parameters" in param_info:
                return 0
        
        # 2. 尝试从 spec 中的 Function signature 解析
        # 格式: "Function signature: func_name(param1: type1, param2: type2)"
        sig_match = re.search(r'Function signature:\s*\w+\s*\(([^)]*)\)', spec, re.IGNORECASE)
        if sig_match:
            params_str = sig_match.group(1).strip()
            if not params_str:
                return 0
            # 计算逗号分隔的参数数量
            params = [p.strip() for p in params_str.split(',') if p.strip()]
            if params:
                return len(params)
        
        # 3. 尝试从 spec 中的 Parameters 部分解析
        # 格式: "Parameters:\n  - param1: ...\n  - param2: ..."
        param_section_match = re.search(r'Parameters:\s*\n((?:\s*[-•]\s*\w+.*\n?)+)', spec, re.IGNORECASE)
        if param_section_match:
            param_lines = param_section_match.group(1)
            # 计算以 - 或 • 开头的行数
            param_count = len(re.findall(r'^\s*[-•]\s*\w+', param_lines, re.MULTILINE))
            if param_count > 0:
                return param_count
        
        # 4. 默认返回 1（保守策略）
        return 1
    
    def _parse_target(self, target: str) -> tuple[str, str]:
        """解析目标字符串，返回 (模块/路径, 函数名)。
        
        支持格式:
        - "module.path:func" -> ("module.path", "func")
        - "/workspace/path/file.py:func" -> ("/workspace/path/file.py", "func")
        - "path/file.py:func" -> ("path/file.py", "func")
        """
        if ":" not in target:
            raise ValueError(f"Target must be in format 'module:func' or 'path:func', got: {target}")
        
        parts = target.rsplit(":", 1)
        return parts[0], parts[1]
    
    def _get_special_method_type(self, method_name: str) -> str | None:
        """Detect the type of special method for appropriate handling.
        
        Returns:
            'constructor' for __new__, __init__
            'binary_op' for __add__, __mul__, etc.
            'unary_op' for __neg__, __len__, etc.
            None for regular methods
        """
        if method_name in self.CONSTRUCTOR_METHODS:
            return 'constructor'
        elif method_name in self.BINARY_OP_METHODS:
            return 'binary_op'
        elif method_name in self.UNARY_OP_METHODS:
            return 'unary_op'
        return None
    
    def _normalize_path(self, path_str: str) -> Path:
        """标准化路径，智能搜索文件的实际位置。
        
        处理以下情况：
        - 绝对路径：直接使用
        - 相对路径：尝试多种可能的位置
        - 包路径：如 "astropy/units/core.py" 可能在 "/workspace/astropy/astropy/units/core.py"
        """
        path = Path(path_str)
        
        # 如果已经是绝对路径且存在，直接使用
        if path.is_absolute():
            if path.exists():
                return path.resolve()
            # 绝对路径但文件不存在，尝试修复
            return path
        
        # 相对路径：尝试多种可能的位置
        candidates = []
        
        # 候选1: working_dir + path
        candidates.append(self.working_dir / path_str)
        
        # 候选2: 如果路径看起来是包路径 (如 "astropy/units/core.py")
        # 尝试在 working_dir 下的子目录中查找
        # 例如: /workspace + astropy/units/core.py -> /workspace/astropy/astropy/units/core.py
        path_parts = path_str.split("/")
        if len(path_parts) >= 2:
            top_dir = path_parts[0]
            # 检查 working_dir/top_dir 是否是一个项目目录
            project_dir = self.working_dir / top_dir
            if project_dir.is_dir():
                # 尝试 project_dir + full_path
                candidates.append(project_dir / path_str)
        
        # 候选3: 递归搜索 working_dir 下的文件
        file_name = path.name
        for subdir in self.working_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / path_str
                if candidate.exists():
                    candidates.append(candidate)
                # 也尝试 subdir/subdir_name/... 的结构
                nested = subdir / subdir.name / "/".join(path_parts[1:]) if len(path_parts) > 1 else None
                if nested and nested.exists():
                    candidates.append(nested)
        
        # 返回第一个存在的候选路径
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        
        # 没有找到，返回第一个候选（用于错误消息）
        return candidates[0] if candidates else self.working_dir / path_str
    
    # Special methods that need different handling
    CONSTRUCTOR_METHODS = {'__new__', '__init__'}
    BINARY_OP_METHODS = {'__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
                         '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__',
                         '__mod__', '__rmod__', '__pow__', '__rpow__',
                         '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
                         '__and__', '__or__', '__xor__'}
    UNARY_OP_METHODS = {'__neg__', '__pos__', '__abs__', '__invert__', '__len__', '__bool__',
                        '__int__', '__float__', '__str__', '__repr__', '__hash__'}
    
    def _generate_import_code(self, target_mod: str, target_func: str) -> str:
        """Generate import code for the target function.
        
        Supports:
        1. Module format: "astropy.units.core:func"
        2. File path format: "/workspace/astropy/astropy/units/core.py:func"
        3. Mixed format: "astropy/units/core.py:func" -> converted to module format
        4. Class method format: "module:Class.method" -> imports class then gets method
        5. Special methods: __new__, __init__, __add__, etc. -> proper wrapping
        """
        # Check if target_func contains a class (e.g., "Quantity.__array_ufunc__")
        is_class_method = "." in target_func
        if is_class_method:
            parts = target_func.split(".", 1)
            class_name = parts[0]
            method_name = parts[1]
        else:
            class_name = None
            method_name = target_func
        
        # Detect special methods and generate appropriate wrapper code
        special_method_type = self._get_special_method_type(method_name) if is_class_method else None
        
        # Generate the getattr code based on whether it's a class method and its type
        if is_class_method:
            if special_method_type == 'constructor':
                # For __new__ and __init__, just call the class directly
                # target_func = Class, so target_func(data) becomes Class(data)
                getattr_code = f'target_func = getattr(_module, "{class_name}")'
                getattr_code_8 = getattr_code
            elif special_method_type == 'binary_op':
                # For binary operators, we need to wrap them
                # data should be a tuple (left, right)
                getattr_code = f'''_class = getattr(_module, "{class_name}")
    _method = getattr(_class, "{method_name}")
    def target_func(data):
        left, right = data
        # If left is not an instance, try to create one
        if not isinstance(left, _class):
            try:
                left = _class(left)
            except:
                pass
        return _method(left, right)'''
                getattr_code_8 = f'''_class = getattr(_module, "{class_name}")
        _method = getattr(_class, "{method_name}")
        def target_func(data):
            left, right = data
            # If left is not an instance, try to create one
            if not isinstance(left, _class):
                try:
                    left = _class(left)
                except:
                    pass
            return _method(left, right)'''
            elif special_method_type == 'unary_op':
                # For unary operators, wrap to create instance first
                getattr_code = f'''_class = getattr(_module, "{class_name}")
    _method = getattr(_class, "{method_name}")
    def target_func(data):
        # Create an instance first, then call the method
        instance = _class(data) if not isinstance(data, _class) else data
        return _method(instance)'''
                getattr_code_8 = f'''_class = getattr(_module, "{class_name}")
        _method = getattr(_class, "{method_name}")
        def target_func(data):
            # Create an instance first, then call the method
            instance = _class(data) if not isinstance(data, _class) else data
            return _method(instance)'''
            else:
                # Regular class method - get the method directly
                getattr_line1 = f'_class = getattr(_module, "{class_name}")'
                getattr_line2 = f'target_func = getattr(_class, "{method_name}")'
                getattr_code = f'{getattr_line1}\n    {getattr_line2}'  # For try block (4 spaces)
                getattr_code_8 = f'{getattr_line1}\n        {getattr_line2}'  # For nested try (8 spaces)
        else:
            getattr_code = f'target_func = getattr(_module, "{target_func}")'
            getattr_code_8 = getattr_code
        
        # Check if it's a file path format
        is_file_path = target_mod.endswith(".py") or "/" in target_mod
        
        # Handle relative path format (e.g., "astropy/units/core.py")
        # Convert to module format and use module import
        if is_file_path and "/" in target_mod and not target_mod.startswith("/"):
            module_name = target_mod.replace("/", ".").replace(".py", "")
            top_package = module_name.split(".")[0]
            return f'''# Import from path (converted to module format)
import sys
import os
import importlib

print("[FuzzHypo] Importing: {module_name}:{target_func}")
print(f"  cwd: {{os.getcwd()}}, sys.path[0]: {{sys.path[0] if sys.path else 'empty'}}")

try:
    _module = importlib.import_module("{module_name}")
    {getattr_code}
    print(f"[FuzzHypo] OK: imported from {module_name}")
except Exception as e:
    print(f"[FuzzHypo] FAILED: {{e}}")
    raise ImportError(f"Failed to import {target_func} from {module_name}: {{e}}")'''
        
        if is_file_path:
            # Absolute file path format
            normalized_path = self._normalize_path(target_mod)
            is_package_file = self._is_file_in_package(normalized_path)
            
            if is_package_file:
                # File belongs to a package, convert to module name
                module_name = self._path_to_module_name(target_mod)
                pkg_root = self._find_package_root(normalized_path)
                top_package = module_name.split(".")[0] if module_name else None
                
                return f'''# Import package module (multi-strategy)
import sys
import os
import importlib

print("[FuzzHypo] Importing: {module_name}:{target_func}")
print(f"  path: {normalized_path}, exists: {{os.path.exists('{normalized_path}')}}")

_module = None
_error = None

# Strategy 1: Direct import (uses installed package)
try:
    _module = importlib.import_module("{module_name}")
    {getattr_code}
    print(f"[FuzzHypo] OK: direct import from {module_name}")
except Exception as e1:
    _error = e1
    print(f"[FuzzHypo] Strategy 1 failed: {{e1}}")
    
    # Strategy 2: Add source dir to sys.path
    _pkg_root = "{pkg_root}"
    if _pkg_root and _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    
    try:
        _top_package = "{top_package}"
        if _top_package in sys.modules:
            importlib.reload(sys.modules[_top_package])
        else:
            importlib.import_module(_top_package)
        
        _module = importlib.import_module("{module_name}")
        {getattr_code_8}
        _error = None
        print(f"[FuzzHypo] OK: imported via sys.path")
    except Exception as e2:
        _error = e2
        print(f"[FuzzHypo] Strategy 2 failed: {{e2}}")

if _error:
    raise ImportError(f"Failed to import {target_func} from {module_name}: {{_error}}")'''
            else:
                # Standalone file
                return f'''# Import standalone file
import sys
import os
import importlib.util

print("[FuzzHypo] Importing standalone: {normalized_path}:{target_func}")
print(f"  exists: {{os.path.exists('{normalized_path}')}}")

_file_path = "{normalized_path}"
_file_dir = "{normalized_path.parent}"
if _file_dir not in sys.path:
    sys.path.insert(0, str(_file_dir))

_spec = importlib.util.spec_from_file_location("_target_module", _file_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module from: {{_file_path}}")
_module = importlib.util.module_from_spec(_spec)
sys.modules["_target_module"] = _module
_spec.loader.exec_module(_module)
{getattr_code}
print(f"[FuzzHypo] OK: imported standalone file")'''
        else:
            # Python module format (e.g., "astropy.units.core")
            top_package = target_mod.split(".")[0]
            return f'''# Import from module path
import sys
import os
import importlib

print("[FuzzHypo] Importing: {target_mod}:{target_func}")
print(f"  cwd: {{os.getcwd()}}, top pkg loaded: {{'{top_package}' in sys.modules}}")

try:
    _module = importlib.import_module("{target_mod}")
    {getattr_code}
    print(f"[FuzzHypo] OK: imported from {target_mod}")
except Exception as e:
    print(f"[FuzzHypo] FAILED: {{e}}")
    raise ImportError(f"Failed to import {target_func} from {target_mod}: {{e}}")'''
    
    def _is_file_in_package(self, file_path: Path) -> bool:
        """Check if file is inside a Python package (direct parent has __init__.py).
        
        Only returns True if the file's DIRECT parent directory has __init__.py.
        This avoids false positives for files like /workspace/astropy/test_xxx.py
        where /workspace/astropy/ is a project root, not a package.
        
        Examples:
        - /workspace/astropy/astropy/units/core.py -> True (parent astropy/units has __init__.py)
        - /workspace/astropy/test_file.py -> False (parent /workspace/astropy has no __init__.py)
        """
        parent = file_path.parent
        init_file = parent / "__init__.py"
        return init_file.exists()
    
    def _path_to_module_name(self, path_str: str) -> str | None:
        """将文件路径转换为 Python 模块名。
        
        例如：
        - "/workspace/astropy/astropy/units/core.py" -> "astropy.units.core"
        - "/workspace/project/src/module.py" -> "src.module" (如果 src 有 __init__.py)
        
        算法：
        1. 从目标文件的父目录开始向上遍历
        2. 收集所有有 __init__.py 的目录名
        3. 当遇到没有 __init__.py 的目录时停止（这是包根目录的父目录）
        4. 将收集的目录名 + 文件名 组成模块名
        """
        # 规范化路径
        path = self._normalize_path(path_str)
        
        # 获取模块名（文件名去掉 .py）
        module_file_name = path.stem  # e.g., "core"
        
        # 从文件的父目录开始向上遍历
        current_dir = path.parent
        package_parts: list[str] = []
        
        while current_dir != current_dir.parent:
            # 检查当前目录是否有 __init__.py
            init_file = current_dir / "__init__.py"
            if init_file.exists():
                # 是一个包，添加到列表
                package_parts.insert(0, current_dir.name)
                current_dir = current_dir.parent
            else:
                # 不是包，停止遍历
                break
        
        # 构建完整模块名
        if package_parts:
            module_name = ".".join(package_parts + [module_file_name])
            return module_name
        
        # 没有找到包结构，返回文件名
        return module_file_name
    
    def _find_package_root(self, file_path: Path) -> str:
        """查找包含给定文件的包的根目录。
        
        向上遍历目录树，查找包含 __init__.py 的最顶层目录的父目录。
        """
        current = file_path.parent
        package_root = None
        
        while current != current.parent:
            init_file = current / "__init__.py"
            if init_file.exists():
                # 找到了一个包，继续向上查找
                package_root = current.parent
            elif package_root is not None:
                # 已经离开了包的范围
                break
            current = current.parent
        
        # 如果没有找到包根目录，使用 working_dir
        return str(package_root) if package_root else str(self.working_dir)
    
    def _extract_failures_json(self, output: str) -> list[dict]:
        """Extract all failures from the JSON output block."""
        failures = []
        
        # Try to extract JSON block first
        json_match = re.search(
            r"\[FuzzHypo:FAILURES_JSON_START\]\s*\n(.+?)\n\s*\[FuzzHypo:FAILURES_JSON_END\]",
            output,
            re.DOTALL
        )
        
        if json_match:
            try:
                failures = json.loads(json_match.group(1).strip())
                return failures
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract from traditional Hypothesis output
        crash = self._extract_crash(output)
        if crash:
            failures.append({
                "input": str(crash),
                "repr": str(crash),
                "error": "Extracted from Hypothesis output"
            })
        
        return failures
    
    def _extract_crash(self, output: str) -> Any:
        """Extract crash data from Hypothesis output (fallback method)."""
        # Match Falsifying example: test_fuzz_task(data=...)
        match = re.search(r"Falsifying example:.*?data=(.*?)\n", output, re.DOTALL)
        if match:
            raw_val = match.group(1).strip()
            try:
                return ast.literal_eval(raw_val)
            except:
                return raw_val
        return None
    
    def _extract_example_count(self, output: str, max_examples: int) -> str:
        """从输出中提取实际运行的例子数量（字符串形式）"""
        count = self._extract_example_count_int(output, max_examples)
        return str(count) if count else "?"
    
    def _extract_example_count_int(self, output: str, max_examples: int) -> int | None:
        """从输出中提取实际运行的例子数量（整数形式）"""
        # 匹配 [FuzzHypo] Total examples tested: X / Y
        match = re.search(r"\[FuzzHypo\] Total examples tested: (\d+)", output)
        if match:
            return int(match.group(1))
        
        # 匹配 Hypothesis 的统计输出（如果使用了 --hypothesis-show-statistics）
        match = re.search(r"(\d+) passing examples", output)
        if match:
            return int(match.group(1))
        
        # 匹配 "Falsifying example at run #X"
        match = re.search(r"at example #(\d+)", output)
        if match:
            return int(match.group(1))
        
        # 如果测试通过且没有异常，假设运行了 max_examples
        if "passed" in output.lower() and "failed" not in output.lower():
            return max_examples
        
        # 无法确定
        return None
    
    def _extract_sample_inputs(self, output: str) -> list[str] | None:
        """从输出中提取样本输入"""
        samples: list[str] = []
        
        # 匹配 Hypothesis 的 Trying example 输出
        for match in re.finditer(r"Trying example: test_fuzz_task\(data=(.*?)\)", output):
            sample = match.group(1).strip()
            if sample and len(sample) < 100:  # 避免过长的输入
                samples.append(sample)
                if len(samples) >= 5:
                    break
        
        # 匹配 Falsifying example
        match = re.search(r"Falsifying example:.*?data=(.*?)(?:\)|,\s*$)", output, re.DOTALL)
        if match:
            samples.insert(0, f"[FAILING] {match.group(1).strip()}")
        
        return samples if samples else None
    
    def _truncate_harness(self, harness_code: str, max_lines: int = 50) -> str:
        """截断 harness 代码，保留关键部分"""
        lines = harness_code.strip().split("\n")
        
        if len(lines) <= max_lines:
            return harness_code.strip()
        
        # 保留前 30 行和后 15 行
        head = lines[:30]
        tail = lines[-15:]
        
        return "\n".join(head) + f"\n\n# ... ({len(lines) - 45} lines omitted) ...\n\n" + "\n".join(tail)
    
    def _extract_fuzz_output(self, full_output: str) -> str:
        """Extract only FuzzHypo-specific output, filtering out Hypothesis/pytest noise.
        
        This is important because:
        1. We catch exceptions in the test harness, so Hypothesis always shows "passing"
        2. This can confuse the agent (FuzzHypo says bug found, Hypothesis says all pass)
        3. We only want to show our own [FuzzHypo] tagged output
        """
        lines = full_output.split('\n')
        fuzz_lines = []
        in_failure_summary = False
        
        for line in lines:
            # Include FuzzHypo tagged lines
            if '[FuzzHypo]' in line:
                fuzz_lines.append(line)
                continue
            
            # Include failure summary section
            if '=== FAILURE SUMMARY ===' in line:
                in_failure_summary = True
                fuzz_lines.append(line)
                continue
            
            if in_failure_summary:
                # End of failure summary
                if line.startswith('===') and 'FAILURE' not in line:
                    in_failure_summary = False
                else:
                    fuzz_lines.append(line)
                continue
            
            # Include import/error diagnostics
            if line.startswith('  ') and ('Error:' in line or 'Input:' in line):
                fuzz_lines.append(line)
        
        if fuzz_lines:
            return '\n'.join(fuzz_lines)
        
        # Fallback: if no FuzzHypo output found, show a clean summary
        return "(No detailed FuzzHypo output captured)"

    def _get_testbed_python(self) -> str:
        """Get the Python executable from the SWE-bench testbed conda environment.
        
        Returns the Python executable path from the testbed conda environment,
        or falls back to sys.executable if conda is not available.
        """
        # Try to use conda run to get Python from testbed environment
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
        
        # Fallback: try to find Python in conda testbed environment directly
        conda_python = "/opt/miniconda3/envs/testbed/bin/python"
        if os.path.exists(conda_python):
            return conda_python
        
        # Last resort: use current Python (agent server's Python)
        return sys.executable
    
    def _ensure_dependencies(self):
        """Ensure hypothesis is installed in the testbed conda environment."""
        python_exe = self._get_testbed_python()
        
        # Check if hypothesis is available in testbed environment
        try:
            result = subprocess.run(
                [python_exe, "-c", "import hypothesis"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return  # Dependencies already available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Install in testbed environment using conda run
        try:
            subprocess.check_call(
                ["conda", "run", "-n", "testbed", "pip", "install", "hypothesis"],
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            # Fallback: try direct pip install
            try:
                subprocess.check_call(
                    [python_exe, "-m", "pip", "install", "hypothesis"],
                    timeout=60,
                )
            except Exception as e:
                # If installation fails, we'll try to run anyway and let it fail with a clear error
                pass

    def _install_extra_packages(self, packages: list[str]) -> None:
        """Install extra packages needed for strategy or post-condition.
        
        Args:
            packages: List of package names to install (e.g., ['numpy', 'pandas'])
        """
        if not packages:
            return
        
        # Filter out standard library modules that don't need installation
        stdlib_modules = {
            'sys', 'os', 'json', 'traceback', 're', 'math', 'random', 'datetime',
            'collections', 'itertools', 'functools', 'operator', 'copy', 'typing',
            'io', 'string', 'time', 'calendar', 'abc', 'contextlib', 'decimal',
            'fractions', 'numbers', 'pathlib', 'tempfile', 'shutil', 'glob',
            'unittest', 'doctest', 'warnings', 'dataclasses', 'enum', 'types',
        }
        
        packages_to_install = [
            pkg for pkg in packages 
            if pkg.lower() not in stdlib_modules and not pkg.startswith('_')
        ]
        
        if not packages_to_install:
            return
        
        # Try to install using conda run first
        cmd = [
            "bash", "-c",
            f"source /opt/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate testbed && "
            f"pip install -q {' '.join(packages_to_install)} 2>/dev/null"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.working_dir),
            )
            if result.returncode == 0:
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: direct pip install
        python_exe = self._get_testbed_python()
        try:
            subprocess.run(
                [python_exe, "-m", "pip", "install", "-q"] + packages_to_install,
                capture_output=True,
                timeout=120,
            )
        except Exception:
            # If installation fails, continue anyway - import error will be reported
            pass
    
    def _generate_extra_imports(
        self, 
        extra_imports: list[str] | None,
        import_statements: list[str] | None
    ) -> str:
        """Generate import code for extra packages.
        
        Args:
            extra_imports: Simple package names to import (e.g., ['numpy', 'pandas'])
            import_statements: Full import statements (e.g., ['import numpy as np'])
        
        Returns:
            Python code string with import statements
        """
        lines = []
        
        # Common package aliases for convenience
        common_aliases = {
            'numpy': 'np',
            'pandas': 'pd',
            'tensorflow': 'tf',
            'matplotlib.pyplot': 'plt',
        }
        
        # Generate imports from package names
        if extra_imports:
            for pkg in extra_imports:
                # Skip if already in standard imports or hypothesis
                if pkg in ('hypothesis', 'pytest', 'sys', 'os', 'json', 'traceback'):
                    continue
                
                # Use common alias if available
                alias = common_aliases.get(pkg)
                if alias:
                    lines.append(f"import {pkg} as {alias}")
                else:
                    lines.append(f"import {pkg}")
        
        # Add custom import statements directly
        if import_statements:
            for stmt in import_statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('#'):
                    # Ensure it's a valid import statement
                    if stmt.startswith(('import ', 'from ')):
                        lines.append(stmt)
                    else:
                        # Assume it's a module name
                        lines.append(f"import {stmt}")
        
        if not lines:
            return "# No extra imports"
        
        # Wrap in try-except to handle missing packages gracefully
        import_block = "\n".join(f"    {line}" for line in lines)
        return f"""try:
{import_block}
except ImportError as _import_err:
    print(f"[FuzzHypo] Warning: Failed to import extra package: {{_import_err}}")"""

    def _load_spec(self, spec: str) -> str:
        """Load spec content from file or use directly if it's content.
        
        The spec parameter can be either:
        1. A file path (relative to working_dir)
        2. The actual spec content as a string
        
        We distinguish by checking if the string contains newlines or YAML/JSON structure.
        """
        # If spec contains newlines or looks like YAML/JSON content, treat it as content directly
        if '\n' in spec or spec.strip().startswith(('inputs:', '{', '[')):
            return spec
        
        # Otherwise, try to treat it as a file path
        try:
            spec_path = self.working_dir / spec
            if spec_path.exists() and spec_path.is_file():
                return spec_path.read_text()
        except (OSError, ValueError):
            # If path construction fails (e.g., invalid path), treat spec as content
            pass
        
        # Fallback: return spec as-is (content)
        return spec