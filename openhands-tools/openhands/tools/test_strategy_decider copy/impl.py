"""Executor for TestStrategyDecider tool."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openhands.sdk.tool import ToolExecutor
from openhands.tools.test_strategy_decider.definition import (
    TestStrategyDeciderAction,
    TestStrategyDeciderObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.llm import LLM


_EMBEDDED_RUBRIC = """## When to use fuzz / PBT (Hypothesis)

Fuzzing / PBT is only worth doing when you can state a **general oracle**
(a verifiable property that does not depend on hidden tests or deep domain knowledge).
Problem types where fuzz / PBT is a good first choice:

- **Round-trip / reversibility**: `write→read`, `encode→decode`, `dump→load`,
  `parse→format→parse`, `serialize→deserialize` should preserve semantics.
- **Differential testing**: there is a trusted reference/alternative implementation to compare against
  (e.g., NumPy/SciPy, a simpler but correct implementation, or a spec algorithm).
- **Algebraic / structural invariants**: idempotence, commutativity/associativity, monotonicity,
  length/shape/sorting invariants, boundary conditions that must not crash, etc.
- **Parser/IO robustness**: invalid inputs, weird delimiters, NUL bytes, whitespace/encoding variations,
  extreme sizes, etc.

Not recommended (or only very small-scale) for fuzz / PBT:

- **Highly test-specific behavior**: exact error-message/format string matching, tightly coupled to internal
  implementation details.
- **Oracles that require deep framework semantics**: you cannot reliably judge correctness without hidden tests.
- **Mostly integration/dependency/environment issues**: the fix is not in a pure function/algorithm/parsing logic.
"""

# This addendum is appended to either the repo rubric file
# (`benchmarks/swebench/prompts/hypothesis_default_lz.j2`) or the embedded rubric.
# Keep it focused on decision quality and output usefulness (not tool discovery).
_RUBRIC_ADDENDUM = r"""## Decision targets (pick exactly one)

- fuzz:
  - Hypothesis/property-based testing.
  - Hard requirement: you can state a general, automation-friendly oracle/invariant.
  - Typical oracles: "never crash", "round-trip", "idempotent", "format/length constraints", "ordering/shape bounds".
- assertion:
  - Deterministic regression test(s) with explicit expected outputs/exceptions/warnings.
  - Prefer when the issue is a specific edge case with stable expected behavior.
- simple_test:
  - Minimal reproduction (1-5 hand-picked cases) without Hypothesis.
  - Prefer when the oracle is ambiguous/expensive, or the bug is integration-heavy.

## Output quality requirements

Your JSON must be actionable. Recommended guidelines:
- rationale: 2-6 sentences, state the oracle availability + why the chosen strategy is lowest-cost.
- recommended_next_steps: 3-8 bullets, each starts with a verb and mentions concrete artifacts
  (file path, test name pattern, input construction, assertion/oracle, and any needed fixtures).

## “Example bank” placement structure (when you include examples in the rubric)

Use this consistent format so the model can pattern-match quickly:

### Example: <swebench_dev_markdown/id> — <pattern name>
- Symptom: <1 sentence>
- Why fuzz/assertion/simple_test: <1-2 sentences>
- Oracle: <property/assertion>
- Minimal test sketch: <very short snippet or pseudocode>

Keep examples short. Prefer 2-4 examples total, each illustrating a distinct decision boundary
(e.g., round-trip invariant vs single edge case vs ambiguous correctness).

## Next-steps templates (use these patterns)

If decision == fuzz:
- Identify the smallest public API surface to test.
- Define 1-2 invariants and guardrails (timeouts, max examples, shrinking).
- Define input strategies (favor boundary values + malformed inputs when parsing/IO).
- Add at least one deterministic regression example discovered during fuzzing (seed the failure).

If decision == assertion:
- Write the narrowest unit/integration test that reproduces the bug.
- Assert the exact post-fix behavior (return value, warning class+message, exception type+message).
- Add parametrization for nearby edge cases only if the oracle remains unambiguous.

If decision == simple_test:
- Start with a minimal repro case that fails before fix and passes after.
- Once behavior/spec stabilizes, promote to assertion or fuzz (do not overfit to incidental details).
"""


class TestStrategyDeciderExecutor(
    ToolExecutor[TestStrategyDeciderAction, TestStrategyDeciderObservation]
):
    def __init__(self, llm: "LLM | None" = None):
        self.llm = llm

    def __call__(
        self, action: TestStrategyDeciderAction, conversation=None
    ) -> TestStrategyDeciderObservation:
        try:
            python_signals = self._collect_python_internal_signals(action)
            should_use_python_internal_info = self._should_use_python_internal_info(
                python_signals
            )
            rubric = self._load_rubric_text()
            raw = self._call_llm(
                rubric=rubric,
                action=action,
                python_signals=python_signals,
                should_use_python_internal_info=should_use_python_internal_info,
            )
            data = self._parse_json_response(raw)
            return self._coerce_observation(
                data=data,
                python_signals=python_signals,
                should_use_python_internal_info=should_use_python_internal_info,
            )
        except Exception as e:
            # Conservative fallback: prefer simple deterministic test unless fuzz is clearly applicable
            heuristic = self._heuristic_fallback(action)
            return TestStrategyDeciderObservation(
                decision=heuristic["decision"],
                rationale=(
                    f"(Tool error; used heuristic fallback)\nError: {type(e).__name__}: {e}\n\n"
                    + heuristic.get("rationale", "")
                ),
                recommended_next_steps=heuristic.get("recommended_next_steps", []),
                suggested_oracle=heuristic.get("suggested_oracle"),
                should_use_python_internal_info=heuristic.get(
                    "should_use_python_internal_info", False
                ),
                python_internal_signals=heuristic.get("python_internal_signals", {}),
                is_error=True,
            )

    # --- rubric loading ---
    def _load_rubric_text(self) -> str:
        """Try to load the rubric from the repo file; fall back to embedded text."""
        env_path = os.getenv("OH_TEST_STRATEGY_RUBRIC_PATH")
        candidates: list[Path] = []
        if env_path:
            candidates.append(Path(env_path).expanduser())

        # Walk parents from this file to find the benchmarks path in monorepo layout.
        here = Path(__file__).resolve()
        for p in here.parents:
            candidates.append(p / "benchmarks" / "swebench" / "prompts" / "hypothesis_default_lz.j2")

        for c in candidates:
            try:
                if c.exists() and c.is_file():
                    return c.read_text(encoding="utf-8") + "\n\n" + _RUBRIC_ADDENDUM
            except Exception:
                continue

        return _EMBEDDED_RUBRIC + "\n\n" + _RUBRIC_ADDENDUM

    # --- llm call ---
    def _call_llm(
        self,
        rubric: str,
        action: TestStrategyDeciderAction,
        python_signals: dict[str, Any],
        should_use_python_internal_info: bool,
    ) -> str:
        system_prompt = f"""You are a senior test engineer. Your job is to decide which testing approach to use.

Use this rubric:
{rubric}

Return ONLY valid JSON (no markdown fences, no extra text) with exactly these keys:
{{
  "decision": "fuzz" | "assertion" | "simple_test",
  "rationale": "string",
  "recommended_next_steps": ["string", "..."],
  "suggested_oracle": "string or null",
  "should_use_python_internal_info": "boolean",
  "python_internal_notes": "string"
}}

Decision meanings:
- fuzz: use Hypothesis / property-based testing (requires a general oracle)
- assertion: write deterministic regression test(s) with clear assertions
- simple_test: write a minimal reproduction script/test without Hypothesis (1-5 cases)

Rules:
- If you cannot propose a reliable, general oracle, DO NOT choose "fuzz".
- "assertion" is preferred over "simple_test" when you can assert stable behavior with explicit expected outputs/errors.
- recommended_next_steps should be concrete and actionable.
- If subclass hierarchy is broad or API behavior depends on polymorphism, prefer using Python internal signals.
"""

        user_prompt = "Issue description:\n" + action.issue_description.strip() + "\n"
        if action.context:
            user_prompt += "\nAdditional context:\n" + action.context.strip() + "\n"
        user_prompt += (
            "\nPython internal signal summary:\n"
            + json.dumps(python_signals, ensure_ascii=False, indent=2)
            + "\n"
            + f"\nDefault recommendation from static heuristics: "
            + f"should_use_python_internal_info={should_use_python_internal_info}\n"
        )

        if self.llm is None:
            raise RuntimeError("LLM is not configured for TestStrategyDeciderExecutor")

        from openhands.sdk.llm.message import Message, TextContent, content_to_str

        messages = [
            Message(role="system", content=[TextContent(text=system_prompt)]),
            Message(role="user", content=[TextContent(text=user_prompt)]),
        ]
        resp = self.llm.completion(messages=messages, temperature=0.1)
        text_parts = content_to_str(resp.message.content)
        return "".join(text_parts)

    # --- parsing/coercion ---
    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        content = (raw or "").strip()

        cleaners = [
            lambda s: s,
            lambda s: re.sub(r"```(?:json)?\s*|\s*```", "", s).strip(),
            self._extract_json_object,
            lambda s: self._extract_json_object("\n".join(line.strip() for line in s.splitlines())),
        ]
        last: Exception | None = None
        for fn in cleaners:
            try:
                cleaned = fn(content)
                if cleaned:
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict):
                        return obj
            except Exception as e:
                last = e
                continue
        raise ValueError(f"Could not parse JSON from LLM output. Last error: {last}")

    def _extract_json_object(self, text: str) -> str:
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]
        return text

    def _coerce_observation(
        self,
        data: dict[str, Any],
        python_signals: dict[str, Any],
        should_use_python_internal_info: bool,
    ) -> TestStrategyDeciderObservation:
        decision = data.get("decision")
        if decision not in ("fuzz", "assertion", "simple_test"):
            decision = "simple_test"
        rationale = str(data.get("rationale") or "").strip()
        steps_raw = data.get("recommended_next_steps") or []
        if not isinstance(steps_raw, list):
            steps_raw = [str(steps_raw)]
        steps = [str(s).strip() for s in steps_raw if str(s).strip()]
        suggested_oracle = data.get("suggested_oracle")
        if suggested_oracle is not None:
            suggested_oracle = str(suggested_oracle).strip() or None
        if decision != "fuzz":
            suggested_oracle = None
        llm_should_use = data.get("should_use_python_internal_info")
        if isinstance(llm_should_use, bool):
            should_use_python_internal_info = llm_should_use
        return TestStrategyDeciderObservation(
            decision=decision,
            rationale=rationale,
            recommended_next_steps=steps,
            suggested_oracle=suggested_oracle,
            should_use_python_internal_info=should_use_python_internal_info,
            python_internal_signals=python_signals,
        )

    # --- fallback heuristic ---
    def _heuristic_fallback(self, action: TestStrategyDeciderAction) -> dict[str, Any]:
        text = (action.issue_description + "\n" + (action.context or "")).lower()
        python_signals = self._collect_python_internal_signals(action)
        should_use_python_internal_info = self._should_use_python_internal_info(
            python_signals
        )

        fuzz_keywords = [
            "serialize",
            "deserialize",
            "round-trip",
            "parse",
            "parser",
            "encode",
            "decode",
            "dump",
            "load",
            "idempotent",
            "invariant",
            "monotonic",
            "robust",
            "io",
        ]
        assertion_keywords = ["regression", "expected", "assert", "exception", "traceback", "error message"]

        if any(k in text for k in fuzz_keywords):
            return {
                "decision": "fuzz",
                "rationale": "Heuristic: problem looks like parsing/round-trip/invariant-friendly; fuzz may be valuable if we can define a general oracle.",
                "suggested_oracle": "Define a round-trip or non-crash invariant that does not depend on hidden tests.",
                "recommended_next_steps": [
                    "Identify a pure/function-level target or a thin wrapper that isolates the bug.",
                    "Propose a general oracle (round-trip / invariant / non-crash) and input constraints.",
                    "If available, use fuzz_hypo_agent (or fuzz_hypo) on the wrapper function.",
                ],
                "should_use_python_internal_info": should_use_python_internal_info,
                "python_internal_signals": python_signals,
            }

        if any(k in text for k in assertion_keywords):
            return {
                "decision": "assertion",
                "rationale": "Heuristic: issue mentions specific expected behavior/errors; a deterministic regression test is appropriate.",
                "recommended_next_steps": [
                    "Write a minimal regression test covering the reported failing input.",
                    "Assert on stable behavior (output or raised exception type), avoid brittle exact error-message matching if not required.",
                ],
                "should_use_python_internal_info": should_use_python_internal_info,
                "python_internal_signals": python_signals,
            }

        return {
            "decision": "simple_test",
            "rationale": "Heuristic: no clear general oracle; start with minimal reproduction to lock in the fix.",
            "recommended_next_steps": [
                "Create a minimal reproduction script/test with 1-3 hand-picked cases.",
                "Once reproduction is stable, convert it into a regression test (assertion-based).",
            ],
            "should_use_python_internal_info": should_use_python_internal_info,
            "python_internal_signals": python_signals,
        }

    # --- python internal signal helpers ---
    def _collect_python_internal_signals(
        self, action: TestStrategyDeciderAction
    ) -> dict[str, Any]:
        signals: dict[str, Any] = {
            "enabled": bool(action.include_python_internal_signals),
            "python_target": action.python_target,
        }
        if not action.include_python_internal_signals or not action.python_target:
            return signals

        file_path, class_name = self._parse_python_target(action.python_target)
        signals["target_file"] = str(file_path) if file_path else None
        signals["target_class"] = class_name
        if not file_path or not class_name:
            signals["error"] = "python_target must be like 'path/to/file.py:ClassName'."
            return signals
        if not file_path.exists():
            signals["error"] = f"Target file does not exist: {file_path}"
            return signals

        class_defs = self._scan_python_class_defs(file_path.parent)
        direct_subclasses = sorted(
            {
                cls for cls, bases in class_defs.items() if class_name in bases and cls != class_name
            }
        )
        method_count = self._count_class_methods(file_path=file_path, class_name=class_name)
        test_ref_count = self._count_tests_referencing_class(
            base_dir=file_path.parent, class_name=class_name
        )
        signals["subclass_count"] = len(direct_subclasses)
        signals["direct_subclasses"] = direct_subclasses[:50]
        signals["method_count"] = method_count
        signals["test_files_referencing_target_class"] = test_ref_count
        return signals

    def _parse_python_target(self, target: str) -> tuple[Path | None, str | None]:
        raw = (target or "").strip()
        if ":" not in raw:
            return None, None
        path_part, symbol_part = raw.rsplit(":", 1)
        path_part = path_part.strip()
        symbol_part = symbol_part.strip()
        if not path_part.endswith(".py") or not symbol_part:
            return None, None
        p = Path(path_part)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve(), symbol_part

    def _scan_python_class_defs(self, base_dir: Path) -> dict[str, set[str]]:
        class_defs: dict[str, set[str]] = {}
        for py_file in base_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases: set[str] = set()
                for base in node.bases:
                    base_name = self._name_from_base_node(base)
                    if base_name:
                        bases.add(base_name)
                class_defs.setdefault(node.name, set()).update(bases)
        return class_defs

    def _name_from_base_node(self, base: ast.expr) -> str | None:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Subscript):
            return self._name_from_base_node(base.value)
        return None

    def _count_class_methods(self, file_path: Path, class_name: str) -> int:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return sum(
                    1
                    for n in node.body
                    if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                )
        return 0

    def _count_tests_referencing_class(self, base_dir: Path, class_name: str) -> int:
        count = 0
        for py_file in base_dir.rglob("test*.py"):
            try:
                text = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if re.search(rf"\b{re.escape(class_name)}\b", text):
                count += 1
        return count

    def _should_use_python_internal_info(self, python_signals: dict[str, Any]) -> bool:
        if not python_signals.get("enabled"):
            return False
        subclass_count = int(python_signals.get("subclass_count") or 0)
        method_count = int(python_signals.get("method_count") or 0)
        test_ref_count = int(python_signals.get("test_files_referencing_target_class") or 0)
        return subclass_count >= 2 or (subclass_count >= 1 and method_count >= 8) or test_ref_count >= 2

