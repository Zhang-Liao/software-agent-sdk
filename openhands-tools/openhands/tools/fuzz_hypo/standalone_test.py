#!/usr/bin/env python3
"""
独立测试脚本：验证 FuzzHypo 生成的测试是否正确运行

这个脚本模拟 FuzzHypo 生成的 test_harness.py，
可以在本地验证 Hypothesis 是否正确运行多个测试用例。

使用方法:
    python standalone_test.py
"""

import sys


def test_hypothesis_basic():
    """基础测试：验证 Hypothesis 是否正常工作"""
    print("\n" + "=" * 60)
    print("测试1: Hypothesis 基础功能")
    print("=" * 60)
    
    import hypothesis.strategies as st
    from hypothesis import given, settings, Verbosity
    
    run_count = 0
    
    @settings(max_examples=50, verbosity=Verbosity.verbose)
    @given(st.integers())
    def test_integers(x):
        nonlocal run_count
        run_count += 1
        assert isinstance(x, int)
    
    try:
        test_integers()
        print(f"✅ 测试通过！运行了 {run_count} 个例子")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_hypothesis_with_strategy():
    """策略测试：验证自定义策略是否工作"""
    print("\n" + "=" * 60)
    print("测试2: 自定义策略")
    print("=" * 60)
    
    import hypothesis.strategies as st
    from hypothesis import given, settings, Verbosity
    
    run_count = 0
    
    # 模拟 LLM 生成的策略
    strategy = st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=20),
        st.none()
    )
    
    @settings(max_examples=100, verbosity=Verbosity.normal)
    @given(strategy)
    def test_mixed(x):
        nonlocal run_count
        run_count += 1
        # 简单的后验条件
        assert x is None or isinstance(x, (int, float, str))
    
    try:
        test_mixed()
        print(f"✅ 测试通过！运行了 {run_count} 个例子")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_hypothesis_with_function():
    """函数测试：模拟真实的 FuzzHypo 场景"""
    print("\n" + "=" * 60)
    print("测试3: 模拟真实 FuzzHypo 场景")
    print("=" * 60)
    
    import hypothesis.strategies as st
    from hypothesis import given, settings, Verbosity
    
    # 目标函数
    def target_func(x):
        """模拟一个可能有 bug 的函数"""
        if isinstance(x, int):
            return x * 2
        elif isinstance(x, float):
            return x / 2
        elif isinstance(x, str):
            return len(x)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")
    
    # 后验条件
    def post_condition(input_obj, output_obj):
        # 检查返回值类型
        if isinstance(input_obj, int):
            return isinstance(output_obj, int)
        elif isinstance(input_obj, float):
            return isinstance(output_obj, float)
        elif isinstance(input_obj, str):
            return isinstance(output_obj, int)
        return True
    
    run_count = 0
    
    strategy = st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
        st.text(max_size=50)
    )
    
    @settings(max_examples=200, verbosity=Verbosity.normal)
    @given(strategy)
    def test_fuzz_task(data):
        nonlocal run_count
        run_count += 1
        result = target_func(data)
        assert post_condition(data, result) is True, "Fuzzing violation found"
    
    try:
        test_fuzz_task()
        print(f"✅ 测试通过！运行了 {run_count} 个例子")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_hypothesis_find_bug():
    """Bug 检测测试：验证能否发现 bug"""
    print("\n" + "=" * 60)
    print("测试4: Bug 检测能力")
    print("=" * 60)
    
    import hypothesis.strategies as st
    from hypothesis import given, settings, Verbosity
    
    # 有 bug 的目标函数
    def buggy_func(x):
        """这个函数在 x == 42 时会失败"""
        if x == 42:
            raise ValueError("Found the bug!")
        return x
    
    run_count = 0
    
    @settings(max_examples=200, verbosity=Verbosity.normal)
    @given(st.integers(min_value=0, max_value=100))
    def test_find_bug(data):
        nonlocal run_count
        run_count += 1
        result = buggy_func(data)
        assert result == data
    
    try:
        test_find_bug()
        print(f"❌ 应该发现 bug，但没有！运行了 {run_count} 个例子")
        return False
    except (ValueError, AssertionError) as e:
        print(f"✅ 成功发现 bug: {e}")
        print(f"   运行了 {run_count} 个例子后发现")
        return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("FuzzHypo 独立测试")
    print("=" * 60)
    print("验证 Hypothesis 是否正常工作并运行多个测试用例")
    
    results = []
    
    results.append(("基础功能", test_hypothesis_basic()))
    results.append(("自定义策略", test_hypothesis_with_strategy()))
    results.append(("模拟场景", test_hypothesis_with_function()))
    results.append(("Bug检测", test_hypothesis_find_bug()))
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n通过: {passed}/{len(results)}")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
