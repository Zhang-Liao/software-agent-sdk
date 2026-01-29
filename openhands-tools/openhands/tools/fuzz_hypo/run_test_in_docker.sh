#!/bin/bash
# FuzzHypo 测试脚本
# 在 Docker 容器中运行此脚本来验证 Hypothesis 是否正常工作
#
# 使用方法:
#   1. 进入正在运行的 SWE-bench Docker 容器
#      docker exec -it <container_id> bash
#   
#   2. 运行此脚本
#      bash /agent-server/.venv/lib/python3.12/site-packages/openhands/tools/fuzz_hypo/run_test_in_docker.sh
#
#   或者直接运行 Python 脚本:
#      python /agent-server/.venv/lib/python3.12/site-packages/openhands/tools/fuzz_hypo/standalone_test.py

echo "=============================================="
echo "FuzzHypo 测试脚本"
echo "=============================================="

# 激活 testbed 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed

# 检查 hypothesis 是否安装
echo "检查 hypothesis 安装..."
python -c "import hypothesis; print(f'Hypothesis version: {hypothesis.__version__}')"
if [ $? -ne 0 ]; then
    echo "安装 hypothesis..."
    pip install hypothesis pytest
fi

# 创建临时测试文件
TEST_FILE=$(mktemp /tmp/fuzz_hypo_test_XXXXXX.py)

cat > $TEST_FILE << 'PYTHON_CODE'
#!/usr/bin/env python3
"""快速测试 Hypothesis 是否正常运行多个例子"""

import hypothesis.strategies as st
from hypothesis import given, settings, Verbosity

print("=" * 60)
print("测试 Hypothesis 多例子运行")
print("=" * 60)

run_count = 0

@settings(max_examples=200, verbosity=Verbosity.normal)
@given(st.integers())
def test_run_count(x):
    global run_count
    run_count += 1
    assert isinstance(x, int)

print("\n运行测试 (应该运行 200 个例子)...")
test_run_count()
print(f"\n✅ 测试完成！实际运行了 {run_count} 个例子")

if run_count >= 100:
    print("✅ Hypothesis 正常工作！")
else:
    print(f"❌ 只运行了 {run_count} 个例子，应该至少 100 个")
PYTHON_CODE

# 运行测试
echo ""
python $TEST_FILE

# 清理
rm -f $TEST_FILE

echo ""
echo "=============================================="
echo "测试完成"
echo "=============================================="
