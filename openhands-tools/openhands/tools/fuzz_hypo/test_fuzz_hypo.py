#!/usr/bin/env python3
"""
本地测试脚本：验证 FuzzHypo 工具的导入逻辑

使用方法 (在 Docker 容器中运行):
    docker exec -it <container_id> bash
    cd /workspace/astropy
    python /agent-server/.venv/lib/python3.12/site-packages/openhands/tools/fuzz_hypo/test_fuzz_hypo.py

或者本地测试 (需要有 astropy 包):
    cd /path/to/astropy
    python test_fuzz_hypo.py

这个脚本模拟 FuzzHypo 生成的 test_harness.py 的导入过程
"""

import sys
import os
from pathlib import Path


def reset_modules(package_prefix: str):
    """重置指定前缀的模块，以便进行干净测试"""
    to_delete = [mod for mod in sys.modules.keys() if mod.startswith(package_prefix)]
    for mod in to_delete:
        del sys.modules[mod]


def test_direct_import(working_dir: str, module_path: str, func_name: str) -> bool:
    """测试：直接导入模块（通常会失败）"""
    print(f"\n=== 测试: 直接模块导入（无顶层包初始化） ===")
    print(f"  module_path: {module_path}")
    
    try:
        import importlib
        
        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
        
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        print(f"  ✅ 成功: {func}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        return False


def test_top_package_first(working_dir: str, module_path: str, func_name: str) -> bool:
    """测试：先导入顶层包，再导入子模块（推荐方法）"""
    print(f"\n=== 测试: 先初始化顶层包（推荐） ===")
    print(f"  module_path: {module_path}")
    
    try:
        import importlib
        
        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
        
        top_package = module_path.split(".")[0]
        
        print(f"  步骤1: 导入顶层包 '{top_package}'")
        top_mod = importlib.import_module(top_package)
        print(f"  步骤1 完成: {top_mod}")
        
        print(f"  步骤2: 导入目标模块 '{module_path}'")
        module = importlib.import_module(module_path)
        
        func = getattr(module, func_name)
        print(f"  ✅ 成功: {func}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_path_import(working_dir: str, file_path: str, func_name: str) -> bool:
    """测试：从文件路径导入"""
    print(f"\n=== 测试: 文件路径导入 ===")
    print(f"  file_path: {file_path}")
    
    try:
        import importlib
        
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = Path(working_dir) / file_path
        
        # 检查是否属于包
        is_in_package = _check_in_package(file_path)
        print(f"  属于包: {is_in_package}")
        
        if is_in_package:
            # 转换为模块路径
            rel_path = file_path.relative_to(working_dir)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
            print(f"  转换后模块路径: {module_path}")
            
            if working_dir not in sys.path:
                sys.path.insert(0, str(working_dir))
            
            top_package = module_path.split(".")[0]
            print(f"  步骤1: 导入顶层包 '{top_package}'")
            importlib.import_module(top_package)
            
            print(f"  步骤2: 导入目标模块")
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
        else:
            # 独立文件
            print(f"  使用 spec_from_file_location")
            import importlib.util
            
            file_dir = str(file_path.parent)
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)
            
            spec = importlib.util.spec_from_file_location("_target", str(file_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, func_name)
        
        print(f"  ✅ 成功: {func}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _check_in_package(file_path: Path) -> bool:
    """检查文件是否属于一个包"""
    current = file_path.parent
    while current != current.parent:
        if (current / "__init__.py").exists():
            return True
        current = current.parent
    return False


def run_all_tests():
    """运行所有测试"""
    # 自动检测工作目录
    cwd = os.getcwd()
    
    # 检查是否在 astropy 目录中
    if (Path(cwd) / "astropy" / "__init__.py").exists():
        working_dir = cwd
        print(f"检测到 astropy 包目录: {working_dir}")
    elif Path("/workspace/astropy").exists():
        working_dir = "/workspace/astropy"
        print(f"使用默认路径: {working_dir}")
    else:
        print("❌ 找不到 astropy 包。请在 astropy 目录中运行此脚本。")
        print("   或者在 Docker 容器中运行。")
        return
    
    print("=" * 70)
    print("FuzzHypo 导入逻辑测试")
    print("=" * 70)
    
    results = []
    
    # 测试1: 模块路径格式
    print(f"\n{'='*70}")
    print("测试组1: 模块路径格式 (astropy.units.core:_condition_arg)")
    print("=" * 70)
    
    reset_modules("astropy")
    r1 = test_direct_import(working_dir, "astropy.units.core", "_condition_arg")
    results.append(("直接导入", r1))
    
    reset_modules("astropy")
    r2 = test_top_package_first(working_dir, "astropy.units.core", "_condition_arg")
    results.append(("顶层包优先", r2))
    
    # 测试2: 文件路径格式
    print(f"\n{'='*70}")
    print("测试组2: 文件路径格式 (/workspace/astropy/astropy/units/core.py)")
    print("=" * 70)
    
    reset_modules("astropy")
    r3 = test_file_path_import(
        working_dir, 
        f"{working_dir}/astropy/units/core.py", 
        "_condition_arg"
    )
    results.append(("文件路径导入", r3))
    
    # 测试3: 另一个模块
    print(f"\n{'='*70}")
    print("测试组3: quantity_helper 模块")
    print("=" * 70)
    
    reset_modules("astropy")
    r4 = test_top_package_first(
        working_dir, 
        "astropy.units.quantity_helper.converters", 
        "can_have_arbitrary_unit"
    )
    results.append(("quantity_helper", r4))
    
    # 汇总
    print(f"\n{'='*70}")
    print("测试汇总")
    print("=" * 70)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n通过: {passed}/{len(results)}")


if __name__ == "__main__":
    run_all_tests()
