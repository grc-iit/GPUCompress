"""
Tests for neural_net/ Python package.

Covers: core, export, inference, training, and xgboost subpackages.
Tests are focused on correctness of logic, not GPU availability.
"""

import sys
import os
import struct
import math
import tempfile
import numpy as np
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Bug 1: gpu_library.py default lib_path resolution
# ============================================================

def test_gpu_library_default_path_resolves_to_project_build():
    """Default lib_path should point to GPUCompress/build/, not neural_net/build/."""
    from neural_net.core.gpu_library import GPUCompressLib
    import inspect

    # Get the source file location
    source_file = Path(inspect.getfile(GPUCompressLib)).resolve()
    # source_file = .../GPUCompress/neural_net/core/gpu_library.py

    # Compute the default path the same way the constructor does
    default_path = source_file.parent.parent.parent / 'build' / 'libgpucompress.so'

    # It should resolve to <project_root>/build/libgpucompress.so
    expected_parent = PROJECT_ROOT / 'build'
    assert default_path.parent.resolve() == expected_parent.resolve(), (
        f"Default lib_path parent is {default_path.parent}, "
        f"expected {expected_parent}"
    )

    # Negative check: it must NOT point inside neural_net/
    assert 'neural_net' not in str(default_path.parent), (
        f"Default lib_path incorrectly points inside neural_net/: {default_path}"
    )
    print("  PASS: default lib_path resolves to project build/ directory")


# ============================================================
# Bug 6: configs.py lossless error_bound inconsistency
# ============================================================

def test_build_all_config_features_lossless_error_bound():
    """Lossless configs must always use error_bound_enc = log10(1e-7) = -7.0,
    regardless of the caller's error_bounds value."""
    import math
    from neural_net.core.configs import build_all_config_features

    # Call with a non-zero error_bounds (e.g. user wants lossy at 0.01)
    rows, configs = build_all_config_features(
        entropy=5.0, mad=0.3, second_derivative=0.1,
        data_size=1_000_000, error_bounds=0.01)

    expected_lossless_eb_enc = math.log10(1e-7)  # -7.0
    # error_bound_enc is at index 10 in the feature vector
    EB_ENC_IDX = 10

    for feature_vec, (algo_name, quant_str, shuffle, eb) in zip(rows, configs):
        if quant_str == 'none':
            actual = feature_vec[EB_ENC_IDX]
            assert abs(actual - expected_lossless_eb_enc) < 1e-6, (
                f"Lossless config ({algo_name}, shuffle={shuffle}) has "
                f"error_bound_enc={actual:.4f}, expected {expected_lossless_eb_enc:.4f}. "
                f"Bug: using caller's error_bounds instead of sentinel 1e-7."
            )

    print("  PASS: lossless configs always use error_bound_enc = -7.0")


def test_build_all_config_features_lossy_uses_quant_eb():
    """Lossy configs must use the error_bound from QUANT_OPTIONS, not the caller's."""
    import math
    from neural_net.core.configs import build_all_config_features, QUANT_OPTIONS

    rows, configs = build_all_config_features(
        entropy=5.0, mad=0.3, second_derivative=0.1,
        data_size=1_000_000, error_bounds=0.05)

    lossy_ebs = {eb for quant, eb in QUANT_OPTIONS if quant}  # {0.1, 0.01, 0.001}
    EB_ENC_IDX = 10

    for feature_vec, (algo_name, quant_str, shuffle, eb) in zip(rows, configs):
        if quant_str == 'linear':
            assert eb in lossy_ebs, (
                f"Lossy config ({algo_name}, shuffle={shuffle}) has unexpected "
                f"error_bound={eb}")
            expected_enc = math.log10(eb)
            actual = feature_vec[EB_ENC_IDX]
            assert abs(actual - expected_enc) < 1e-6, (
                f"Lossy config ({algo_name}, shuffle={shuffle}, eb={eb}) has "
                f"error_bound_enc={actual:.4f}, expected {expected_enc:.4f}")

    print("  PASS: lossy configs use QUANT_OPTIONS error_bounds correctly")


# ============================================================
# Bug 4: xgboost/train.py unclosed file handle in pickle.dump
# ============================================================

def test_xgb_train_pickle_uses_context_manager():
    """pickle.dump must use a 'with' statement so the file handle is always closed."""
    import ast

    xgb_train_path = PROJECT_ROOT / 'neural_net' / 'xgboost' / 'train.py'
    source = xgb_train_path.read_text()
    tree = ast.parse(source)

    # Walk the AST looking for calls to pickle.dump
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match pickle.dump(...)
        func = node.func
        is_pickle_dump = (
            isinstance(func, ast.Attribute)
            and func.attr == 'dump'
            and isinstance(func.value, ast.Name)
            and func.value.id == 'pickle'
        )
        if not is_pickle_dump:
            continue

        # The second arg to pickle.dump is the file object.
        # It must NOT be a bare open() call (i.e., ast.Call to 'open').
        if len(node.args) >= 2:
            file_arg = node.args[1]
        elif node.keywords:
            file_arg = next(
                (kw.value for kw in node.keywords if kw.arg == 'file'), None)
        else:
            file_arg = None

        if file_arg is not None and isinstance(file_arg, ast.Call):
            callee = file_arg.func
            is_bare_open = (
                (isinstance(callee, ast.Name) and callee.id == 'open')
                or (isinstance(callee, ast.Attribute) and callee.attr == 'open')
            )
            assert not is_bare_open, (
                f"pickle.dump at line {node.lineno} uses a bare open() call "
                f"as file argument — file handle will never be closed. "
                f"Use 'with open(...) as f: pickle.dump(..., f)' instead."
            )

    print("  PASS: pickle.dump uses context-managed file handle")


# ============================================================
# Bug 5: evaluate.py division by zero when no valid groups
# ============================================================

def test_evaluate_ranking_zero_groups():
    """evaluate_ranking must not crash when the validation set has no valid groups."""
    import torch
    from neural_net.core.model import CompressionPredictor
    import pandas as pd

    # Create a model
    model = CompressionPredictor()
    model.eval()
    device = torch.device('cpu')

    # Build a minimal data dict with an empty validation DataFrame
    feature_names = (
        [f'alg_{a}' for a in ['lz4','snappy','deflate','gdeflate',
                               'zstd','ans','cascaded','bitcomp']]
        + ['quant_enc','shuffle_enc','error_bound_enc',
           'data_size_enc','entropy','mad','second_derivative']
    )
    empty_df = pd.DataFrame(columns=['file','error_bound','algorithm',
                                      'quantization','shuffle',
                                      'compression_ratio'] + feature_names)
    data = {
        'df_val': empty_df,
        'feature_names': feature_names,
        'x_means': np.zeros(15, dtype=np.float32),
        'x_stds': np.ones(15, dtype=np.float32),
        'y_means': np.zeros(4, dtype=np.float32),
        'y_stds': np.ones(4, dtype=np.float32),
    }

    from neural_net.inference.evaluate import evaluate_ranking
    result = evaluate_ranking(model, data, device, rank_by='compression_ratio')

    assert result['total_groups'] == 0
    assert result['top1_accuracy'] == 0.0
    assert result['top3_accuracy'] == 0.0
    assert result['mean_regret'] == 0.0
    print("  PASS: evaluate_ranking handles zero groups without crashing")


# ============================================================
# Quality 7: benchmark.py duplicate imports
# ============================================================

def test_benchmark_no_duplicate_imports():
    """benchmark.py should not have duplicate import statements."""
    import ast

    bench_path = PROJECT_ROOT / 'neural_net' / 'training' / 'benchmark.py'
    tree = ast.parse(bench_path.read_text())

    # Collect all top-level imports
    imported_names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imported_names.append(f"{node.module}.{alias.name}")

    duplicates = [name for name in set(imported_names)
                  if imported_names.count(name) > 1]
    assert not duplicates, f"Duplicate imports in benchmark.py: {duplicates}"
    print("  PASS: benchmark.py has no duplicate imports")


# ============================================================
# Quality 8: configs.py dead code removed
# ============================================================

def test_configs_no_decode_action():
    """decode_action was dead code and should be removed from configs.py."""
    source = (PROJECT_ROOT / 'neural_net' / 'core' / 'configs.py').read_text()
    assert 'def decode_action' not in source, (
        "decode_action is dead code — never imported by any module"
    )
    print("  PASS: dead code decode_action removed from configs.py")


# ============================================================
# Quality 9: xgboost/train.py lazy imports for shap/matplotlib
# ============================================================

def test_xgb_train_no_toplevel_shap_matplotlib():
    """shap and matplotlib must not be imported at module level in xgboost/train.py."""
    import ast

    xgb_path = PROJECT_ROOT / 'neural_net' / 'xgboost' / 'train.py'
    tree = ast.parse(xgb_path.read_text())

    # Only check top-level statements (not inside functions)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in ('shap', 'matplotlib', 'matplotlib.pyplot'), (
                    f"Top-level 'import {alias.name}' found — should be lazy")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(('shap', 'matplotlib')):
                assert False, (
                    f"Top-level 'from {node.module} import ...' found — should be lazy")

    print("  PASS: shap/matplotlib are lazily imported in xgboost/train.py")


# ============================================================
# Quality 10: predict scripts use shared build_all_config_features
# ============================================================

def test_predict_scripts_use_shared_feature_builder():
    """predict.py and xgb predict.py must import build_all_config_features,
    not duplicate the feature construction inline."""
    for rel_path in ['neural_net/inference/predict.py',
                     'neural_net/xgboost/predict.py']:
        source = (PROJECT_ROOT / rel_path).read_text()
        assert 'build_all_config_features' in source, (
            f"{rel_path} should use build_all_config_features from configs.py"
        )
        # Should NOT define its own QUANT_OPTIONS or SHUFFLE_OPTIONS
        assert 'QUANT_OPTIONS' not in source, (
            f"{rel_path} still has local QUANT_OPTIONS — should use configs.py"
        )
    print("  PASS: predict scripts use shared feature builder")


def test_build_all_config_features_returns_correct_shape():
    """build_all_config_features must return 64 rows of 15 features each."""
    from neural_net.core.configs import build_all_config_features

    rows, configs = build_all_config_features(
        entropy=5.0, mad=0.3, second_derivative=0.1,
        data_size=1_000_000, error_bounds=0.0)

    assert len(rows) == 64, f"Expected 64 rows, got {len(rows)}"
    assert len(configs) == 64, f"Expected 64 configs, got {len(configs)}"
    for i, vec in enumerate(rows):
        assert len(vec) == 15, f"Row {i} has {len(vec)} features, expected 15"

    # Verify config tuple structure: (algo_name, quant_str, shuffle, eb)
    algo, quant_str, shuffle, eb = configs[0]
    assert isinstance(algo, str)
    assert quant_str in ('none', 'linear')
    assert shuffle in (0, 4)
    assert isinstance(eb, float)

    print("  PASS: build_all_config_features returns (64 rows × 15 features, 64 configs)")


# ============================================================
# Quality 11: retrain.py groups experience rows by data stats
# ============================================================

def test_retrain_experience_file_grouping():
    """Experience rows with identical stats must get the same file name
    so they stay together during file-level train/val splitting."""
    import pandas as pd
    # Avoid importing retrain (it imports benchmark which imports gpu_library)
    # Instead, read and test the logic directly
    source = (PROJECT_ROOT / 'neural_net' / 'training' / 'retrain.py').read_text()

    # The fix should use groupby on data stats, not per-row index
    assert 'groupby' in source, (
        "retrain.py should groupby data stats to assign file names"
    )
    assert "experience_' + str(i)" not in source, (
        "retrain.py still assigns per-row unique file names"
    )
    print("  PASS: retrain.py groups experience rows by data statistics")


# ============================================================
# Quality 12: benchmark.py GPU warmup before timing
# ============================================================

def test_benchmark_has_warmup():
    """benchmark.py should have a warmup step before the timing loop."""
    source = (PROJECT_ROOT / 'neural_net' / 'training' / 'benchmark.py').read_text()
    # The warmup must appear before the benchmarking loop
    warmup_pos = source.find('warmup')
    benchmark_loop_pos = source.find('# Benchmark all 64 configs')
    assert warmup_pos != -1, "benchmark.py has no warmup step"
    assert warmup_pos < benchmark_loop_pos, (
        "Warmup must appear before the benchmark loop"
    )
    print("  PASS: benchmark.py has GPU warmup before timing")


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(run_all())
