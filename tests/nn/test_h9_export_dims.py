#!/usr/bin/env python3
"""
test_h9_export_dims.py

H-9 / C-2: export_weights.py uses hardcoded defaults in CompressionPredictor()
and verify_export() asserts hardcoded dims (15, 128, 4).

Tests:
  1. Default dims export + verify round-trip works correctly.
  2. Non-default hidden_dim model → load_state_dict raises RuntimeError
     (C-2 defense: prevents corrupt export).
  3. Manually crafted .nnwt with wrong dims → verify_export rejects it.
  4. Binary header dims match what was written.

Usage: python tests/nn/test_h9_export_dims.py
"""

import os
import sys
import struct
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from neural_net.core.model import CompressionPredictor

MAGIC = 0x4E4E5754
VERSION = 2

pass_count = 0
fail_count = 0

def PASS(msg):
    global pass_count
    print(f"  PASS: {msg}")
    pass_count += 1

def FAIL(msg):
    global fail_count
    print(f"  FAIL: {msg}")
    fail_count += 1


def make_fake_checkpoint(input_dim=15, hidden_dim=128, output_dim=4):
    """Create a minimal checkpoint dict with random weights."""
    model = CompressionPredictor(input_dim, hidden_dim, output_dim)
    return {
        'model_state_dict': model.state_dict(),
        'x_means': np.random.randn(input_dim).astype(np.float32),
        'x_stds': np.abs(np.random.randn(input_dim)).astype(np.float32) + 0.1,
        'y_means': np.random.randn(output_dim).astype(np.float32),
        'y_stds': np.abs(np.random.randn(output_dim)).astype(np.float32) + 0.1,
        'x_mins': np.random.randn(input_dim).astype(np.float32) - 5.0,
        'x_maxs': np.random.randn(input_dim).astype(np.float32) + 5.0,
    }


def test_default_dims_export():
    """Test 1: Export with default dims (15, 128, 4) succeeds."""
    print("\n--- Test 1: Default dims export ---")

    checkpoint = make_fake_checkpoint(15, 128, 4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        pt_path = f.name
        torch.save(checkpoint, pt_path)

    nnwt_path = pt_path.replace('.pt', '.nnwt')
    try:
        from neural_net.export.export_weights import export_weights
        export_weights(pt_path, nnwt_path)

        # Verify header
        with open(nnwt_path, 'rb') as f:
            magic, version, n_layers, in_dim, hid_dim, out_dim = struct.unpack('<6I', f.read(24))

        if magic == MAGIC and in_dim == 15 and hid_dim == 128 and out_dim == 4:
            PASS("default dims export + verify succeeded")
        else:
            FAIL(f"header mismatch: magic={magic:#x} dims=({in_dim},{hid_dim},{out_dim})")
    except Exception as e:
        FAIL(f"export raised: {e}")
    finally:
        os.unlink(pt_path)
        if os.path.exists(nnwt_path):
            os.unlink(nnwt_path)


def test_nondefault_hidden_dim_rejected():
    """Test 2: Non-default hidden_dim → load_state_dict throws."""
    print("\n--- Test 2: Non-default hidden_dim rejected by load_state_dict ---")

    # Train with hidden_dim=256
    checkpoint_256 = make_fake_checkpoint(15, 256, 4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        pt_path = f.name
        torch.save(checkpoint_256, pt_path)

    try:
        # export_weights() creates CompressionPredictor() with defaults (128)
        # then calls load_state_dict with 256-dim weights → should throw
        from neural_net.export.export_weights import export_weights
        nnwt_path = pt_path.replace('.pt', '.nnwt')
        try:
            export_weights(pt_path, nnwt_path)
            FAIL("export should have raised RuntimeError for hidden_dim=256")
            if os.path.exists(nnwt_path):
                os.unlink(nnwt_path)
        except RuntimeError as e:
            if "size mismatch" in str(e).lower() or "shape" in str(e).lower():
                PASS(f"load_state_dict correctly rejected: {e}")
            else:
                PASS(f"load_state_dict raised RuntimeError: {e}")
        except Exception as e:
            FAIL(f"unexpected exception type: {type(e).__name__}: {e}")
    finally:
        os.unlink(pt_path)


def test_verify_rejects_wrong_dims():
    """Test 3: verify_export rejects .nnwt with wrong header dims."""
    print("\n--- Test 3: verify_export rejects wrong dims in header ---")

    from neural_net.export.export_weights import verify_export

    model = CompressionPredictor(15, 128, 4)
    x_means = np.zeros(15, dtype=np.float32)
    x_stds = np.ones(15, dtype=np.float32)
    y_means = np.zeros(4, dtype=np.float32)
    y_stds = np.ones(4, dtype=np.float32)
    x_mins = np.full(15, -1.0, dtype=np.float32)
    x_maxs = np.full(15, 1.0, dtype=np.float32)

    # Write a .nnwt with wrong hidden_dim in header
    with tempfile.NamedTemporaryFile(suffix='.nnwt', delete=False) as f:
        nnwt_path = f.name
        # Header with hidden_dim=256 (wrong)
        f.write(struct.pack('<6I', MAGIC, VERSION, 3, 15, 256, 4))
        # Write enough dummy data so it doesn't hit EOF first
        f.write(b'\x00' * 200000)

    try:
        try:
            verify_export(nnwt_path, model, x_means, x_stds, y_means, y_stds,
                         x_mins, x_maxs)
            FAIL("verify_export should have asserted on hid_dim=256")
        except AssertionError:
            PASS("verify_export correctly rejected hid_dim=256 in header")
        except Exception as e:
            if isinstance(e, AssertionError) or "assert" in type(e).__name__.lower():
                PASS(f"verify_export correctly rejected: {e}")
            else:
                FAIL(f"unexpected exception: {type(e).__name__}: {e}")
    finally:
        os.unlink(nnwt_path)


def test_header_dims_match_written():
    """Test 4: Binary header dims match the model that was exported."""
    print("\n--- Test 4: Header dims match model ---")

    checkpoint = make_fake_checkpoint(15, 128, 4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        pt_path = f.name
        torch.save(checkpoint, pt_path)

    nnwt_path = pt_path.replace('.pt', '.nnwt')
    try:
        from neural_net.export.export_weights import export_weights
        export_weights(pt_path, nnwt_path)

        with open(nnwt_path, 'rb') as f:
            magic, version, n_layers, in_dim, hid_dim, out_dim = struct.unpack('<6I', f.read(24))

        # Verify all header fields
        ok = True
        if magic != MAGIC:
            FAIL(f"magic={magic:#x} expected={MAGIC:#x}"); ok = False
        if version != VERSION:
            FAIL(f"version={version} expected={VERSION}"); ok = False
        if n_layers != 3:
            FAIL(f"n_layers={n_layers} expected=3"); ok = False
        if in_dim != 15:
            FAIL(f"in_dim={in_dim} expected=15"); ok = False
        if hid_dim != 128:
            FAIL(f"hid_dim={hid_dim} expected=128"); ok = False
        if out_dim != 4:
            FAIL(f"out_dim={out_dim} expected=4"); ok = False

        if ok:
            PASS("all header fields correct")

        # Verify file size matches expected
        file_size = os.path.getsize(nnwt_path)
        # Header(24) + x_means(60) + x_stds(60) + y_means(16) + y_stds(16)
        # + Layer1(15*128*4 + 128*4) + Layer2(128*128*4 + 128*4) + Layer3(128*4*4 + 4*4)
        # + x_mins(60) + x_maxs(60)
        expected = (24 + 60 + 60 + 16 + 16
                   + 15*128*4 + 128*4
                   + 128*128*4 + 128*4
                   + 128*4*4 + 4*4
                   + 60 + 60)
        if file_size == expected:
            PASS(f"file size correct ({file_size} bytes)")
        else:
            FAIL(f"file size={file_size} expected={expected}")

    except Exception as e:
        FAIL(f"export raised: {e}")
    finally:
        os.unlink(pt_path)
        if os.path.exists(nnwt_path):
            os.unlink(nnwt_path)


if __name__ == '__main__':
    print("=== H-9 / C-2: Export dimension handling tests ===")

    test_default_dims_export()
    test_nondefault_hidden_dim_rejected()
    test_verify_rejects_wrong_dims()
    test_header_dims_match_written()

    print(f"\n=== Summary: {pass_count} pass, {fail_count} fail ===")
    print("OVERALL: PASS" if fail_count == 0 else "OVERALL: FAIL")
    sys.exit(0 if fail_count == 0 else 1)
