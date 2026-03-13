"""
Generate synthetic float32 data with varying characteristics and predict
compression performance for all 64 configs using the trained NN (CPU only).
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_net.core.data import compute_stats_cpu, inverse_transform_outputs
from neural_net.core.configs import build_all_config_features, ALGORITHM_NAMES
from neural_net.core.model import CompressionPredictor


def generate_synthetic_datasets(n_elements=65536):
    """Generate diverse synthetic float32 arrays (~256 KB each)."""
    rng = np.random.RandomState(42)
    datasets = {}

    # 1. Uniform random — high entropy, low compressibility
    datasets['uniform_random'] = rng.uniform(-1.0, 1.0, n_elements).astype(np.float32)

    # 2. Smooth sine wave — low entropy, high compressibility
    t = np.linspace(0, 8 * np.pi, n_elements, dtype=np.float32)
    datasets['smooth_sine'] = np.sin(t)

    # 3. Constant — trivially compressible
    datasets['constant'] = np.full(n_elements, 3.14159, dtype=np.float32)

    # 4. Gaussian noise — moderate entropy
    datasets['gaussian_noise'] = rng.normal(0.0, 1.0, n_elements).astype(np.float32)

    # 5. Step function — blocky, good for delta/cascaded
    steps = np.repeat(rng.uniform(-10, 10, 64), n_elements // 64).astype(np.float32)
    datasets['step_function'] = steps

    # 6. Linear ramp — perfectly smooth 2nd derivative ~0
    datasets['linear_ramp'] = np.linspace(-100.0, 100.0, n_elements, dtype=np.float32)

    # 7. Spiky (mostly zero with sparse peaks) — high sparsity
    spiky = np.zeros(n_elements, dtype=np.float32)
    spike_idx = rng.choice(n_elements, size=n_elements // 100, replace=False)
    spiky[spike_idx] = rng.normal(0, 50, len(spike_idx)).astype(np.float32)
    datasets['sparse_spiky'] = spiky

    # 8. Turbulence-like (sum of sines at different freqs)
    t = np.linspace(0, 1, n_elements, dtype=np.float32)
    turb = np.zeros(n_elements, dtype=np.float32)
    for freq in [1, 3, 7, 15, 31, 63]:
        turb += rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2*np.pi)).astype(np.float32)
    datasets['turbulence'] = turb

    return datasets


def predict_for_data(name, raw_bytes, model, checkpoint, top_n=5):
    """Run NN prediction on raw bytes, print top configs per metric."""
    original_size = len(raw_bytes)
    entropy, mad, second_deriv = compute_stats_cpu(raw_bytes)

    x_means = checkpoint['x_means']
    x_stds = checkpoint['x_stds']
    y_means = checkpoint['y_means']
    y_stds = checkpoint['y_stds']

    rows, configs = build_all_config_features(entropy, mad, second_deriv, original_size, error_bounds=0.0)
    X_raw = np.array(rows, dtype=np.float32)
    X_norm = (X_raw - x_means) / x_stds

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_norm)).numpy()

    pred = inverse_transform_outputs(pred_norm, y_means, y_stds)

    print(f"\n{'='*100}")
    print(f"  Dataset: {name}  |  Size: {original_size:,} bytes")
    print(f"  Stats: entropy={entropy:.4f} bits  MAD={mad:.6f}  2nd_deriv={second_deriv:.6f}")
    print(f"{'='*100}")

    # Full table ranked by compression ratio
    print(f"\n  All 64 configs ranked by compression ratio (higher is better):\n")
    print(f"  {'Rank':>4}  {'Algorithm':<12} {'Quant':<8} {'Shuf':>4}  {'ErrBound':>8}  "
          f"{'Ratio':>8}  {'PSNR dB':>8}  {'Comp ms':>8}  {'Decomp ms':>9}")
    print(f"  {'-'*95}")

    ratio_rank = np.argsort(-pred['compression_ratio'])
    for rank, idx in enumerate(ratio_rank):
        algo, quant, shuffle, eb = configs[idx]
        marker = " <-- best" if rank == 0 else ""
        print(f"  {rank+1:>4}  {algo:<12} {quant:<8} {shuffle:>4}  {eb:>8.4f}  "
              f"{pred['compression_ratio'][idx]:>8.2f}  "
              f"{pred['psnr_db'][idx]:>8.1f}  "
              f"{pred['compression_time_ms'][idx]:>8.3f}  "
              f"{pred['decompression_time_ms'][idx]:>9.3f}{marker}")

    # Summary: best config per metric
    metrics = [
        ('compression_ratio', True),
        ('psnr_db', True),
        ('compression_time_ms', False),
        ('decompression_time_ms', False),
    ]
    print(f"\n  Best config per metric:")
    print(f"  {'Metric':<25} {'Best Value':>12}  {'Algorithm':<12} {'Quant':<8} {'Shuf':>4}")
    print(f"  {'-'*70}")
    for metric, higher_better in metrics:
        vals = pred[metric]
        best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
        algo, quant, shuffle, eb = configs[best_idx]
        fmt = f"{vals[best_idx]:>12.3f}" if 'time' in metric else f"{vals[best_idx]:>12.2f}"
        print(f"  {metric:<25} {fmt}  {algo:<12} {quant:<8} {shuffle:>4}")


def main():
    weights_path = Path(__file__).parent.parent / 'neural_net' / 'weights' / 'model.pt'
    if not weights_path.exists():
        print(f"No trained model at {weights_path}. Run train.py first.")
        sys.exit(1)

    device = torch.device('cpu')
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = CompressionPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Loaded model:", f"{checkpoint['input_dim']}→{checkpoint['hidden_dim']}→{checkpoint['output_dim']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    datasets = generate_synthetic_datasets()

    for name, arr in datasets.items():
        raw_bytes = arr.tobytes()
        predict_for_data(name, raw_bytes, model, checkpoint)

    print(f"\n{'='*100}")
    print(f"  Done — predicted all 4 metrics × 64 configs × {len(datasets)} synthetic datasets")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
