#!/usr/bin/env python3
"""
Generate preset combinations for synthetic HDF5 datasets.

Goal: Create presets with VARIETY in entropy levels (1-8 bits) across
      different distributions for comprehensive compression benchmarking.

Note: Entropy targets are goals; actual achieved entropy may vary slightly
      for skewed distributions at high entropy levels. This is acceptable
      as long as we achieve good variety across the spectrum.
"""

import yaml
import itertools


def generate_preset_combinations(
    output_file='config/presets_small.yaml',
    entropy_steps=8,
    distributions=None,
    entropy_min=1.0,
    entropy_max=8.0
):
    """
    Generate all combinations of distributions and entropy levels.

    Args:
        output_file: Output YAML filename
        entropy_steps: Number of entropy levels (default: 8)
        distributions: List of distributions (default: all except constant)
        entropy_min: Minimum entropy in bits (default: 1.0)
        entropy_max: Maximum entropy in bits (default: 8.0)

    Total combinations = distributions × entropy_steps
    """

    # Default: exclude 'constant' since it can't achieve entropy > 0
    if distributions is None:
        distributions = ['uniform', 'normal', 'exponential', 'bimodal']

    # Entropy: entropy_min to entropy_max bits
    entropy_range = entropy_max - entropy_min
    entropy_levels = [round(entropy_min + i * (entropy_range / (entropy_steps - 1)), 8)
                      for i in range(entropy_steps)]

    # Default shape
    default_shape = [1000, 1000]  # 4MB

    # Generate all combinations
    combinations = itertools.product(distributions, entropy_levels)

    presets = {}
    combo_id = 0

    total_expected = len(distributions) * len(entropy_levels)

    print("=" * 70)
    print("Generating preset combinations...")
    print("=" * 70)
    print(f"  Distributions:     {len(distributions)}")
    print(f"  Distribution list: {distributions}")
    print(f"  Entropy levels:    {len(entropy_levels)}")
    print(f"  Entropy range:     {min(entropy_levels):.8f} - {max(entropy_levels):.8f} bits")
    print(f"  Expected total:    {total_expected:,} combinations")
    print()

    for dist, entropy in combinations:
        combo_id += 1

        # Create preset name
        preset_name = f"combo_{combo_id:05d}"

        # Create description
        desc = f"{dist}, E={entropy:.8f}"

        # Create preset config
        presets[preset_name] = {
            'distribution': dist,
            'entropy_target': float(entropy),
            'shape': default_shape.copy(),
            'seed': combo_id,
            'description': desc
        }

        if combo_id % 100 == 0:
            print(f"  Generated {combo_id} combinations...")

    print(f"\n✓ Generated {combo_id} total combinations")

    # Create YAML structure (no defaults needed)
    yaml_data = {
        'presets': presets
    }

    # Write to file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, width=1000)

    print(f"✓ Saved {combo_id} presets to {output_file}")
    print(f"✓ File size: {len(open(output_file).read()) / 1024:.1f} KB")

    return combo_id


def print_usage():
    """Print usage information."""
    print("=" * 70)
    print("Preset Generator - Updated Version")
    print("=" * 70)
    print()
    print("This script generates presets with only distribution and entropy.")
    print("Correlation and smoothness have been removed from the pipeline.")
    print()
    print("Usage:")
    print("  python preset_generator.py [mode]")
    print()
    print("Modes:")
    print("  small   - 32 presets  (4 distributions × 8 entropy levels, 1-8 bits)")
    print("  medium  - 64 presets  (4 distributions × 16 entropy levels, 1-8 bits)")
    print("  large   - 128 presets (4 distributions × 32 entropy levels, 1-8 bits)")
    print("  custom=N[,MIN,MAX][,file.yaml] - Custom configuration")
    print()
    print("Custom Mode Examples:")
    print("  custom=100                    - 400 presets, 1-8 bits (default)")
    print("  custom=100,1,16               - 400 presets, 1-16 bits")
    print("  custom=100,0.5,20             - 400 presets, 0.5-20 bits")
    print("  custom=50,1,12,my_file.yaml   - 200 presets, 1-12 bits, custom file")
    print("  custom=200,5,15               - 800 presets, 5-15 bits range")
    print()
    print("Distributions used:")
    print("  uniform, normal, exponential, bimodal")
    print("  (constant excluded - can't achieve entropy > 0)")
    print()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print_usage()
        mode = input("Select mode [small]: ").strip() or 'small'

    print()

    if mode == 'small':
        output_file = 'config/presets_small.yaml'
        count = generate_preset_combinations(
            output_file,
            entropy_steps=8
        )
    elif mode == 'medium':
        output_file = 'config/presets_medium.yaml'
        count = generate_preset_combinations(
            output_file,
            entropy_steps=16
        )
    elif mode == 'large':
        output_file = 'config/presets_large.yaml'
        count = generate_preset_combinations(
            output_file,
            entropy_steps=32
        )
    elif mode.startswith('custom='):
        # Custom mode: custom=100 or custom=100,1,16 or custom=100,1,16,output.yaml
        parts = mode.split('=', 1)[1].split(',')
        entropy_steps = int(parts[0])
        entropy_min = float(parts[1]) if len(parts) > 1 and parts[1].replace('.','').isdigit() else 1.0
        entropy_max = float(parts[2]) if len(parts) > 2 and parts[2].replace('.','').isdigit() else 8.0

        # Output file is last part if it's not a number
        if len(parts) > 1:
            last_part = parts[-1]
            if not last_part.replace('.','').replace('-','').isdigit():
                output_file = last_part
            else:
                output_file = f'config/presets_custom_{entropy_steps}_{entropy_min}-{entropy_max}.yaml'
        else:
            output_file = f'config/presets_custom_{entropy_steps}.yaml'

        print(f"Custom mode: {entropy_steps} entropy levels")
        print(f"Entropy range: {entropy_min} - {entropy_max} bits")
        print(f"Output: {output_file}")
        print()

        count = generate_preset_combinations(
            output_file,
            entropy_steps=entropy_steps,
            entropy_min=entropy_min,
            entropy_max=entropy_max
        )
    else:
        print(f"Unknown mode: {mode}")
        print("Use: small, medium, large, or custom=N[,MIN,MAX][,output.yaml]")
        print("Examples:")
        print("  custom=100                  (default 1-8 bits)")
        print("  custom=100,1,16             (1-16 bits range)")
        print("  custom=50,0.5,20,file.yaml  (custom range and file)")
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"✓ Complete! Generated {count} preset combinations")
    print("=" * 70)
    print()
    print("To use:")
    print(f"  python generator.py batch -c {output_file} -o datasets/")
    print()
    print("Or generate specific presets:")
    print("  python generator.py generate --preset combo_00001 \\")
    print(f"    --config {output_file} -o test.h5")
    print()
