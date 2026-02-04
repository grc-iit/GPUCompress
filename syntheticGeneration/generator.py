#!/usr/bin/env python3
"""
Synthetic HDF5 Dataset Generator

Command-line tool for generating synthetic HDF5 datasets with variety in
entropy levels (1-8 bits) for compression benchmarking.

Goal: Generate datasets with different entropy characteristics to test
      compressor performance across the full entropy spectrum.

Usage:
    python generator.py generate --preset combo_00001 --output test.h5
    python generator.py generate --distribution normal --entropy 5.0 --shape 500 500 --output test.h5
    python generator.py batch --config config/presets_small.yaml --output-dir datasets/
"""

import click
import numpy as np
import yaml
import os
import sys
from typing import Union, Tuple

# Import attribute generators
from attributes import distributions, entropy
from utils import hdf5_writer, binary_writer, metrics


def parse_size_to_bytes(size_str: str) -> int:
    """
    Parse size string like '4MB', '64MB', '1GB' to bytes.

    Args:
        size_str: Size string (e.g., '4MB', '1GB', '1024' for bytes)

    Returns:
        Size in bytes

    Examples:
        >>> parse_size_to_bytes('4MB')
        4194304
        >>> parse_size_to_bytes('1GB')
        1073741824
    """
    size_str = size_str.strip().upper()

    units = {
        'TB': 1024**4,
        'GB': 1024**3,
        'MB': 1024**2,
        'KB': 1024,
        'B': 1
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)])
            return int(number * multiplier)

    # If no unit, assume bytes
    return int(size_str)


def calculate_square_shape(total_bytes: int, dtype_bytes: int = 4, dimensions: int = 2) -> tuple:
    """
    Calculate square array dimensions from total byte size.

    Args:
        total_bytes: Total size in bytes
        dtype_bytes: Bytes per element (default: 4 for float32)
        dimensions: Number of dimensions (default: 2 for 2D array)

    Returns:
        Tuple of dimensions for square array

    Examples:
        >>> calculate_square_shape(4194304, dtype_bytes=4, dimensions=2)
        (1024, 1024)
        >>> calculate_square_shape(67108864, dtype_bytes=4, dimensions=2)
        (4096, 4096)
    """
    total_elements = total_bytes // dtype_bytes
    side_length = int(total_elements ** (1.0 / dimensions))

    if dimensions == 1:
        return (total_elements,)
    elif dimensions == 2:
        return (side_length, side_length)
    else:
        return tuple([side_length] * dimensions)


def generate_dataset(
    distribution: str = 'normal',
    shape: Union[int, Tuple[int, ...]] = (100, 100),
    entropy_target: float = 4.0,
    seed: int = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate synthetic dataset with specified attributes.

    Applies transformations in order:
    1. Generate base distribution
    2. Adjust entropy to target value

    Args:
        distribution: Distribution type ('uniform', 'normal', 'exponential', 'bimodal', 'constant')
        shape: Shape of output array (int for 1D or tuple for multi-D)
        entropy_target: Target Shannon entropy in bits
        seed: Random seed for reproducibility
        verbose: Print progress messages

    Returns:
        Generated dataset as float32 numpy array
    """
    if verbose:
        print(f"\nGenerating synthetic dataset...")
        print(f"  Distribution:  {distribution}")
        print(f"  Shape:         {shape}")
        print(f"  Entropy:       {entropy_target:.8f} bits")
        print(f"  Seed:          {seed}")

    # Step 1: Generate base distribution
    if verbose:
        print("\n[1/2] Generating base distribution...")

    data = distributions.generate(
        distribution=distribution,
        shape=shape,
        seed=seed
    )

    # Step 2: Adjust entropy to target value
    if verbose:
        print(f"[2/2] Adjusting entropy to {entropy_target} bits...")
    entropy.adjust_entropy(data, entropy_target, seed=seed)

    if verbose:
        print("✓ Dataset generation complete!")

    return data


def load_preset(preset_name: str, config_file: str = 'config/presets.yaml') -> dict:
    """
    Load preset configuration from YAML file.

    Args:
        preset_name: Name of the preset to load
        config_file: Path to YAML configuration file

    Returns:
        Dictionary of preset parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If preset name not found
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'presets' not in config or preset_name not in config['presets']:
        available = list(config.get('presets', {}).keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")

    preset = config['presets'][preset_name].copy()

    # Apply defaults for missing values
    if 'defaults' in config:
        for key, value in config['defaults'].items():
            if key not in preset:
                preset[key] = value

    return preset


@click.group()
def cli():
    """Synthetic HDF5 Dataset Generator"""
    pass


@cli.command()
@click.option('--preset', '-p', help='Preset name from config file')
@click.option('--config', type=str, default='config/presets.yaml',
              help='Config file to load preset from (default: config/presets.yaml)')
@click.option('--distribution', '-d', default=None,
              help='Distribution type (uniform/normal/exponential/bimodal/constant)')
@click.option('--shape', '-s', multiple=True, type=int,
              help='Shape dimensions (e.g., -s 100 -s 100 for 100x100)')
@click.option('--dataset-size', type=str,
              help='Dataset size in bytes (e.g., 4MB, 1GB). Overrides --shape if provided.')
@click.option('--entropy', '-e', 'entropy_target', type=float,
              help='Target entropy in bits')
@click.option('--seed', type=int, help='Random seed')
@click.option('--output', '-o', required=True, help='Output HDF5 filename')
@click.option('--compression', type=click.Choice(['gzip', 'lzf', 'none']),
              default='none', help='HDF5 compression (default: none)')
@click.option('--chunk-size', '-c', multiple=True, type=int,
              help='HDF5 chunk dimensions (e.g., -c 4096 -c 4096)')
@click.option('--chunk-bytes', type=str,
              help='Chunk size in bytes (e.g., 64MB, 128MB). Overrides --chunk-size if provided.')
@click.option('--validate/--no-validate', default=True,
              help='Validate generated data against targets')
@click.option('--binary/--no-binary', default=True,
              help='Also output raw binary file for GPUCompress (default: enabled)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def generate(preset, config, distribution, shape, dataset_size, entropy_target,
            seed, output, compression, chunk_size, chunk_bytes, validate, binary, quiet):
    """
    Generate a single synthetic dataset.

    Examples:
        # Using preset
        python generator.py generate --preset high_entropy_uniform -o test.h5

        # Using dimensions
        python generator.py generate -d normal -s 500 -s 500 -e 5.0 -o test.h5

        # Using byte sizes (NEW)
        python generator.py generate -d normal --dataset-size 4MB -e 5.0 -o test.h5
        python generator.py generate --dataset-size 1GB --chunk-bytes 64MB -o test.h5
    """
    verbose = not quiet

    try:
        # Load parameters from preset or command line
        if preset:
            if verbose:
                print(f"Loading preset: {preset}")
                if config != 'config/presets.yaml':
                    print(f"  From: {config}")
            params = load_preset(preset, config)
        else:
            params = {}

        # Override with command-line arguments
        if distribution:
            params['distribution'] = distribution
        if shape:
            params['shape'] = list(shape)
        if dataset_size:
            # Calculate shape from byte size
            size_bytes = parse_size_to_bytes(dataset_size)
            calculated_shape = calculate_square_shape(size_bytes, dtype_bytes=4, dimensions=2)
            params['shape'] = list(calculated_shape)
            if verbose:
                print(f"Calculated shape from dataset size {dataset_size}: {calculated_shape}")
        if entropy_target is not None:
            params['entropy_target'] = entropy_target
        if seed is not None:
            params['seed'] = seed

        # Set defaults if still missing
        params.setdefault('distribution', 'normal')
        params.setdefault('shape', [100, 100])
        params.setdefault('entropy_target', 4.0)

        # Convert shape to tuple
        if isinstance(params['shape'], list):
            params['shape'] = tuple(params['shape'])

        # Generate dataset
        data = generate_dataset(
            distribution=params['distribution'],
            shape=params['shape'],
            entropy_target=params['entropy_target'],
            seed=params.get('seed'),
            verbose=verbose
        )

        # Write to HDF5
        if verbose:
            print(f"\nWriting to {output}...")

        compression_filter = None if compression == 'none' else compression

        # Prepare chunk size
        chunks_param = None  # Default: contiguous layout (no chunks)
        if chunk_bytes:
            # Calculate chunk dimensions from byte size
            chunk_size_bytes = parse_size_to_bytes(chunk_bytes)
            num_dims = len(params['shape'])
            calculated_chunks = calculate_square_shape(chunk_size_bytes, dtype_bytes=4, dimensions=num_dims)
            chunks_param = calculated_chunks
            if verbose:
                print(f"  Calculated chunk dimensions from {chunk_bytes}: {calculated_chunks}")
        elif chunk_size:
            chunks_param = tuple(chunk_size)
            if verbose:
                print(f"  HDF5 chunk size: {chunks_param}")

        hdf5_writer.write_dataset(
            filename=output,
            data=data,
            generation_params=params,
            compression=compression_filter,
            chunks=chunks_param,
            measure_metrics=True
        )

        if verbose:
            print(f"✓ Saved to {output}")

        # Write binary file if requested
        if binary:
            # Replace .h5 extension with .bin
            if output.endswith('.h5'):
                binary_output = output[:-3] + '.bin'
            elif output.endswith('.hdf5'):
                binary_output = output[:-5] + '.bin'
            else:
                binary_output = output + '.bin'

            if verbose:
                print(f"Writing binary file to {binary_output}...")

            binary_writer.write_binary(binary_output, data)

            if verbose:
                print(f"✓ Saved binary to {binary_output}")

        if verbose:
            # Print metrics
            measured = metrics.measure_all_attributes(data)
            metrics.print_metrics(measured, title=f"Generated Dataset: {output}")

            # Validate if requested
            if validate:
                validation = metrics.validate_attributes(
                    data,
                    target_entropy=params.get('entropy_target')
                )
                metrics.print_validation_results(validation)

        click.echo(f"\n✓ Successfully generated: {output}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', default='config/presets.yaml',
              help='Path to presets YAML file')
@click.option('--output-dir', '-o', default='datasets',
              help='Output directory for datasets')
@click.option('--presets', '-p', multiple=True,
              help='Specific presets to generate (default: all)')
@click.option('--binary/--no-binary', default=True,
              help='Also output raw binary files for GPUCompress (default: enabled)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def batch(config, output_dir, presets, binary, quiet):
    """
    Generate multiple datasets from preset file.

    Example:
        python generator.py batch -c config/presets.yaml -o datasets/
        python generator.py batch -p high_entropy_uniform -p low_entropy_normal
    """
    verbose = not quiet

    try:
        # Load config
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

        if 'presets' not in config_data:
            click.echo("✗ No presets found in config file", err=True)
            sys.exit(1)

        # Determine which presets to generate
        if presets:
            preset_names = presets
        else:
            preset_names = list(config_data['presets'].keys())

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            click.echo(f"\nGenerating {len(preset_names)} datasets...")
            click.echo(f"Output directory: {output_dir}\n")

        # Generate each preset
        for i, preset_name in enumerate(preset_names, 1):
            if verbose:
                click.echo(f"\n{'=' * 70}")
                click.echo(f"[{i}/{len(preset_names)}] Preset: {preset_name}")
                click.echo(f"{'=' * 70}")

            output_file = os.path.join(output_dir, f"{preset_name}.h5")

            # Load preset
            preset_params = load_preset(preset_name, config)

            # Generate
            data = generate_dataset(
                distribution=preset_params.get('distribution', 'normal'),
                shape=tuple(preset_params.get('shape', [100, 100])),
                entropy_target=preset_params.get('entropy_target', 4.0),
                seed=preset_params.get('seed'),
                verbose=verbose
            )

            # Write
            hdf5_writer.write_dataset(
                filename=output_file,
                data=data,
                generation_params=preset_params,
                measure_metrics=True
            )

            if verbose:
                click.echo(f"✓ Saved: {output_file}")

            # Write binary file if requested
            if binary:
                binary_output = os.path.join(output_dir, f"{preset_name}.bin")
                binary_writer.write_binary(binary_output, data)
                if verbose:
                    click.echo(f"✓ Saved binary: {binary_output}")

        if verbose:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"✓ Batch generation complete! Generated {len(preset_names)} files.")
            click.echo(f"{'=' * 70}\n")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filename')
@click.option('--dataset', '-d', default='data', help='Dataset name within HDF5 file')
def info(filename, dataset):
    """
    Display information about an HDF5 or binary file.

    Examples:
        python generator.py info datasets/high_entropy_uniform.h5
        python generator.py info datasets/high_entropy_uniform.bin
    """
    try:
        # Detect file type by extension
        if filename.endswith('.bin'):
            binary_writer.print_binary_info(filename)
        else:
            hdf5_writer.print_file_info(filename, dataset)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filename')
@click.option('--dataset', '-d', default='data', help='Dataset name within HDF5 file')
def verify(filename, dataset):
    """
    Verify HDF5 file validity.

    Example:
        python generator.py verify datasets/high_entropy_uniform.h5
    """
    try:
        is_valid = hdf5_writer.verify_hdf5_file(filename, dataset, verbose=True)
        if is_valid:
            click.echo("\n✓ File is valid!")
        else:
            click.echo("\n✗ File validation failed!", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=str, default='config/presets.yaml',
              help='Config file to list presets from (default: config/presets.yaml)')
def list_presets(config):
    """
    List all available presets.

    Examples:
        python generator.py list-presets
        python generator.py list-presets --config config/presets_custom.yaml
    """
    try:
        config_file = config
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        if 'presets' not in config_data:
            click.echo("No presets found")
            return

        click.echo(f"\nAvailable Presets from: {config_file}")
        click.echo("=" * 70)

        preset_count = len(config_data['presets'])
        click.echo(f"Total presets: {preset_count}\n")

        for name, params in config_data['presets'].items():
            desc = params.get('description', 'No description')
            click.echo(f"{name}")
            click.echo(f"  {desc}")
            click.echo(f"  Shape: {params.get('shape', 'N/A')}")
            click.echo(f"  Distribution: {params.get('distribution', 'N/A')}")
            entropy_val = params.get('entropy_target', 'N/A')
            if isinstance(entropy_val, (int, float)):
                click.echo(f"  Entropy: {entropy_val:.8f} bits")
            else:
                click.echo(f"  Entropy: {entropy_val} bits")
            click.echo()

        click.echo(f"{'=' * 70}\n")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
