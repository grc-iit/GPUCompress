#!/usr/bin/env python3
"""
Visualization script for GPUCompress compression results.

Generates charts from the CSV output of run_simple_tests.sh:
1. Compression ratio comparison by configuration
2. Error vs compression ratio trade-off
3. PSNR comparison across patterns
4. Best configurations summary
5. Improvement heatmap

Charts are organized in subfolders per algorithm.

Usage:
    python3 visualize_results.py [csv_file] [output_dir]

    Defaults:
        csv_file: test_data/compression_results.csv
        output_dir: test_data/charts/
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_file):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_file)

    # Create a configuration label (short form for readability)
    def config_label(row):
        quant = row['quantization']
        eb = row['error_bound']
        shuf = row['shuffle'] != 0

        if quant == 'none' and not shuf:
            return "Baseline"
        elif quant == 'none' and shuf:
            return "Shuffle"
        elif quant != 'none' and not shuf:
            return f"Q({eb})"
        else:
            return f"Q({eb})+S"

    df['config'] = df.apply(config_label, axis=1)

    # Replace inf with a large number for plotting
    df['psnr_db'] = df['psnr_db'].replace('inf', 999)
    df['psnr_db'] = pd.to_numeric(df['psnr_db'], errors='coerce')

    return df

def plot_compression_ratios(df, output_dir, algorithm):
    """Bar chart: Compression ratios by pattern and configuration."""
    patterns = df['pattern'].unique()
    n_patterns = len(patterns)

    # Calculate grid dimensions based on number of patterns
    ncols = min(3, n_patterns)
    nrows = (n_patterns + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle(f'Compression Ratios by Data Pattern ({algorithm})', fontsize=14, fontweight='bold', y=0.98)

    # Define config order for consistent display
    config_order = ['Baseline', 'Shuffle', 'Q(0.01)', 'Q(0.001)', 'Q(0.0001)',
                    'Q(0.01)+S', 'Q(0.001)+S', 'Q(0.0001)+S']

    for idx, pattern in enumerate(patterns):
        ax = axes[idx // ncols, idx % ncols]
        pattern_df = df[df['pattern'] == pattern]

        # Group by config and reorder
        pivot = pattern_df.groupby('config')['ratio'].mean()
        pivot = pivot.reindex([c for c in config_order if c in pivot.index])

        bars = ax.bar(range(len(pivot)), pivot.values, color='steelblue')
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'{pattern.capitalize()} Pattern', fontsize=12)
        ax.set_ylabel('Compression Ratio')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=2)

    # Hide any unused subplots
    total_subplots = nrows * ncols
    for idx in range(n_patterns, total_subplots):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_ratios.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: compression_ratios.png")

def plot_error_vs_ratio(df, output_dir, algorithm):
    """Scatter plot: Error bound vs compression ratio trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to only quantized data
    quant_df = df[df['quantization'] == 'linear'].copy()

    if quant_df.empty:
        plt.close()
        return

    available_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    available_colors = ['blue', 'red', 'green', 'orange', 'purple']

    patterns = quant_df['pattern'].unique()
    markers = {p: available_markers[i % len(available_markers)] for i, p in enumerate(patterns)}
    colors = {p: available_colors[i % len(available_colors)] for i, p in enumerate(patterns)}

    for pattern in patterns:
        subset = quant_df[quant_df['pattern'] == pattern]
        ax.scatter(
            subset['error_bound'],
            subset['ratio'],
            marker=markers[pattern],
            c=colors[pattern],
            label=pattern,
            s=80,
            alpha=0.7
        )

    ax.set_xscale('log')
    ax.set_xlabel('Error Bound (log scale)', fontsize=11)
    ax.set_ylabel('Compression Ratio', fontsize=11)
    ax.set_title(f'Compression Ratio vs Error Bound ({algorithm})', fontsize=12, fontweight='bold')
    ax.legend(title='Pattern', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_ratio.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: error_vs_ratio.png")

def plot_psnr_comparison(df, output_dir, algorithm):
    """Bar chart: PSNR by pattern and error bound."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to quantized data only
    quant_df = df[df['quantization'] == 'linear'].copy()
    quant_df = quant_df[quant_df['psnr_db'] < 500]  # Exclude inf values

    if quant_df.empty:
        plt.close()
        return

    # Group by pattern and error_bound
    pivot = quant_df.pivot_table(
        values='psnr_db',
        index='pattern',
        columns='error_bound',
        aggfunc='mean'
    )

    pivot.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_xlabel('Data Pattern', fontsize=11)
    ax.set_title(f'Signal Quality (PSNR) by Pattern and Error Bound ({algorithm})', fontsize=12, fontweight='bold')
    ax.legend(title='Error Bound')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: psnr_comparison.png")

def plot_config_comparison(df, output_dir, algorithm):
    """Grouped bar chart: All configurations compared."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Use the same short config names from the 'config' column
    pivot = df.pivot_table(
        values='ratio',
        index='config',
        columns='pattern',
        aggfunc='mean'
    )

    # Reorder for logical presentation
    config_order = ['Baseline', 'Shuffle', 'Q(0.01)', 'Q(0.001)', 'Q(0.0001)',
                    'Q(0.01)+S', 'Q(0.001)+S', 'Q(0.0001)+S']
    pivot = pivot.reindex([c for c in config_order if c in pivot.index])

    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Compression Ratio', fontsize=11)
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_title(f'Compression Ratio by Configuration ({algorithm})', fontsize=12, fontweight='bold')
    ax.legend(title='Pattern', loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'config_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: config_comparison.png")

def plot_best_configs(df, output_dir, algorithm):
    """Summary table: Best configuration for each pattern."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Find best config for each pattern
    results = []
    for pattern in df['pattern'].unique():
        pattern_df = df[df['pattern'] == pattern]

        # Best lossless
        lossless = pattern_df[pattern_df['quantization'] == 'none']
        if not lossless.empty:
            best_lossless = lossless.loc[lossless['ratio'].idxmax()]
            results.append({
                'Pattern': pattern,
                'Type': 'Best Lossless',
                'Config': f"{'shuffle' if best_lossless['shuffle'] else 'no shuffle'}",
                'Ratio': f"{best_lossless['ratio']:.2f}x",
                'Max Error': '0 (exact)'
            })

        # Best lossy (highest ratio with reasonable error)
        lossy = pattern_df[pattern_df['quantization'] == 'linear']
        if not lossy.empty:
            best_lossy = lossy.loc[lossy['ratio'].idxmax()]
            results.append({
                'Pattern': pattern,
                'Type': 'Best Lossy',
                'Config': f"quant(eb={best_lossy['error_bound']}) + {'shuffle' if best_lossy['shuffle'] else 'no shuffle'}",
                'Ratio': f"{best_lossy['ratio']:.2f}x",
                'Max Error': f"{best_lossy['max_error']:.2e}"
            })

    if not results:
        plt.close()
        return

    results_df = pd.DataFrame(results)

    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(results_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(results_df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title(f'Best Configurations Summary ({algorithm})', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_configs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: best_configs.png")

def plot_improvement_heatmap(df, output_dir, algorithm):
    """Heatmap: Improvement over baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate improvement over baseline for each pattern
    improvements = []

    for pattern in df['pattern'].unique():
        pattern_df = df[df['pattern'] == pattern]
        baseline = pattern_df[(pattern_df['quantization'] == 'none') & (pattern_df['shuffle'] == 0)]

        if baseline.empty:
            continue
        baseline_ratio = baseline['ratio'].values[0]

        for _, row in pattern_df.iterrows():
            improvement = row['ratio'] / baseline_ratio
            improvements.append({
                'pattern': pattern,
                'config': row['config'],  # Use the short config label
                'improvement': improvement
            })

    if not improvements:
        plt.close()
        return

    imp_df = pd.DataFrame(improvements)
    pivot = imp_df.pivot_table(values='improvement', index='config', columns='pattern')

    # Reorder configs for consistent display
    config_order = ['Baseline', 'Shuffle', 'Q(0.01)', 'Q(0.001)', 'Q(0.0001)',
                    'Q(0.01)+S', 'Q(0.001)+S', 'Q(0.0001)+S']
    pivot = pivot.reindex([c for c in config_order if c in pivot.index])

    im = ax.imshow(pivot.values, cmap='YlGn', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement Factor vs Baseline')

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.mean() else 'black'
            ax.text(j, i, f'{val:.1f}x', ha='center', va='center', color=color, fontsize=8)

    ax.set_title(f'Improvement Over Baseline ({algorithm})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Created: improvement_heatmap.png")

def main():
    # Parse arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_data/compression_results.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'test_data/charts'

    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if not os.path.isabs(csv_file):
        csv_file = os.path.join(project_dir, csv_file)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_dir, output_dir)

    print(f"Loading data from: {csv_file}")
    print(f"Saving charts to: {output_dir}")
    print()

    # Load data
    df = load_data(csv_file)
    print(f"Loaded {len(df)} test results")
    print(f"Patterns: {', '.join(df['pattern'].unique())}")
    print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    print()

    # Generate charts per algorithm
    algorithms = df['algorithm'].unique()

    for algorithm in algorithms:
        algo_dir = os.path.join(output_dir, algorithm)
        os.makedirs(algo_dir, exist_ok=True)

        algo_df = df[df['algorithm'] == algorithm]

        print(f"Generating visualizations for {algorithm}...")
        plot_compression_ratios(algo_df, algo_dir, algorithm)
        plot_error_vs_ratio(algo_df, algo_dir, algorithm)
        plot_psnr_comparison(algo_df, algo_dir, algorithm)
        plot_config_comparison(algo_df, algo_dir, algorithm)
        plot_best_configs(algo_df, algo_dir, algorithm)
        plot_improvement_heatmap(algo_df, algo_dir, algorithm)
        print()

    print(f"All charts saved to: {output_dir}")

if __name__ == '__main__':
    main()
