"""
Generate publication-quality visualizations for language evolution metrics.

EMNLP/Nature journal style:
- High resolution (300 DPI)
- Professional color schemes
- Clear typography
- Publication-ready figures
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict

# Nature/EMNLP style settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# Nature journal figure parameters
NATURE_FIG_WIDTH = 7.2  # inches (Nature single column)
NATURE_FIG_HEIGHT = 5.0
DPI = 300  # High resolution for publication
FONT_SIZE = 9
TICK_SIZE = 8
LABEL_SIZE = 10
TITLE_SIZE = 11

# Set matplotlib parameters for Nature/EMNLP style
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': TITLE_SIZE + 1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.7,
    'ytick.minor.width': 0.7,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (Nature/EMNLP style)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#6C757D',      # Gray
    'light': '#E9ECEF',        # Light gray
}

# Professional color scheme
PALETTE = sns.color_palette("husl", 8)


def load_csv(csv_path: Path) -> List[Dict]:
    """Load CSV file into a list of dictionaries."""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def plot_lifecycle_metrics(data_dir: Path, output_dir: Path):
    """Plot pattern life cycle metrics."""
    csv_path = data_dir / 'lifecycle_metrics.csv'
    data = load_csv(csv_path)
    
    # Extract metrics
    metrics = {row['Metric']: float(row['Value']) for row in data}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(NATURE_FIG_WIDTH * 1.2, NATURE_FIG_HEIGHT * 1.2))
    fig.suptitle('Pattern Life Cycle Metrics', fontsize=TITLE_SIZE + 2, fontweight='bold', y=0.98)
    
    # 1. First Appearance Distribution
    ax = axes[0, 0]
    categories = ['Early\n(1-3)', 'Mid\n(4-7)', 'Late\n(8-14)']
    values = [
        metrics['First Appearance - Early (1-3)'],
        metrics['First Appearance - Mid (4-7)'],
        metrics['First Appearance - Late (8-14)']
    ]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.set_ylabel('Number of Patterns', fontsize=LABEL_SIZE)
    ax.set_title('First Appearance Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=TICK_SIZE)
    
    # 2. Stabilization Time (bar chart)
    ax = axes[0, 1]
    avg_stab = metrics['Avg Stabilization Time']
    # Simple bar chart
    ax.bar(['Avg Stabilization'], [avg_stab], color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.set_ylabel('Time (Rounds)', fontsize=LABEL_SIZE)
    ax.set_title('Average Stabilization Time', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.text(0, avg_stab, f'{avg_stab:.2f}', ha='center', va='bottom', fontsize=TICK_SIZE + 1, fontweight='bold')
    
    # 3. Survival vs Extinction
    ax = axes[1, 0]
    categories = ['Survival', 'Extinction']
    values = [
        metrics['Survival Rate (%)'],
        metrics['Extinction Rate (%)']
    ]
    colors = [COLORS['success'], COLORS['neutral']]
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.set_ylabel('Rate (%)', fontsize=LABEL_SIZE)
    ax.set_title('Pattern Survival vs Extinction', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=TICK_SIZE)
    
    # 4. Summary metrics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    Life Cycle Summary:
    
    Total Patterns: {sum(values[:3]):.0f}
    Early Patterns: {metrics['First Appearance - Early (1-3)']:.0f}
    Avg Stabilization: {avg_stab:.2f} rounds
    Survival Rate: {metrics['Survival Rate (%)']:.1f}%
    Extinction Rate: {metrics['Extinction Rate (%)']:.1f}%
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=FONT_SIZE + 1,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'figure_lifecycle_metrics.pdf'
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_lexical_evolution(data_dir: Path, output_dir: Path):
    """Plot lexical evolution metrics."""
    csv_path = data_dir / 'lexical_metrics.csv'
    data = load_csv(csv_path)
    
    rounds = [int(row['Round']) for row in data]
    innovation = [int(row['Innovation Count']) for row in data]
    reuse = [float(row['Reuse Rate (%)']) for row in data]
    
    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT))
    
    # Innovation count (bars)
    color1 = COLORS['primary']
    ax1.set_xlabel('Round', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_ylabel('Innovation Count (New Tokens)', fontsize=LABEL_SIZE, fontweight='bold', color=color1)
    bars = ax1.bar(rounds, innovation, color=color1, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(rounds)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Reuse rate (line)
    ax2 = ax1.twinx()
    color2 = COLORS['accent']
    ax2.set_ylabel('Reuse Rate (%)', fontsize=LABEL_SIZE, fontweight='bold', color=color2)
    line = ax2.plot(rounds, reuse, color=color2, marker='o', linewidth=2, markersize=6, label='Reuse Rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.title('Lexical Evolution: Innovation and Reuse', fontsize=TITLE_SIZE + 1, fontweight='bold', pad=15)
    
    # Add legend
    ax1.legend([mpatches.Patch(color=color1, alpha=0.7), plt.Line2D([0], [0], color=color2, marker='o')],
               ['Innovation Count', 'Reuse Rate'], loc='upper left', fontsize=FONT_SIZE - 1)
    
    plt.tight_layout()
    output_path = output_dir / 'figure_lexical_evolution.pdf'
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_structural_evolution(data_dir: Path, output_dir: Path):
    """Plot structural evolution metrics."""
    csv_path = data_dir / 'structural_metrics.csv'
    data = load_csv(csv_path)
    
    rounds = [int(row['Round']) for row in data]
    msg_length = [float(row['Avg Message Length']) for row in data]
    bigram_ratio = [float(row['Bigram Ratio (%)']) for row in data]
    trigram_ratio = [float(row['Trigram Ratio (%)']) for row in data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 1.3))
    fig.suptitle('Structural Evolution Metrics', fontsize=TITLE_SIZE + 2, fontweight='bold', y=0.98)
    
    # 1. Message Length Evolution
    ax = axes[0]
    ax.plot(rounds, msg_length, color=COLORS['primary'], marker='o', linewidth=2, markersize=6)
    ax.fill_between(rounds, msg_length, alpha=0.2, color=COLORS['primary'])
    ax.set_ylabel('Average Message Length\n(Tokens)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Message Complexity Evolution', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(rounds)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add simple trend annotation
    if len(rounds) > 1:
        trend = (msg_length[-1] - msg_length[0]) / (rounds[-1] - rounds[0])
        ax.text(0.05, 0.95, f'Trend: {trend:+.2f} tokens/round', 
                transform=ax.transAxes, fontsize=FONT_SIZE - 1,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 2. N-gram Ratio Evolution
    ax = axes[1]
    ax.plot(rounds, bigram_ratio, color=COLORS['secondary'], marker='s', linewidth=2, markersize=5, label='2-gram')
    ax.plot(rounds, trigram_ratio, color=COLORS['accent'], marker='^', linewidth=2, markersize=5, label='3-gram')
    ax.fill_between(rounds, bigram_ratio, alpha=0.2, color=COLORS['secondary'])
    ax.fill_between(rounds, trigram_ratio, alpha=0.2, color=COLORS['accent'])
    ax.set_xlabel('Round', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('N-gram Pattern Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(rounds)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=FONT_SIZE - 1, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'figure_structural_evolution.pdf'
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_evolution_paths(data_dir: Path, output_dir: Path):
    """Plot pattern evolution paths."""
    csv_path = data_dir / 'evolution_paths.csv'
    data = load_csv(csv_path)
    
    compositional = [row for row in data if row['Type'] == 'Compositional']
    simplification = [row for row in data if row['Type'] == 'Simplification']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_FIG_WIDTH * 1.4, NATURE_FIG_HEIGHT))
    fig.suptitle('Pattern Evolution Paths', fontsize=TITLE_SIZE + 2, fontweight='bold', y=0.98)
    
    # 1. Compositional Evolution
    ax = axes[0]
    comp_rounds = [int(row['From Round']) for row in compositional]
    comp_counts = Counter(comp_rounds)
    rounds_sorted = sorted(comp_counts.keys())
    counts = [comp_counts[r] for r in rounds_sorted]
    
    bars = ax.bar(rounds_sorted, counts, color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.set_xlabel('Round', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Number of Compositional Paths', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Compositional Evolution\n(2-gram → 3-gram)', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(rounds_sorted)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=TICK_SIZE)
    
    # 2. Simplification Evolution
    ax = axes[1]
    simp_rounds = [int(row['From Round']) for row in simplification]
    simp_counts = Counter(simp_rounds)
    rounds_sorted = sorted(simp_counts.keys()) if simp_counts else []
    counts = [simp_counts[r] for r in rounds_sorted] if rounds_sorted else []
    
    if rounds_sorted:
        bars = ax.bar(rounds_sorted, counts, color=COLORS['accent'], alpha=0.8, edgecolor='black', linewidth=1.0)
        ax.set_xticks(rounds_sorted)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=TICK_SIZE)
    else:
        ax.text(0.5, 0.5, 'No simplification\npaths found', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=FONT_SIZE + 1, style='italic')
    
    ax.set_xlabel('Round', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Number of Simplification Paths', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Simplification Evolution\n(3-gram → 2-gram)', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'figure_evolution_paths.pdf'
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_diffusion_metrics(data_dir: Path, output_dir: Path):
    """Plot pattern diffusion metrics."""
    csv_path = data_dir / 'diffusion_metrics.csv'
    data = load_csv(csv_path)
    
    metrics = {row['Metric']: float(row['Value']) for row in data}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH * 0.8, NATURE_FIG_HEIGHT * 0.8))
    
    categories = ['Adoption\nRate', 'Cross-Pair\nCount']
    values = [
        metrics['Avg Adoption Rate'],
        metrics['Avg Cross-Pair Count']
    ]
    colors = [COLORS['primary'], COLORS['secondary']]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.0)
    ax.set_ylabel('Average Count', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Pattern Diffusion Metrics', fontsize=TITLE_SIZE + 1, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=TICK_SIZE + 1)
    
    plt.tight_layout()
    output_path = output_dir / 'figure_diffusion_metrics.pdf'
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(data_dir: Path, output_dir: Path):
    """Generate LaTeX summary table."""
    # Load all metrics
    lifecycle = load_csv(data_dir / 'lifecycle_metrics.csv')
    diffusion = load_csv(data_dir / 'diffusion_metrics.csv')
    lexical = load_csv(data_dir / 'lexical_metrics.csv')
    
    lifecycle_dict = {row['Metric']: row['Value'] for row in lifecycle}
    diffusion_dict = {row['Metric']: row['Value'] for row in diffusion}
    
    # Calculate lexical averages
    innovation_values = [int(row['Innovation Count']) for row in lexical]
    reuse_values = [float(row['Reuse Rate (%)']) for row in lexical]
    avg_innovation = sum(innovation_values) / len(innovation_values) if innovation_values else 0
    avg_reuse = sum(reuse_values) / len(reuse_values) if reuse_values else 0
    
    # Generate LaTeX table
    # Note: Use double braces {{}} to escape braces in format strings
    latex = """\\begin{{table}}[t]
\\centering
\\caption{{Language Evolution Metrics Summary}}
\\label{{tab:evolution_summary}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textbf{{Life Cycle Metrics}}}} \\\\
\\cmidrule(lr){{1-2}}
First Appearance - Early (Rounds 1-3) & {:.0f} \\\\
First Appearance - Mid (Rounds 4-7) & {:.0f} \\\\
First Appearance - Late (Rounds 8-14) & {:.0f} \\\\
Avg Stabilization Time (Rounds) & {:.2f} \\\\
Survival Rate (\\%) & {:.1f} \\\\
Extinction Rate (\\%) & {:.1f} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textbf{{Diffusion Metrics}}}} \\\\
\\cmidrule(lr){{1-2}}
Avg Adoption Rate & {:.2f} \\\\
Avg Cross-Pair Count & {:.2f} \\\\
\\midrule
\\multicolumn{{2}}{{l}}{{\\textbf{{Lexical Metrics}}}} \\\\
\\cmidrule(lr){{1-2}}
Avg Innovation Count (per round) & {:.1f} \\\\
Avg Reuse Rate (\\%) & {:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""".format(
        float(lifecycle_dict['First Appearance - Early (1-3)']),
        float(lifecycle_dict['First Appearance - Mid (4-7)']),
        float(lifecycle_dict['First Appearance - Late (8-14)']),
        float(lifecycle_dict['Avg Stabilization Time']),
        float(lifecycle_dict['Survival Rate (%)']),
        float(lifecycle_dict['Extinction Rate (%)']),
        float(diffusion_dict['Avg Adoption Rate']),
        float(diffusion_dict['Avg Cross-Pair Count']),
        avg_innovation,
        avg_reuse
    )
    
    output_path = output_dir / 'table_evolution_summary.tex'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate language evolution visualizations")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("results/language_evolution"),
        help="Directory containing evolution metrics CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/language_evolution/figures"),
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating language evolution visualizations...")
    print("=" * 80)
    
    # Generate all visualizations
    plot_lifecycle_metrics(args.data_dir, args.output_dir)
    plot_lexical_evolution(args.data_dir, args.output_dir)
    plot_structural_evolution(args.data_dir, args.output_dir)
    plot_evolution_paths(args.data_dir, args.output_dir)
    plot_diffusion_metrics(args.data_dir, args.output_dir)
    generate_summary_table(args.data_dir, args.output_dir)
    
    print("=" * 80)
    print(f"All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

