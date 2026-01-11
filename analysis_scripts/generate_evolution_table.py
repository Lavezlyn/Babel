"""
Generate LaTeX summary table for language evolution metrics.

This script can be run independently to generate the summary table.
"""

import argparse
import csv
from pathlib import Path


def load_csv(csv_path: Path) -> list:
    """Load CSV file into a list of dictionaries."""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def generate_summary_table(data_dir: Path, output_dir: Path):
    """Generate LaTeX summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    latex = """\\begin{table}[t]
\\centering
\\caption{Language Evolution Metrics Summary}
\\label{tab:evolution_summary}
\\begin{tabular}{lc}
\\toprule
\\textbf{Metric} & \\textbf{Value} \\\\
\\midrule
\\multicolumn{2}{l}{\\textbf{Life Cycle Metrics}} \\\\
\\cmidrule(lr){1-2}
First Appearance - Early (Rounds 1-3) & {:.0f} \\\\
First Appearance - Mid (Rounds 4-7) & {:.0f} \\\\
First Appearance - Late (Rounds 8-14) & {:.0f} \\\\
Avg Stabilization Time (Rounds) & {:.2f} \\\\
Survival Rate (\\%) & {:.1f} \\\\
Extinction Rate (\\%) & {:.1f} \\\\
\\midrule
\\multicolumn{2}{l}{\\textbf{Diffusion Metrics}} \\\\
\\cmidrule(lr){1-2}
Avg Adoption Rate & {:.2f} \\\\
Avg Cross-Pair Count & {:.2f} \\\\
\\midrule
\\multicolumn{2}{l}{\\textbf{Lexical Metrics}} \\\\
\\cmidrule(lr){1-2}
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
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX summary table for language evolution metrics")
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
        help="Output directory for the table"
    )
    
    args = parser.parse_args()
    
    print("Generating language evolution summary table...")
    print("=" * 80)
    
    try:
        output_path = generate_summary_table(args.data_dir, args.output_dir)
        print("=" * 80)
        print(f"Table successfully generated: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

