"""
Simple script to generate only the LaTeX summary table.
"""

import csv
from pathlib import Path

data_dir = Path('results/language_evolution')
output_dir = Path('results/language_evolution/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
lifecycle = []
with open(data_dir / 'lifecycle_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    lifecycle = list(reader)

diffusion = []
with open(data_dir / 'diffusion_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    diffusion = list(reader)

lexical = []
with open(data_dir / 'lexical_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    lexical = list(reader)

lifecycle_dict = {row['Metric']: row['Value'] for row in lifecycle}
diffusion_dict = {row['Metric']: row['Value'] for row in diffusion}

innovation_values = [int(row['Innovation Count']) for row in lexical]
reuse_values = [float(row['Reuse Rate (%)']) for row in lexical]
avg_innovation = sum(innovation_values) / len(innovation_values) if innovation_values else 0
avg_reuse = sum(reuse_values) / len(reuse_values) if reuse_values else 0

# Generate LaTeX table (using double braces to escape in format strings)
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

print(f'Successfully generated: {output_path}')

