"""
Nature journal-quality visualizations for emergent language analysis.

Generates publication-ready figures for:
- Rating distributions
- Language pattern correlations with ratings
- Temporal trends
- Pattern emergence analysis

Usage:
    python visualize_emergent_language.py \
        --data_dir results/emergent_language \
        --output_dir results/emergent_language/figures
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from collections import Counter, defaultdict
import re

# Nature journal style settings
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

# Nature journal figure parameters
NATURE_FIG_WIDTH = 7.2  # inches (Nature single column)
NATURE_FIG_HEIGHT = 5.0
DPI = 300  # High resolution for publication
FONT_SIZE = 8
TICK_SIZE = 7
LABEL_SIZE = 9
TITLE_SIZE = 10

# Set matplotlib parameters for Nature style
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.titlesize': TITLE_SIZE + 2,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_messages_csv(csv_path: Path) -> List[Dict]:
    """Load messages.csv into a list of dictionaries."""
    records = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_tokens_csv(csv_path: Path) -> List[Dict]:
    """Load unknown_tokens.csv into a list of dictionaries."""
    records = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_ngrams_csv(csv_path: Path) -> List[Dict]:
    """Load tail_ngrams.csv into a list of dictionaries."""
    records = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def plot_rating_distribution(messages: List[Dict], output_path: Path):
    """Figure 1: Rating distribution and correlation with trade success."""
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    # Panel A: Rating distribution
    ratings = [int(r['partner_rating']) for r in messages if r.get('partner_rating') and r['partner_rating'].strip()]
    rating_counts = Counter(ratings)
    
    ax = axes[0]
    rating_values = [1, 2, 3, 4]
    counts = [rating_counts.get(r, 0) for r in rating_values]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red to blue gradient
    
    bars = ax.bar(rating_values, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Rating (1=not teammate, 4=teammate)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_xticks(rating_values)
    ax.set_xticklabels(['1\n(not)', '2', '3', '4\n(teammate)'])
    ax.set_title('A. Rating Distribution', fontweight='bold', pad=10)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=TICK_SIZE)
    
    # Panel B: Rating vs Trade Success
    ax = axes[1]
    success_ratings = [int(r['partner_rating']) for r in messages 
                      if r.get('partner_rating') and r['partner_rating'].strip() 
                      and r.get('has_successful_trade') == '1']
    no_success_ratings = [int(r['partner_rating']) for r in messages 
                          if r.get('partner_rating') and r['partner_rating'].strip() 
                          and r.get('has_successful_trade') == '0']
    
    data_to_plot = [success_ratings, no_success_ratings]
    labels = ['With successful\ntrade', 'Without successful\ntrade']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True, meanline=True)
    
    # Color the boxes
    colors_box = ['#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=0.8)
    
    ax.set_ylabel('Rating', fontweight='bold')
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_title('B. Rating vs Trade Success', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_temporal_trends(messages: List[Dict], output_path: Path):
    """Figure 2: Rating trends across rounds."""
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    # Group by round
    round_ratings: Dict[int, List[int]] = defaultdict(list)
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                round_num = int(r['round'])
                rating = int(r['partner_rating'])
                round_ratings[round_num].append(rating)
            except (ValueError, KeyError):
                continue
    
    rounds = sorted(round_ratings.keys())
    means = [np.mean(round_ratings[r]) for r in rounds]
    stds = [np.std(round_ratings[r]) for r in rounds]
    counts = [len(round_ratings[r]) for r in rounds]
    
    # Plot with error bars
    ax.errorbar(rounds, means, yerr=stds, fmt='o-', color='#1f77b4', 
                linewidth=1.5, markersize=5, capsize=3, capthick=1,
                elinewidth=1, label='Mean rating')
    
    # Add sample size annotations
    for r, mean, count in zip(rounds, means, counts):
        if count < max(counts) * 0.8:  # Only annotate if sample size is notably smaller
            ax.annotate(f'n={count}', xy=(r, mean), xytext=(5, 5),
                       textcoords='offset points', fontsize=TICK_SIZE-1, alpha=0.7)
    
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Mean Rating', fontweight='bold')
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks(rounds)
    ax.set_title('Rating Trends Across Game Rounds', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pattern_rating_correlation(tokens: List[Dict], ngrams: List[Dict], output_path: Path):
    """Figure 3: Language patterns correlated with high ratings."""
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.7))
    
    # Panel A: Top unknown tokens by average rating
    ax = axes[0]
    tokens_with_rating = [(t['token'], float(t['avg_rating']), int(t['rating_count'])) 
                          for t in tokens 
                          if t.get('avg_rating') and t['avg_rating'].strip() 
                          and float(t['avg_rating']) > 0 and int(t['rating_count']) >= 3]
    
    # Sort by average rating, take top 15
    tokens_with_rating.sort(key=lambda x: x[1], reverse=True)
    top_tokens = tokens_with_rating[:15]
    
    if top_tokens:
        tokens_list = [t[0] for t in top_tokens]
        ratings_list = [t[1] for t in top_tokens]
        counts_list = [t[2] for t in top_tokens]
        
        y_pos = np.arange(len(tokens_list))
        bars = ax.barh(y_pos, ratings_list, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Color by count (darker = more samples)
        max_count = max(counts_list)
        for i, (bar, count) in enumerate(zip(bars, counts_list)):
            alpha = 0.5 + 0.5 * (count / max_count) if max_count > 0 else 0.7
            bar.set_alpha(alpha)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens_list, fontsize=TICK_SIZE-1)
        ax.set_xlabel('Average Rating', fontweight='bold')
        ax.set_xlim(0, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title('A. Unknown Tokens (Top by Avg Rating)', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
        
        # Add count annotations
        for i, (rating, count) in enumerate(zip(ratings_list, counts_list)):
            ax.text(rating + 0.1, i, f'n={count}', va='center', fontsize=TICK_SIZE-2, alpha=0.7)
    
    # Panel B: Top n-grams by average rating
    ax = axes[1]
    ngrams_with_rating = [(n['ngram'], float(n['avg_rating']), int(n['rating_count']), int(n['n'])) 
                          for n in ngrams 
                          if n.get('avg_rating') and n['avg_rating'].strip() 
                          and float(n['avg_rating']) > 0 and int(n['rating_count']) >= 3]
    
    ngrams_with_rating.sort(key=lambda x: x[1], reverse=True)
    top_ngrams = ngrams_with_rating[:15]
    
    if top_ngrams:
        ngrams_list = [f"{n[0]} (n={n[3]})" for n in top_ngrams]
        ratings_list = [n[1] for n in top_ngrams]
        counts_list = [n[2] for n in top_ngrams]
        
        y_pos = np.arange(len(ngrams_list))
        bars = ax.barh(y_pos, ratings_list, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        max_count = max(counts_list)
        for i, (bar, count) in enumerate(zip(bars, counts_list)):
            alpha = 0.5 + 0.5 * (count / max_count) if max_count > 0 else 0.7
            bar.set_alpha(alpha)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ngrams_list, fontsize=TICK_SIZE-1)
        ax.set_xlabel('Average Rating', fontweight='bold')
        ax.set_xlim(0, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title('B. Tail N-grams (Top by Avg Rating)', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
        
        for i, (rating, count) in enumerate(zip(ratings_list, counts_list)):
            ax.text(rating + 0.1, i, f'n={count}', va='center', fontsize=TICK_SIZE-2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rating_by_pattern_frequency(messages: List[Dict], output_path: Path):
    """Figure 4: Relationship between pattern frequency and rating."""
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    # Analyze unknown tokens frequency vs rating
    token_freq_by_rating: Dict[int, List[int]] = defaultdict(list)
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                unknown_tokens = r.get('unknown_tokens', '').strip()
                if unknown_tokens:
                    token_count = len(unknown_tokens.split())
                    token_freq_by_rating[rating].append(token_count)
            except (ValueError, KeyError):
                continue
    
    # Create box plot
    rating_values = sorted(token_freq_by_rating.keys())
    data_to_plot = [token_freq_by_rating[r] for r in rating_values]
    labels = [f'Rating {r}' for r in rating_values]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True, meanline=True)
    
    # Color by rating (gradient)
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for i, (patch, rating) in enumerate(zip(bp['boxes'], rating_values)):
        if rating <= 4:
            patch.set_facecolor(colors[rating-1])
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=0.8)
    
    ax.set_xlabel('Rating', fontweight='bold')
    ax.set_ylabel('Number of Unknown Tokens\nper Message', fontweight='bold')
    ax.set_title('Unknown Token Usage by Rating', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rating_by_reciprocity(messages: List[Dict], output_path: Path):
    """Figure 5: Rating comparison between reciprocal and non-reciprocal trades."""
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    reciprocal_ratings = []
    non_reciprocal_ratings = []
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                is_reciprocal = r.get('is_reciprocal_trade', '').strip()
                if is_reciprocal == '1':
                    reciprocal_ratings.append(rating)
                elif is_reciprocal == '0':
                    non_reciprocal_ratings.append(rating)
            except (ValueError, KeyError):
                continue
    
    if reciprocal_ratings or non_reciprocal_ratings:
        data_to_plot = []
        labels = []
        if reciprocal_ratings:
            data_to_plot.append(reciprocal_ratings)
            labels.append('Reciprocal\ntrade')
        if non_reciprocal_ratings:
            data_to_plot.append(non_reciprocal_ratings)
            labels.append('Non-reciprocal\ntrade')
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        colors_box = ['#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_ylabel('Rating', fontweight='bold')
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_title('Rating by Trade Reciprocity', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rating_by_trade_value(messages: List[Dict], output_path: Path):
    """Figure 6: Rating by total value transferred."""
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    # Group by value ranges
    value_ranges = {
        'Low (1-2)': [],
        'Medium (3-4)': [],
        'High (5+)': []
    }
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                value_str = r.get('total_value_transferred', '').strip()
                if value_str:
                    value = int(value_str)
                    if value > 0:
                        if value <= 2:
                            value_ranges['Low (1-2)'].append(rating)
                        elif value <= 4:
                            value_ranges['Medium (3-4)'].append(rating)
                        else:
                            value_ranges['High (5+)'].append(rating)
            except (ValueError, KeyError):
                continue
    
    # Filter out empty ranges
    data_to_plot = []
    labels = []
    colors_list = ['#d62728', '#ff7f0e', '#2ca02c']
    
    for i, (label, ratings) in enumerate(value_ranges.items()):
        if ratings:
            data_to_plot.append(ratings)
            labels.append(label)
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_ylabel('Rating', fontweight='bold')
        ax.set_xlabel('Total Value Transferred', fontweight='bold')
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_title('Rating by Trade Value', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trade_quality_metrics(messages: List[Dict], output_path: Path):
    """Figure 7: Comprehensive trade quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT))
    
    # Panel A: Number of successful exchanges by rating
    ax = axes[0, 0]
    rating_by_success_count: Dict[int, List[int]] = defaultdict(list)
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                success_count_str = r.get('num_successful_exchanges', '').strip()
                if success_count_str:
                    success_count = int(success_count_str)
                    rating_by_success_count[rating].append(success_count)
            except (ValueError, KeyError):
                continue
    
    if rating_by_success_count:
        rating_values = sorted(rating_by_success_count.keys())
        data_to_plot = [rating_by_success_count[r] for r in rating_values]
        labels = [f'Rating {r}' for r in rating_values]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        for i, (patch, rating) in enumerate(zip(bp['boxes'], rating_values)):
            if rating <= 4:
                patch.set_facecolor(colors[rating-1])
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_xlabel('Rating', fontweight='bold')
        ax.set_ylabel('Number of Successful\nExchanges', fontweight='bold')
        ax.set_title('A. Successful Exchanges by Rating', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Panel B: Total value transferred by rating
    ax = axes[0, 1]
    rating_by_value: Dict[int, List[int]] = defaultdict(list)
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                value_str = r.get('total_value_transferred', '').strip()
                if value_str:
                    value = int(value_str)
                    if value > 0:
                        rating_by_value[rating].append(value)
            except (ValueError, KeyError):
                continue
    
    if rating_by_value:
        rating_values = sorted(rating_by_value.keys())
        data_to_plot = [rating_by_value[r] for r in rating_values]
        labels = [f'Rating {r}' for r in rating_values]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        for i, (patch, rating) in enumerate(zip(bp['boxes'], rating_values)):
            if rating <= 4:
                patch.set_facecolor(colors[rating-1])
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_xlabel('Rating', fontweight='bold')
        ax.set_ylabel('Total Value Transferred', fontweight='bold')
        ax.set_title('B. Trade Value by Rating', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Panel C: Reciprocity rate by rating
    ax = axes[1, 0]
    rating_reciprocity: Dict[int, Dict[str, int]] = defaultdict(lambda: {'reciprocal': 0, 'non_reciprocal': 0})
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                is_reciprocal = r.get('is_reciprocal_trade', '').strip()
                if is_reciprocal == '1':
                    rating_reciprocity[rating]['reciprocal'] += 1
                elif is_reciprocal == '0':
                    rating_reciprocity[rating]['non_reciprocal'] += 1
            except (ValueError, KeyError):
                continue
    
    if rating_reciprocity:
        rating_values = sorted(rating_reciprocity.keys())
        reciprocal_counts = [rating_reciprocity[r]['reciprocal'] for r in rating_values]
        non_reciprocal_counts = [rating_reciprocity[r]['non_reciprocal'] for r in rating_values]
        
        x = np.arange(len(rating_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, reciprocal_counts, width, label='Reciprocal', 
                       color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, non_reciprocal_counts, width, label='Non-reciprocal',
                       color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Rating', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rating {r}' for r in rating_values])
        ax.set_title('C. Reciprocity by Rating', fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Panel D: Success rate (successful / total exchanges) by rating
    ax = axes[1, 1]
    rating_success_rate: Dict[int, List[float]] = defaultdict(list)
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                success_str = r.get('num_successful_exchanges', '').strip()
                failed_str = r.get('num_failed_exchanges', '').strip()
                if success_str and failed_str:
                    success = int(success_str)
                    failed = int(failed_str)
                    total = success + failed
                    if total > 0:
                        success_rate = success / total
                        rating_success_rate[rating].append(success_rate)
            except (ValueError, KeyError):
                continue
    
    if rating_success_rate:
        rating_values = sorted(rating_success_rate.keys())
        data_to_plot = [rating_success_rate[r] for r in rating_values]
        labels = [f'Rating {r}' for r in rating_values]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        for i, (patch, rating) in enumerate(zip(bp['boxes'], rating_values)):
            if rating <= 4:
                patch.set_facecolor(colors[rating-1])
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_xlabel('Rating', fontweight='bold')
        ax.set_ylabel('Success Rate\n(Successful / Total)', fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('D. Exchange Success Rate by Rating', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ngram_heatmap(ngrams: List[Dict], output_path: Path):
    """Figure 8: Heatmap of n-gram frequency vs average rating."""
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
    
    # Filter n-grams with sufficient data
    filtered_ngrams = [
        n for n in ngrams
        if n.get('avg_rating') and n['avg_rating'].strip()
        and float(n['avg_rating']) > 0
        and int(n.get('rating_count', 0)) >= 3
    ]
    
    # Separate 2-grams and 3-grams
    bigrams = [n for n in filtered_ngrams if int(n.get('n', 0)) == 2]
    trigrams = [n for n in filtered_ngrams if int(n.get('n', 0)) == 3]
    
    # Sort by count, take top 20
    bigrams.sort(key=lambda x: int(x.get('count', 0)), reverse=True)
    trigrams.sort(key=lambda x: int(x.get('count', 0)), reverse=True)
    top_bigrams = bigrams[:20]
    top_trigrams = trigrams[:20]
    
    # Panel A: Bigrams
    ax = axes[0]
    if top_bigrams:
        ngram_labels = [n['ngram'] for n in top_bigrams]
        counts = [int(n.get('count', 0)) for n in top_bigrams]
        ratings = [float(n.get('avg_rating', 0)) for n in top_bigrams]
        
        # Create scatter plot: count vs rating, size by rating_count
        rating_counts = [int(n.get('rating_count', 0)) for n in top_bigrams]
        sizes = [rc * 10 for rc in rating_counts]  # Scale for visibility
        
        scatter = ax.scatter(counts, ratings, s=sizes, alpha=0.6, 
                            c=ratings, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        # Add labels for top n-grams
        for i, (ngram, count, rating) in enumerate(zip(ngram_labels, counts, ratings)):
            if rating > 3.5 or count > 50:  # Label high-rating or high-frequency
                ax.annotate(ngram, (count, rating), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=TICK_SIZE-2, alpha=0.8)
        
        ax.set_xlabel('Frequency (Count)', fontweight='bold')
        ax.set_ylabel('Average Rating', fontweight='bold')
        ax.set_title('A. Bigrams: Frequency vs Rating', fontweight='bold', pad=10)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Rating')
    
    # Panel B: Trigrams
    ax = axes[1]
    if top_trigrams:
        ngram_labels = [n['ngram'] for n in top_trigrams]
        counts = [int(n.get('count', 0)) for n in top_trigrams]
        ratings = [float(n.get('avg_rating', 0)) for n in top_trigrams]
        rating_counts = [int(n.get('rating_count', 0)) for n in top_trigrams]
        sizes = [rc * 10 for rc in rating_counts]
        
        scatter = ax.scatter(counts, ratings, s=sizes, alpha=0.6,
                            c=ratings, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
        
        for i, (ngram, count, rating) in enumerate(zip(ngram_labels, counts, ratings)):
            if rating > 3.5 or count > 30:
                ax.annotate(ngram, (count, rating),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=TICK_SIZE-2, alpha=0.8)
        
        ax.set_xlabel('Frequency (Count)', fontweight='bold')
        ax.set_ylabel('Average Rating', fontweight='bold')
        ax.set_title('B. Trigrams: Frequency vs Rating', fontweight='bold', pad=10)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Rating')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ngram_temporal_trends(messages: List[Dict], output_path: Path):
    """Figure 9: N-gram usage trends across rounds."""
    fig, axes = plt.subplots(2, 1, figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.8))
    
    # Collect n-grams by round
    bigrams_by_round: Dict[int, Counter] = defaultdict(Counter)
    trigrams_by_round: Dict[int, Counter] = defaultdict(Counter)
    high_rating_bigrams: Dict[int, Counter] = defaultdict(Counter)
    high_rating_trigrams: Dict[int, Counter] = defaultdict(Counter)
    
    for r in messages:
        try:
            round_num = int(r.get('round', 0))
            if round_num <= 0:
                continue
            
            rating = None
            if r.get('partner_rating') and r['partner_rating'].strip():
                rating = int(r['partner_rating'])
            
            # Extract bigrams and trigrams
            bigram = r.get('tail_bigram', '').strip()
            trigram = r.get('tail_trigram', '').strip()
            
            if bigram:
                bigrams_by_round[round_num][bigram] += 1
                if rating and rating >= 4:
                    high_rating_bigrams[round_num][bigram] += 1
            
            if trigram:
                trigrams_by_round[round_num][trigram] += 1
                if rating and rating >= 4:
                    high_rating_trigrams[round_num][trigram] += 1
        except (ValueError, KeyError):
            continue
    
    # Panel A: Bigram diversity over time
    ax = axes[0]
    rounds = sorted(bigrams_by_round.keys())
    diversity = [len(bigrams_by_round[r]) for r in rounds]
    high_rating_diversity = [len(high_rating_bigrams[r]) for r in rounds]
    
    ax.plot(rounds, diversity, 'o-', color='#1f77b4', linewidth=1.5, 
            markersize=5, label='All bigrams', alpha=0.8)
    ax.plot(rounds, high_rating_diversity, 's-', color='#2ca02c', linewidth=1.5,
            markersize=5, label='High-rating bigrams (rating≥4)', alpha=0.8)
    
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Number of Unique Bigrams', fontweight='bold')
    ax.set_title('A. Bigram Diversity Across Rounds', fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Panel B: Trigram diversity over time
    ax = axes[1]
    rounds = sorted(trigrams_by_round.keys())
    diversity = [len(trigrams_by_round[r]) for r in rounds]
    high_rating_diversity = [len(high_rating_trigrams[r]) for r in rounds]
    
    ax.plot(rounds, diversity, 'o-', color='#1f77b4', linewidth=1.5,
            markersize=5, label='All trigrams', alpha=0.8)
    ax.plot(rounds, high_rating_diversity, 's-', color='#2ca02c', linewidth=1.5,
            markersize=5, label='High-rating trigrams (rating≥4)', alpha=0.8)
    
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Number of Unique Trigrams', fontweight='bold')
    ax.set_title('B. Trigram Diversity Across Rounds', fontweight='bold', pad=10)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ngram_cooccurrence(messages: List[Dict], output_path: Path):
    """Figure 10: Co-occurrence network of tokens in high-rating n-grams."""
    from collections import defaultdict as dd
    
    # Extract tokens from high-rating n-grams
    token_pairs: Dict[Tuple[str, str], int] = Counter()
    high_rating_ngrams = []
    
    for r in messages:
        if r.get('partner_rating') and r['partner_rating'].strip():
            try:
                rating = int(r['partner_rating'])
                if rating >= 4:  # High rating
                    bigram = r.get('tail_bigram', '').strip()
                    trigram = r.get('tail_trigram', '').strip()
                    
                    if bigram:
                        tokens = bigram.split()
                        if len(tokens) == 2:
                            token_pairs[tuple(sorted(tokens))] += 1
                            high_rating_ngrams.append(bigram)
                    
                    if trigram:
                        tokens = trigram.split()
                        if len(tokens) == 3:
                            # Add all pairs from trigram
                            token_pairs[tuple(sorted([tokens[0], tokens[1]]))] += 1
                            token_pairs[tuple(sorted([tokens[1], tokens[2]]))] += 1
                            token_pairs[tuple(sorted([tokens[0], tokens[2]]))] += 1
                            high_rating_ngrams.append(trigram)
            except (ValueError, KeyError):
                continue
    
    # Get top token pairs
    top_pairs = token_pairs.most_common(20)
    
    if not top_pairs:
        # Create empty figure
        fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.6))
        ax.text(0.5, 0.5, 'Insufficient data for co-occurrence analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=LABEL_SIZE)
        ax.set_title('Token Co-occurrence in High-Rating N-grams', fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path} (empty)")
        return
    
    # Create bar chart of top pairs
    fig, ax = plt.subplots(figsize=(NATURE_FIG_WIDTH, NATURE_FIG_HEIGHT * 0.7))
    
    pairs_labels = [f"{p[0][0]} - {p[0][1]}" for p in top_pairs]
    counts = [p[1] for p in top_pairs]
    
    y_pos = np.arange(len(pairs_labels))
    bars = ax.barh(y_pos, counts, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs_labels, fontsize=TICK_SIZE-1)
    ax.set_xlabel('Co-occurrence Count', fontweight='bold')
    ax.set_title('Top Token Pairs in High-Rating N-grams (Rating≥4)', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count + max(counts) * 0.01, i, f'{count}',
               va='center', fontsize=TICK_SIZE-2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nature journal-quality visualizations for emergent language analysis."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="results/emergent_language",
        help="Directory containing CSV files from analyze_emergent_language.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/emergent_language/figures",
        help="Directory to save figure files",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    messages = load_messages_csv(data_dir / "messages.csv")
    tokens = load_tokens_csv(data_dir / "unknown_tokens.csv")
    ngrams = load_ngrams_csv(data_dir / "tail_ngrams.csv")
    
    print(f"Loaded {len(messages)} messages, {len(tokens)} tokens, {len(ngrams)} n-grams")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Original figures
    plot_rating_distribution(messages, output_dir / "figure1_rating_distribution.pdf")
    plot_temporal_trends(messages, output_dir / "figure2_temporal_trends.pdf")
    plot_pattern_rating_correlation(tokens, ngrams, output_dir / "figure3_pattern_rating.pdf")
    plot_rating_by_pattern_frequency(messages, output_dir / "figure4_pattern_frequency.pdf")
    
    # New figures with enhanced trade quality metrics
    plot_rating_by_reciprocity(messages, output_dir / "figure5_rating_by_reciprocity.pdf")
    plot_rating_by_trade_value(messages, output_dir / "figure6_rating_by_trade_value.pdf")
    plot_trade_quality_metrics(messages, output_dir / "figure7_trade_quality_metrics.pdf")
    
    # N-gram specific visualizations
    plot_ngram_heatmap(ngrams, output_dir / "figure8_ngram_heatmap.pdf")
    plot_ngram_temporal_trends(messages, output_dir / "figure9_ngram_temporal_trends.pdf")
    plot_ngram_cooccurrence(messages, output_dir / "figure10_ngram_cooccurrence.pdf")
    
    # Also save as PNG for easy viewing
    plot_rating_distribution(messages, output_dir / "figure1_rating_distribution.png")
    plot_temporal_trends(messages, output_dir / "figure2_temporal_trends.png")
    plot_pattern_rating_correlation(tokens, ngrams, output_dir / "figure3_pattern_rating.png")
    plot_rating_by_pattern_frequency(messages, output_dir / "figure4_pattern_frequency.png")
    plot_rating_by_reciprocity(messages, output_dir / "figure5_rating_by_reciprocity.png")
    plot_rating_by_trade_value(messages, output_dir / "figure6_rating_by_trade_value.png")
    plot_trade_quality_metrics(messages, output_dir / "figure7_trade_quality_metrics.png")
    plot_ngram_heatmap(ngrams, output_dir / "figure8_ngram_heatmap.png")
    plot_ngram_temporal_trends(messages, output_dir / "figure9_ngram_temporal_trends.png")
    plot_ngram_cooccurrence(messages, output_dir / "figure10_ngram_cooccurrence.png")
    
    print(f"\nAll figures saved to: {output_dir}")
    print("Figures are saved in both PDF (publication-ready) and PNG (preview) formats.")
    print("\nNew figures added:")
    print("  - Figure 5: Rating by Reciprocity")
    print("  - Figure 6: Rating by Trade Value")
    print("  - Figure 7: Comprehensive Trade Quality Metrics")
    print("  - Figure 8: N-gram Heatmap (Frequency vs Rating)")
    print("  - Figure 9: N-gram Temporal Trends")
    print("  - Figure 10: Token Co-occurrence in High-Rating N-grams")


if __name__ == "__main__":
    main()

