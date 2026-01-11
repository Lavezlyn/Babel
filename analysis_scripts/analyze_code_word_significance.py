"""
Analyze code word significance and distinguish from resource names.

This script:
1. Identifies potential code words (n-grams with high rating)
2. Distinguishes code words from resource names
3. Calculates significance metrics beyond avg_rating
4. Provides context analysis
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import re

# Resource types in the game
RESOURCE_TYPES = {"meat", "grain", "water", "fruit", "fish"}

# Vocabulary meanings that might be confused with resources
VOCAB_MEANINGS_WITH_RESOURCES = {
    "meat", "grain", "water", "fruit", "fish", "give", "want", "have"
}


def load_csv(csv_path: Path) -> List[Dict]:
    """Load CSV file."""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def load_messages_csv(csv_path: Path) -> List[Dict]:
    """Load messages CSV."""
    return load_csv(csv_path)


def get_vocab_meanings(log_path: str) -> Dict[str, str]:
    """Get vocabulary word -> meaning mapping from log file."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = data.get('vocabulary', {})
        return vocab
    except:
        return {}


def contains_resource_name(ngram: str, vocab_meanings: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Check if n-gram contains resource names.
    Returns: (contains_resource, list_of_resources_found)
    """
    tokens = ngram.lower().split()
    found_resources = []
    
    # Check direct resource names
    for token in tokens:
        if token in RESOURCE_TYPES:
            found_resources.append(token)
    
    # Check if any token maps to resource meaning in vocabulary
    for token in tokens:
        meaning = vocab_meanings.get(token, "").lower()
        if meaning in RESOURCE_TYPES:
            found_resources.append(f"{token} ({meaning})")
    
    return len(found_resources) > 0, found_resources


def calculate_significance_metrics(ngram: str, messages: List[Dict], vocab_meanings: Dict[str, str]) -> Dict:
    """
    Calculate comprehensive significance metrics for an n-gram.
    """
    # Filter messages containing this n-gram
    ngram_messages = []
    for msg in messages:
        bigram = msg.get('tail_bigram', '').strip()
        trigram = msg.get('tail_trigram', '').strip()
        if bigram == ngram or trigram == ngram:
            ngram_messages.append(msg)
    
    if not ngram_messages:
        return {}
    
    # Basic metrics
    ratings = []
    teammate_cases = []
    opponent_cases = []
    successful_trades = []
    reciprocal_trades = []
    
    for msg in ngram_messages:
        rating_str = msg.get('partner_rating', '').strip()
        if rating_str:
            try:
                rating = float(rating_str)
                ratings.append(rating)
                
                # Ground truth analysis
                is_teammate_gt = msg.get('is_teammate_gt', '').strip()
                if is_teammate_gt == '1':
                    teammate_cases.append(rating)
                elif is_teammate_gt == '0':
                    opponent_cases.append(rating)
            except:
                pass
        
        if msg.get('has_successful_trade') == '1':
            successful_trades.append(msg)
        if msg.get('is_reciprocal_trade') == '1':
            reciprocal_trades.append(msg)
    
    # Calculate metrics
    metrics = {
        'count': len(ngram_messages),
        'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
        'rating_count': len(ratings),
        'rating_std': 0,
        'teammate_avg_rating': sum(teammate_cases) / len(teammate_cases) if teammate_cases else None,
        'opponent_avg_rating': sum(opponent_cases) / len(opponent_cases) if opponent_cases else None,
        'teammate_count': len(teammate_cases),
        'opponent_count': len(opponent_cases),
        'success_rate': len(successful_trades) / len(ngram_messages) * 100 if ngram_messages else 0,
        'reciprocal_rate': len(reciprocal_trades) / len(ngram_messages) * 100 if ngram_messages else 0,
    }
    
    # Calculate standard deviation
    if len(ratings) > 1:
        mean = metrics['avg_rating']
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        metrics['rating_std'] = variance ** 0.5
    
    # Significance indicators
    # 1. High rating consistency (low std, high avg)
    metrics['rating_consistency'] = 1.0 / (1.0 + metrics['rating_std']) if metrics['rating_std'] > 0 else 1.0
    
    # 2. Teammate discrimination (higher rating for teammates)
    if metrics['teammate_avg_rating'] and metrics['opponent_avg_rating']:
        metrics['discrimination_score'] = metrics['teammate_avg_rating'] - metrics['opponent_avg_rating']
    else:
        metrics['discrimination_score'] = None
    
    # 3. Code word likelihood (high avg_rating, consistent, good discrimination)
    code_word_score = 0
    if metrics['avg_rating'] > 3.5:
        code_word_score += 0.4
    if metrics['rating_consistency'] > 0.7:
        code_word_score += 0.2
    if metrics['discrimination_score'] and metrics['discrimination_score'] > 0.5:
        code_word_score += 0.2
    if metrics['count'] >= 5:
        code_word_score += 0.2
    metrics['code_word_score'] = code_word_score
    
    # Check if contains resource name
    contains_resource, found_resources = contains_resource_name(ngram, vocab_meanings)
    metrics['contains_resource'] = contains_resource
    metrics['found_resources'] = found_resources
    
    return metrics


def analyze_code_words(data_dir: Path, output_dir: Path):
    """Main analysis function."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    ngrams_data = load_csv(data_dir / 'tail_ngrams.csv')
    messages_data = load_messages_csv(data_dir / 'messages.csv')
    
    # Get vocabulary mappings from log files
    log_files = {}
    for msg in messages_data[:100]:  # Sample to get log files
        log_path = msg.get('log_path', '')
        if log_path and log_path not in log_files:
            log_files[log_path] = get_vocab_meanings(log_path)
    
    # Analyze each n-gram
    print("Analyzing n-grams...")
    results = []
    
    for ngram_row in ngrams_data:
        ngram = ngram_row.get('ngram', '').strip()
        if not ngram:
            continue
        
        # Get vocab meanings (use first available)
        vocab_meanings = log_files.get(list(log_files.keys())[0] if log_files else '', {})
        
        # Calculate significance metrics
        metrics = calculate_significance_metrics(ngram, messages_data, vocab_meanings)
        if not metrics:
            continue
        
        # Add n-gram info
        metrics['ngram'] = ngram
        metrics['n'] = ngram_row.get('n', '')
        metrics['game_count'] = ngram_row.get('game_count', '')
        
        results.append(metrics)
    
    # Sort by code word score (descending)
    results.sort(key=lambda x: x.get('code_word_score', 0), reverse=True)
    
    # Write detailed analysis
    with open(output_dir / 'code_word_significance_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'ngram', 'n', 'count', 'game_count', 'avg_rating', 'rating_count', 'rating_std',
            'rating_consistency', 'teammate_avg_rating', 'opponent_avg_rating',
            'teammate_count', 'opponent_count', 'discrimination_score',
            'success_rate', 'reciprocal_rate', 'code_word_score',
            'contains_resource', 'found_resources'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            # Convert found_resources list to string
            r_copy = r.copy()
            r_copy['found_resources'] = ', '.join(r.get('found_resources', []))
            writer.writerow(r_copy)
    
    # Generate summary report
    with open(output_dir / 'code_word_significance_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CODE WORD SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This analysis distinguishes code words from resource names and evaluates significance.\n\n")
        
        # High-confidence code words (score > 0.7, no resource names)
        high_confidence = [r for r in results if r.get('code_word_score', 0) > 0.7 and not r.get('contains_resource', False)]
        f.write(f"High-Confidence Code Words (score > 0.7, no resource names): {len(high_confidence)}\n")
        f.write("-" * 80 + "\n")
        for r in high_confidence[:20]:
            f.write(f"\nN-gram: {r['ngram']}\n")
            f.write(f"  Code Word Score: {r.get('code_word_score', 0):.2f}\n")
            f.write(f"  Avg Rating: {r.get('avg_rating', 0):.2f} (n={r.get('rating_count', 0)})\n")
            f.write(f"  Rating Consistency: {r.get('rating_consistency', 0):.2f}\n")
            if r.get('discrimination_score') is not None:
                f.write(f"  Discrimination Score: {r.get('discrimination_score', 0):.2f}\n")
            f.write(f"  Count: {r.get('count', 0)}, Games: {r.get('game_count', 0)}\n")
        
        # Potential resource names (contains resource)
        resource_ngrams = [r for r in results if r.get('contains_resource', False) and r.get('code_word_score', 0) > 0.5]
        f.write(f"\n\nPotential Resource Name Usage (contains resource, score > 0.5): {len(resource_ngrams)}\n")
        f.write("-" * 80 + "\n")
        for r in resource_ngrams[:10]:
            f.write(f"\nN-gram: {r['ngram']}\n")
            f.write(f"  Contains Resources: {', '.join(r.get('found_resources', []))}\n")
            f.write(f"  Code Word Score: {r.get('code_word_score', 0):.2f}\n")
            f.write(f"  Avg Rating: {r.get('avg_rating', 0):.2f}\n")
            f.write(f"  Note: May be resource reference rather than code word\n")
        
        # Statistical summary
        f.write(f"\n\nStatistical Summary\n")
        f.write("-" * 80 + "\n")
        all_scores = [r.get('code_word_score', 0) for r in results]
        resource_scores = [r.get('code_word_score', 0) for r in results if r.get('contains_resource', False)]
        non_resource_scores = [r.get('code_word_score', 0) for r in results if not r.get('contains_resource', False)]
        
        f.write(f"Total n-grams analyzed: {len(results)}\n")
        f.write(f"  With resource names: {len(resource_scores)}\n")
        f.write(f"  Without resource names: {len(non_resource_scores)}\n")
        f.write(f"\nAverage Code Word Score:\n")
        f.write(f"  All: {sum(all_scores)/len(all_scores):.3f}\n")
        if resource_scores:
            f.write(f"  With resources: {sum(resource_scores)/len(resource_scores):.3f}\n")
        if non_resource_scores:
            f.write(f"  Without resources: {sum(non_resource_scores)/len(non_resource_scores):.3f}\n")
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - code_word_significance_analysis.csv")
    print(f"  - code_word_significance_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze code word significance")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("results/emergent_language"),
        help="Directory containing analysis CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/code_word_analysis"),
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    analyze_code_words(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

