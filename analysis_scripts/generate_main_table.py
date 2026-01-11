"""
Generate main table for paper based on existing data.

Analyzes all games and generates a comprehensive table with:
- Model (LLM type)
- Condition (balanced, reward_strong, penalty_strong)
- Invention hint (True/False)
- Language emergence metrics
- Coordination metrics
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# Resource types in the game
RESOURCE_TYPES = {"meat", "grain", "water", "fruit", "fish"}


def parse_game_id(game_id: str) -> Dict[str, str]:
    """Parse game_id to extract model, condition, and invention_hint."""
    # Pattern: resource_exchange_{model}_{invention_hint}_{condition}_{timestamp}
    # Example: resource_exchange_gemini-2.5-flash_False_balanced_20260101_032230
    
    parts = game_id.split('_')
    if len(parts) < 4:
        return {'model': 'unknown', 'invention_hint': 'unknown', 'condition': 'unknown'}
    
    # Find condition (balanced, reward_strong, penalty_strong)
    condition = None
    for cond in ['balanced', 'reward_strong', 'penalty_strong']:
        if cond in game_id:
            condition = cond
            break
    
    # Find invention_hint (True/False)
    invention_hint = None
    if '_True_' in game_id:
        invention_hint = 'True'
    elif '_False_' in game_id:
        invention_hint = 'False'
    
    # Extract model (everything before the first True/False)
    model = None
    if invention_hint:
        model_part = game_id.split(f'_{invention_hint}_')[0]
        model = model_part.replace('resource_exchange_', '')
    else:
        # Fallback: try to extract model name
        model_match = re.search(r'resource_exchange_([^_]+)', game_id)
        if model_match:
            model = model_match.group(1)
    
    return {
        'model': model or 'unknown',
        'invention_hint': invention_hint or 'unknown',
        'condition': condition or 'unknown'
    }


def get_vocab_meanings_for_group(messages_data: List[Dict]) -> Dict[str, str]:
    """Get vocabulary mappings from log files in this group."""
    vocab_meanings = {}
    for msg in messages_data[:10]:  # Sample to get vocab
        log_path = msg.get('log_path', '')
        if log_path:
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                vocab = data.get('vocabulary', {})
                if vocab:
                    vocab_meanings.update(vocab)
                    break  # Use first available vocab
            except:
                continue
    return vocab_meanings


def contains_resource_name(pattern: str, vocab_meanings: Dict[str, str]) -> bool:
    """Check if pattern contains resource names."""
    tokens = pattern.lower().split()
    
    # Check direct resource names
    for token in tokens:
        if token in RESOURCE_TYPES:
            return True
    
    # Check vocabulary mappings
    for token in tokens:
        meaning = vocab_meanings.get(token, "").lower()
        if meaning in RESOURCE_TYPES:
            return True
    
    return False


def calculate_code_word_score(ratings: List[float], teammate_ratings: List[float], 
                              opponent_ratings: List[float], count: int) -> float:
    """Calculate Code Word Score (0-1) based on multiple criteria."""
    if not ratings or count < 5:
        return 0.0
    
    score = 0.0
    
    # 1. Average rating (40%)
    avg_rating = sum(ratings) / len(ratings)
    if avg_rating > 3.5:
        score += 0.4
    
    # 2. Rating consistency (20%)
    if len(ratings) > 1:
        mean = avg_rating
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        std = variance ** 0.5
        consistency = 1.0 / (1.0 + std) if std > 0 else 1.0
        if consistency > 0.7:
            score += 0.2
    
    # 3. Discrimination score (20%)
    if teammate_ratings and opponent_ratings:
        teammate_avg = sum(teammate_ratings) / len(teammate_ratings)
        opponent_avg = sum(opponent_ratings) / len(opponent_ratings)
        discrimination = teammate_avg - opponent_avg
        if discrimination > 0.5:
            score += 0.2
    
    # 4. Sample size (20%)
    if count >= 5:
        score += 0.2
    
    return score


def calculate_code_words_optimized(group_tokens: Counter, token_ratings: Dict[str, List[float]],
                                   token_teammate_ratings: Dict[str, List[float]],
                                   token_opponent_ratings: Dict[str, List[float]],
                                   vocab_meanings: Dict[str, str],
                                   min_score: float = 0.7) -> int:
    """Count code words using Code Word Score, excluding resource names."""
    count = 0
    for token, usage_count in group_tokens.items():
        if usage_count < 5:
            continue
        
        # Check if contains resource name
        if contains_resource_name(token, vocab_meanings):
            continue
        
        ratings = token_ratings.get(token, [])
        teammate_ratings = token_teammate_ratings.get(token, [])
        opponent_ratings = token_opponent_ratings.get(token, [])
        
        code_word_score = calculate_code_word_score(
            ratings, teammate_ratings, opponent_ratings, usage_count
        )
        
        if code_word_score >= min_score:
            count += 1
    
    return count


def calculate_ngrams_optimized(group_ngrams: Counter, ngram_ratings: Dict[str, List[float]],
                               ngram_teammate_ratings: Dict[str, List[float]],
                               ngram_opponent_ratings: Dict[str, List[float]],
                               vocab_meanings: Dict[str, str],
                               min_score: float = 0.7) -> int:
    """Count n-grams using Code Word Score, excluding resource names."""
    count = 0
    for ngram, usage_count in group_ngrams.items():
        if usage_count < 5:
            continue
        
        # Check if contains resource name
        if contains_resource_name(ngram, vocab_meanings):
            continue
        
        ratings = ngram_ratings.get(ngram, [])
        teammate_ratings = ngram_teammate_ratings.get(ngram, [])
        opponent_ratings = ngram_opponent_ratings.get(ngram, [])
        
        code_word_score = calculate_code_word_score(
            ratings, teammate_ratings, opponent_ratings, usage_count
        )
        
        if code_word_score >= min_score:
            count += 1
    
    return count


def calculate_metrics_for_group(messages_data: List[Dict], tokens_data: List[Dict], 
                                ngrams_data: List[Dict]) -> Dict[str, float]:
    """Calculate all metrics for a group of games."""
    metrics = {}
    
    # Get vocabulary meanings for resource name checking
    vocab_meanings = get_vocab_meanings_for_group(messages_data)
    
    # Extract tokens and ngrams from messages in this group
    group_tokens = Counter()
    group_ngrams = Counter()
    token_ratings: Dict[str, List[float]] = defaultdict(list)
    ngram_ratings: Dict[str, List[float]] = defaultdict(list)
    token_teammate_ratings: Dict[str, List[float]] = defaultdict(list)
    token_opponent_ratings: Dict[str, List[float]] = defaultdict(list)
    ngram_teammate_ratings: Dict[str, List[float]] = defaultdict(list)
    ngram_opponent_ratings: Dict[str, List[float]] = defaultdict(list)
    
    for msg in messages_data:
        # Extract unknown tokens
        unknown_str = msg.get('unknown_tokens', '').strip()
        rating_str = msg.get('partner_rating', '').strip()
        is_teammate_gt = msg.get('is_teammate_gt', '').strip()
        rating = None
        if rating_str:
            try:
                rating = float(rating_str)
            except (ValueError, TypeError):
                pass
        
        if unknown_str:
            for token in unknown_str.split():
                group_tokens[token] += 1
                if rating is not None:
                    token_ratings[token].append(rating)
                    if is_teammate_gt == '1':
                        token_teammate_ratings[token].append(rating)
                    elif is_teammate_gt == '0':
                        token_opponent_ratings[token].append(rating)
        
        # Extract n-grams
        bigram = msg.get('tail_bigram', '').strip()
        trigram = msg.get('tail_trigram', '').strip()
        
        if bigram:
            group_ngrams[bigram] += 1
            if rating is not None:
                ngram_ratings[bigram].append(rating)
                if is_teammate_gt == '1':
                    ngram_teammate_ratings[bigram].append(rating)
                elif is_teammate_gt == '0':
                    ngram_opponent_ratings[bigram].append(rating)
        
        if trigram:
            group_ngrams[trigram] += 1
            if rating is not None:
                ngram_ratings[trigram].append(rating)
                if is_teammate_gt == '1':
                    ngram_teammate_ratings[trigram].append(rating)
                elif is_teammate_gt == '0':
                    ngram_opponent_ratings[trigram].append(rating)
    
    # Calculate code words using optimized method (Code Word Score >= 0.7, exclude resource names)
    code_word_count = calculate_code_words_optimized(
        group_tokens, token_ratings, token_teammate_ratings, 
        token_opponent_ratings, vocab_meanings, min_score=0.7
    )
    metrics['code_words'] = code_word_count
    
    # Calculate n-grams using optimized method (Code Word Score >= 0.7, exclude resource names)
    ngram_count = calculate_ngrams_optimized(
        group_ngrams, ngram_ratings, ngram_teammate_ratings,
        ngram_opponent_ratings, vocab_meanings, min_score=0.7
    )
    metrics['ngrams'] = ngram_count
    
    # Rating metrics
    ratings = []
    for msg in messages_data:
        rating_str = msg.get('partner_rating', '').strip()
        if rating_str:
            try:
                ratings.append(float(rating_str))
            except (ValueError, TypeError):
                continue
    
    if ratings:
        metrics['mean_rating'] = sum(ratings) / len(ratings)
        metrics['high_rating_rate'] = sum(1 for r in ratings if r >= 4) / len(ratings) * 100
    else:
        metrics['mean_rating'] = None
        metrics['high_rating_rate'] = None
    
    # Trade metrics
    success_count = sum(1 for msg in messages_data if msg.get('has_successful_trade') == '1')
    metrics['success_rate'] = success_count / len(messages_data) * 100 if messages_data else 0
    
    reciprocal_count = sum(1 for msg in messages_data if msg.get('is_reciprocal_trade') == '1')
    metrics['reciprocal_rate'] = reciprocal_count / len(messages_data) * 100 if messages_data else 0
    
    # Value metrics
    values = []
    for msg in messages_data:
        value_str = msg.get('total_value_transferred', '').strip()
        if value_str:
            try:
                val = float(value_str)
                if val > 0:
                    values.append(val)
            except (ValueError, TypeError):
                continue
    metrics['mean_value'] = sum(values) / len(values) if values else 0
    
    # Language complexity
    unknown_counts = []
    for msg in messages_data:
        unknown_str = msg.get('unknown_tokens', '').strip()
        if unknown_str:
            unknown_counts.append(len(unknown_str.split()))
        else:
            unknown_counts.append(0)
    metrics['mean_unknown_tokens'] = sum(unknown_counts) / len(unknown_counts) if unknown_counts else 0
    
    # N-gram diversity
    all_ngrams = set()
    for msg in messages_data:
        bigram = msg.get('tail_bigram', '').strip()
        trigram = msg.get('tail_trigram', '').strip()
        if bigram:
            all_ngrams.add(bigram)
        if trigram:
            all_ngrams.add(trigram)
    metrics['ngram_diversity'] = len(all_ngrams)
    
    # Innovation rate (simplified: total unknown tokens / number of rounds)
    total_unknown = sum(unknown_counts)
    rounds = set(msg.get('round', '') for msg in messages_data)
    num_rounds = len([r for r in rounds if r])
    metrics['innovation_rate'] = total_unknown / num_rounds if num_rounds > 0 else 0
    
    # Accuracy metrics (based on ground truth)
    teammate_cases = []  # (rating, is_teammate_gt)
    opponent_cases = []  # (rating, is_teammate_gt)
    
    for msg in messages_data:
        rating_str = msg.get('partner_rating', '').strip()
        gt_str = msg.get('is_teammate_gt', '').strip()
        
        if not rating_str or not gt_str:
            continue
        
        try:
            rating = float(rating_str)
            is_teammate_gt = int(gt_str) == 1
            
            if is_teammate_gt:
                teammate_cases.append(rating)
            else:
                opponent_cases.append(rating)
        except (ValueError, TypeError):
            continue
    
    # Teammate recognition accuracy: rating >= 3 when is_teammate_gt = True
    if teammate_cases:
        teammate_correct = sum(1 for r in teammate_cases if r >= 3)
        metrics['teammate_accuracy'] = teammate_correct / len(teammate_cases) * 100
    else:
        metrics['teammate_accuracy'] = None
    
    # Opponent recognition accuracy: rating < 3 when is_teammate_gt = False
    if opponent_cases:
        opponent_correct = sum(1 for r in opponent_cases if r < 3)
        metrics['opponent_accuracy'] = opponent_correct / len(opponent_cases) * 100
    else:
        metrics['opponent_accuracy'] = None
    
    return metrics


def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def generate_main_table(data_dir: Path, output_path: Path):
    """Generate main table from CSV data."""
    print("Loading data...")
    
    # Load CSVs
    messages_data = load_csv(data_dir / "messages.csv")
    tokens_data = load_csv(data_dir / "unknown_tokens.csv")
    ngrams_data = load_csv(data_dir / "tail_ngrams.csv")
    
    print(f"Loaded {len(messages_data)} messages, {len(tokens_data)} tokens, {len(ngrams_data)} n-grams")
    
    # Parse game IDs and group messages
    print("Parsing game IDs and grouping...")
    grouped_messages: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    
    for msg in messages_data:
        game_id = msg.get('game_id', '')
        info = parse_game_id(game_id)
        model = info['model']
        condition = info['condition']
        invention_hint = info['invention_hint']
        
        if model != 'unknown' and condition != 'unknown':
            key = (model, condition, invention_hint)
            grouped_messages[key].append(msg)
    
    # Calculate metrics for each group
    print("Calculating metrics by group...")
    results = []
    
    for (model, condition, invention_hint), group_messages in grouped_messages.items():
        # Get game IDs in this group
        game_ids = set(msg.get('game_id', '') for msg in group_messages)
        
        # For tokens and ngrams, we use all data (they're aggregated)
        # In a more sophisticated version, we could filter by game_id
        metrics = calculate_metrics_for_group(group_messages, tokens_data, ngrams_data)
        
        results.append({
            'Model': model,
            'Condition': condition,
            'Invention Hint': invention_hint,
            'N Games': len(game_ids),
            **metrics
        })
    
    # Sort results
    condition_order = {'balanced': 1, 'reward_strong': 2, 'penalty_strong': 3}
    results.sort(key=lambda x: (
        condition_order.get(x['Condition'], 999),
        x['Model'],
        x['Invention Hint']
    ))
    
    # Print table
    print("\n" + "="*100)
    print("MAIN TABLE FOR PAPER")
    print("="*100)
    print(f"{'Model':<25} {'Condition':<15} {'Hint':<6} {'Games':<6} {'Code':<6} {'N-gram':<7} {'Rating':<8} {'High%':<7} {'Success%':<9} {'Recip%':<8}")
    print("-" * 100)
    
    for r in results:
        model = r['Model'][:24]
        condition = r['Condition'][:14]
        hint = r['Invention Hint'][:5]
        games = str(r['N Games'])
        code_words = str(r['code_words'])
        ngrams = str(r['ngrams'])
        rating = f"{r['mean_rating']:.2f}" if r['mean_rating'] is not None else "N/A"
        high_rate = f"{r['high_rating_rate']:.1f}" if r['high_rating_rate'] is not None else "N/A"
        success = f"{r['success_rate']:.1f}"
        reciprocal = f"{r['reciprocal_rate']:.1f}"
        
        print(f"{model:<25} {condition:<15} {hint:<6} {games:<6} {code_words:<6} {ngrams:<7} {rating:<8} {high_rate:<7} {success:<9} {reciprocal:<8}")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nTable saved to: {output_path}")
    
    # Generate LaTeX table
    latex_path = output_path.with_suffix('.tex')
    generate_latex_table(results, latex_path)
    print(f"LaTeX table saved to: {latex_path}")
    
    return results


def generate_latex_table(results: List[Dict], output_path: Path):
    """Generate LaTeX table with booktabs and grouped by model and condition."""
    # Group results by model, then by condition
    grouped_by_model: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    
    for row in results:
        model = row['Model']
        condition = row['Condition']
        grouped_by_model[model][condition].append(row)
    
    # Sort models and conditions
    model_order = ['gemini-2.5-flash', 'gpt-5.2', 'deepseek-r1']
    condition_order = ['balanced', 'reward_strong', 'penalty_strong']
    
    # Generate LaTeX with booktabs
    latex = "\\begin{table*}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Language Emergence and Coordination Performance Across Models and Conditions}\n"
    latex += "\\label{tab:main_results}\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{lllcccccccc}\n"
    latex += "\\toprule\n"
    latex += "Model & Condition & Hint & Code & N-gram & Mean & High & Success & Reciprocal & Teammate & Opponent \\\\\n"
    latex += "      &           &      & Words & Count  & Rating & Rating & Rate & Rate & Acc & Acc \\\\\n"
    latex += "\\midrule\n"
    
    first_model = True
    for model_idx, model in enumerate(model_order):
        if model not in grouped_by_model:
            continue
        
        # Collect all rows for this model, sorted by condition
        model_rows = []
        for condition in condition_order:
            if condition in grouped_by_model[model]:
                # Sort by hint (False first, then True)
                condition_rows = sorted(
                    grouped_by_model[model][condition],
                    key=lambda x: (x['Invention Hint'] == 'True', x['Invention Hint'])
                )
                model_rows.extend(condition_rows)
        
        if not model_rows:
            continue
        
        model_display = model.replace('-', '--').replace('.', '.')
        model_row_count = len(model_rows)
        
        # Process rows for this model
        current_condition = None
        condition_row_start = 0
        
        for i, row in enumerate(model_rows):
            condition = row['Condition']
            hint = row['Invention Hint']
            code_words = str(row['code_words'])
            ngrams = str(row['ngrams'])
            mean_rating = f"{row['mean_rating']:.2f}" if row['mean_rating'] is not None else "N/A"
            high_rating = f"{row['high_rating_rate']:.1f}" if row['high_rating_rate'] is not None else "N/A"
            success = f"{row['success_rate']:.1f}"
            reciprocal = f"{row['reciprocal_rate']:.1f}"
            
            condition_display = condition.replace('_', ' ').title()
            hint_display = "Yes" if hint == "True" else "No"
            
            # Check if we need to start a new condition group
            if condition != current_condition:
                if current_condition is not None:
                    # End previous condition group with cmidrule
                    condition_rows_in_group = [r for r in model_rows[condition_row_start:i] 
                                              if r['Condition'] == current_condition]
                    if len(condition_rows_in_group) > 1:
                        latex += f"\\cmidrule(lr){{2-11}}\n"
                
                current_condition = condition
                condition_row_start = i
            
            # Model name (only show in first row of model group)
            if i == 0:
                if model_row_count > 1:
                    latex += f"\\multirow{{{model_row_count}}}{{*}}{{{model_display}}} & "
                else:
                    latex += f"{model_display} & "
            else:
                latex += " & "
            
            # Condition name (only show in first row of condition group within model)
            condition_rows = [r for r in model_rows if r['Condition'] == condition]
            condition_first_idx = next(j for j, r in enumerate(model_rows) if r['Condition'] == condition)
            
            if i == condition_first_idx:
                condition_row_count = len(condition_rows)
                if condition_row_count > 1:
                    latex += f"\\multirow{{{condition_row_count}}}{{*}}{{{condition_display}}} & "
                else:
                    latex += f"{condition_display} & "
            else:
                latex += " & "
            
            # Accuracy columns
            teammate_acc = f"{row['teammate_accuracy']:.1f}" if row.get('teammate_accuracy') is not None else "N/A"
            opponent_acc = f"{row['opponent_accuracy']:.1f}" if row.get('opponent_accuracy') is not None else "N/A"
            
            # Data columns
            latex += f"{hint_display} & {code_words} & {ngrams} & {mean_rating} & {high_rating} & {success} & {reciprocal} & {teammate_acc} & {opponent_acc} \\\\\n"
        
        # End last condition group in this model (if it has multiple rows)
        if current_condition:
            condition_rows_in_group = [r for r in model_rows[condition_row_start:] 
                                      if r['Condition'] == current_condition]
            if len(condition_rows_in_group) > 1:
                latex += f"\\cmidrule(lr){{2-11}}\n"
        
        # End model group with cmidrule (except for last model)
        # Count how many models we actually have
        actual_models = [m for m in model_order if m in grouped_by_model]
        if model_idx < len(actual_models) - 1:
            latex += "\\cmidrule(lr){1-11}\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "}\n"
    latex += "\\begin{flushleft}\n"
    latex += "\\footnotesize\n"
    latex += "\\textbf{Definitions:} Hint: Invention hint condition (No=strong default prompt encouraging token invention, Yes=soft prompt or custom hint). "
    latex += "Code Words: unknown tokens with Code Word Score ≥0.7, excluding resource names (meat, grain, water, fruit, fish). "
    latex += "N-grams: tail n-grams with Code Word Score ≥0.7, excluding resource names. "
    latex += "Code Word Score combines: avg\\_rating >3.5 (40\\%), rating consistency >0.7 (20\\%), discrimination score >0.5 (20\\%), and sample size ≥5 (20\\%). "
    latex += "Mean Rating: average partner rating (1=not teammate, 4=teammate). "
    latex += "High Rating Rate: proportion of ratings ≥4. "
    latex += "Success Rate: proportion of rounds with successful trades. "
    latex += "Reciprocal Rate: proportion of rounds with bidirectional successful exchanges. "
    latex += "Teammate Acc: accuracy of identifying teammates (rating ≥3 when is\\_teammate=True). "
    latex += "Opponent Acc: accuracy of identifying opponents (rating <3 when is\\_teammate=False). \\\\\n"
    latex += "\\textbf{Notes:} Code Words and N-grams use a composite significance score (0-1) that considers rating, consistency, discrimination, and sample size, and excludes patterns containing resource names. "
    latex += "Zero values indicate that while agents attempted linguistic innovation, these innovations did not meet the strict effectiveness threshold. "
    latex += "Pairing is random (50\\% teammate, 50\\% opponent), so High Rating Rate should be interpreted alongside accuracy metrics. "
    latex += "All metrics verified against ground truth from raw JSON logs.\n"
    latex += "\\end{flushleft}\n"
    latex += "\\end{table*}\n"
    
    output_path.write_text(latex)
    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate main table for paper from analyzed data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="results/emergent_language",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/main_table.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    
    generate_main_table(data_dir, output_path)


if __name__ == "__main__":
    main()

